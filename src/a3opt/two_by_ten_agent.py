from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _jobs, _npz, _record_once
from a3opt.backends.advanced_trajectory import fit_pose
from a3opt.backends.two_by_ten import COUNTS, RECIPES, SOURCE_COLUMN, TwoByTenPolicy
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, PROFILES
from a3opt.refinement_eval import read_json, sha256, simulator_identity, wilson_interval
from a3opt.two_by_ten_sim import COLLECT_STEPS, EVAL_RULES, SHORT_STEPS, check_collection, check_evaluation, collect, evaluate


MIN_PER_COUNT = 2
EVAL_REPEATS_PER_COUNT = 2
MIN_GAIN_PER_EPISODE = .5


@dataclass(frozen=True)
class Budget:
    seed_block: int = 82000
    initial_repeats: int = 1
    feedback_attempts: int = 30

    @property
    def initial_attempts(self) -> int:
        return self.initial_repeats * len(COUNTS) * len(PROFILES)

    def __post_init__(self) -> None:
        if (self.seed_block < 80000 or self.seed_block % 1000 or not 1 <= self.initial_repeats <= 5
                or not 0 <= self.feedback_attempts <= 150):
            raise ValueError("use a fresh 1000-aligned seed block >= 80000, 1–5 initial repeats and 0–150 feedback attempts")

    def evaluation_seeds(self, split: str) -> list[int]:
        from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan

        offset = {"development": 600, "confirmation": 800}[split]
        counts = Counter()
        selected = []
        for seed in range(self.seed_block + offset, self.seed_block + offset + 200):
            first_grasp = Cookie2x10Plan.from_seed("random", seed).first_grasp
            if counts[first_grasp] < EVAL_REPEATS_PER_COUNT:
                selected.append(seed)
                counts[first_grasp] += 1
        if len(selected) != len(COUNTS) * EVAL_REPEATS_PER_COUNT:
            raise ValueError(f"seed block does not contain a balanced {split} split")
        return selected


def initial_plan(budget: Budget) -> list[dict]:
    strata = [(first_grasp, profile) for first_grasp in COUNTS for profile in PROFILES] * budget.initial_repeats
    return [{"seed": budget.seed_block + index, "first_grasp": count, "profile": profile, "reason": "initial_coverage"}
            for index, (count, profile) in enumerate(strata)]


def uniform_plan(budget: Budget) -> list[dict]:
    strata = [(count, PROFILES[(count + shift) % len(PROFILES)]) for shift in range(len(PROFILES)) for count in COUNTS]
    return [{"seed": budget.seed_block + 400 + index, "first_grasp": count, "profile": profile, "reason": "uniform"}
            for index, (count, profile) in enumerate(strata * 5) if index < budget.feedback_attempts]


def feedback_plan(budget: Budget, attempts: list[dict], development: list[dict]) -> list[dict]:
    accepted = Counter(row["first_grasp"] for row in attempts if row["accepted"])
    exposures = Counter((row["first_grasp"], row["profile"]) for row in attempts)
    deficit = {count: max(0, MIN_PER_COUNT - accepted[count]) for count in COUNTS}
    error = {count: np.mean(losses) if (losses := [20 - row["requested_score"] for row in development
                                                   if row["first_grasp"] == count]) else 0. for count in COUNTS}
    allocated = Counter()
    plan = []
    for index in range(budget.feedback_attempts):
        count = max(COUNTS, key=lambda value: (max(0, deficit[value] - allocated[value]),
                                                error[value] / (1 + allocated[value]),
                                                -accepted[value] - allocated[value]))
        need_coverage = deficit[count] > allocated[count]
        order = PROFILES if need_coverage else tuple(reversed(PROFILES))
        profile = min(order, key=lambda name: exposures[count, name])
        exposures[count, profile] += 1
        allocated[count] += 1
        reason = "missing_demonstrations" if need_coverage else ("development_shortfall" if development else "coverage_only")
        plan.append({"seed": budget.seed_block + 200 + index, "first_grasp": count,
                     "profile": profile, "reason": reason, "accepted_before_round": accepted[count],
                     "development_shortfall": float(error[count]) if development else None})
    return plan


def collect_plan(pool, root: Path, name: str, plan: list[dict]) -> list[dict]:
    _record_once(root / name / "sampling-plan.json", {"attempts": plan})
    shards = root / "shards"
    jobs = [(row["profile"], row["seed"], row["first_grasp"], shards) for row in plan]
    return _jobs(pool, collect, jobs, root / "training", lambda job: str(job[1]), check_collection)


def fit_candidates(root: Path, directory: Path, attempts: list[dict]) -> dict[str, Path] | None:
    counts = Counter(row["first_grasp"] for row in attempts if row["accepted"])
    manifest = {"counts": {str(count): counts[count] for count in COUNTS}, "attempts": attempts}
    dataset_path = directory / "dataset.json"
    _record_once(dataset_path, manifest)
    if any(counts[count] < MIN_PER_COUNT for count in COUNTS):
        return None
    paths = {recipe: directory / f"{recipe}.npz" for recipe in RECIPES}
    fit_path = directory / "fit.json"
    dataset_digest = sha256(dataset_path)
    if fit_path.exists():
        fit = read_json(fit_path)
        if fit["dataset_sha256"] != dataset_digest or fit["checkpoints"] != {
            recipe: sha256(path) for recipe, path in paths.items()
        }:
            raise ValueError("2x10 fit or training manifest changed on resume")
        return paths
    data, accepted = [], []
    for row in attempts:
        with np.load(root / "shards" / f"{row['seed']}.npz", allow_pickle=False) as archive:
            values = dict(archive)
        data.append(values)
        if row["accepted"]:
            accepted.append(values)
    visual, diagnostics = fit_pose(*(np.stack([sample[key] for sample in data])
                                    for key in ("outline", "silhouette", "pose", "seed")))
    common = {"format": TwoByTenPolicy.name, "counts": np.array([sample["first_grasp"] for sample in accepted])}
    common.update({name: np.stack([sample[key] for sample in accepted])
                   for name, key in (("traces", "traces"), ("lengths", "lengths"), ("poses", "pose"))})
    for recipe, path in paths.items():
        _npz(path, recipe=recipe, **common, **(visual if recipe == "visual" else {}))
    _record_once(fit_path, {"dataset_sha256": dataset_digest, "calibration": diagnostics,
                           "checkpoints": {recipe: sha256(path) for recipe, path in paths.items()}})
    return paths


def stratified_ci(rows: list[dict], values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    counts = np.asarray([row["first_grasp"] for row in rows])
    totals = np.zeros(BOOTSTRAP_SAMPLES)
    for count in sorted(set(counts)):
        values_at_count = values[counts == count]
        totals += rng.choice(values_at_count, (BOOTSTRAP_SAMPLES, len(values_at_count)), replace=True).sum(axis=1)
    return np.quantile(totals / len(values), [.025, .975]).tolist()


def score(rows: list[dict]) -> dict:
    scores = np.asarray([row["requested_score"] for row in rows])
    successes = sum(row["success"] and row["requested_score"] == 20 for row in rows)
    return {"episodes": len(rows), "mean_score": float(np.mean([row["score"] for row in rows])),
            "mean_requested_score": float(scores.mean()), "requested_score_ci95": stratified_ci(rows, scores),
            "upstream_successes": sum(row["success"] for row in rows), "requested_successes": successes,
            "success_ci95": wilson_interval(successes, len(rows)),
            "wrong_source_episodes": sum(row["score"] != row["requested_score"] for row in rows),
            "safety_stops": sum(bool(row["safety_reason"]) for row in rows),
            "first_grasp_counts": dict(sorted(Counter(row["first_grasp"] for row in rows).items())),
            "mean_planned_steps": float(np.mean([row["planned_steps"] for row in rows]))}


def compare(reference: list[dict], candidate: list[dict]) -> dict:
    changes = []
    safety_regressions = 0
    for baseline, current in zip(reference, candidate, strict=True):
        if any(baseline[key] != current[key] for key in ("seed", "profile", "task_prompt", "scene_sha256")):
            raise ValueError("2x10 policies were evaluated on different scenarios")
        changes.append(current["requested_score"] - baseline["requested_score"])
        safety_regressions += bool(current["safety_reason"] and not baseline["safety_reason"])
    baseline_stats, candidate_stats = score(reference), score(candidate)
    return {"reference": baseline_stats, "candidate": candidate_stats, "mean_paired_gain": float(np.mean(changes)),
            "paired_gain_ci95": stratified_ci(reference, np.asarray(changes)), "new_safety_stops": safety_regressions,
            "success_delta": candidate_stats["requested_successes"] - baseline_stats["requested_successes"],
            "gain_count": sum(delta > 0 for delta in changes), "loss_count": sum(delta < 0 for delta in changes)}


def admissible(result: dict) -> bool:
    return not (result["wrong_source_episodes"] or result["safety_stops"])


def choose_recipe(comparison: dict) -> str | None:
    reference_ok = admissible(comparison["reference"])
    visual_ok = admissible(comparison["candidate"])
    if visual_ok and (not reference_ok or (comparison["success_delta"] >= 0
                                           and comparison["mean_paired_gain"] >= MIN_GAIN_PER_EPISODE)):
        return "visual"
    return "mean" if reference_ok else None


def evaluate_split(pool, directory: Path, checkpoint: Path, profiles: tuple[str, ...], seeds: list[int]) -> dict:
    jobs = [(checkpoint, profile, seed) for profile in profiles for seed in seeds]
    digest = sha256(checkpoint)
    _record_once(directory / "request.json", {"checkpoint_sha256": digest, "profiles": list(profiles), "seeds": seeds})
    rows = _jobs(pool, evaluate, jobs, directory / "episodes", lambda job: f"{job[1]}/{job[2]}", check_evaluation)
    if sha256(checkpoint) != digest:
        raise ValueError("2x10 checkpoint changed during evaluation")
    return {profile: [row for row in rows if row["profile"] == profile] for profile in profiles}


def make_protocol(budget: Budget) -> dict:
    from a3_dual_arm_sim.paths import resource_root

    source = Path(__file__).parent
    code_paths = ("two_by_ten_agent.py", "two_by_ten_sim.py", "advanced_agent.py", "advanced_sim.py",
                  "difficulty_eval.py", "refinement_eval.py", "backends/two_by_ten.py", "backends/trajectory.py",
                  "backends/advanced_phase.py", "backends/advanced_trajectory.py")
    return {"budget": asdict(budget), "rules": EVAL_RULES, "collection_max_steps": COLLECT_STEPS,
            "short_horizon_steps": SHORT_STEPS, "acquisition_arms": ["initial", "initial+feedback", "initial+uniform"],
            "evaluation_repeats_per_count": EVAL_REPEATS_PER_COUNT,
            "success_interval": "descriptive_binomial_wilson", "score_interval": "bootstrap_within_grasp_count",
            "source_column": SOURCE_COLUMN, "profiles": list(PROFILES), "minimum_demonstrations_per_count": MIN_PER_COUNT,
            "minimum_development_gain": MIN_GAIN_PER_EPISODE, "environment": simulator_identity(),
            "opencv_version": importlib.metadata.version("opencv-python"),
            "configs": {name: sha256(resource_root() / "configs" / name)
                        for name in ("cookie_2x10.yaml", "randomization_2x10.yaml")},
            "code": {name: sha256(source / name) for name in code_paths},
            "development_seeds": budget.evaluation_seeds("development"),
            "confirmation_seeds": budget.evaluation_seeds("confirmation")}


def acquisition_summary(attempts: list[dict]) -> dict:
    return {"attempts": len(attempts), "accepted": sum(row["accepted"] for row in attempts),
            "by_count": [{"count": count, "profile": profile, "attempts": len(group),
                          "accepted": sum(row["accepted"] for row in group),
                          "expert_full": sum(row["success"] for row in group),
                          "best_requested_score": max((row["requested_score"] for row in group), default=None)}
                         for count in COUNTS for profile in PROFILES
                         for group in [[row for row in attempts if row["first_grasp"] == count and row["profile"] == profile]]],
            "rejections": dict(Counter(row["rejection"] for row in attempts if not row["accepted"]))}


def develop(pool, root: Path, arm: str, attempts: list[dict], seeds: list[int]) -> dict:
    directory = root / arm
    candidates = fit_candidates(root, directory, attempts)
    if candidates is None:
        decision = {"status": "insufficient_training_coverage"}
        rows = []
    else:
        results = {recipe: evaluate_split(pool, directory / "development" / recipe, checkpoint,
                                          ("advanced",), seeds)["advanced"]
                   for recipe, checkpoint in candidates.items()}
        comparison = compare(results["mean"], results["visual"])
        winner = choose_recipe(comparison)
        rows = results[winner] if winner else []
        decision = {"status": "eligible" if winner else "development_rejected", "recipe": winner,
                    "comparison": comparison, "score": score(rows) if rows else None}
    _record_once(directory / "development-decision.json", decision)
    print(f"{arm}: {decision['status']}", flush=True)
    return {"decision": decision, "candidates": candidates, "rows": rows}


def horizon_rows(rows: list[dict]) -> list[dict]:
    return [{**row, **row["at_500"]} for row in rows]


def confirm(pool, root: Path, protocol: dict, arms: dict, name: str) -> dict:
    development = arms[name]["decision"]
    chosen_recipe = development["recipe"]
    chosen_label = f"{name}/{chosen_recipe}"
    chosen_path = arms[name]["candidates"][chosen_recipe]
    required = {chosen_label: chosen_path}
    other_recipe = next(recipe for recipe in RECIPES if recipe != chosen_recipe)
    required[f"{name}/{other_recipe}"] = arms[name]["candidates"][other_recipe]
    pair = ("feedback", "uniform")
    if all(arm in arms and arms[arm]["candidates"] for arm in pair):
        required.update({f"{arm}/{chosen_recipe}": arms[arm]["candidates"][chosen_recipe] for arm in pair})

    lock = {"candidate": chosen_label, "protocol_sha256": sha256(root / "protocol.json"),
            "development_decisions": {arm: sha256(root / arm / "development-decision.json") for arm in arms},
            "checkpoints": {label: sha256(path) for label, path in required.items()}}
    lock_path = root / "confirmation-lock.json"
    _record_once(lock_path, lock)
    print(f"confirmation: {len(required)} locked checkpoints", flush=True)
    results = {label: evaluate_split(pool, root / "confirmation" / label, checkpoint, PROFILES,
                                     protocol["confirmation_seeds"]) for label, checkpoint in required.items()}
    cells = {label: {profile: {"native": score(rows), "at_500": score(horizon_rows(rows))}
                     for profile, rows in profiles.items()} for label, profiles in results.items()}
    mean_label, visual_label = (f"{name}/{recipe}" for recipe in RECIPES)
    contrasts = {"same_data_visual_minus_mean": {profile: compare(results[mean_label][profile], results[visual_label][profile])
                                                for profile in PROFILES}}
    if all(f"{arm}/{chosen_recipe}" in results for arm in pair):
        contrasts["equal_attempt_feedback_minus_uniform"] = {
            profile: compare(results[f"uniform/{chosen_recipe}"][profile], results[f"feedback/{chosen_recipe}"][profile])
            for profile in PROFILES}
    accepted = all(admissible(cell["native"]) and admissible(cell["at_500"]) for cell in cells[chosen_label].values())
    if chosen_recipe == "visual":
        visual_test = contrasts["same_data_visual_minus_mean"]
        accepted = accepted and visual_test["advanced"]["paired_gain_ci95"][0] > 0 and all(
            cell["mean_paired_gain"] >= 0 and cell["success_delta"] >= 0 for cell in visual_test.values())
    return {"status": "confirmation_passed" if accepted else "confirmation_rejected", "candidate": chosen_label,
            "selected_checkpoint": str(chosen_path) if accepted else None, "confirmation_lock_sha256": sha256(lock_path),
            "cells": cells, "paired_contrasts": contrasts}


def run_agent(root: Path, budget: Budget, workers: int) -> dict:
    protocol = make_protocol(budget)
    protocol_path = root / "protocol.json"
    _record_once(protocol_path, protocol)
    arms = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        plan = initial_plan(budget)
        print(f"initial: {len(plan)} scheduled expert attempts", flush=True)
        initial = collect_plan(pool, root, "initial", plan)
        attempts_by_arm = {"initial": initial}
        arms["initial"] = develop(pool, root, "initial", initial, protocol["development_seeds"])
        if budget.feedback_attempts:
            plans = {"feedback": feedback_plan(budget, initial, arms["initial"]["rows"]), "uniform": uniform_plan(budget)}
            for name, plan in plans.items():
                print(f"{name}: {len(plan)} scheduled expert attempts", flush=True)
                attempts_by_arm[name] = collect_plan(pool, root, name, plan)
                arms[name] = develop(pool, root, name, initial + attempts_by_arm[name], protocol["development_seeds"])
        eligible = [name for name, arm in arms.items() if arm["decision"]["status"] == "eligible"]
        if eligible:
            chosen = max(eligible, key=lambda name: (arms[name]["decision"]["score"]["mean_requested_score"],
                                                     arms[name]["decision"]["score"]["requested_successes"]))
            result = confirm(pool, root, protocol, arms, chosen)
        else:
            fitted = any(arm["candidates"] for arm in arms.values())
            result = {"status": "development_rejected" if fitted else "insufficient_training_coverage"}
        result.update(expert_attempts=sum(len(rows) for rows in attempts_by_arm.values()),
                      acquisition={name: acquisition_summary(rows) for name, rows in attempts_by_arm.items()},
                      development={name: arm["decision"] for name, arm in arms.items()})
        _record_once(root / "result.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous acquisition, fitting and paired 2x10 evaluation.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=Budget.seed_block)
    parser.add_argument("--initial-repeats", type=int, default=Budget.initial_repeats)
    parser.add_argument("--feedback-attempts", type=int, default=Budget.feedback_attempts)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan", action="store_true", help="Print the locked protocol and initial acquisition plan without running")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    budget = Budget(args.seed_block, args.initial_repeats, args.feedback_attempts)
    if args.plan:
        result = {"protocol": make_protocol(budget), "initial": initial_plan(budget), "uniform": uniform_plan(budget)}
    else:
        with _run_lock(args.out.resolve()):
            result = run_agent(args.out.resolve(), budget, args.workers)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
