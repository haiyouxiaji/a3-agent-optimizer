from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple, replace
from pathlib import Path

import numpy as np

from a3opt.adaptive_fit import fit_grip, rank_training, select_columns
from a3opt.adaptive_sim import adaptive_episode, check_adaptive_row, phase_model
from a3opt.advanced_agent import MILESTONE, SUITES, TARGET, _jobs, _npz, _record_once, summarize
from a3opt.backends.adaptive_phase import Recipe
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES, _bootstrap
from a3opt.refinement_eval import read_json, sha256, simulator_identity, write_json


GRIP_RECIPES = {
    "baseline": Recipe(), "transit": Recipe(transit_scale=.80),
    "source-x-minus": Recipe(source_dx=-.006), "source-x-plus": Recipe(source_dx=.006),
    "source-y-minus": Recipe(source_dy=-.006), "source-y-plus": Recipe(source_dy=.006),
    "close": Recipe(close_bias=.03), "transit-close": Recipe(transit_scale=.80, close_bias=.03),
}
PLACEMENT_RECIPES = {
    "baseline": Recipe(), "joint-placement": Recipe(joint_placement=True), "settle": Recipe(settle_steps=12),
    "joint-settle": Recipe(joint_placement=True, settle_steps=12),
    "target-x-minus": Recipe(target_dx=-.007), "target-x-plus": Recipe(target_dx=.007),
    "target-y-minus": Recipe(target_dy=-.007), "target-y-plus": Recipe(target_dy=.007),
}
RECIPE_FAMILIES = {"grip": GRIP_RECIPES, "placement": PLACEMENT_RECIPES}
COLUMNS = (1, 2, 3, 4)


def seed_schedule(block: int) -> dict:
    if block < 10000 or block % 4000:
        raise ValueError("expected a new 4000-seed block")
    return {
        "scout": list(range(block, block + 12)),
        "focus": list(range(block + 128, block + 144)),
        "focus_extra": list(range(block + 256, block + 260)),
        "development": {"1": list(range(block + 2000, block + 2012)),
                        "random": list(range(block + 2100, block + 2116))},
        "confirmation": {"1": list(range(block + 2400, block + 2420)),
                         "random": list(range(block + 2500, block + 2520))},
    }


def _training_episode(checkpoint: Path, profile: str, suite: str, seed: int, name: str, recipe: Recipe) -> dict:
    return {**adaptive_episode(checkpoint, profile, suite, seed, recipe), "training_recipe": name}


def _check_training_episode(job: tuple, row: dict) -> None:
    check_adaptive_row(job, row)
    if row["training_recipe"] != job[4]:
        raise ValueError("training evidence does not match the recipe")


def _run_samples(pool, reference: Path, directory: Path, plan: dict[str, list[tuple[str, int]]], recipes: dict) -> dict:
    _record_once(directory / "request.json", {"reference_sha256": sha256(reference),
                 "samples": {key: [list(item) for item in values] for key, values in plan.items()}})
    jobs = [(reference, "advanced", suite, seed, name, recipes[name])
            for name, inputs in plan.items() for suite, seed in inputs]
    rows = _jobs(pool, _training_episode, jobs, directory, lambda job: f"{job[4]}/{job[3]}", _check_training_episode)
    result = {name: [row for row in rows if row["training_recipe"] == name] for name in plan}
    print(f"{directory.name}: {len(rows)} first-batch episodes", flush=True)
    return result


def _focus_plan(scout: dict, balanced: list[int], extra: list[int]) -> tuple[dict, dict]:
    from a3_dual_arm_sim.sim.randomization import choose_column

    shortlisted = {column: [name for name in rank_training(scout, column) if name != "baseline"][:2]
                   for column in COLUMNS}
    risk = {column: sum(5 - min(5, row["requested_score"]) for row in scout["baseline"] if row["column"] == column)
            for column in COLUMNS}
    weakest = max(COLUMNS, key=risk.get)
    schedule = [("random", seed, choose_column("random", seed)) for seed in balanced]
    schedule.extend((str(weakest), seed, weakest) for seed in extra)
    plan: dict[str, list[tuple[str, int]]] = {}
    for suite, seed, column in schedule:
        for name in ("baseline", *shortlisted[column]):
            plan.setdefault(name, []).append((suite, seed))
    return plan, {"weakest_column": weakest, "baseline_shortfall": {str(key): value for key, value in risk.items()},
                  "shortlist": {str(key): names for key, names in shortlisted.items()}}


def _grip_samples(waves: tuple[tuple[str, dict], ...]) -> list[dict]:
    samples = []
    for wave, rows in waves:
        for name, group in rows.items():
            for row in group:
                event = row["batches"][0]
                if row["safety_reason"] or not {"grip_features", "lifted_requested", "lifted_other"} <= event.keys():
                    continue
                samples.append({"seed": row["seed"], "suite": row["suite"], "column": row["column"], "recipe": name,
                                "row": f"{wave}/{name}/{row['seed']}.json", "action_sha256": row["action_sha256"],
                                "features": event["grip_features"],
                                "complete_grasp": event["lifted_requested"] == 5 and event["lifted_other"] == 0})
    return samples


def _checkpoint(path: Path, reference: Path, recipes: tuple[Recipe, ...], detector: dict) -> Path:
    manifest = path.with_suffix(".json")
    if not manifest.exists():
        _npz(path, **phase_model(reference), adaptive_recipes=np.asarray([astuple(recipe) for recipe in recipes]),
             **{f"grip_{name}": value for name, value in detector.items()})
    _record_once(manifest, {"reference_sha256": sha256(reference), "checkpoint_sha256": sha256(path),
                            "recipes": [list(astuple(recipe)) for recipe in recipes]})
    return path


def _evaluate(pool, checkpoint: Path, directory: Path, schedule: dict, profile: str = "advanced") -> list[dict]:
    _record_once(directory / "request.json", {"checkpoint_sha256": sha256(checkpoint), "seeds": schedule,
                                               "profile": profile})
    jobs = [(checkpoint, profile, suite, seed) for suite in schedule for seed in schedule[suite]]
    rows = _jobs(pool, adaptive_episode, jobs, directory, lambda job: f"{job[2]}/{job[3]}", check_adaptive_row)
    _record_once(directory / "results.json", {"rows": rows})
    print(f"{directory.parent.name}/{directory.name}: {len(rows)} full episodes", flush=True)
    return rows


def _release_observations(group: list[dict]) -> dict:
    first = [row["batches"][0] for row in group]
    reached_five = [row["first_requested_five"] for row in group if row["first_requested_five"] is not None]
    return {"episodes": len(group), "lifted_five": sum(event.get("lifted_requested") == 5 for event in first),
            "retained_five_through_transfer": sum(event.get("retained_through_transfer") == 5 for event in first),
            "released_five": sum(event.get("requested_after_release", 0) >= 5 for event in first),
            "five_in_target_region_after_retract": sum(event.get("released_in_target_region") == 5 for event in first),
            "median_open_step": float(np.median([event["open_start"] for event in first if "open_start" in event]))
            if any("open_start" in event for event in first) else None,
            "reached_requested_five": len(reached_five),
            "median_first_requested_five_step": float(np.median(reached_five)) if reached_five else None,
            "retries": sum(sum(row["retries"]) for row in group),
            "wrong_column_items": sum(row["score"] - row["requested_score"] for row in group),
            "safety_stops": sum(bool(row["safety_reason"]) for row in group)}


def paired_summary(baseline: list[dict], selected: list[dict]) -> dict:
    output = {}
    for suite in SUITES:
        base = {row["seed"]: row for row in baseline if row["suite"] == suite}
        choice = {row["seed"]: row for row in selected if row["suite"] == suite}
        if not base or list(base) != list(choice) or any(base[seed]["column"] != choice[seed]["column"] for seed in base):
            raise ValueError("paired results require identical ordered evaluation scenes")
        columns = [row["column"] for row in base.values()]
        differences = np.asarray([choice[seed]["requested_score"] - base[seed]["requested_score"] for seed in base])
        subset = lambda rows, column: [row for row in rows.values() if row["column"] == column]
        output[suite] = {"baseline": summarize(list(base.values())), "selected": summarize(list(choice.values())),
                         "paired_difference": float(differences.mean()), "paired_bootstrap_95": _bootstrap(differences, columns),
                         "first_batch_by_column": {str(column): {"baseline": _release_observations(subset(base, column)),
                                                       "selected": _release_observations(subset(choice, column))}
                                                   for column in sorted(set(columns))}}
        for label in ("baseline", "selected"):
            for field in ("by_requested_column", "requested_distribution"):
                output[suite][label][field] = {str(key): value for key, value in output[suite][label][field].items()}
    return output


def run(out: Path, reference: Path, seed_block: int, workers: int, family: str = "placement") -> dict:
    out, reference = out.resolve(), reference.resolve()
    schedule = seed_schedule(seed_block)
    recipes = RECIPE_FAMILIES[family]
    locked = read_json(reference / "confirmation-lock.json")
    old_protocol = reference / "protocol.json"
    checkpoint = Path(locked["selected"]["checkpoint"])
    if (sha256(old_protocol) != locked["protocol_sha256"] or sha256(checkpoint) != locked["checkpoint_sha256"]
            or seed_block == read_json(old_protocol)["budget"]["seed_block"]):
        raise ValueError("the reference must be intact and use a different seed block")
    package = Path(__file__).parent
    code_files = ("adaptive_agent.py", "adaptive_fit.py", "adaptive_sim.py", "advanced_agent.py", "advanced_sim.py",
                  "difficulty_eval.py", "refinement_eval.py", "backends/adaptive_phase.py", "backends/advanced_phase.py",
                  "backends/advanced_trajectory.py", "backends/trajectory.py")
    protocol = {"environment": simulator_identity(), "rules": RULES, "profile": "advanced", "schedule": schedule,
                "reference_sha256": sha256(checkpoint), "reference_protocol_sha256": sha256(old_protocol),
                "family": family, "recipes": {name: list(astuple(recipe)) for name, recipe in recipes.items()},
                "code_sha256": {path: sha256(package / path) for path in code_files}}
    _record_once(out / "protocol.json", protocol)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scout_plan = {name: [("random", seed) for seed in schedule["scout"]] for name in recipes}
        scout = _run_samples(pool, checkpoint, out / "scout", scout_plan, recipes)
        focus_plan, diagnosis = _focus_plan(scout, schedule["focus"], schedule["focus_extra"])
        _record_once(out / "training-focus.json", diagnosis)
        print(f"Agent prioritized source column {diagnosis['weakest_column']}", flush=True)
        focus = _run_samples(pool, checkpoint, out / "focus", focus_plan, recipes)
        samples = _grip_samples((("scout", scout), ("focus", focus)))
        _record_once(out / "grip-dataset.json", {"samples": samples})
        detector, validation = fit_grip(samples)
        _record_once(out / "grip-validation.json", validation)
        by_placement = tuple(recipes[rank_training(focus, column)[0]] for column in COLUMNS)
        by_lift = tuple(recipes[rank_training(focus, column, prioritize_lift=True)[0]] for column in COLUMNS)
        alternatives = {"baseline": (Recipe(),) * 4}
        for name, recipes in (("placement", by_placement), ("lift", by_lift), ("feedback", tuple(
                replace(recipe, retry_probability=validation["retry_probability"]) for recipe in by_placement))):
            if recipes not in alternatives.values():
                alternatives[name] = recipes
        checkpoints, evidence = {}, {}
        for name, recipes in alternatives.items():
            candidate_dir = out / "candidates" / name
            checkpoints[name] = _checkpoint(candidate_dir / "policy.npz", checkpoint, recipes, detector)
            evidence[name] = _evaluate(pool, checkpoints[name], candidate_dir / "development", schedule["development"])
        selected = select_columns(evidence)
        selected_recipes = tuple(alternatives[name][index] for index, name in enumerate(selected))
        baseline = evidence["baseline"]
        selected_rows = [evidence[selected[row["column"] - 1]][index] for index, row in enumerate(baseline)]
        result = {"status": "baseline_retained", "selected_by_column": selected,
                  "candidate_recipes": {name: [list(astuple(recipe)) for recipe in recipes]
                                        for name, recipes in alternatives.items()},
                  "training_episodes": sum(len(rows) for rows in (*scout.values(), *focus.values())),
                  "grip_samples": len(samples), "development": paired_summary(baseline, selected_rows)}
        _record_once(out / "development-selection.json", result)
        if selected_recipes != alternatives["baseline"]:
            winner = _checkpoint(out / "selected" / "policy.npz", checkpoint, selected_recipes, detector)
            _record_once(out / "confirmation-lock.json", {"checkpoint_sha256": sha256(winner),
                         "reference_checkpoint_sha256": sha256(checkpoints["baseline"]),
                         "protocol_sha256": sha256(out / "protocol.json"), "selected_by_column": selected,
                         "development_sha256": sha256(out / "development-selection.json"),
                         "seeds": schedule["confirmation"]})
            print("Agent froze the promoted checkpoint before confirmation", flush=True)
            control = _evaluate(pool, checkpoints["baseline"], out / "confirmation" / "baseline", schedule["confirmation"])
            trial = _evaluate(pool, winner, out / "confirmation" / "selected", schedule["confirmation"])
            confirmation = paired_summary(control, trial)
            scores = [group["selected"]["mean_requested_score"] for group in confirmation.values()]
            lower = [group["selected"]["requested_mean_bootstrap_95"][0] for group in confirmation.values()]
            result.update(status="confirmed", checkpoint=str(winner), confirmation=confirmation, score=min(scores),
                          milestone_met=min(scores) >= MILESTONE, target_point_estimate=min(scores) > TARGET,
                          target_lower_bound=min(lower) > TARGET)
    write_json(out / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and confirm a phase adaptation of a frozen advanced policy")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed-block", default=52000, type=int)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--family", choices=tuple(RECIPE_FAMILIES), default="placement")
    args = parser.parse_args()
    if not 1 <= args.workers <= 61:
        parser.error("workers must be between 1 and 61")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    with _run_lock(args.out):
        result = run(args.out, args.reference, args.seed_block, args.workers, args.family)
    print(f"Agent finished: {result['status']}; confirmation floor: {result.get('score')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
