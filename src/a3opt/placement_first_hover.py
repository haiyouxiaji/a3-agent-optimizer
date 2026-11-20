from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_descent_agent as descent, placement_hover_agent as hover
from a3opt import placement_search, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import PLACE, RETRACT, TRANSFER, task_column
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementPolicy, PlacementTrace
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


BATCH = 0
RECIPES = {"rise12": (0., 0., .012), "rise20": (0., 0., .020), "rise28": (0., 0., .028),
           "rise20-x-minus": (-.004, 0., .020), "rise20-x-plus": (.004, 0., .020)}
CASE_LIMIT = 4
DEVELOPMENT_PER_COLUMN = 8
CONFIRMATION_SCENES = 64


@dataclass(frozen=True)
class FirstHover:
    column: int
    recipe: str
    batch = BATCH

    def __post_init__(self):
        if self.column not in range(1, 5) or self.recipe not in RECIPES:
            raise ValueError("invalid first-batch hover recipe")

    def record(self):
        return [self.column, BATCH, self.recipe]


class FirstHoverPolicy(descent.DescentPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        self.first_hover = {item.column: item.recipe for item in corrections}
        if len(self.first_hover) != len(corrections):
            raise ValueError("expected at most one first-batch hover recipe per column")
        super().__init__(checkpoint, focus, scale, incumbent)

    def act(self, observation, task=""):
        if self.first_hover and self.actions is not None and self.cursor == (BATCH, PLACE, 0):
            column = task_column(task)
            if column in self.first_hover:
                if (column, BATCH) in self.profiles:
                    descent.adjust(self.actions[BATCH], observation["observation.eef_pose"], self.profiles[column, BATCH])
                self.shift_placement(BATCH, np.asarray(RECIPES[self.first_hover[column]]))
                return PlacementPolicy.act(self, observation, task)
        return super().act(observation, task)


def acquisition_plan(episodes):
    groups = {}
    for column in range(1, 5):
        failures, controls = [], []
        for row in episodes:
            if row["column"] != column or row["score"] != row["requested_score"] or row["safety_reason"]:
                continue
            stages = row["phase_geometry"][BATCH]
            transfer = stages.get("transfer", {}).get("batches", [[]])[BATCH]
            if len(transfer) != 5 or min(cookie["abs_vertical_axis"] for cookie in transfer) < .98:
                continue
            landed = stages.get("retract", {}).get("batches", [[]])[BATCH]
            missed = 5 - sum(cookie["scored"] for cookie in landed)
            tilt = min((cookie["abs_vertical_axis"] for name in ("descent", "open") for cookie in
                        stages.get(name, {}).get("batches", [[]])[BATCH]), default=1.)
            if missed > 0 and tilt < .97:
                failures.append((-missed, tilt, row["seed"]))
            elif row["score"] == 10 and row["success"]:
                controls.append(row["seed"])
        if min(len(failures), len(controls)) >= CASE_LIMIT:
            groups[column] = {"first_batch_lost": -sum(case[0] for case in failures), "failure_count": len(failures),
                              "failures": [case[-1] for case in sorted(failures)[:CASE_LIMIT]],
                              "controls": sorted(controls)[:CASE_LIMIT]}
    return dict(sorted(groups.items(), key=lambda item: (-item[1]["first_batch_lost"], item[0]))[:2])


def continue_batch(env, policy, task, ids, column, recipe):
    branch = copy(env)
    branch.data = copy(env.data)
    controller = copy(policy)
    controller.actions = [[stage.copy() for stage in policy.actions[BATCH]], policy.actions[1]]
    controller.first_hover = {column: recipe} if recipe else {}
    observation = branch._observation()
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated or (controller.batch == BATCH and controller.stage == RETRACT and controller.ended):
            return placement_train.outcome(branch, info, ids, digest.hexdigest())
    raise ValueError("first-batch fork could not reach the episode boundary")


class FirstHoverTrace(PlacementTrace):
    def reset(self, context=None):
        super().reset(context)
        self.task = ""
        self.trials = self.actual = self.suffix = None

    def act(self, observation, task):
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        policy = self.policy
        ids = self.lifted_ids[BATCH]
        if self.suffix is not None:
            self.suffix.update(placement_train.command_bytes(action, info))
            if self.safety or self.step == RULES["max_steps"] or (policy.stage == RETRACT and policy.ended):
                self.actual = placement_train.outcome(env, info, ids, self.suffix.hexdigest())
                self.suffix = None
        if policy.batch == BATCH and policy.stage == TRANSFER and policy.ended and not self.safety:
            self.trials = {"reference": continue_batch(env, policy, self.task, ids, self.column, None)}
            self.trials.update({recipe: continue_batch(env, policy, self.task, ids, self.column, recipe) for recipe in RECIPES})
            self.suffix = hashlib.sha256()


def collect(checkpoint, timing, policy_type, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = FirstHoverTrace(policy_type(checkpoint, *timing))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "steps": result.steps, "score": result.score, "target_cookie_ids": np.flatnonzero(trace.target).tolist(),
                "success": result.success, "safety_reason": trace.safety, "action_sha256": trace.digest.hexdigest(),
                "applied_action_sha256": trace.applied_digest.hexdigest(), "trials": trace.trials, "continuation": trace.actual,
                "first_transfer": trace.stages[BATCH].get("transfer")}
    finally:
        env.close()


def audit_training(job, measured, *, expected):
    original = expected[job[-1]]
    if (any(measured[key] != original[key] for key in ("seed", "column", "profile", *hover.MATCH_FIELDS))
            or measured["first_transfer"] != original["phase_geometry"][BATCH]["transfer"]
            or set(measured["trials"] or ()) != {"reference", *RECIPES}
            or measured["trials"]["reference"] != measured["continuation"]):
        raise ValueError("first-hover samples do not reproduce the unchanged physics and action prefix")


def fit(dataset, plan):
    candidates, model = {}, []
    for column, group in plan.items():
        controls = set(group["controls"])
        episodes = [item for item in dataset if item["column"] == column]
        metrics = {}
        for recipe in RECIPES:
            changes, unsafe, control_regressions = {}, 0, 0
            for item in episodes:
                actual, proposed = (item["trials"][name] for name in ("reference", recipe))
                count = proposed["current_batch_count"] - actual["current_batch_count"]
                seed = item["seed"]
                changes[str(seed)] = count
                unsafe += bool(proposed["safety_reason"] or len(proposed["counted_ids"]) != proposed["current_batch_count"])
                control_regressions += seed in controls and (count < 0 or proposed["filled_slots"] < actual["filled_slots"])
            gains = sum(value > 0 for value in changes.values())
            eligible = gains >= 2 and not unsafe and not control_regressions and min(changes.values()) >= 0
            metrics[recipe] = {"deltas": changes, "gains": gains, "total": sum(changes.values()), "unsafe": unsafe,
                               "control_regressions": control_regressions, "eligible": eligible}
        chosen = max((name for name, row in metrics.items() if row["eligible"]),
                     key=lambda name: (metrics[name]["total"], metrics[name]["gains"], -RECIPES[name][2]), default=None)
        if chosen is not None:
            model.append(FirstHover(column, chosen))
        candidates[str(column)] = {"options": metrics, "selected": chosen}
    return tuple(model), candidates


def historical_rows(release, sources, frozen, common, checkpoint, timing, published_model):
    episodes, fingerprints = [], {}
    for filename, digest in frozen["rollout_sha256"].items():
        raw = gzip.decompress((release / filename).read_bytes())
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("published rollout hash differs from its manifest")
        fingerprints[filename] = digest
        for entry in json.loads(raw)["selected"]:
            placement_benchmark.check((checkpoint, *timing, entry["strict-500"]["seed"], published_model), entry)
            episodes.append(entry["strict-500"])
    for root in sources:
        protocol = read_json(root / "protocol.json")
        lock = read_json(root / "confirmation-lock.json")
        folder = root / "confirmation" / "rollouts"
        if (any(protocol[key] != value for key, value in common.items())
                or lock["checkpoint_sha256"] != frozen["checkpoint_sha256"]
                or lock["protocol_sha256"] != sha256(root / "protocol.json")
                or read_json(root / "confirmation" / "audit.json")["matched_native_replays"] != 8):
            raise ValueError(f"training reference does not match the frozen published controller: {root}")
        for seed, entry in zip(lock["seeds"], read_json(folder / "results.json")["reference"], strict=True):
            placement_benchmark.check((checkpoint, *timing, seed, ()), entry)
            if entry != read_json(folder / "episodes" / "reference" / f"{seed}.json"):
                raise ValueError("training reference aggregate differs from its original receipt")
            episodes.append(entry["strict-500"])
        fingerprints[str(root)] = sha256(folder / "results.json")
    if len({row["seed"] for row in episodes}) != len(episodes):
        raise ValueError("duplicate training seed in historical rollouts")
    return episodes, fingerprints


def supports_confirmation(stats):
    return stats["default"]["mean"] > placement_search.TARGET and all(
        placement_search.passes(row) and row["paired_count_ci95"][0] > 0 for row in stats.values())


def run(root, release, sources, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected an aligned seed block and a positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint = release / frozen["checkpoint"]
    timing = tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*item) for item in frozen["corrections"])
    environment = simulator_identity()
    checkpoint_hash = sha256(checkpoint)
    if (checkpoint_hash != frozen["checkpoint_sha256"] or environment["sim_revision"] != frozen["sim_revision"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("the installed controller differs from the published reference")
    common = {"environment": environment, "timing": list(timing), "incumbent": frozen["corrections"],
              "rules": RULES, "strict_rules": STRICT_RULES, "reference_sha256": sha256(manifest)}
    history, fingerprints = historical_rows(release, sources, frozen, common, checkpoint, timing, incumbent)
    plan = acquisition_plan(history)
    training = [seed for group in plan.values() for seed in group["failures"] + group["controls"]]
    development = [seed for seed in range(block, block + 4 * DEVELOPMENT_PER_COLUMN) if choose_column("random", seed) in plan]
    confirmation = list(range(block + 400, block + 400 + CONFIRMATION_SCENES))
    reserved = set(development + confirmation)
    paths = [*(source / "protocol.json" for source in sources), *(root.parent.glob("*/protocol.json"))]
    if reserved & {row["seed"] for row in history} or any(
        path.parent != root and reserved & hover.scheduled_seeds(read_json(path).get("schedule", {})) for path in paths):
        raise ValueError("first-hover expects fresh development and confirmation seeds")
    protocol = root / "protocol.json"
    _record_once(protocol, {**common, "profile": "advanced", "schedule": {"training": training, "development": development,
                                "confirmation": confirmation}, "diagnostic_receipts": fingerprints,
                            "code_sha256": sha256(Path(__file__)), "evaluator_sha256": frozen["evaluator_sha256"],
                            "offsets_m": {name: list(values) for name, values in RECIPES.items()},
                            "confirmation_requires": ["course_mean_above_9", "positive_lower_bound_on_paired_gain_both_rules",
                                                      "no_major_losses", "no_safety_or_wrong_source", "no_success_count_regression"]})
    _record_once(root / "acquisition-plan.json", {str(column): cases for column, cases in plan.items()})
    print(f"Selected {len(training)} historical cases for first-batch physics forks in columns {sorted(plan)}", flush=True)
    if not training:
        return {"status": "reference_retained_without_training_support"}
    policy_type = partial(FirstHoverPolicy, incumbent=incumbent)
    expected = {row["seed"]: row for row in history}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(checkpoint, timing, policy_type, seed) for seed in training]
        dataset = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]), partial(audit_training, expected=expected))
        dataset_path = root / "training" / "dataset.json"
        _record_once(dataset_path, {"rows": dataset})
        model, candidates = fit(dataset, plan)
        fit_path = root / "fit.json"
        corrections = [item.record() for item in model]
        _record_once(fit_path, {"dataset_sha256": sha256(dataset_path), "corrections": corrections, "candidates": candidates})
        if not model:
            return {"status": "reference_retained_after_training"}
        seeds = [seed for seed in development if choose_column("random", seed) in {item.column for item in model}]
        print(f"Fitted {corrections}; evaluating {len(seeds)} fresh development pairs", flush=True)
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, seeds,
                                              {"reference": (), "selected": model}, policy_type=policy_type)
        progress = hover.pair_stats(paired, model)
        selection = root / "development-selection.json"
        _record_once(selection, progress)
        if not all(placement_search.passes(stats) for stats in progress.values()):
            return {"status": "reference_retained_after_development", "development": progress}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                     "timing": list(timing), "incumbent": frozen["corrections"], "corrections": corrections, "seeds": confirmation,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)})
        print(f"First-hover candidate frozen before {len(confirmation)} fresh confirmation pairs", flush=True)
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                   model, confirmation[:4], policy_type=policy_type)
    balance = Counter({str(column): CONFIRMATION_SCENES // 4 for column in range(1, 5)})
    if any(Counter(row["scored_pairs_by_column"]) != balance for row in final.values()):
        raise ValueError("first-hover confirmation is not balanced across all four source columns")
    return {"status": "candidate_promoted" if supports_confirmation(final) else "reference_retained_after_confirmation",
            "development": progress, "confirmation": final}


def main():
    parser = argparse.ArgumentParser(description="Learn first-batch hover release from historical advanced placement failures.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=144000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), [path.resolve() for path in args.source], args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
