from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER, task_column
from a3opt.backends.release_residual import _ramp
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_agent import reference_setup
from a3opt.placement_search import TARGET, compare, confirm_combined, encode, passes, reuse_training
from a3opt.placement_sim import PlacementPolicy, PlacementTrace
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES


RECIPES = {
    "reference": (1., 0., 0.),
    "faster-opening": (2., 0., 0.),
    "wider-opening": (1., .04, 0.),
    "raised-and-faster": (2., 0., .006),
}


@dataclass(frozen=True)
class Release:
    column: int
    batch: int
    recipe: str

    def __post_init__(self) -> None:
        if self.column not in range(1, 5) or self.batch not in (0, 1) or self.recipe not in RECIPES or self.recipe == "reference":
            raise ValueError("invalid release intervention")

    def record(self) -> list:
        return [self.column, self.batch, self.recipe]


def adjust_release(policy: PlacementPolicy, batch: int, recipe: str) -> None:
    speed, opening_delta, height = RECIPES[recipe]
    if height:
        policy.shift_placement(batch, np.array([0., 0., height]))
    opening, retreat = (policy.actions[batch][stage] for stage in (OPEN, RETRACT))
    frames = np.arange(len(opening))
    opening[:, 7] = np.interp(speed * frames, frames, opening[:, 7])
    if opening_delta:
        opening[:, 7] = np.minimum(.55, opening[:, 7] + opening_delta * _ramp(np.linspace(0., 1., len(opening))))
        weights = 1. - _ramp(np.linspace(0., 1., len(retreat)))
        retreat[:, 7] = np.minimum(.55, retreat[:, 7] + opening_delta * weights)


class ReleasePolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.release_recipes = {(item.column, item.batch): item.recipe for item in corrections}
        if len(self.release_recipes) != len(corrections):
            raise ValueError("duplicate release intervention")
        super().__init__(checkpoint, focus, scale)

    def act(self, observation, task=""):
        if self.cursor is not None and self.cursor[1:] == (PLACE, 0):
            batch = self.cursor[0]
            recipe = self.release_recipes.get((task_column(task), batch))
            if recipe:
                adjust_release(self, batch, recipe)
        return super().act(observation, task)


def terminal(env, info, columns, column, digest):
    measured = placement_train.outcome(env, info, np.empty(0, dtype=int), digest.hexdigest())
    return {**measured, "success": bool(info["success"] and not measured["safety_reason"]),
            "wrong_source": int(np.count_nonzero(columns[measured["counted_ids"]] != column))}


def fork(env, policy, task, batch, recipe, columns, column):
    branch = copy(env)
    branch.data = copy(env.data)
    controller = copy(policy)
    if recipe != "reference":
        controller.actions = policy.actions.copy()
        controller.actions[batch] = [phase.copy() for phase in policy.actions[batch]]
        adjust_release(controller, batch, recipe)
    observation = branch._observation()
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated:
            break
    return terminal(branch, info, columns, column, digest)


class ReleaseTrace(PlacementTrace):
    def __init__(self, policy, batch):
        self.branch_batch = batch
        super().__init__(policy)

    def reset(self, context=None):
        super().reset(context)
        self.task = ""
        self.trials = self.suffix = self.control = None

    def act(self, observation, task):
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.suffix is not None:
            self.suffix.update(placement_train.command_bytes(action, info))
            if self.safety or self.step == RULES["max_steps"]:
                self.control = terminal(env, info, self.columns, self.column, self.suffix)
        if (self.policy.batch == self.branch_batch and self.policy.stage == TRANSFER and self.policy.ended
                and not self.safety and self.step < RULES["max_steps"]):
            self.trials = {name: fork(env, self.policy, self.task, self.branch_batch, name, self.columns, self.column)
                           for name in RECIPES}
            self.suffix = hashlib.sha256()


def collect(checkpoint, focus, scale, seed, batch):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = ReleaseTrace(PlacementPolicy(checkpoint, focus, scale), batch)
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "batch": batch, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "timing": [focus, scale], "score": result.score,
                "trials": trace.trials, "control": trace.control, "terminal_ids": np.flatnonzero(trace.target).tolist(),
                "prefix_sha256": trace.stages[batch].get("transfer", {}).get("prefix_sha256"),
                "wall_seconds": result.wall_seconds}
    finally:
        env.close()


def audit(job, row):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, focus, scale, seed, batch = job
    if (row["seed"] != seed or row["batch"] != batch or row["column"] != choose_column("random", seed)
            or row["profile"] != "advanced" or row["checkpoint_sha256"] != sha256(checkpoint)
            or row["timing"] != [focus, scale] or row["score"] != len(row["terminal_ids"])):
        raise ValueError("release episode differs from its training request")
    if row["trials"] is not None and (not row["prefix_sha256"] or set(row["trials"]) != set(RECIPES)
            or row["control"] != row["trials"]["reference"] or row["control"]["counted_ids"] != row["terminal_ids"]):
        raise ValueError("release control fork did not reproduce the actual continuation")


def training_plan(rows: list[dict]) -> list[tuple[int, int]]:
    plan = []
    for column in range(1, 5):
        group = [row for row in rows if row["column"] == column and not row["safety_reason"]]
        candidates = []
        for batch in (0, 1):
            losses = []
            for row in group:
                transfer = row["phase_geometry"][batch].get("transfer", {}).get("batches", ((), ()))[batch]
                if len(transfer) == 5:
                    terminal_ids = set(row["terminal"]["counted_ids"])
                    lost = sum(cookie["id"] not in terminal_ids for cookie in transfer)
                    losses.append((lost, row["seed"]))
            candidates.append(losses)
        batch = max(range(2), key=lambda index: sum(lost for lost, _ in candidates[index]))
        failures = sorted(((lost, seed) for lost, seed in candidates[batch] if lost), reverse=True)
        controls = sorted(seed for lost, seed in candidates[batch] if not lost)
        if len(failures) >= 2 and controls:
            plan.extend((seed, batch) for seed in [*(seed for _, seed in failures[:4]), *controls[:2]])
    return plan


def fit(rows: list[dict]) -> tuple[tuple[Release, ...], dict]:
    model, summary = [], {}
    for column, batch in sorted({(row["column"], row["batch"]) for row in rows}):
        group = [row["trials"] for row in rows if (row["column"], row["batch"]) == (column, batch) and row["trials"]]
        options = {}
        for name in RECIPES:
            deltas = [len(forks[name]["counted_ids"]) - len(forks["reference"]["counted_ids"]) for forks in group]
            options[name] = {"gain": sum(deltas), "improved": sum(delta > 0 for delta in deltas),
                             "worst_delta": min(deltas, default=0),
                             "complete_delta": sum(int(forks[name]["success"]) - int(forks["reference"]["success"]) for forks in group),
                             "violations": sum(bool(forks[name]["wrong_source"] or forks[name]["safety_reason"]) for forks in group)}
        eligible = [name for name, stats in options.items() if stats["gain"] > 0 and stats["improved"] >= 2
                    and stats["worst_delta"] >= -1 and stats["complete_delta"] >= 0 and not stats["violations"]]
        selected = max(eligible, key=lambda name: (options[name]["gain"], options[name]["complete_delta"]), default=None)
        if selected:
            model.append(Release(column, batch, selected))
        summary[f"{column}/{batch}"] = {"episodes": len(group), "options": options, "selected": selected}
    return tuple(model), summary


def run(root, reference, timing_run, sources, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and native evaluation audit are required")
    development, confirmation = list(range(block, block + 12)), list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    training_sources = [reuse_training(source, checkpoint, timing, identity) for source in sources]
    rows = [row for group, _ in training_sources for row in group]
    source_seeds = {row["seed"] for row in rows}
    if len(source_seeds) != len(rows) or source_seeds.intersection(development + confirmation):
        raise ValueError("release training data must be unique and disjoint from evaluation")
    plan = training_plan(rows)
    splits["training"] = [seed for seed, _ in plan]
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "training_plan": [list(item) for item in plan],
                 "training_sources": [evidence for _, evidence in training_sources], "recipes": {k: list(v) for k, v in RECIPES.items()},
                 "training_code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "parity_sha256": sha256(parity), "target_mean_strictly_above": TARGET})
    print(f"Reserved {len(plan)} training forks", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(checkpoint, *timing, seed, batch) for seed, batch in plan]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-2]), audit)
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"rows": rows})
        model, training = fit(rows)
        _record_once(root / "fit.json", {"dataset_sha256": sha256(dataset), "corrections": encode(model), "training": training})
        print(f"Collected {len(rows)} terminal receipts and selected {len(model)} release interventions", flush=True)
        if not model:
            return {"status": "reference_retained_after_training"}
        outcomes = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                {"reference": (), "selected": model}, policy_type=ReleasePolicy)
        scores = {rule: compare([row[rule] for row in outcomes["reference"]], [row[rule] for row in outcomes["selected"]], model)
                  for rule in ("default", "strict-500")}
        _record_once(root / "development-selection.json", scores)
        if not all(passes(stats) for stats in scores.values()):
            return {"status": "reference_retained_after_development", "development": scores}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model, confirmation[:4],
                                   policy_type=ReleasePolicy)
    primary = checked["default"]
    above = primary["mean"] > TARGET
    accepted = above and all(passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation",
            "target_point_estimate": above, "target_lower_bound": primary["mean_ci95"][0] > TARGET,
            "active_policy_above_target": accepted, "development": scores, "confirmation": checked}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_RELEASE_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("release policy requires the frozen checkpoint")
    return ReleasePolicy(checkpoint, *lock["timing"], tuple(Release(*item) for item in lock["corrections"]))


def main():
    parser = argparse.ArgumentParser(description="Train release-phase interventions from terminal-scored simulator forks.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-block", type=int, default=117000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), [source.resolve() for source in args.source],
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
