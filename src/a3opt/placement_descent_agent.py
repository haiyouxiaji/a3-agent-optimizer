from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_release, placement_search, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_agent import reference_setup
from a3opt.placement_sim import PlacementPolicy, PlacementTrace
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES


PROFILES = {"vertical": (0., 0.), "midpoint": (.5, 0.), "align-early": (1., 0.), "align-raised": (1., .004)}
SCOPES = ((0,), (1,), (0, 1))
MAX_XY = .008
FINISH_ALIGNMENT = .45
PROBES_PER_COLUMN = 2


@dataclass(frozen=True)
class Descent:
    column: int
    batch: int
    profile: str

    def __post_init__(self):
        if self.column not in range(1, 5) or self.batch not in (0, 1) or self.profile not in PROFILES:
            raise ValueError("invalid constrained descent profile")

    def record(self):
        return [self.column, self.batch, self.profile]


def tool_xyz(solver, position):
    solver.data.qpos[solver.positions] = position
    solver.mujoco.mj_kinematics(solver.model, solver.data)
    return solver.data.site_xpos[solver.site].copy()


def horizontal_profile(progress, start, endpoint, weight):
    fraction = _ramp(progress / FINISH_ALIGNMENT)
    return start + fraction * weight * (endpoint - start)


def adjust(phases, measured_xyz, name):
    weight, height = PROFILES[name]
    solver = nominal_kinematics()
    descent, opening, retract = (phases[phase] for phase in (PLACE, OPEN, RETRACT))
    start = np.asarray(measured_xyz[:2])
    endpoint = tool_xyz(solver, descent[-1, :7])[:2]
    for stage, actions in ((PLACE, descent), (OPEN, opening)):
        for index, action in enumerate(actions, start=1):
            progress = index / len(actions) if stage == PLACE else 1.
            target_xy = horizontal_profile(progress, start, endpoint, weight)
            original = tool_xyz(solver, action[:7])[:2]
            displacement = np.r_[np.clip(target_xy - original, -MAX_XY, MAX_XY), height * _ramp(progress)]
            action[:7] = solver.shift(action[:7], displacement)
    retract_offset = np.r_[np.clip(horizontal_profile(1., start, endpoint, weight) - endpoint, -MAX_XY, MAX_XY), height]
    for index, action in enumerate(retract, start=1):
        displacement = retract_offset * (1. - _ramp(index / len(retract)))
        if displacement.any():
            action[:7] = solver.shift(action[:7], displacement)


class DescentPolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.profiles = {(item.column, item.batch): item.profile for item in corrections}
        if len(self.profiles) != len(corrections):
            raise ValueError("duplicate constrained descent")
        super().__init__(checkpoint, focus, scale)

    def act(self, observation, task=""):
        if self.actions is not None and self.cursor is not None and self.cursor[1:] == (PLACE, 0):
            batch = self.cursor[0]
            option = self.profiles.get((task_column(task), batch))
            if option:
                adjust(self.actions[batch], observation["observation.eef_pose"], option)
        return super().act(observation, task)


def interventions():
    return {"reference": ((), None), **{f"{name}/{'+'.join(map(str, scope))}": (scope, name)
            for scope, name in product(SCOPES, PROFILES)}}


def fork(env, policy, task, scope, name, columns, column):
    branch = copy(env)
    branch.data = copy(env.data)
    controller = copy(policy)
    controller.actions = list(policy.actions)
    for batch in scope:
        controller.actions[batch] = [phase.copy() for phase in policy.actions[batch]]
    controller.profiles = {(column, batch): name for batch in scope}
    observation = branch._observation()
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated:
            break
    return placement_release.terminal(branch, info, columns, column, digest)


class DescentTrace(PlacementTrace):
    def reset(self, context=None):
        super().reset(context)
        self.task = ""
        self.options = self.prefix = self.digest_suffix = self.continuation = None

    def act(self, observation, task):
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.digest_suffix is not None:
            self.digest_suffix.update(placement_train.command_bytes(action, info))
            if self.step == RULES["max_steps"] or self.safety:
                self.continuation = placement_release.terminal(env, info, self.columns, self.column, self.digest_suffix)
        if (self.policy.batch == 0 and self.policy.stage == TRANSFER and self.policy.ended
                and not self.safety and self.step < RULES["max_steps"]):
            self.prefix = self.stages[0]["transfer"]["prefix_sha256"]
            self.options = {key: fork(env, self.policy, self.task, scope, profile, self.columns, self.column)
                            for key, (scope, profile) in interventions().items()}
            self.digest_suffix = hashlib.sha256()


def collect(checkpoint, focus, scale, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = DescentTrace(DescentPolicy(checkpoint, focus, scale))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "timing": [focus, scale], "score": result.score,
                "prefix_sha256": trace.prefix, "continuation": trace.continuation, "options": trace.options,
                "terminal_ids": np.flatnonzero(trace.target).tolist(), "wall_seconds": result.wall_seconds}
    finally:
        env.close()


def audit(job, row):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, focus, scale, seed = job
    if (row["seed"] != seed or row["column"] != choose_column("random", seed) or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != [focus, scale]
            or len(row["terminal_ids"]) != row["score"]):
        raise ValueError("constrained descent training receipt differs from its request")
    if row["options"] is None:
        return
    if (not row["prefix_sha256"] or set(row["options"]) != set(interventions())
            or row["options"]["reference"] != row["continuation"] or row["continuation"]["counted_ids"] != row["terminal_ids"]):
        raise ValueError("constrained descent reference fork did not reproduce the full episode")


def training_plan(rows):
    failures = {column: [] for column in range(1, 5)}
    controls = {column: [] for column in range(1, 5)}
    for row in rows:
        if row["safety_reason"]:
            continue
        for stages in row["phase_geometry"]:
            if "transfer" not in stages or "descent" not in stages:
                break
        else:
            if row["score"] < 10:
                failures[row["column"]].append((row["score"], row["seed"]))
            else:
                controls[row["column"]].append(row["seed"])
    return [seed for column in range(1, 5) for seed in (
            [seed for _, seed in sorted(failures[column])[:PROBES_PER_COLUMN]] + sorted(controls[column])[:PROBES_PER_COLUMN])]


def fit(rows):
    options = interventions()
    model, report = [], {}
    for column in range(1, 5):
        subset = [row for row in rows if row["column"] == column and row["options"] is not None]
        scores = {}
        for key in options:
            old = [row["options"]["reference"] for row in subset]
            new = [row["options"][key] for row in subset]
            gains = [len(trial["counted_ids"]) - len(reference["counted_ids"])
                     for trial, reference in zip(new, old, strict=True)]
            scores[key] = {"gains": gains, "seeds": [row["seed"] for row in subset], "sum": sum(gains),
                           "controls": sum(len(row["counted_ids"]) == 10 for row in old),
                           "improved": sum(gain > 0 for gain in gains), "major_losses": sum(gain <= -4 for gain in gains),
                           "success_delta": sum(row["success"] for row in new) - sum(row["success"] for row in old),
                           "violations": sum(bool(row["wrong_source"] or row["safety_reason"]) for row in new)}
        eligible = [key for key, entry in scores.items() if entry["controls"] >= 2 and entry["improved"] >= 2
                    and entry["sum"] > 0 and entry["success_delta"] >= 0 and not (entry["violations"] or entry["major_losses"])]
        winner = max(eligible, key=lambda key: (scores[key]["sum"], scores[key]["success_delta"], -len(options[key][0])), default=None)
        if winner:
            scope, profile = options[winner]
            model.extend(Descent(column, batch, profile) for batch in scope)
        report[str(column)] = {"winner": winner, "options": scores}
    return tuple(model), report


def run(root, reference, timing_run, sources, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("descent search requires balanced seeds and verified scoring")
    fresh = list(range(block, block + 8))
    development = list(range(block + 100, block + 120))
    confirmation = list(range(block + 200, block + 240))
    splits = {"scout": {"random": []}, "training": fresh, "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    records = [placement_search.reuse_training(source, checkpoint, timing, identity) for source in sources]
    training = [row for rows, _ in records for row in rows]
    scheduled = fresh + development + confirmation
    if set(scheduled).intersection(row["seed"] for row in training) or len({row["seed"] for row in training}) != len(training):
        raise ValueError("descent training acquisition requires disjoint, unique seeds")
    focused = training_plan(training)
    splits["training"] = focused + fresh
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "training_sources": [item for _, item in records],
                 "focused": focused, "fresh": fresh, "profiles": {key: list(values) for key, values in PROFILES.items()},
                 "scopes": [list(scope) for scope in SCOPES], "code_sha256": sha256(Path(__file__)),
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__)), "parity_sha256": sha256(parity),
                 "target_mean_strictly_above": placement_search.TARGET})
    print(f"Reserved {len(splits['training'])} descent probes including {len(fresh)} new advanced scenes", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(checkpoint, *timing, seed) for seed in splits["training"]]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]), audit)
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"rows": rows})
        model, report = fit(rows)
        _record_once(root / "fit.json", {"columns": report, "dataset_sha256": sha256(dataset),
                                        "corrections": placement_search.encode(model)})
        print(f"Audited {len(rows)} descent probes; selected {placement_search.encode(model)}", flush=True)
        if not model:
            return {"status": "reference_retained_after_training", "training_episodes": len(rows)}
        observed = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                {"reference": (), "selected": model}, policy_type=DescentPolicy)
        dev = {rule: placement_search.compare([row[rule] for row in observed["reference"]],
                                               [row[rule] for row in observed["selected"]], model)
               for rule in ("default", "strict-500")}
        _record_once(root / "development-selection.json", dev)
        if not all(placement_search.passes(stats) for stats in dev.values()):
            return {"status": "reference_retained_after_development", "development": dev}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": placement_search.encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                    confirmation[:4], policy_type=DescentPolicy)
    primary = checked["default"]
    above = primary["mean"] > placement_search.TARGET
    accepted = above and all(placement_search.passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation", "development": dev,
            "confirmation": checked, "active_policy_above_target": accepted, "target_point_estimate": above,
            "target_lower_bound": primary["mean_ci95"][0] > placement_search.TARGET}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_DESCENT_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("constrained descent requires the frozen reference checkpoint")
    return DescentPolicy(checkpoint, *lock["timing"], tuple(Descent(*item) for item in lock["corrections"]))


def main():
    parser = argparse.ArgumentParser(description="Learn bounded constrained descent profiles from paired advanced rollouts.")
    for name in ("out", "reference", "timing", "parity"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed-block", type=int, default=130000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), [source.resolve() for source in args.source],
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
