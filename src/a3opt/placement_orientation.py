from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_sim, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER, task_column
from a3opt.backends.release_residual import MAX_JOINT_CORRECTION, _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_agent import reference_setup
from a3opt.placement_search import MAJOR_LOSS, TARGET, best_option, branch_stats, compare, confirm_combined, encode, passes, reuse_training
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES


ANGLES = {"reference": 0., "tilt-negative": -.06, "tilt-positive": .06}


@dataclass(frozen=True)
class Orientation:
    column: int
    batch: int
    tilt_rad: float

    def __post_init__(self):
        if self.column not in range(1, 5) or self.batch not in (0, 1) or self.tilt_rad not in ANGLES.values():
            raise ValueError("invalid orientation correction")

    def record(self):
        return [self.column, self.batch, self.tilt_rad]


def correct_orientation(policy, batch, tilt_rad):
    solver = nominal_kinematics()
    low, high = solver.ranges.T
    axis = np.array([-np.sin(policy.pose[4]), np.cos(policy.pose[4]), 0.])
    for stage in (PLACE, OPEN, RETRACT):
        actions = policy.actions[batch][stage]
        weights = _ramp(np.arange(1, len(actions) + 1) / len(actions))
        if stage == OPEN:
            weights[:] = 1.
        elif stage == RETRACT:
            weights = 1. - weights
        for action, weight in zip(actions, weights, strict=True):
            if weight == 0.:
                continue
            start = np.clip(action[:7], low, high)
            solver.data.qpos[solver.positions] = start
            solver.mujoco.mj_kinematics(solver.model, solver.data)
            solver.mujoco.mj_comPos(solver.model, solver.data)
            solver.mujoco.mj_jacSite(solver.model, solver.data, solver.jac_position, solver.jac_rotation, solver.site)
            jacobian = np.vstack((solver.jac_position[:, solver.dofs], .1 * solver.jac_rotation[:, solver.dofs]))
            twist = np.r_[np.zeros(3), .1 * weight * tilt_rad * axis]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .004**2 * np.eye(6), twist)
            largest = np.max(np.abs(delta))
            if largest > MAX_JOINT_CORRECTION:
                delta *= MAX_JOINT_CORRECTION / largest
            action[:7] = np.clip(start + delta, low, high)


class OrientationPolicy(placement_sim.PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.orientations = {(item.column, item.batch): item.tilt_rad for item in corrections}
        if len(self.orientations) != len(corrections):
            raise ValueError("duplicate orientation correction")
        super().__init__(checkpoint, focus, scale)

    def act(self, observation, task=""):
        if self.actions is not None and self.cursor is not None and self.cursor[1:] == (PLACE, 0):
            batch = self.cursor[0]
            tilt = self.orientations.get((task_column(task), batch), 0.)
            if tilt:
                correct_orientation(self, batch, tilt)
        return super().act(observation, task)


def rollout_branch(env, reference, task, batch, angle, ids):
    state = copy(env)
    state.data = copy(env.data)
    controller = copy(reference)
    if angle:
        controller.actions = reference.actions.copy()
        controller.actions[batch] = [phase.copy() for phase in reference.actions[batch]]
        correct_orientation(controller, batch, angle)
    observation = state._observation()
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = state.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated or (batch == 0 and controller.stage == RETRACT and controller.ended):
            break
    return placement_train.outcome(state, info, ids, digest.hexdigest())


class OrientationTrace(placement_sim.PlacementTrace):
    def reset(self, context=None):
        super().reset(context)
        self.task = ""
        self.digests, self.trials, self.actual = ([None, None] for _ in range(3))

    def act(self, observation, task):
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        batch = self.policy.batch
        digest = self.digests[batch]
        ids = self.lifted_ids[batch]
        if digest is not None:
            digest.update(placement_train.command_bytes(action, info))
            if (self.safety or self.step == RULES["max_steps"]
                    or (batch == 0 and self.policy.stage == RETRACT and self.policy.ended)):
                self.actual[batch] = placement_train.outcome(env, info, ids, digest.hexdigest())
                self.digests[batch] = None
        if self.policy.stage == TRANSFER and self.policy.ended and not self.safety and self.step < RULES["max_steps"]:
            self.trials[batch] = {"outcomes": {name: rollout_branch(env, self.policy, self.task, batch, angle, ids)
                                               for name, angle in ANGLES.items()}, "step": self.step}
            self.digests[batch] = hashlib.sha256()


def collect(checkpoint, focus, scale, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = OrientationTrace(placement_sim.PlacementPolicy(checkpoint, focus, scale))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "timing": [focus, scale], "score": result.score,
                "actual": trace.actual, "trials": trace.trials, "terminal_ids": np.flatnonzero(trace.target).tolist(),
                "wall_seconds": result.wall_seconds}
    finally:
        env.close()


def audit(job, row):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, focus, scale, seed = job
    if (row["seed"] != seed or row["column"] != choose_column("random", seed) or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != [focus, scale]
            or row["score"] != len(row["terminal_ids"])):
        raise ValueError("orientation training request differs from its replay receipt")
    for fork, actual in zip(row["trials"], row["actual"], strict=True):
        if fork is None and actual is None:
            continue
        if (fork is None or set(fork["outcomes"]) != set(ANGLES) or actual != fork["outcomes"]["reference"]
                or any(not fork["step"] < item["step"] <= RULES["max_steps"] for item in fork["outcomes"].values())):
            raise ValueError("orientation forks failed the control continuation check")


def run(root, reference, timing_run, sources, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and native evaluation audit are required")
    development = list(range(block, block + 12))
    confirmation = list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    sources = [reuse_training(source, checkpoint, timing, identity) for source in sources]
    source_rows = [row for group, _ in sources for row in group if not row["safety_reason"]]
    seeds = []
    for column in range(1, 5):
        rows = sorted((row for row in source_rows if row["column"] == column), key=lambda row: (row["score"], row["seed"]))
        seeds.extend([row["seed"] for row in rows if row["score"] < 10][:2])
        seeds.extend([row["seed"] for row in rows if row["score"] == 10][:2])
    if len(seeds) != len(set(seeds)) or set(seeds).intersection(development + confirmation):
        raise ValueError("orientation training and evaluation seeds must be disjoint")
    splits["training"] = seeds
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "sources": [meta for _, meta in sources],
                 "native_parity_sha256": sha256(parity), "orientation_search_rad": ANGLES,
                 "training_code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "minimum_training_gains": 2, "target": TARGET})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(checkpoint, *timing, seed) for seed in seeds]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]), audit)
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"rows": rows})
        models, options = [], {}
        for batch in range(2):
            trials = [row["trials"][batch] for row in rows if row["trials"][batch] is not None]
            pooled = {name: branch_stats(trials, name) for name in ANGLES}
            selected = best_option(pooled, minimum_gains=2)
            by_column = {}
            for column in range(1, 5):
                group = [row["trials"][batch] for row in rows if row["column"] == column and row["trials"][batch] is not None]
                stats = {name: branch_stats(group, name) for name in ANGLES}
                by_column[str(column)] = stats
                if selected != "reference" and stats[selected]["improved"] and not stats[selected]["worse"]:
                    models.append(Orientation(column, batch, ANGLES[selected]))
            options[str(batch)] = {"pooled": pooled, "by_column": by_column, "selected": selected}
        model = tuple(models)
        _record_once(root / "fit.json", {"dataset_sha256": sha256(dataset), "corrections": encode(model), "train": options})
        print(f"Collected {len(rows)} orientation training receipts and selected {len(model)} interventions", flush=True)
        if not model:
            return {"status": "reference_retained_after_training"}
        outcomes = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                {"reference": (), "selected": model}, policy_type=OrientationPolicy)
        scores = {rule: compare([row[rule] for row in outcomes["reference"]], [row[rule] for row in outcomes["selected"]], model)
                  for rule in ("default", "strict-500")}
        _record_once(root / "development-selection.json", scores)
        if not all(passes(stats) for stats in scores.values()):
            return {"status": "reference_retained_after_development", "development": scores}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                   confirmation[:4], policy_type=OrientationPolicy)
    accepted = all(passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation",
            "active_policy_above_target": accepted and checked["default"]["mean"] > TARGET,
            "development": scores, "confirmation": checked}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_ORIENTATION_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("orientation policy requires the frozen checkpoint")
    return OrientationPolicy(checkpoint, *lock["timing"], tuple(Orientation(*item) for item in lock["corrections"]))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate bounded orientation residuals for placement.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-block", type=int, default=116000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), [source.resolve() for source in args.source],
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
