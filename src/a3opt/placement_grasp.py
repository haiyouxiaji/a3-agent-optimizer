from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import CLOSE, LIFT, OPEN, grip_features
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_agent import reference_setup
from a3opt.placement_release import terminal
from a3opt.placement_sim import PlacementPolicy, PlacementTrace
from a3opt.placement_train import command_bytes
from a3opt.refinement_eval import sha256
from a3opt.release_timing_agent import STRICT_RULES


OPTIONS = ("reference", "settle", "squeeze", "lower-reclose")
EXTRA_STEPS = 8


def recover(policy, option: str) -> None:
    if option == "reference":
        return
    phases = policy.actions[1]
    closed = phases[CLOSE][-1]
    prefix = np.repeat(closed[None], EXTRA_STEPS, axis=0)
    ramp = _ramp(np.linspace(0., 1., EXTRA_STEPS))
    if option == "squeeze":
        amount = .025
        prefix[:, 7] = np.maximum(0., closed[7] - amount * ramp)
        for stage in range(LIFT, OPEN + 1):
            rollout = phases[stage]
            strength = np.linspace(1., 0., len(rollout)) if stage == OPEN else 1.
            rollout[:, 7] = np.maximum(0., rollout[:, 7] - amount * strength)
    elif option == "lower-reclose":
        new = nominal_kinematics().shift(closed[:7], np.array([0., 0., -.003]))
        down = _ramp(np.linspace(0., 1., EXTRA_STEPS // 2))
        prefix[:len(down), :7] = closed[:7] + down[:, None] * (new - closed[:7])
        prefix[len(down):, :7] = new
        prefix[:, 7] += .075 * np.sin(np.pi * np.linspace(0., 1., EXTRA_STEPS))
        blend = min(7, len(phases[LIFT]))
        fade = 1. - _ramp(np.linspace(0., 1., blend))
        phases[LIFT][:blend, :7] += fade[:, None] * (new - closed[:7])
    elif option != "settle":
        raise ValueError("unknown grasp recovery option")
    phases[LIFT] = np.concatenate((prefix, phases[LIFT]))


def cloned(env):
    branch = copy(env)
    branch.data = copy(env.data)
    return branch


def continuation(env, policy, task, option, columns, column):
    branch = cloned(env)
    controller = copy(policy)
    controller.actions = [policy.actions[0], [stage.copy() for stage in policy.actions[1]]]
    recover(controller, option)
    observation = branch._observation()
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(command_bytes(action, info))
        if terminated or truncated:
            break
    return terminal(branch, info, columns, column, digest)


class GraspTrace(PlacementTrace):
    def __init__(self, policy, control):
        super().__init__(policy)
        self.sample_control = control

    def reset(self, context=None):
        super().reset(context)
        self.task = ""
        self.point = self.branches = self.features = self.actual = self.suffix = None

    def act(self, observation, task):
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.suffix is not None:
            self.suffix.update(command_bytes(action, info))
            if self.safety or self.step == RULES["max_steps"]:
                self.actual = terminal(env, info, self.columns, self.column, self.suffix)
        if self.policy.batch != 1 or not self.policy.ended or self.safety or self.step >= RULES["max_steps"]:
            return
        if self.policy.stage == CLOSE:
            next_observation = env._observation()
            self.features = grip_features(next_observation, self.policy.actions[1][CLOSE][-1]).tolist()
            self.point = cloned(env), copy(self.policy)
            self.suffix = hashlib.sha256()
        if self.policy.stage == LIFT and self.point is not None:
            if self.sample_control or len(self.lifted_ids[1]) < 5:
                saved_env, saved_policy = self.point
                self.branches = {option: continuation(saved_env, saved_policy, self.task, option, self.columns, self.column)
                                 for option in OPTIONS}
            self.point = None


def collect(checkpoint, focus, scale, seed, control):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = GraspTrace(PlacementPolicy(checkpoint, focus, scale), control)
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "timing": [focus, scale], "control": control,
                "features": trace.features, "lifted": trace.events[1].get("lifted_requested"), "score": result.score,
                "safety_reason": trace.safety, "terminal_ids": np.flatnonzero(trace.target).tolist(),
                "actual": trace.actual, "branches": trace.branches, "wall_seconds": result.wall_seconds}
    finally:
        env.close()


def audit(job, row):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, focus, scale, seed, control = job
    if (row["seed"] != seed or row["column"] != choose_column("random", seed) or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != [focus, scale] or row["control"] != control
            or row["score"] != len(row["terminal_ids"])):
        raise ValueError("grasp receipt disagrees with the training request")
    if row["branches"] is not None and (set(row["branches"]) != set(OPTIONS) or row["actual"] != row["branches"]["reference"]
            or row["actual"]["counted_ids"] != row["terminal_ids"]):
        raise ValueError("unchanged grasp branch did not reproduce its original continuation")


def summarize(rows):
    samples = [row for row in rows if row["branches"] is not None]
    comparisons = {}
    for option in OPTIONS[1:]:
        effects = []
        for row in samples:
            reference, trial = (row["branches"][key] for key in ("reference", option))
            effects.append({"seed": row["seed"], "column": row["column"], "lifted": row["lifted"],
                            "count_change": len(trial["counted_ids"]) - len(reference["counted_ids"]),
                            "success_change": int(trial["success"]) - int(reference["success"]),
                            "unsafe": bool(trial["wrong_source"] or trial["safety_reason"])})
        comparisons[option] = effects
    return {"scenes": len(rows), "grasp_failures": sum(row["lifted"] is not None and row["lifted"] < 5 for row in rows),
            "counterfactual_scenes": len(samples), "options": comparisons}


def run(root, reference, timing_run, block, workers):
    if block % 4:
        raise ValueError("grasp acquisition requires a balanced seed block")
    train = list(range(block, block + 16))
    splits = {"scout": {"random": []}, "training": train, "development": [], "confirmation": {}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "control_seeds": train[:4], "recipes": list(OPTIONS),
                 "extra_steps": EXTRA_STEPS, "acquisition_code_sha256": sha256(Path(__file__))})
    print(f"Reserved {len(train)} balanced grasp-training scenes", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(checkpoint, *timing, seed, seed in train[:4]) for seed in train]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-2]), audit)
    dataset = root / "training" / "dataset.json"
    _record_once(dataset, {"rows": rows})
    return {"status": "training_data_collected", "dataset_sha256": sha256(dataset), **summarize(rows)}


def main():
    parser = argparse.ArgumentParser(description="Collect paired counterfactuals for second-batch grasp recovery.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed-block", type=int, default=123000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
