from __future__ import annotations

import hashlib
from copy import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import RETRACT, TRANSFER
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import Correction, PlacementPolicy, PlacementTrace
from a3opt.refinement_eval import sha256
from a3opt.release_timing_agent import STRICT_RULES


OPTIONS = {"reference": (0., 0.), "raise4": (0., .004), "raise8": (0., .008),
           "raise12": (0., .012), "raise8_inset3": (.003, .008)}


def correction(column: int, batch: int, option: str) -> Correction:
    inset, height = OPTIONS[option]
    return Correction(column, batch, (inset * (-1 if batch == 0 else 1), 0., height))


def physics_digest(env) -> str:
    state = (env.data.qpos, env.data.qvel, env.data.qacc_warmstart, env.data.ctrl)
    return hashlib.sha256(np.concatenate(state).astype("<f8", copy=False).tobytes()).hexdigest()


def command_bytes(action: np.ndarray, info: dict) -> bytes:
    return np.asarray([action, info["applied_action"]], dtype="<f8").tobytes()


def outcome(env, info: dict, ids: np.ndarray, digest: str) -> dict:
    counted = np.flatnonzero(info["cookies_in_target_mask"])
    return {"step": int(info["step"]), "counted_ids": counted.tolist(),
            "current_batch_count": int(np.intersect1d(ids, counted).size),
            "filled_slots": sum(cookie >= 0 for cookie in info["target_slot_occupancy"]),
            "safety_reason": info.get("safety_reason"), "source_remaining": int(info["cookies_in_source"]),
            "command_sha256": digest, "physics_sha256": physics_digest(env)}


def run_branch(env, reference: PlacementPolicy, task: str, ids: np.ndarray, option: str, column: int) -> dict:
    branch = copy(env)
    branch.data = copy(env.data)
    controller = copy(reference)
    batch = reference.batch
    if option != "reference":
        controller.actions = reference.actions.copy()
        controller.actions[batch] = [actions.copy() for actions in reference.actions[batch]]
        controller.shift_placement(batch, np.asarray(correction(column, batch, option).xyz_m))
    observation = branch._observation()
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(command_bytes(action, info))
        if terminated or truncated or (controller.stage == RETRACT and controller.ended):
            break
    return outcome(branch, info, ids, digest.hexdigest())


class CollectionTrace(PlacementTrace):
    def reset(self, context=None) -> None:
        super().reset(context)
        self.trials = [None, None]
        self.suffix_digest = None
        self.actual = [None, None]
        self.task = None

    def act(self, observation: dict, task: str) -> np.ndarray:
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context) -> None:
        super().add_frame(observation, action, info, env=env, **context)
        policy = self.policy
        batch = policy.batch
        ids = self.lifted_ids[batch]
        if self.suffix_digest is not None:
            self.suffix_digest.update(command_bytes(action, info))
            if policy.stage == RETRACT and policy.ended or self.safety or self.step == RULES["max_steps"]:
                measured = outcome(env, info, ids, self.suffix_digest.hexdigest())
                if measured != self.trials[batch]["outcomes"]["reference"]:
                    raise ValueError("unchanged training fork did not reproduce the control state and commands")
                self.actual[batch] = measured
                self.suffix_digest = None
        if policy.stage == TRANSFER and policy.ended and not self.safety and self.step < RULES["max_steps"]:
            self.trials[batch] = {
                "step": self.step, "lifted_ids": ids.tolist(), "prefix_physics_sha256": physics_digest(env),
                "prefix_action_sha256": self.stages[batch]["transfer"]["prefix_sha256"],
                "outcomes": {option: run_branch(env, policy, self.task, ids, option, self.column) for option in OPTIONS},
            }
            self.suffix_digest = hashlib.sha256()


def collect(checkpoint: Path, focus: int, scale: float, seed: int) -> dict:
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        policy = PlacementPolicy(checkpoint, focus, scale)
        trace = CollectionTrace(policy)
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"checkpoint_sha256": sha256(checkpoint), "timing": [focus, scale], "seed": result.seed,
                "column": result.source_column, "profile": result.randomization["profile"],
                "steps": result.steps, "score": result.score, "safety_reason": trace.safety,
                "trials": trace.trials, "actual_continuation": trace.actual,
                "phase_geometry": trace.stages, "terminal": trace.terminal_snapshot,
                "descent_observations": trace.measurements, "wall_seconds": result.wall_seconds}
    finally:
        env.close()


def audit(job: tuple, row: dict) -> None:
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, focus, scale, seed = job
    if (row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != [focus, scale]
            or row["profile"] != "advanced" or row["seed"] != seed or row["column"] != choose_column("random", seed)
            or row["score"] != len(row["terminal"]["counted_ids"]) or row["steps"] != row["terminal"]["step"]):
        raise ValueError("training receipts differ from the acquisition schedule")
    for trial, actual in zip(row["trials"], row["actual_continuation"], strict=True):
        if trial is None and actual is None:
            continue
        if trial is None or set(trial["outcomes"]) != set(OPTIONS) or trial["outcomes"]["reference"] != actual:
            raise ValueError("training forks require an exact replay of the reference commands and state")
        for outcome_row in trial["outcomes"].values():
            if not trial["step"] < outcome_row["step"] <= RULES["max_steps"]:
                raise ValueError("training fork exceeded its control step budget")


def acquire(pool, folder: Path, checkpoint: Path, timing: tuple[int, float], seeds: list[int]) -> list[dict]:
    _record_once(folder / "request.json", {"checkpoint_sha256": sha256(checkpoint), "timing": list(timing),
                 "seeds": seeds, "options": {name: list(offset) for name, offset in OPTIONS.items()}})
    jobs = [(checkpoint, *timing, seed) for seed in seeds]
    rows = _jobs(pool, collect, jobs, folder / "episodes", lambda job: str(job[-1]), audit)
    _record_once(folder / "dataset.json", {"rows": rows})
    print(f"Collected {len(rows)} scene receipts with placement branches", flush=True)
    return rows
