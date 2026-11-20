from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import LIFT, OPEN, PLACE, RETRACT, TRANSFER, task_column
from a3opt.backends.advanced_trajectory import pose_label
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.difficulty_eval import RULES, check_row
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES, ReleaseTimingPolicy, TimingTrace


BOUND_M = np.array([.012, .015, .015])
PHASES = {LIFT: "lift", TRANSFER: "transfer", PLACE: "descent", OPEN: "open", RETRACT: "retract"}


@dataclass(frozen=True)
class Correction:
    column: int
    batch: int
    xyz_m: tuple[float, float, float]
    minimum_tool_x_m: float | None = None
    requires_first_correction: bool = False

    def __post_init__(self) -> None:
        offset = np.asarray(self.xyz_m)
        if (self.column not in range(1, 5) or self.batch not in (0, 1)
                or offset.shape != (3,) or not np.all(np.abs(offset) <= BOUND_M)
                or (self.requires_first_correction and self.batch != 1)
                or (self.minimum_tool_x_m is not None and not np.isfinite(self.minimum_tool_x_m))):
            raise ValueError("invalid placement correction")

    def record(self) -> list:
        values = [self.column, self.batch, list(self.xyz_m)]
        if self.minimum_tool_x_m is not None or self.requires_first_correction:
            values.append(self.minimum_tool_x_m)
        if self.requires_first_correction:
            values.append(True)
        return values


def projected_tool_x(eef_xyz: np.ndarray, estimated_pose: np.ndarray) -> float:
    yaw = estimated_pose[4]
    return float((np.asarray(eef_xyz[:2]) - estimated_pose[2:4]) @ [np.cos(yaw), np.sin(yaw)])


class PlacementPolicy(ReleaseTimingPolicy):
    def __init__(self, checkpoint: Path, focus: int, scale: float, corrections: tuple[Correction, ...] = ()):
        self.corrections = {(item.column, item.batch): item for item in corrections}
        if len(self.corrections) != len(corrections):
            raise ValueError("expected at most one correction per column and batch")
        super().__init__(checkpoint, focus, scale)

    def reset(self, context=None) -> None:
        super().reset(context)
        self.first_corrected = False

    def act(self, observation: dict, task: str = "") -> np.ndarray:
        if self.actions is not None and self.cursor is not None and self.cursor[1:] == (PLACE, 0):
            batch = self.cursor[0]
            correction = self.corrections.get((task_column(task), batch))
            if correction is not None and (not correction.requires_first_correction or self.first_corrected):
                limit = correction.minimum_tool_x_m
                if limit is None or projected_tool_x(observation["observation.eef_pose"], self.pose) >= limit:
                    offset = np.asarray(correction.xyz_m, dtype=float)
                    if np.any(offset):
                        self.shift_placement(batch, offset)
                        if batch == 0:
                            self.first_corrected = True
        return super().act(observation, task)

    def shift_placement(self, batch: int, offset: np.ndarray) -> None:
        cosine, sine = np.cos(self.pose[4]), np.sin(self.pose[4])
        lateral, longitudinal, vertical = offset
        displacement = np.array([cosine * lateral - sine * longitudinal,
                                 sine * lateral + cosine * longitudinal, vertical])
        solver = nominal_kinematics()
        for phase in (PLACE, OPEN, RETRACT):
            actions = self.actions[batch][phase]
            weights = _ramp(np.arange(1, len(actions) + 1) / len(actions))
            if phase == OPEN:
                weights[:] = 1.
            elif phase == RETRACT:
                weights = 1. - weights
            for action, weight in zip(actions, weights, strict=True):
                if weight:
                    action[:7] = solver.shift(action[:7], weight * displacement)


class PlacementTrace(TimingTrace):
    def __init__(self, policy: PlacementPolicy):
        super().__init__(policy, first_batch_only=False)

    def bind_env(self, env) -> None:
        super().bind_env(env)
        from a3_dual_arm_sim.controllers.placement_order import front_camera_column_order

        slots = np.asarray(env.TARGET_SLOTS_LOCAL)
        x_values = np.unique(slots[:, 0])
        self.goals_xy = [slots[slots[:, 0] == x_values[index]].mean(axis=0).tolist()
                         for index in front_camera_column_order(env)]
        self.floor_z = env.config.cookie_transfer.target_floor_z_m + env.config.cookie_transfer.bin_wall_thickness_m / 2

    def reset(self, context=None) -> None:
        super().reset(context)
        self.stages = [{}, {}]
        self.terminal_snapshot = None
        self.measurements = [None, None]

    def act(self, observation: dict, task: str) -> np.ndarray:
        if self.policy.actions is not None and self.policy.cursor is not None and self.policy.cursor[1:] == (PLACE, 0):
            batch = self.policy.cursor[0]
            self.measurements[batch] = {
                "eef_xyz_m": np.asarray(observation["observation.eef_pose"][:3]).tolist(),
                "force": np.asarray(observation["observation.force"][:6]).tolist(),
                "predicted_pose": self.policy.pose.tolist(),
            }
        return super().act(observation, task)

    def snapshot(self, env, info) -> dict:
        target_rotation = env.data.xmat[env._target_bin_body].reshape(3, 3)
        slots = {int(cookie): slot for slot, cookie in enumerate(info["target_slot_occupancy"]) if cookie >= 0}
        batches = []
        for ids in self.lifted_ids:
            cookies = []
            for cookie in map(int, ids):
                local = env.privileged_cookie_target_position(cookie)
                rotation = target_rotation.T @ env.data.xmat[env._cookie_bodies[cookie]].reshape(3, 3)
                extent = np.abs(rotation) @ env.COOKIE_HALF_SIZE
                margin = env.TARGET_INNER_HALF_SIZE + env.CONTACT_CONTAINMENT_TOLERANCE_M - np.abs(local[:2]) - extent[:2]
                linear, angular = env._cookie_speed(cookie, relative_to_target=True)
                cookies.append({"id": cookie, "xyz_m": local.tolist(), "bottom_above_floor_m": float(local[2] - extent[2] - self.floor_z),
                                "footprint_margin_m": margin.tolist(), "abs_vertical_axis": abs(float(rotation[2, 2])),
                                "speed": [float(linear), float(angular)], "pad_contacts": list(env.privileged_left_finger_contacts(cookie)),
                                "scored": bool(self.target[cookie]), "slot": slots.get(cookie)})
            batches.append(cookies)
        return {"step": self.step, "batches": batches, "counted_ids": np.flatnonzero(self.target).tolist()}

    def add_frame(self, observation, action, info, *, env, **context) -> None:
        super().add_frame(observation, action, info, env=env, **context)
        policy = self.policy
        snapshot = None
        if policy.ended and policy.stage in PHASES:
            snapshot = self.snapshot(env, info)
            if policy.stage == TRANSFER:
                snapshot["prefix_sha256"] = [self.digest.hexdigest(), self.applied_digest.hexdigest()]
            self.stages[policy.batch][PHASES[policy.stage]] = snapshot
        if self.step == RULES["max_steps"] or self.safety or (not env.task_config.require_released and info["success"]):
            self.terminal_snapshot = snapshot if snapshot is not None else self.snapshot(env, info)


def episode(checkpoint: Path, focus: int, scale: float, suite: str, seed: int,
            corrections: tuple[Correction, ...], strict: bool, *, policy_type=PlacementPolicy) -> dict:
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column=suite, **(STRICT_RULES if strict else RULES))
    env = engine.create_env(render_cameras=False)
    try:
        if strict:
            env.task_config = replace(env.task_config, terminate_on_success=False)
        policy = policy_type(checkpoint, focus, scale, corrections)
        trace = PlacementTrace(policy)
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        terminal = trace.terminal_snapshot
        if terminal is None or result.steps != trace.step or result.score != len(terminal["counted_ids"]):
            raise ValueError("terminal snapshot disagrees with benchmark")
        columns = np.bincount(trace.columns[trace.target], minlength=5)[1:].tolist()
        meta = result.randomization
        return {"checkpoint_sha256": sha256(checkpoint), "focus": focus, "scale": scale, "strict": strict,
                "corrections": [item.record() for item in corrections],
                "seed": seed, "suite": suite, "column": result.source_column, "profile": meta["profile"],
                "score": result.score, "steps": result.steps, "success": result.success,
                "requested_score": columns[result.source_column - 1], "by_column": columns,
                "target_cookie_ids": terminal["counted_ids"], "safety_reason": trace.safety,
                "source_remaining": result.cookies_in_source, "terminal": trace.terminal, "physical_terminal": terminal,
                "color": meta["cookie_color"], "cameras": meta["cameras"], "light_diffuse": meta["light_diffuse"],
                "initial_pose": pose_label(meta).tolist(), "estimated_pose": policy.pose.tolist(),
                "task_prompt": engine._episode_task(result.source_column, seed), "goals_xy_m": trace.goals_xy,
                "batches": trace.events, "phase_geometry": trace.stages, "descent_observations": trace.measurements,
                "second_open_step": trace.events[1].get("open_start"), "first_placement_prefix": trace.first_batch_prefix,
                "action_sha256": trace.digest.hexdigest(), "applied_action_sha256": trace.applied_digest.hexdigest(),
                "wall_seconds": result.wall_seconds}
    finally:
        env.close()


def audit(job: tuple, row: dict) -> None:
    checkpoint, focus, scale, suite, seed, corrections, strict = job
    hold = (STRICT_RULES if strict else RULES)["success_hold_steps"]
    official_success = row["terminal"]["hold_steps"] >= RULES["success_hold_steps"] and row["score"] == 10 and (
        row["source_remaining"] == 70 and not row["safety_reason"])
    check_row((checkpoint, "advanced", suite, seed), {**row, "success": official_success})
    if (row["checkpoint_sha256"] != sha256(checkpoint) or (row["focus"], row["scale"], row["strict"]) != (focus, scale, strict)
            or row["corrections"] != [item.record() for item in corrections]
            or row["success"] != (official_success and row["terminal"]["hold_steps"] >= hold)
            or (row["steps"] != RULES["max_steps"] and not (row["safety_reason"] or (not strict and row["success"])))):
        raise ValueError("placement receipt differs from the frozen run")
    terminal = row["physical_terminal"]
    if terminal["step"] != row["steps"] or terminal["counted_ids"] != row["target_cookie_ids"]:
        raise ValueError("geometry and terminal provenance disagree")
    for batch in terminal["batches"]:
        if any(cookie["scored"] != (cookie["id"] in row["target_cookie_ids"]) for cookie in batch):
            raise ValueError("batch IDs disagree with terminal provenance")


def evaluate(pool, folder: Path, checkpoint: Path, timing: tuple[int, float], suite: str, seeds: list[int],
             corrections: tuple[Correction, ...] = (), *, strict: bool = True) -> list[dict]:
    jobs = [(checkpoint, *timing, suite, seed, corrections, strict) for seed in seeds]
    _record_once(folder / "request.json", {"checkpoint_sha256": sha256(checkpoint), "timing": list(timing), "suite": suite,
                 "strict": strict, "seeds": seeds, "corrections": [item.record() for item in corrections]})
    rows = _jobs(pool, episode, jobs, folder / "episodes", lambda job: str(job[4]), audit)
    _record_once(folder / "results.json", {"rows": rows})
    print(f"{folder.parent.name}/{folder.name}: {len(rows)} episodes", flush=True)
    return rows


def upstream_policy() -> PlacementPolicy:
    lock = read_json(Path(os.environ["A3OPT_PLACEMENT_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("placement policy requires the frozen checkpoint")
    corrections = tuple(Correction(column, batch, tuple(xyz), *limit) for column, batch, xyz, *limit in lock["corrections"])
    return PlacementPolicy(checkpoint, *lock["timing"], corrections)
