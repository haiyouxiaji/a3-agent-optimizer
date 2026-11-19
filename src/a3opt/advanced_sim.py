from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from a3opt.backends.advanced_trajectory import AdvancedTrajectoryPolicy, align_batch, pose_label, visual_features
from a3opt.refinement_eval import render_frames, sha256


def benchmark(column: str, *, collection: bool = False):
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    return make_cookie_benchmark(
        task="2x5", profile="advanced", source_column=column, prompt_style="task",
        max_steps=750 if collection else 500, complete_demonstrations=collection,
        success_hold_steps=5, slot_tolerance_m=(0.015, 0.005), wall_contact_tolerance_m=0.03,
    )


class Probe:
    action_mode = "joint_position"
    finished = True

    def bind_env(self, env) -> None:
        front = render_frames(env, ("front",))["observation.images.front"]
        self.visual = visual_features(front)
        self.pose = pose_label(env.randomization_metadata)
        self.color = env.randomization_metadata["cookie_color"]

    def act(self, observation: dict, task: str) -> np.ndarray:
        return observation["observation.state"]


def probe_training_scenes(seeds: tuple[int, ...]) -> list[tuple]:
    import cv2

    cv2.setNumThreads(1)
    bench, probe = benchmark("1"), Probe()
    env = bench.create_env(render_cameras=False)
    try:
        records = []
        for seed in seeds:
            bench.run_episode(probe, seed=seed, env=env)
            records.append((seed, *probe.visual, probe.pose, probe.color))
        return records
    finally:
        env.close()


class ExpertTrace:
    def __init__(self, adapter, env, column: int):
        self.adapter = adapter
        self.column = column
        self.pose = None
        self.frames = {0: [], 1: []}
        self.phases = {0: [], 1: []}
        self.verified: set[int] = set()
        self.last_info = {}

    def add_frame(self, observation, action, info, *, env, **context) -> None:
        if self.pose is None:
            self.pose = pose_label(env.randomization_metadata)
        expert = self.adapter.expert
        batch = expert.batch_index
        if batch in self.frames:
            self.frames[batch].append(np.asarray(info["applied_action"], dtype=np.float32))
            self.phases[batch].append(expert.phase.name)
        if batch in (1, 2):
            required = batch * 5
            target = np.flatnonzero(info["cookies_in_target_mask"])
            source = np.asarray(env.SOURCE_POSITIONS)[:, 0]
            correct_x = np.unique(source)[self.column - 1]
            if (len(target) == required and info["cookies_in_source"] == 80 - required
                    and np.all(source[target] == correct_x) and not info.get("safety_reason")):
                self.verified.add(batch - 1)
        self.last_info = info


def collect_expert(seed: int, column: int, folder: Path) -> dict:
    from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert, A3VariedColumnBatchExpert
    from a3_dual_arm_sim.workflows.policy_adapters import ExpertPolicyAdapter

    def expert_factory(env):
        if column == 1:
            return A3SameColumnBatchExpert(env)
        expert = A3VariedColumnBatchExpert(env)
        expert.requested_source_column_index = column - 1
        return expert

    started = time.monotonic()
    bench = benchmark(str(column), collection=True)
    env = bench.create_env(render_cameras=False)
    adapter = ExpertPolicyAdapter(expert_factory, name="upstream_same_column")
    trace = ExpertTrace(adapter, env, column)
    try:
        try:
            result = bench.run_episode(adapter, seed=seed, env=env, diagnostic=trace)
        except RuntimeError as error:
            if "unreachable with vertical grasp" not in str(error) and "Cookies in the source column are unavailable" not in str(error):
                raise
            return {"seed": seed, "column": column, "score": trace.last_info.get("cookies_in_target", 0),
                    "success": False, "steps": sum(map(len, trace.frames.values())), "phase": adapter.phase_name,
                    "color": bench.last_attempt_metadata.get("cookie_color"), "batches": [], "shard": None,
                    "failure_reason": str(error), "elapsed_seconds": time.monotonic() - started}
        row = {
            "seed": seed, "column": column, "score": result.score, "success": result.success,
            "steps": result.steps, "failure_reason": result.failure_reason, "phase": adapter.phase_name,
            "color": result.randomization.get("cookie_color"), "elapsed_seconds": time.monotonic() - started,
            "batches": [], "shard": None, "rejected_batches": {},
        }
        content = {"pose": trace.pose, "seed": seed, "column": column}
        if not trace.last_info.get("safety_reason"):
            for batch in sorted(trace.verified):
                if batch == 1 and (0 not in row["batches"] or not result.success):
                    continue
                actions = np.asarray(trace.frames[batch])
                phases = np.asarray(trace.phases[batch])
                try:
                    aligned, lengths = align_batch(actions, phases)
                except ValueError as error:
                    row["rejected_batches"][str(batch)] = str(error)
                    continue
                row["batches"].append(batch)
                content.update({f"actions_{batch}": actions, f"phases_{batch}": phases,
                                f"trace_{batch}": aligned, f"lengths_{batch}": lengths})
        if row["batches"]:
            path = folder / f"{column}-{seed}.npz"
            with path.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream, **content)
            path.with_suffix(".tmp").replace(path)
            row.update(shard=path.name, sha256=sha256(path), pose=trace.pose.tolist())
        return row
    finally:
        env.close()


class EvaluatedPolicy:
    action_mode = "joint_position"
    finished = False

    def __init__(self, policy: AdvancedTrajectoryPolicy):
        self.policy = policy

    def bind_env(self, env) -> None:
        self.cameras = lambda names: render_frames(env, names)
        source_x = np.asarray(env.SOURCE_POSITIONS)[:, 0]
        self.columns = np.searchsorted(np.unique(source_x), source_x) + 1

    def reset(self, context=None) -> None:
        self.policy.reset()
        self.digest = hashlib.sha256()
        self.target = np.zeros(len(self.columns), dtype=bool)
        self.first_batch = None
        self.step = 0
        self.safety = None

    def act(self, observation: dict, task: str) -> np.ndarray:
        if self.policy.camera_names:
            observation = {**observation, **self.cameras(self.policy.camera_names)}
        self.step += 1
        action = self.policy.act(observation, task)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("invalid evaluation action")
        self.digest.update(np.asarray(action, dtype="<f8").tobytes())
        return action

    def add_frame(self, observation: dict, action: np.ndarray, info: dict, **context) -> None:
        self.target = np.asarray(info["cookies_in_target_mask"], dtype=bool)
        self.safety = info.get("safety_reason")
        if self.first_batch is None and info["cookies_in_target"] >= 5:
            self.first_batch = self.step


def evaluate_episode(checkpoint: Path, column_mode: str, seed: int) -> dict:
    import cv2

    cv2.setNumThreads(1)
    bench = benchmark(column_mode)
    env = bench.create_env(render_cameras=False)
    policy = EvaluatedPolicy(AdvancedTrajectoryPolicy(checkpoint))
    try:
        result = bench.run_episode(policy, seed=seed, env=env, diagnostic=policy)
        provenance = np.bincount(policy.columns[policy.target], minlength=5)[1:]
        if result.score != provenance.sum() or result.steps != policy.step:
            raise RuntimeError("upstream score does not match the recorded steps and cookie IDs")
        return {
            "seed": seed, "column": result.source_column, "score": result.score, "success": result.success,
            "requested_score": int(provenance[result.source_column - 1]), "by_column": provenance.tolist(),
            "steps": result.steps, "wall_seconds": result.wall_seconds, "first_batch_step": policy.first_batch,
            "failure_reason": result.failure_reason, "safety_reason": policy.safety,
            "source_remaining": result.cookies_in_source, "action_sha256": policy.digest.hexdigest(),
            "color": result.randomization["cookie_color"], "estimated_pose": policy.policy.pose.tolist(),
            "initial_pose": pose_label(result.randomization).tolist(),
        }
    finally:
        env.close()
