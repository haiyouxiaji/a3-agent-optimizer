from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256 as hash_actions
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _npz
from a3opt.backends.advanced_phase import SAMPLES, align_phases
from a3opt.backends.advanced_trajectory import pose_label, visual_features
from a3opt.backends.two_by_ten import SOURCE_COLUMN, TASK, TwoByTenPolicy, active_batches
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import render_frames, sha256


SHORT_STEPS = RULES["max_steps"]
NATIVE_STEPS = 3200
EVAL_RULES = {**RULES, "task": TASK, "first_grasp": "random", "max_steps": NATIVE_STEPS}
COLLECT_STEPS = NATIVE_STEPS


class TransferTrace:
    def __init__(self):
        self.steps = 0
        self.raw = hash_actions()
        self.applied = hash_actions()
        self.peak_count = 0
        self.info = None
        self.expert_step = None
        self.short = None
        self.times = {"first_target": None, "first_target_column_full": None, "target_full": None}
        self.batches = [([], []) for _ in range(4)]

    def capture_expert_action(self, expert, act, observation, task: str):
        batch, phase = expert.batch_index, expert.phase.name
        action = act(observation, task)
        if phase == "SELECT_COOKIE":
            batch = expert.batch_index
        self.expert_step = batch, phase
        return action

    def add_frame(self, observation, action, info, *, phase, env, **context) -> None:
        applied = np.asarray(info["applied_action"], dtype="<f8")
        self.raw.update(np.asarray(action, dtype="<f8").tobytes())
        self.applied.update(applied.tobytes())
        self.steps += 1
        self.info = info
        self.peak_count = max(self.peak_count, info["cookies_in_target"])
        for key, reached in zip(self.times, (info["cookies_in_target"] > 0, 10 in info["target_column_counts"],
                                             info["cookies_in_target"] == 20), strict=True):
            if reached and self.times[key] is None:
                self.times[key] = self.steps
        if self.expert_step is not None:
            batch, executed_phase = self.expert_step
            if batch < 4:
                actions, phases = self.batches[batch]
                actions.append(applied)
                phases.append(executed_phase)
        if self.steps == SHORT_STEPS:
            self.short = self.snapshot(env)

    def snapshot(self, env) -> dict:
        info = self.info
        target_ids = np.flatnonzero(info["cookies_in_target_mask"])
        source_x = np.asarray(env.SOURCE_POSITIONS)[:, 0]
        columns = np.searchsorted(np.unique(source_x), source_x)
        by_column = np.bincount(columns[target_ids], minlength=4).tolist()
        success = (info["success"] and info["cookies_in_target"] == 20
                   and info["cookies_in_source"] == 60 and not info.get("safety_reason"))
        return {"steps": self.steps, "score": info["cookies_in_target"], "requested_score": by_column[SOURCE_COLUMN - 1],
                "by_column": by_column, "target_cookie_ids": target_ids.tolist(),
                "target_column_counts": list(info["target_column_counts"]), "source_remaining": info["cookies_in_source"],
                "hold_steps": info["success_hold_count"], "success": bool(success), "peak_count": self.peak_count,
                "safety_reason": info.get("safety_reason"), "action_sha256": self.raw.hexdigest(),
                "applied_action_sha256": self.applied.hexdigest()}

    def result(self, result, env) -> dict:
        terminal = self.snapshot(env)
        if (result.score, result.steps, result.cookies_in_source) != (
            terminal["score"], terminal["steps"], terminal["source_remaining"],
        ):
            raise ValueError("upstream 2x10 result disagrees with the action or cookie trace")
        scene_sha256 = hash_actions(json.dumps(env.randomization_metadata, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return {
            **terminal,
            "task": TASK, "profile": result.randomization["profile"], "seed": result.seed,
            "column": result.source_column, "task_prompt": result.randomization["task_instruction"],
            "first_grasp": result.randomization["first_grasp_count"], "success": result.success,
            "color": result.randomization["cookie_color"], "initial_pose": pose_label(result.randomization).tolist(),
            "scene_sha256": scene_sha256, "milestone_steps": self.times, "failure_reason": result.failure_reason,
        }


def benchmark(profile: str, **overrides):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    rules = {**EVAL_RULES, **overrides}
    return make_cookie_benchmark(profile=profile, source_column=str(SOURCE_COLUMN), **rules)


def collect(profile: str, seed: int, first_grasp: int, directory: Path) -> dict:
    from a3_dual_arm_sim.controllers.cookie_2x10_expert import A3Cookie2x10Expert
    from a3_dual_arm_sim.workflows.policy_adapters import ExpertPolicyAdapter

    trace = TransferTrace()
    bench = benchmark(profile, first_grasp=first_grasp, max_steps=COLLECT_STEPS, complete_demonstrations=True)
    env = bench.create_env(render_cameras=False)
    content = {"seed": seed, "first_grasp": first_grasp, "profile": profile}

    class RecordingExpert(A3Cookie2x10Expert):
        def act(self, observation=None, task=""):
            return trace.capture_expert_action(self, super().act, observation, task)

    def expert_factory(environment):
        front = render_frames(environment, ("front",))["observation.images.front"]
        content["outline"], content["silhouette"] = visual_features(front)
        content["pose"] = pose_label(environment.randomization_metadata)
        return RecordingExpert(environment, first_grasp=first_grasp, sampling_seed=seed)

    expert = ExpertPolicyAdapter(expert_factory)
    try:
        result = bench.run_episode(expert, seed=seed, env=env, diagnostic=trace)
        row = trace.result(result, env)
        row["grasp_reports"] = result.randomization["grasp_reports"]
        reason = "" if row["success"] and row["requested_score"] == 20 else "incomplete_or_wrong_source"
        if not reason:
            traces = np.zeros((4, sum(SAMPLES), 16), dtype=np.float32)
            lengths = np.zeros((4, len(SAMPLES)), dtype=np.int32)
            for batch in active_batches(first_grasp):
                try:
                    actions, phases = trace.batches[batch]
                    traces[batch], lengths[batch] = align_phases(np.asarray(actions), np.asarray(phases))
                except ValueError as error:
                    reason = str(error)
                    break
            if not reason:
                content.update(traces=traces, lengths=lengths)
        path = directory / f"{seed}.npz"
        _npz(path, **content)
        return {**row, "accepted": not reason, "rejection": reason, "shard_sha256": sha256(path)}
    finally:
        env.close()


def evaluate(checkpoint: Path, profile: str, seed: int) -> dict:
    from a3opt.advanced_sim import EvaluatedPolicy

    bench, trace = benchmark(profile), TransferTrace()
    env = bench.create_env(render_cameras=False)
    try:
        policy = EvaluatedPolicy(TwoByTenPolicy(checkpoint))
        result = bench.run_episode(policy, seed=seed, env=env, diagnostic=trace)
        return {**trace.result(result, env), "checkpoint_sha256": sha256(checkpoint),
                "planned_steps": len(policy.policy.actions), "at_500": trace.short or trace.snapshot(env)}
    finally:
        env.close()


@lru_cache(maxsize=1)
def source_columns() -> np.ndarray:
    from a3_dual_arm_sim.config import load_config
    from a3_dual_arm_sim.paths import resource_root

    source_x = np.asarray(load_config(resource_root() / "configs/cookie_2x10.yaml").cookie_transfer.cookie_source_positions_m)[:, 0]
    return np.searchsorted(np.unique(source_x), source_x)


def check_terminal(row: dict, max_steps: int) -> None:
    ids = row["target_cookie_ids"]
    if not (0 <= row["requested_score"] <= row["score"] <= row["peak_count"] <= 20
            and ids == sorted(set(ids)) and all(0 <= cookie < 80 for cookie in ids)
            and len(row["by_column"]) == 4 and all(count >= 0 for count in row["by_column"])
            and len(row["target_column_counts"]) == 2 and all(count >= 0 for count in row["target_column_counts"])
            and row["score"] == len(ids) == sum(row["by_column"]) == sum(row["target_column_counts"])
            and row["requested_score"] == row["by_column"][SOURCE_COLUMN - 1]
            and 0 <= row["source_remaining"] <= 80 and 0 <= row["hold_steps"] <= row["steps"]
            and 1 <= row["steps"] <= max_steps):
        raise ValueError("receipt does not match terminal provenance or the control-step budget")
    observed = np.bincount(source_columns()[ids], minlength=4).tolist()
    if observed != row["by_column"]:
        raise ValueError("cookie identities disagree with source-column counts")
    for name in ("action_sha256", "applied_action_sha256"):
        digest = row[name]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("receipt has an invalid action digest")


def check_rollout(row: dict, profile: str, seed: int, first_grasp: int | str, max_steps: int) -> None:
    from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan

    plan = Cookie2x10Plan.from_seed(first_grasp, seed)
    if (row["task"], row["profile"], row["seed"], row["first_grasp"], row["column"], row["task_prompt"]) != (
        TASK, profile, seed, plan.first_grasp, SOURCE_COLUMN, plan.prompt(SOURCE_COLUMN),
    ):
        raise ValueError("receipt does not match the requested 2x10 scenario")
    check_terminal(row, max_steps)


def check_collection(job: tuple, row: dict) -> None:
    profile, seed, first_grasp, directory = job
    check_rollout(row, profile, seed, first_grasp, COLLECT_STEPS)
    path = directory / f"{seed}.npz"
    if sha256(path) != row["shard_sha256"]:
        raise ValueError("2x10 training shard differs from its recorded digest")
    with np.load(path, allow_pickle=False) as archive:
        has_actions = {"traces", "lengths"}.issubset(archive.files)
        if (int(archive["seed"]), str(archive["profile"]), int(archive["first_grasp"])) != (seed, profile, first_grasp):
            raise ValueError("2x10 training shard does not match its scheduled scenario")
        if has_actions != row["accepted"] or row["accepted"] and not (
            row["success"] and row["requested_score"] == 20 and not row["safety_reason"]
        ):
            raise ValueError("2x10 shard contains actions without a verified successful demonstration")
        if has_actions:
            traces, lengths = archive["traces"], archive["lengths"]
            active = np.asarray([batch in active_batches(first_grasp) for batch in range(4)])
            if (traces.shape != (4, sum(SAMPLES), 16) or lengths.shape != (4, len(SAMPLES))
                    or not np.isfinite(traces).all() or not (lengths[active] > 0).all()
                    or traces[~active].any() or lengths[~active].any() or lengths.sum() > row["steps"]):
                raise ValueError("invalid aligned actions or skipped-grasp data")


def check_evaluation(job: tuple, row: dict) -> None:
    checkpoint, profile, seed = job
    check_rollout(row, profile, seed, "random", EVAL_RULES["max_steps"])
    prefix = row["at_500"]
    check_terminal(prefix, min(row["steps"], SHORT_STEPS))
    for snapshot in (row, prefix):
        success = (snapshot["target_column_counts"] == [10, 10] and snapshot["source_remaining"] == 60
                   and snapshot["hold_steps"] >= EVAL_RULES["success_hold_steps"] and not snapshot["safety_reason"])
        if snapshot["success"] != success:
            raise ValueError("2x10 evaluation disagrees with the success contract")
    if (prefix["steps"] != min(row["steps"], SHORT_STEPS) or prefix["peak_count"] > row["peak_count"]
            or row["steps"] <= SHORT_STEPS and any(row[key] != value for key, value in prefix.items())):
        raise ValueError("500-step snapshot disagrees with the native rollout")
    if row["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("2x10 evaluation disagrees with the checkpoint or the success contract")
