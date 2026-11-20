from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.advanced_trajectory import pose_label
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import Correction, PlacementPolicy, PlacementTrace, audit
from a3opt.refinement_eval import sha256
from a3opt.release_timing_agent import STRICT_RULES


class CombinedTrace(PlacementTrace):
    def bind_env(self, env) -> None:
        super().bind_env(env)
        self.released_config = replace(env.task_config, require_released=True)

    def reset(self, context=None) -> None:
        super().reset(context)
        self.course = self.released = None
        self.released_hold = 0

    def _terminal(self, env, info, hold: int, success: bool) -> dict:
        ids = np.flatnonzero(self.target)
        columns = np.bincount(self.columns[ids], minlength=5)[1:].tolist()
        return {"steps": self.step, "score": len(ids), "requested_score": columns[self.column - 1], "by_column": columns,
                "success": success, "target_cookie_ids": ids.tolist(), "source_remaining": int(info["cookies_in_source"]),
                "safety_reason": self.safety, "terminal": {"slot_occupancy": list(info["target_slot_occupancy"]),
                    "touches_all_walls": bool(info["target_touches_all_walls"]), "hold_steps": hold,
                    "exact_fill": bool(info["exact_2x5_fill"])}, "physical_terminal": self.snapshot(env, info),
                "action_sha256": self.digest.hexdigest(), "applied_action_sha256": self.applied_digest.hexdigest()}

    def add_frame(self, observation, action, info, *, env, **context) -> None:
        super().add_frame(observation, action, info, env=env, **context)
        final = self.step == RULES["max_steps"] or bool(self.safety)
        if self.course is None and (info["success"] or final):
            self.course = self._terminal(env, info, int(info["success_hold_count"]), bool(info["success"] and not self.safety))
        if not info["exact_2x5_fill"] and not final:
            self.released_hold = 0
            return
        course_config, course_target = env.task_config, self.target
        try:
            env.task_config = self.released_config
            self.target = np.zeros_like(course_target)
            for index in map(int, np.flatnonzero(course_target)):
                self.target[index] = env.privileged_cookie_in_target(index)
            exact = bool(info["exact_2x5_fill"] and np.array_equal(self.target, course_target))
            self.released_hold = self.released_hold + 1 if exact else 0
            if final:
                mask = tuple(map(bool, self.target))
                released_info = {**info, "exact_2x5_fill": exact, "target_slot_occupancy": env._target_slot_occupancy(mask),
                                 "target_touches_all_walls": env._collection_touches_all_walls(
                                     mask, env.TARGET_CENTER, env.TARGET_INNER_HALF_SIZE, is_target=True)}
                success = self.released_hold >= STRICT_RULES["success_hold_steps"] and not self.safety
                self.released = self._terminal(env, released_info, self.released_hold, success)
        finally:
            self.target = course_target
            env.task_config = course_config


def episode(checkpoint: Path, focus: int, scale: float, seed: int, model: tuple[Correction, ...], *,
            policy_type=PlacementPolicy) -> dict:
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        policy = policy_type(checkpoint, focus, scale, model)
        trace = CombinedTrace(policy)
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        if trace.course is None or trace.released is None or trace.step != result.steps:
            raise ValueError("combined evaluation did not reach both frozen score endpoints")
        meta = result.randomization
        common = {"checkpoint_sha256": sha256(checkpoint), "focus": focus, "scale": scale,
                  "corrections": [item.record() for item in model],
                  "seed": seed, "suite": "random", "column": result.source_column, "profile": meta["profile"],
                  "color": meta["cookie_color"], "cameras": meta["cameras"], "light_diffuse": meta["light_diffuse"],
                  "initial_pose": pose_label(meta).tolist(), "estimated_pose": policy.pose.tolist(),
                  "task_prompt": engine._episode_task(result.source_column, seed), "wall_seconds": result.wall_seconds}
        output = {}
        for name, terminal in (("default", trace.course), ("strict-500", trace.released)):
            stop = terminal["steps"]
            stages = [{key: event for key, event in geometry.items() if event["step"] <= stop} for geometry in trace.stages]
            output[name] = {**common, **terminal, "strict": name != "default", "phase_geometry": stages}
        return output
    finally:
        env.close()


def check(job: tuple, receipt: dict) -> None:
    checkpoint, focus, scale, seed, model = job
    for name, strict in (("default", False), ("strict-500", True)):
        audit((checkpoint, focus, scale, "random", seed, model, strict), receipt[name])


def evaluate(pool, folder: Path, checkpoint: Path, timing: tuple[int, float], seeds: list[int],
             models: dict[str, tuple[Correction, ...]], *, policy_type=PlacementPolicy) -> dict:
    _record_once(folder / "request.json", {"checkpoint_sha256": sha256(checkpoint), "timing": list(timing), "seeds": seeds,
                 "models": {name: [item.record() for item in model] for name, model in models.items()},
                 "rules": [RULES, STRICT_RULES]})
    names = list(models)
    labels = {model: name for name, model in models.items()}
    jobs = [(checkpoint, *timing, seed, models[name]) for seed in seeds for name in names]
    rollout = partial(episode, policy_type=policy_type)
    rows = _jobs(pool, rollout, jobs, folder / "episodes", lambda job: f"{labels[job[-1]]}/{job[-2]}", check)
    result = {name: rows[index::len(names)] for index, name in enumerate(names)}
    _record_once(folder / "results.json", result)
    return result
