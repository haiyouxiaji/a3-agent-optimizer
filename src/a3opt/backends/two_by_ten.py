from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from a3opt.backends.advanced_phase import SAMPLES
from a3opt.backends.advanced_trajectory import FRONT, POSE_SCALE, predict_pose, visual_features
from a3opt.backends.trajectory import resample


TASK = "2x10"
SOURCE_COLUMN = 1
COUNTS = tuple(range(10))
RECIPES = ("mean", "visual")
PROMPT = re.compile(
    r"transfer 20 cookies from source column ([1-4]) into the 2x10 target box\. "
    r"(?:grasp ([1-9]) cookies first,|skip grasp 1 and transfer)"
)


def instruction(task: str) -> tuple[int, int]:
    from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan

    normalized = " ".join(task.casefold().split())
    match = PROMPT.match(normalized)
    if match is None:
        raise ValueError("expected a native 2x10 instruction with source column and grasp plan")
    column, first_grasp = int(match[1]), int(match[2] or 0)
    if normalized != Cookie2x10Plan(first_grasp, first_grasp).prompt(column).casefold():
        raise ValueError("inconsistent native 2x10 grasp plan")
    return column, first_grasp


def active_batches(first_grasp: int) -> tuple[int, ...]:
    return (0, 1, 2, 3) if first_grasp else (1, 3)


class TwoByTenPolicy:
    action_mode = "joint_position"
    name = "a3opt_two_by_ten_v1"
    finished = False

    def __init__(self, checkpoint: Path):
        with np.load(checkpoint, allow_pickle=False) as archive:
            self.model = dict(archive)
        self.recipe = str(self.model["recipe"])
        if str(self.model["format"]) != self.name or self.recipe not in RECIPES:
            raise ValueError("incorrect native 2x10 checkpoint format or recipe")
        self.reset()

    def reset(self, context=None) -> None:
        self.actions = None
        self.pose = None
        self.step = 0

    def close(self) -> None:
        self.reset()

    @property
    def camera_names(self) -> tuple[str, ...]:
        return ("front",) if self.actions is None and self.recipe == "visual" else ()

    def _plan(self, first_grasp: int) -> np.ndarray:
        model = self.model
        selected = np.flatnonzero(model["counts"] == first_grasp)
        if len(selected) < 2:
            raise ValueError(f"insufficient 2x10 demonstrations for first grasp {first_grasp}")
        weights = np.ones(len(selected))
        if self.recipe == "visual":
            distance = np.linalg.norm((model["poses"][selected] - self.pose) / POSE_SCALE, axis=1)
            bandwidth = max(.45, float(np.median(distance)))
            log_weights = -.5 * (distance / bandwidth) ** 2
            weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        stages = []
        for batch in active_batches(first_grasp):
            start = 0
            for stage, size in enumerate(SAMPLES):
                batch_stage = model["traces"][selected, batch, start:start + size]
                length = max(1, round(float(weights @ model["lengths"][selected, batch, stage])))
                actions = resample(np.tensordot(weights, batch_stage, axes=1), length)
                if stages:
                    blend_size = min(length, 3)
                    blend = np.arange(1, blend_size + 1)[:, None] / blend_size
                    actions[:blend_size] = (1 - blend) * stages[-1][-1] + blend * actions[:blend_size]
                stages.append(actions)
                start += size
        actions = np.concatenate(stages)
        actions[:, [7, 15]] = np.clip(actions[:, [7, 15]], 0, 1)
        if not np.isfinite(actions).all():
            raise ValueError("invalid native 2x10 joint action plan")
        return actions

    def act(self, observation: dict, task: str = "") -> np.ndarray:
        if self.actions is None:
            column, first_grasp = instruction(task)
            if column != SOURCE_COLUMN:
                raise ValueError("this 2x10 checkpoint supports source column 1")
            if self.recipe == "visual":
                self.pose = predict_pose(self.model, *visual_features(observation[FRONT]))
            self.actions = self._plan(first_grasp)
        action = self.actions[min(self.step, len(self.actions) - 1)]
        self.step += 1
        return action.copy()


def create_policy() -> TwoByTenPolicy:
    return TwoByTenPolicy(Path(os.environ["A3OPT_2X10_CHECKPOINT"]))
