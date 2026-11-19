from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from a3opt.backends.advanced_phase import JOINT, SAMPLES, AdvancedPhasePolicy, _coefficients, _design, _factors
from a3opt.backends.advanced_trajectory import POSE_SCALE, predict_pose, requested_column, visual_features
from a3opt.backends.trajectory import FRONT, resample


APPROACH, PRE_CLOSE, DESCEND, CLOSE, LIFT, TRANSFER, PLACE, OPEN, RETRACT = range(len(SAMPLES))
TRANSIT = (APPROACH, LIFT, TRANSFER, RETRACT)
SHORT_PROMPT = "transfer 10 cookies into target box"
FEATURES = 8
MAX_RETRY_STEPS = 8
MAX_CLOSE_BIAS = .04


@dataclass(frozen=True)
class Recipe:
    transit_scale: float = 1.0
    source_dx: float = 0.0
    source_dy: float = 0.0
    close_bias: float = 0.0
    retry_probability: float = 0.0
    target_dx: float = 0.0
    target_dy: float = 0.0
    settle_steps: int = 0
    joint_placement: bool = False

    def __post_init__(self) -> None:
        if not (0.75 <= self.transit_scale <= 1 and abs(self.source_dx) <= .008 and abs(self.source_dy) <= .008
                and 0 <= self.close_bias <= MAX_CLOSE_BIAS and 0 <= self.retry_probability <= 1
                and abs(self.target_dx) <= .01 and abs(self.target_dy) <= .01
                and self.settle_steps in range(17) and self.joint_placement in (False, True)):
            raise ValueError("invalid adaptive phase recipe")


def task_column(task: str) -> int:
    if task.strip().casefold() == SHORT_PROMPT:
        return 1
    if not task.strip().casefold().startswith("transfer 10 cookies from source column "):
        raise ValueError("this policy accepts the upstream 2x5 task instructions")
    return requested_column(task)


def grip_features(observation: dict, action: np.ndarray) -> np.ndarray:
    state = np.asarray(observation["observation.state"])
    velocity = np.asarray(observation["observation.velocity"])
    forces = np.asarray(observation["observation.force"])
    return np.array([state[7], state[7] - action[7], abs(velocity[7]), np.linalg.norm(state[:7] - action[:7]),
                     np.log1p(np.linalg.norm(forces[:3])), np.log1p(np.linalg.norm(forces[3:6])),
                     np.log1p(np.linalg.norm(forces[12:14])), np.log1p(abs(forces[16]))])


def grasp_probability(features: np.ndarray, detector: dict) -> float:
    logit = detector["intercept"] + ((features - detector["center"]) / detector["scale"]) @ detector["coefficients"]
    return float(1 / (1 + np.exp(-np.clip(logit, -35., 35.))))


class AdaptivePhasePolicy(AdvancedPhasePolicy):
    def __init__(self, model: dict, recipes: tuple[Recipe, ...], detector: dict | None = None):
        if len(recipes) != 4 or str(model["format"]) != AdvancedPhasePolicy.name:
            raise ValueError("expected a phase-model checkpoint and four column recipes")
        if detector is None and any(recipe.retry_probability for recipe in recipes):
            raise ValueError("grasp feedback requires a trained detector")
        self.model = model
        self.recipes = recipes
        self.detector = detector
        self.reset()

    def reset(self, context=None) -> None:
        super().reset(context)
        self.batch = self.stage = self.frame = 0
        self.started = self.ended = False
        self.retry_remaining = 0
        self.retries = [0, 0]
        self.recovery_bias = 0.
        self.last_probability = None
        self.pending_features = None

    def _phase_actions(self, column: int, recipe: Recipe) -> list[list[np.ndarray]]:
        model = self.model
        poses = model["poses"] / POSE_SCALE
        query = self.pose.copy()
        query[:2] += [recipe.source_dx, recipe.source_dy]
        query[2:4] += [recipe.target_dx, recipe.target_dy]
        query /= POSE_SCALE
        batches, previous = [], None
        for batch in range(2):
            rows = np.flatnonzero((model["columns"] == column) & (model["batches"] == batch))
            if not len(rows):
                raise ValueError(f"no phase demonstrations for source column {column}, batch {batch}")
            stages, start = [], 0
            for stage, size in enumerate(SAMPLES):
                factors = list(JOINT if recipe.joint_placement and stage >= PLACE else _factors(batch, stage))
                design, weights = _design(poses[rows][:, factors] - query[factors])
                coefficients = _coefficients(design, weights, float(model["phase_penalties"][batch, stage]))
                length = max(1, round(float(weights @ model["lengths"][rows, stage])))
                if stage in TRANSIT and recipe.transit_scale != 1.:
                    length = min(length, max(2, round(length * recipe.transit_scale)))
                trace = np.tensordot(coefficients, model["traces"][rows, start:start + size], axes=1)
                actions = resample(trace, length)
                if previous is not None:
                    blend_length = min(3, len(actions))
                    blend = np.linspace(0, 1, blend_length + 1)[1:, None]
                    actions[:blend_length] = (1 - blend) * previous + blend * actions[:blend_length]
                previous = actions[-1].copy()
                if stage == CLOSE:
                    actions[:, 7] -= recipe.close_bias * np.linspace(0., 1., len(actions))
                elif LIFT <= stage <= OPEN:
                    fade = np.linspace(1., 0., len(actions)) if stage == OPEN else 1.
                    actions[:, 7] -= recipe.close_bias * fade
                actions[:, [7, 15]] = np.clip(actions[:, [7, 15]], 0., 1.)
                if not np.isfinite(actions).all():
                    raise ValueError("invalid phase action plan")
                if stage == OPEN and recipe.settle_steps:
                    actions = np.concatenate((actions, np.repeat(actions[-1:], int(recipe.settle_steps), axis=0)))
                stages.append(actions)
                start += size
            batches.append(stages)
        return batches

    def _retry(self) -> np.ndarray:
        self.retry_remaining -= 1
        fraction = (MAX_RETRY_STEPS - self.retry_remaining) / MAX_RETRY_STEPS
        action = self.actions[self.batch][CLOSE][-1].copy()
        action[7] = max(0., action[7] - fraction * self.recovery_bias)
        self.stage, self.frame = CLOSE, len(self.actions[self.batch][CLOSE]) - 1
        return action

    def act(self, observation: dict, task: str = "") -> np.ndarray:
        if self.actions is None:
            column = task_column(task)
            self.recipe = self.recipes[column - 1]
            self.pose = predict_pose(self.model, *visual_features(observation[FRONT]))
            self.actions = self._phase_actions(column, self.recipe)
            self.cursor = (0, 0, 0)
        self.step += 1
        self.started = self.ended = False
        self.pending_features = None
        if self.retry_remaining:
            return self._retry()
        if self.cursor is None:
            return self.actions[-1][-1][-1].copy()
        self.batch, self.stage, self.frame = self.cursor
        stage = self.actions[self.batch][self.stage]
        if self.stage == LIFT and self.frame == 0 and not self.retries[self.batch]:
            self.pending_features = grip_features(observation, self.actions[self.batch][CLOSE][-1])
            if self.batch == 0 and self.detector is not None and self.recipe.retry_probability:
                self.last_probability = grasp_probability(self.pending_features, self.detector)
                time_to_open = sum(len(actions) for actions in self.actions[self.batch][LIFT:OPEN + 1])
                recovery_bias = min(.025, MAX_CLOSE_BIAS - self.recipe.close_bias)
                if (recovery_bias > 0 and self.last_probability < self.recipe.retry_probability
                        and self.step + MAX_RETRY_STEPS + time_to_open <= 500):
                    self.retries[self.batch] += 1
                    self.recovery_bias = recovery_bias
                    self.retry_remaining = MAX_RETRY_STEPS
                    return self._retry()
        self.started, self.ended = self.frame == 0, self.frame + 1 == len(stage)
        action = stage[self.frame].copy()
        if LIFT <= self.stage <= OPEN and self.recovery_bias:
            fade = (1. - self.frame / max(1, len(stage) - 1)) if self.stage == OPEN else 1.
            action[7] = max(0., action[7] - self.recovery_bias * fade)
        if not self.ended:
            self.cursor = self.batch, self.stage, self.frame + 1
        elif self.stage != RETRACT:
            self.cursor = self.batch, self.stage + 1, 0
        else:
            self.cursor = (1, 0, 0) if self.batch == 0 else None
            self.recovery_bias = 0.
        return action


def load_checkpoint(checkpoint: Path) -> AdaptivePhasePolicy:
    with np.load(checkpoint, allow_pickle=False) as archive:
        model = dict(archive)
    recipes = tuple(Recipe(*map(float, row)) for row in model.pop("adaptive_recipes"))
    detector = {name: model.pop(f"grip_{name}") for name in ("center", "scale", "coefficients", "intercept")}
    return AdaptivePhasePolicy(model, recipes, detector)


def upstream_policy() -> AdaptivePhasePolicy:
    return load_checkpoint(Path(os.environ["A3OPT_ADAPTIVE_CHECKPOINT"]))
