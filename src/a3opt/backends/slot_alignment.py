from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, AdaptivePhasePolicy, Recipe, load_checkpoint as load_adaptive
from a3opt.backends.release_residual import _ramp, nominal_kinematics


MAX_SHIFT_M = .012
FEATURE_COUNT = 7
KEYS = ("center", "scale", "weights", "bias", "column")


def release_features(pose: np.ndarray, phases: list[np.ndarray]) -> np.ndarray:
    solver = nominal_kinematics()
    cosine, sine = np.cos(pose[4]), np.sin(pose[4])
    target_y_axis = np.array([-sine, cosine])
    coordinates = []
    for stage in (PLACE, OPEN):
        limits = solver.ranges
        solver.data.qpos[solver.positions] = np.clip(phases[stage][-1, :7], limits[:, 0], limits[:, 1])
        solver.mujoco.mj_kinematics(solver.model, solver.data)
        tool_xy = solver.data.site_xpos[solver.site, :2]
        coordinates.append(float((tool_xy - pose[2:4]) @ target_y_axis))
    return np.r_[pose, coordinates]


def predict_shift(parameters: dict, features: np.ndarray) -> np.ndarray:
    distance = np.clip((features - parameters["center"]) / parameters["scale"], -3., 3.)
    return np.clip(parameters["bias"] + distance @ parameters["weights"], -MAX_SHIFT_M, MAX_SHIFT_M)


class SlotAlignmentPolicy(AdaptivePhasePolicy):
    name = "a3opt_slot_alignment"

    def __init__(self, reference: AdaptivePhasePolicy, parameters: dict):
        self.parameters = {key: np.asarray(parameters[key]) for key in KEYS}
        self.column = int(self.parameters["column"])
        vectors = [self.parameters[key] for key in ("center", "scale", "weights")]
        if (self.column not in range(1, 5) or any(vector.shape != (FEATURE_COUNT,) for vector in vectors)
                or self.parameters["bias"].shape != () or np.any(self.parameters["scale"] <= 0)
                or any(not np.isfinite(values).all() for values in self.parameters.values())):
            raise ValueError("invalid first-batch slot calibration")
        super().__init__(reference.model, reference.recipes, reference.detector)

    def _phase_actions(self, column: int, recipe: Recipe) -> list[list[np.ndarray]]:
        batches = super()._phase_actions(column, recipe)
        if column != self.column:
            return batches
        phases = batches[0]
        shift = float(predict_shift(self.parameters, release_features(self.pose, phases)))
        if shift == 0.:
            return batches
        direction = shift * np.array([-np.sin(self.pose[4]), np.cos(self.pose[4]), 0.])
        solver = nominal_kinematics()
        for stage in (PLACE, OPEN, RETRACT):
            actions = phases[stage]
            progress = _ramp(np.arange(1, len(actions) + 1) / len(actions))
            if stage == OPEN:
                progress[:] = 1.
            elif stage == RETRACT:
                progress = 1. - progress
            for action, weight in zip(actions, progress, strict=True):
                if weight:
                    action[:7] = solver.shift(action[:7], direction * weight)
        return batches


def from_adaptive(reference: AdaptivePhasePolicy) -> AdaptivePhasePolicy:
    if "slot_column" not in reference.model:
        return reference
    return SlotAlignmentPolicy(reference, {name: reference.model[f"slot_{name}"] for name in KEYS})


def upstream_policy() -> AdaptivePhasePolicy:
    return from_adaptive(load_adaptive(Path(os.environ["A3OPT_SLOT_CHECKPOINT"])))
