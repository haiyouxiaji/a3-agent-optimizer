from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, AdaptivePhasePolicy, Recipe, load_checkpoint as load_adaptive
from a3opt.backends.release_residual import _ramp, nominal_kinematics


MAX_OFFSET_M = np.array([.010, .010, .012])
MAX_OPEN_HOLD = 8


@dataclass(frozen=True)
class ReleaseRecipe:
    offset_m: tuple[float, float, float] = (0., 0., 0.)
    open_hold: int = 0

    def __post_init__(self) -> None:
        offset = np.asarray(self.offset_m)
        if (offset.shape != (3,) or not np.all(np.abs(offset) <= MAX_OFFSET_M)
                or self.open_hold not in range(MAX_OPEN_HOLD + 1)):
            raise ValueError("invalid second-batch release recipe")


class SecondBatchPolicy(AdaptivePhasePolicy):
    name = "a3opt_second_batch"

    def __init__(self, reference: AdaptivePhasePolicy, column: int, release: ReleaseRecipe):
        if column not in range(1, 5):
            raise ValueError("invalid source column for second-batch release")
        self.release_column = column
        self.release = release
        super().__init__(reference.model, reference.recipes, reference.detector)

    def _phase_actions(self, column: int, recipe: Recipe) -> list[list[np.ndarray]]:
        batches = super()._phase_actions(column, recipe)
        if column != self.release_column:
            return batches
        phases = batches[1]
        lateral_x, lateral_y, height = self.release.offset_m
        if any(self.release.offset_m):
            cosine, sine = np.cos(self.pose[4]), np.sin(self.pose[4])
            offset = np.array([cosine * lateral_x - sine * lateral_y, sine * lateral_x + cosine * lateral_y, height])
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
                        action[:7] = solver.shift(action[:7], offset * weight)
        if self.release.open_hold:
            opening = phases[OPEN]
            phases[OPEN] = np.concatenate((opening, np.repeat(opening[-1:], self.release.open_hold, axis=0)))
        return batches


def load_checkpoint(checkpoint: Path) -> AdaptivePhasePolicy:
    with np.load(checkpoint, allow_pickle=False) as archive:
        if "second_batch_column" not in archive:
            return load_adaptive(checkpoint)
        column = int(archive["second_batch_column"])
        release = ReleaseRecipe(tuple(archive["second_batch_offset_m"].tolist()), int(archive["second_batch_open_hold"]))
    return SecondBatchPolicy(load_adaptive(checkpoint), column, release)


def upstream_policy() -> AdaptivePhasePolicy:
    return load_checkpoint(Path(os.environ["A3OPT_SECOND_BATCH_CHECKPOINT"]))
