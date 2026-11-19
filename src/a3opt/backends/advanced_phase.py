from pathlib import Path

import numpy as np

from a3opt.backends.advanced_trajectory import AdvancedTrajectoryPolicy, POSE_SCALE
from a3opt.backends.trajectory import resample


BOUNDARIES = ("PRE_CLOSE", "DESCEND", "CLOSE", "LIFT", "MOVE_TO_SLOT", "DESCEND_TO_PLACE", "OPEN", "RETRACT")
SAMPLES = (28, 16, 32, 32, 32, 40, 28, 18, 28)
SOURCE = (0, 1)
TARGET = (2, 3, 4)
JOINT = (*SOURCE, *TARGET)
FACTORS = (SOURCE, SOURCE, SOURCE, SOURCE, SOURCE, JOINT, TARGET, TARGET, TARGET)
PENALTIES = (.0001, .001, .006, .025)


def align_phases(actions: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = [0]
    for boundary in BOUNDARIES:
        matches = np.flatnonzero(phases[starts[-1] + 1:] == boundary)
        if not len(matches):
            raise ValueError(f"retained expert has no ordered {boundary} phase")
        starts.append(starts[-1] + 1 + int(matches[0]))
    split = np.split(actions, starts[1:])
    return np.concatenate([resample(part, size) for part, size in zip(split, SAMPLES, strict=True)]), np.diff(
        [*starts, len(actions)]
    )


def _factors(batch: int, stage: int) -> tuple[int, ...]:
    return JOINT if batch == 1 and stage == 0 else FACTORS[stage]


def _design(offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(offset, axis=1)
    bandwidth = max(.45, float(np.quantile(distance, .75)) * .85)
    weights = np.exp(-.5 * (distance / bandwidth)**2)
    weights /= weights.sum()
    return np.column_stack((np.ones(len(offset)), offset)), weights


def _coefficients(design: np.ndarray, weights: np.ndarray, ridge: float) -> np.ndarray:
    weighted_design = design.T * weights
    penalty = np.diag([0., *([ridge] * (design.shape[1] - 1))])
    return np.linalg.solve(weighted_design @ design + penalty, weighted_design)[0]


def select_penalties(dataset: dict) -> tuple[np.ndarray, dict]:
    poses = dataset["poses"] / POSE_SCALE
    traces = dataset["traces"]
    scores = np.zeros((2, len(SAMPLES), len(PENALTIES)))
    counts = np.zeros((2, len(SAMPLES)))
    for column in range(1, 5):
        for batch in range(2):
            group = np.flatnonzero((dataset["columns"] == column) & (dataset["batches"] == batch))
            if len(group) < 4:
                continue
            start = 0
            for stage, size in enumerate(SAMPLES):
                factors = list(_factors(batch, stage))
                for held in group:
                    training = group[group != held]
                    design, weights = _design(poses[training][:, factors] - poses[held, factors])
                    target = traces[held, start:start + size]
                    examples = traces[training, start:start + size]
                    for index, ridge in enumerate(PENALTIES):
                        coefficients = _coefficients(design, weights, ridge)
                        prediction = np.tensordot(coefficients, examples, axes=1)
                        residual = prediction - target
                        scores[batch, stage, index] += np.mean(residual**2) + np.mean(residual[-1]**2)
                    counts[batch, stage] += 1
                start += size
    if np.any(counts == 0):
        raise ValueError("insufficient groups for train-only phase validation")
    scores /= counts[..., None]
    penalties = np.asarray(PENALTIES)[scores.argmin(axis=-1)]
    return penalties, {"grid": list(PENALTIES), "train_only_loo_loss": scores.tolist(),
                       "retained_batches": counts.astype(int).tolist(), "selected": penalties.tolist()}


class AdvancedPhasePolicy(AdvancedTrajectoryPolicy):
    name = "a3opt_advanced_phase"

    def close(self) -> None:
        self.reset()

    def _action_plan(self, column: int, pose: np.ndarray) -> np.ndarray:
        model = self.model
        poses = model["poses"] / POSE_SCALE
        query = pose / POSE_SCALE
        stages = []
        for batch in range(2):
            rows = np.flatnonzero((model["columns"] == column) & (model["batches"] == batch))
            if not len(rows):
                raise ValueError(f"no phase demonstrations for source column {column}, batch {batch}")
            start = 0
            for stage, size in enumerate(SAMPLES):
                factors = list(_factors(batch, stage))
                design, weights = _design(poses[rows][:, factors] - query[factors])
                coefficients = _coefficients(design, weights, float(model["phase_penalties"][batch, stage]))
                length = max(1, round(float(weights @ model["lengths"][rows, stage])))
                trace = np.tensordot(coefficients, model["traces"][rows, start:start + size], axes=1)
                actions = resample(trace, length)
                if stages:
                    blend_length = min(3, len(actions))
                    blend = np.linspace(0, 1, blend_length + 1)[1:, None]
                    actions[:blend_length] = (1 - blend) * stages[-1][-1] + blend * actions[:blend_length]
                stages.append(actions)
                start += size
        plan = np.concatenate(stages)
        plan[:, [7, 15]] = np.clip(plan[:, [7, 15]], 0, 1)
        if not np.isfinite(plan).all():
            raise ValueError("invalid phase action plan")
        return plan


def load_advanced_policy(checkpoint: Path) -> AdvancedTrajectoryPolicy:
    with np.load(checkpoint, allow_pickle=False) as archive:
        policy_type = AdvancedPhasePolicy if str(archive["format"]) == AdvancedPhasePolicy.name else AdvancedTrajectoryPolicy
    return policy_type(checkpoint)
