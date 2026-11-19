from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from a3opt.backends.trajectory import FRONT, resample


POSE_SCALE = np.array([0.020, 0.020, 0.020, 0.020, 0.080])
BOUNDARIES = ("DESCEND", "LIFT", "MOVE_TO_SLOT", "DESCEND_TO_PLACE", "OPEN", "RETRACT")
PHASE_SAMPLES = (40, 22, 26, 30, 18, 18, 26)
FACTORS = ((0, 1), (0, 1), (0, 1), (0, 1, 2, 3, 4), (2, 3, 4), (2, 3, 4), (2, 3, 4))
COLUMN = re.compile(r"\bsource column ([1-4])\b", flags=re.IGNORECASE)


def requested_column(task: str) -> int:
    match = COLUMN.search(task)
    if match is None:
        raise ValueError("the policy requires an explicit source column from 1 to 4")
    return int(match.group(1))


def _outline(mask: np.ndarray) -> np.ndarray:
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [cv2.convexHull(contour)[:, 0, :] for contour in contours if cv2.contourArea(contour) >= 40]
    if len(candidates) < 2:
        return np.zeros(24)
    trays = sorted(sorted(candidates, key=cv2.contourArea, reverse=True)[:2], key=lambda hull: hull[:, 0].mean())
    features = []
    for hull in trays:
        top, right, bottom, left = (hull[hull[:, axis].argmin() if direction < 0 else hull[:, axis].argmax()]
                                    for axis, direction in ((1, -1), (0, 1), (1, 1), (0, -1)))
        features.extend([*top, *right, *bottom, *left, *hull.mean(axis=0),
                         np.sqrt(cv2.contourArea(hull)), len(hull)])
    return np.asarray(features, dtype=float)


def visual_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    rgb = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    peak = rgb.max(axis=-1)
    chroma = peak - rgb.min(axis=-1)
    neutral = np.zeros((256, 256), dtype=np.uint8)
    neutral[75:203, 11:236] = (chroma[75:203, 11:236] < peak[75:203, 11:236] * .30) & (
        (peak[75:203, 11:236] > 54) & (blue[75:203, 11:236] > red[75:203, 11:236] * .88)
    )
    mask = cv2.morphologyEx(neutral, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    table = (red - blue > 24) & (red > green + 7) & (green > blue + 7)
    first_on_table = []
    for left in (*range(5, 71, 8), *range(158, 251, 8)):
        visible = np.nonzero(table[:83, left:left + 4].mean(axis=1) >= .75)[0]
        first_on_table.append(float(visible[0]) if len(visible) else 83.0)
    anchors = cv2.resize(table[:78].astype(np.float32), (32, 10), interpolation=cv2.INTER_AREA).ravel()
    silhouette = cv2.resize(mask.astype(np.float32)[74:204], (28, 16), interpolation=cv2.INTER_AREA).ravel()
    outline = np.r_[_outline(mask), first_on_table]
    return outline, np.r_[anchors, silhouette]


def pose_label(metadata: dict) -> np.ndarray:
    quat_w, quat_x, quat_y, quat_z = metadata["target_bin_quaternion"]
    yaw = np.arctan2(2 * (quat_w * quat_z + quat_x * quat_y), 1 - 2 * (quat_y**2 + quat_z**2))
    return np.array([*metadata["source_bin_position"][:2], *metadata["target_bin_position"][:2], yaw])


def _basis(outlines, silhouettes, center, scale, variant):
    contours = np.clip((outlines - center) / scale, -6, 6)
    if variant == "silhouette":
        return np.column_stack((contours, 3 * silhouettes))
    if variant == "quadratic":
        rows, cols = np.triu_indices(contours.shape[1])
        return np.column_stack((contours, 0.3 * contours[:, rows] * contours[:, cols], silhouettes))
    raise ValueError(f"unknown visual feature variant {variant}")


def fit_pose(outlines: np.ndarray, silhouettes: np.ndarray, labels: np.ndarray, seeds: np.ndarray) -> tuple[dict, dict]:
    valid = seeds % 5 == 0
    if valid.sum() < 3 or (~valid).sum() < 10 or not np.isfinite(labels).all():
        raise ValueError("insufficient training-only samples for pose calibration")
    center = outlines[~valid].mean(axis=0)
    scale = np.maximum(outlines[~valid].std(axis=0), 1.)
    choices, models = [], []
    for variant in ("silhouette", "quadratic"):
        design = _basis(outlines, silhouettes, center, scale, variant)
        mean_features = design[~valid].mean(axis=0)
        mean_pose = labels[~valid].mean(axis=0)
        train = design[~valid] - mean_features
        truth = labels[~valid] - mean_pose
        u_matrix, singular, vt_matrix = np.linalg.svd(train, full_matrices=False)
        projected = u_matrix.T @ truth
        for penalty in (0.1, 1.0, 8.0):
            coeff = (vt_matrix.T * (singular / (singular**2 + penalty))) @ projected
            errors = np.abs((design[valid] - mean_features) @ coeff + mean_pose - labels[valid])
            relative_error = float(np.mean(errors / POSE_SCALE))
            choices.append({"variant": variant, "ridge": penalty, "relative_mae": relative_error,
                            "mae_mm_and_mrad": (1000 * errors.mean(axis=0)).tolist()})
            models.append({"pose_variant": variant, "pose_center": center, "pose_scale": scale,
                           "pose_design_mean": mean_features, "pose_mean": mean_pose, "pose_coeff": coeff})
    winner = min(range(len(choices)), key=lambda index: choices[index]["relative_mae"])
    return models[winner], {"training": int((~valid).sum()), "validation": int(valid.sum()),
                            "choices": choices, "selected": choices[winner]}


def predict_pose(model: dict, outline: np.ndarray, silhouette: np.ndarray) -> np.ndarray:
    design = _basis(outline[None], silhouette[None], model["pose_center"], model["pose_scale"],
                    str(model["pose_variant"]))[0]
    return model["pose_mean"] + (design - model["pose_design_mean"]) @ model["pose_coeff"]


def align_batch(actions: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if actions.shape != (len(phases), 16) or not np.isfinite(actions).all():
        raise ValueError("invalid phase-aligned applied actions")
    starts = [0]
    for phase in BOUNDARIES:
        matches = np.flatnonzero((phases == phase) & (np.arange(len(phases)) > starts[-1]))
        if not len(matches):
            raise ValueError(f"missing expert phase {phase}")
        starts.append(int(matches[0]))
    stages = np.split(actions, starts[1:])
    return np.concatenate([resample(stage, count) for stage, count in zip(stages, PHASE_SAMPLES)]), np.diff(
        [*starts, len(actions)]
    )


class AdvancedTrajectoryPolicy:
    action_mode = "joint_position"
    name = "a3opt_advanced_trajectory"
    finished = False

    def __init__(self, checkpoint: str | Path):
        with np.load(checkpoint, allow_pickle=False) as archive:
            self.model = dict(archive)
        if str(self.model["format"]) != self.name:
            raise ValueError("incorrect advanced trajectory checkpoint format")
        self.reset()

    def reset(self, context=None) -> None:
        self.actions = None
        self.pose = None
        self.step = 0

    @property
    def camera_names(self) -> tuple[str, ...]:
        return ("front",) if self.actions is None else ()

    def _action_plan(self, column: int, pose: np.ndarray) -> np.ndarray:
        stages = []
        model = self.model
        for batch in range(2):
            eligible = np.flatnonzero((model["columns"] == column) & (model["batches"] == batch))
            if not len(eligible):
                if stages:
                    break
                raise ValueError(f"no training demonstrations for source column {column}")
            start = 0
            for stage, (sample_count, factors) in enumerate(zip(PHASE_SAMPLES, FACTORS)):
                if batch == 1 and stage == 0 or str(model["mode"]) == "joint":
                    factors = tuple(range(5))
                offset = (model["poses"][eligible][:, factors] - pose[list(factors)]) / POSE_SCALE[list(factors)]
                distances = np.linalg.norm(offset, axis=1)
                nearest = np.argsort(distances, kind="stable")[:int(model["neighbors"])]
                selected = eligible[nearest]
                bandwidth = max(0.35, float(distances[nearest[-1]]) * .65)
                weights = np.exp(-.5 * (distances[nearest] / bandwidth)**2)
                weights /= weights.sum()
                design = np.column_stack((np.ones(len(selected)), offset[nearest]))
                weighted = design.T * weights
                penalty = np.diag([0., *([float(model["ridge"])] * len(factors))])
                coefficients = np.linalg.solve(weighted @ design + penalty, weighted)[0]
                source = model["traces"][selected, start:start + sample_count]
                length = max(1, round(float(weights @ model["lengths"][selected, stage])))
                actions = resample(np.tensordot(coefficients, source, axes=1), length)
                if stages:
                    blend_length = min(3, length)
                    blend = np.linspace(0, 1, blend_length + 1)[1:, None]
                    actions[:blend_length] = (1 - blend) * stages[-1][-1] + blend * actions[:blend_length]
                stages.append(actions)
                start += sample_count
        actions = np.concatenate(stages)
        actions[:, [7, 15]] = np.clip(actions[:, [7, 15]], 0, 1)
        if not np.isfinite(actions).all():
            raise ValueError("invalid planned joint actions")
        return actions

    def act(self, observation: dict, task: str = "") -> np.ndarray:
        if self.actions is None:
            column = requested_column(task)
            self.pose = predict_pose(self.model, *visual_features(observation[FRONT]))
            self.actions = self._action_plan(column, self.pose)
        action = self.actions[min(self.step, len(self.actions) - 1)]
        self.step += 1
        return action.copy()
