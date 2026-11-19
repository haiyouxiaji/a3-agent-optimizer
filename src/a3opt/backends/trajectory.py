from __future__ import annotations

from pathlib import Path

import numpy as np

FRONT = "observation.images.front"
IMAGE_SIZE = 128
JOINTS = 16
GRIPPER = 7
GRID = (12, 180, 16, 200)


def layout_features(image: np.ndarray) -> np.ndarray:
    height, width, channels = image.shape
    if height % IMAGE_SIZE or width % IMAGE_SIZE or channels != 3:
        raise ValueError(f"unsupported front image shape {image.shape}")
    rgb = image.reshape(IMAGE_SIZE, height // IMAGE_SIZE, IMAGE_SIZE, width // IMAGE_SIZE, 3)
    rgb = rgb.mean(axis=(1, 3)).astype(np.uint8).astype(float)
    source = rgb[38:91, 43:116]
    red, green, blue = np.moveaxis(source, -1, 0)
    source_weights = np.maximum(np.minimum(red - 120, green - 72), 0) * (red - blue > 55)
    target = rgb[58:96, 11:44]
    red, green, blue = np.moveaxis(target, -1, 0)
    target_weights = np.clip((blue - 72) / 30, 0, 1) * np.clip((blue - red + 4) / 12, 0, 1)

    def moments(weights: np.ndarray) -> tuple[float, float, float]:
        total = weights.sum()
        if total < 30:
            raise ValueError("could not locate a box in the initial observation")
        rows, cols = np.indices(weights.shape)
        row = (rows * weights).sum() / total
        col = (cols * weights).sum() / total
        row_difference, col_difference = rows - row, cols - col
        covariance = (weights * row_difference * col_difference).sum()
        difference = (weights * (col_difference**2 - row_difference**2)).sum()
        angle = 0.5 * np.arctan2(2 * covariance, difference)
        return row, col, angle

    source_row, source_col, _ = moments(source_weights)
    target_row, target_col, target_angle = moments(target_weights)
    return np.array([source_row, source_col, target_row, target_col, 8 * target_angle])


def load_legacy_demos(path: Path, allowed_seeds: set[int]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        seeds = data["seeds"]
        actions = data["action"]
        if seeds.ndim != 1 or not np.issubdtype(seeds.dtype, np.integer):
            raise ValueError("expected one integer seed per frame")
        if not len(seeds) or actions.shape != (len(seeds), JOINTS):
            raise ValueError("invalid or empty demonstration actions")
        starts = np.r_[0, np.flatnonzero(np.diff(seeds)) + 1]
        ends = np.r_[starts[1:], len(seeds)]
        episode_seeds = seeds[starts]
        if len(set(episode_seeds)) != len(episode_seeds) or not set(episode_seeds) <= allowed_seeds:
            raise ValueError("duplicate or non-training seed in legacy demonstrations")
        frames = data["front"][starts]
    features = np.stack([layout_features(frame) for frame in frames])
    return features, [actions[start:end] for start, end in zip(starts, ends)], episode_seeds


def split_trajectory(actions: np.ndarray) -> list[np.ndarray]:
    if actions.ndim != 2 or actions.shape[1] != JOINTS or not np.all(np.isfinite(actions)):
        raise ValueError("invalid joint-position trace")
    open_gripper = actions[:, GRIPPER] > 0.75
    events = np.flatnonzero(open_gripper[1:] != open_gripper[:-1]) + 1
    if not open_gripper[0] or len(events) not in (3, 4):
        raise ValueError(f"expected two grasps in the demonstration, found transitions {events.tolist()}")
    return list(np.split(actions, events[:3]))


def resample(actions: np.ndarray, count: int) -> np.ndarray:
    axis = np.linspace(0, 1, len(actions))
    target = np.linspace(0, 1, count)
    return np.stack([np.interp(target, axis, column) for column in actions.T], axis=1)


def fit_trajectory(
    output: Path,
    features: np.ndarray,
    actions: list[np.ndarray],
    seeds: np.ndarray,
    *,
    neighbors: int = 18,
    ridge: float = 0.02,
) -> None:
    if len(features) != len(actions) or len(seeds) != len(actions) or len(set(seeds)) != len(seeds):
        raise ValueError("layout, action and unique seed counts must match")
    if not actions or neighbors < 1 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("invalid trajectory training configuration")
    demos = [split_trajectory(episode) for episode in actions]
    lengths = np.array([[len(stage) for stage in demo] for demo in demos])
    traces = np.stack([
        np.concatenate([resample(stage, count) for stage, count in zip(demo, GRID)])
        for demo in demos
    ]).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, features=features, traces=traces, lengths=lengths, seeds=seeds,
                            neighbors=neighbors, ridge=ridge)


class TrajectoryPolicy:
    action_mode = "joint_position"
    name = "layout_conditioned_trajectory"
    finished = False

    def __init__(self, checkpoint: str | Path):
        with np.load(checkpoint, allow_pickle=False) as data:
            self.features = data["features"]
            self.traces = data["traces"]
            self.lengths = data["lengths"]
            self.neighbors = int(data["neighbors"])
            self.ridge = float(data["ridge"])
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.actions = None
        self.layout = None

    @property
    def camera_names(self) -> tuple[str, ...]:
        return ("front",) if self.actions is None else ()

    def plan(self, layout: np.ndarray) -> np.ndarray:
        scales = np.maximum(self.features.std(axis=0), [0.2, 0.2, 0.2, 0.2, 0.30])
        offsets = (self.features - layout) / scales
        distances = np.linalg.norm(offsets, axis=1)
        near = np.argsort(distances, kind="stable")[:self.neighbors]
        local = offsets[near]
        bandwidth = max(float(distances[near[-1]]) / 2, 0.25)
        weights = np.exp(-0.5 * (distances[near] / bandwidth)**2)
        design = np.column_stack((np.ones(len(near)), local))
        weighted = design.T * weights
        penalty = np.diag([0, *([self.ridge * weights.sum()] * local.shape[1])])
        coefficients = np.linalg.solve(weighted @ design + penalty, weighted)[0]
        trace = np.tensordot(coefficients, self.traces[near], axes=1)
        length_weights = weights / weights.sum()
        lengths = np.rint(length_weights @ self.lengths[near]).astype(int)
        phases = np.split(trace, np.cumsum(GRID[:-1]))
        actions = np.concatenate([resample(phase, length) for phase, length in zip(phases, lengths)])
        actions[:, GRIPPER] = np.clip(actions[:, GRIPPER], 0, 1)
        actions[:, 15] = np.clip(actions[:, 15], 0, 1)
        return actions

    def act(self, observation: dict, task: str = "") -> np.ndarray:
        if self.actions is None:
            self.layout = layout_features(observation[FRONT])
            self.actions = self.plan(self.layout)
        action = self.actions[min(self.step, len(self.actions) - 1)]
        self.step += 1
        return action.copy()
