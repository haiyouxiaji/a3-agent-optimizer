from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from a3opt.contracts import (
    MAX_SCORE_2X5,
    STATUS_FAILED,
    STATUS_OK,
    CandidateSpec,
    EpisodeRow,
    EvalResult,
    ExperimentBudget,
    ExperimentOutcome,
    TrainResult,
    now_iso,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "artifacts" / "datasets" / "expert_2x5_multicam_20ep.npz"
DEFAULT_ROOT = PROJECT_ROOT / "artifacts" / "local_sim_v2"
DEFAULT_SEEDS = tuple(range(1000, 1006))
DATA_IMAGE_SIZE = 128
CAMERAS = ("front", "left_wrist", "right_wrist")
FRONT = "observation.images.front"
LEFT = "observation.images.left_wrist"
RIGHT = "observation.images.right_wrist"
STATE = "observation.state"
MODEL_FILE = "model.pt"
SPEC_FILE = "spec.json"
TRAIN_LOG_FILE = "train_log.json"
FORMAT = "a3opt-bc2-v1"
HIDDEN = 128
IMAGE_HIDDEN = 96
EXECUTION_HORIZON = 8
STEP_SCALE = 600.0
STATE_DIM = 16
ACTION_DIM = 16


@dataclass
class BC2Config:
    dataset_path: Path = DEFAULT_DATASET
    experiment_root: Path = DEFAULT_ROOT
    device: str = "cpu"
    image_size: int = DATA_IMAGE_SIZE
    execution_horizon: int = EXECUTION_HORIZON
    record_seeds: tuple[int, ...] = tuple(range(1000, 1020))
    record_script: Path = PROJECT_ROOT / "scripts" / "record_multicam.py"
    record_env: dict[str, str] = field(default_factory=dict)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_device(requested: str | None) -> str:
    if requested is None or requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"
    return "cpu"


def downsample(image: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(image)
    height, width = int(array.shape[0]), int(array.shape[1])
    if height == size and width == size:
        return np.ascontiguousarray(array)
    factor_y = height // size
    factor_x = width // size
    if factor_y * size == height and factor_x * size == width:
        blocks = array.reshape(size, factor_y, size, factor_x, array.shape[2])
        return blocks.mean(axis=(1, 3)).astype(np.uint8)
    rows = np.minimum((np.arange(size) * height) // size, height - 1)
    cols = np.minimum((np.arange(size) * width) // size, width - 1)
    return np.ascontiguousarray(array[rows][:, cols])


def _images(tensor: torch.Tensor, size: int) -> torch.Tensor:
    images = tensor.float().div(255.0)
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    if int(images.shape[-2]) != int(size) or int(images.shape[-1]) != int(size):
        images = F.interpolate(images, size=(int(size), int(size)), mode="bilinear", align_corners=False)
    return images.contiguous()


class CameraEncoder(nn.Module):
    def __init__(self, hidden: int = IMAGE_HIDDEN) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)


class BC2Model(nn.Module):
    def __init__(
        self,
        camera_count: int = 3,
        encoder: str = "tiny_cnn",
        fusion: str = "concat",
        action_head: str = "chunk",
        chunk_size: int = 20,
        image_size: int = DATA_IMAGE_SIZE,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = HIDDEN,
    ) -> None:
        super().__init__()
        self.encoder = str(encoder)
        self.fusion = str(fusion)
        self.action_head = str(action_head)
        self.chunk_len = max(1, int(chunk_size)) if self.action_head == "chunk" else 1
        self.image_size = int(image_size)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.camera_count = int(camera_count)
        self.use_images = self.encoder == "tiny_cnn" and self.image_size > 0
        self.cameras = nn.ModuleList([CameraEncoder(IMAGE_HIDDEN) for _ in range(self.camera_count)]) if self.use_images else nn.ModuleList()
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim * 2 + 1, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        head_input = self.hidden
        if self.use_images:
            image_output = IMAGE_HIDDEN * self.camera_count
            if self.fusion == "film":
                self.film = nn.Linear(self.hidden, 2 * image_output)
                head_input = image_output + self.hidden
            else:
                head_input = image_output + self.hidden
        self.head = nn.Sequential(
            nn.Linear(head_input, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.chunk_len * self.action_dim),
        )

    def architecture(self) -> dict[str, Any]:
        return {
            "camera_count": self.camera_count,
            "encoder": self.encoder,
            "fusion": self.fusion,
            "action_head": self.action_head,
            "chunk_size": self.chunk_len,
            "image_size": self.image_size,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden": self.hidden,
        }

    def forward(
        self,
        images: tuple[torch.Tensor, ...] | None,
        state: torch.Tensor,
        previous_action: torch.Tensor,
        step_fraction: torch.Tensor,
    ) -> torch.Tensor:
        features = [state, previous_action, step_fraction]
        memory = self.state_encoder(torch.cat(features, dim=-1))
        pieces = [memory]
        if self.use_images:
            if images is None:
                raise ValueError("images are required by the tiny_cnn encoder")
            encoded = [camera(_images(image, self.image_size)) for camera, image in zip(self.cameras, images)]
            image_feature = torch.cat(encoded, dim=-1)
            if self.fusion == "film":
                modulation = self.film(memory)
                gamma, beta = modulation.chunk(2, dim=-1)
                image_feature = image_feature * (1.0 + torch.tanh(gamma)) + beta
            pieces.insert(0, image_feature)
        return self.head(torch.cat(pieces, dim=-1)).reshape(-1, self.chunk_len, self.action_dim)


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        payload = {
            "front": np.asarray(data["front"], dtype=np.uint8),
            "left_wrist": np.asarray(data["left_wrist"], dtype=np.uint8),
            "right_wrist": np.asarray(data["right_wrist"], dtype=np.uint8),
            "state": np.asarray(data["state"], dtype=np.float32),
            "action": np.asarray(data["action"], dtype=np.float32),
            "seeds": np.asarray(data["seeds"], dtype=np.int64),
        }
    count = payload["state"].shape[0]
    if count == 0:
        raise ValueError("dataset is empty")
    for key in ("front", "left_wrist", "right_wrist"):
        if payload[key].shape[0] != count:
            raise ValueError(f"{key} frame count does not match state")
    return payload


def _episode_bounds(seeds: np.ndarray) -> list[tuple[int, int]]:
    changes = np.flatnonzero(seeds[1:] != seeds[:-1]) + 1
    starts = np.concatenate((np.array([0]), changes))
    ends = np.concatenate((changes, np.array([int(seeds.size)])))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _chunk_targets(actions: np.ndarray, seeds: np.ndarray, chunk_length: int) -> np.ndarray:
    targets = np.empty((actions.shape[0], int(chunk_length), actions.shape[1]), dtype=np.float32)
    for start, end in _episode_bounds(seeds):
        segment = actions[start:end]
        for offset in range(segment.shape[0]):
            available = min(int(chunk_length), segment.shape[0] - offset)
            targets[start + offset, :available] = segment[offset : offset + available]
            if available < int(chunk_length):
                targets[start + offset, available:] = segment[-1]
    return targets


def _history(actions: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    history = np.zeros_like(actions)
    for start, end in _episode_bounds(seeds):
        if end - start > 1:
            history[start + 1 : end] = actions[start : end - 1]
    return history


def _step_fraction(seeds: np.ndarray) -> np.ndarray:
    fraction = np.zeros((seeds.shape[0], 1), dtype=np.float32)
    for start, end in _episode_bounds(seeds):
        steps = np.arange(end - start, dtype=np.float32) / STEP_SCALE
        fraction[start:end, 0] = np.clip(steps, 0.0, 1.0)
    return fraction


def _configure_threads() -> None:
    import os

    value = os.environ.get("A3OPT_TORCH_THREADS", "").strip()
    if value:
        try:
            torch.set_num_threads(max(1, int(value)))
        except ValueError:
            pass


def train_candidate(
    candidate: CandidateSpec,
    dataset_path: str | Path,
    train_budget: ExperimentBudget,
    out_dir: str | Path,
    config: BC2Config | None = None,
) -> TrainResult:
    started = time.perf_counter()
    cfg = config or BC2Config()
    _configure_threads()
    try:
        payload = _load_dataset(Path(dataset_path))
        actions = payload["action"]
        seeds = payload["seeds"]
        action_mean = actions.mean(axis=0)
        action_std = actions.std(axis=0)
        action_std = np.where(action_std < 1e-4, 1e-4, action_std)
        normalized = (actions - action_mean) / action_std
        history = _history(normalized, seeds)
        fraction = _step_fraction(seeds)
        chunk_targets = torch.from_numpy(_chunk_targets(normalized, seeds, candidate.chunk_size if candidate.action_head == "chunk" else 1))
        states = torch.from_numpy((payload["state"] - payload["state"].mean(axis=0)) / np.where(payload["state"].std(axis=0) < 1e-4, 1e-4, payload["state"].std(axis=0)))
        previous = torch.from_numpy(history)
        steps = torch.from_numpy(fraction)
        images = {}
        use_images = candidate.encoder == "tiny_cnn" and int(candidate.image_size) > 0
        if use_images:
            for camera in CAMERAS:
                key = "front" if camera == "front" else camera
                arrays = payload[key]
                target_size = int(candidate.image_size)
                if int(arrays.shape[1]) != target_size:
                    arrays = np.stack([downsample(frame, target_size) for frame in arrays], axis=0)
                images[key] = torch.from_numpy(arrays)
        device = _resolve_device(cfg.device)
        torch.manual_seed(int(candidate.seed))
        np.random.seed(int(candidate.seed))
        model = BC2Model(
            camera_count=len(CAMERAS),
            encoder=candidate.encoder,
            fusion=candidate.fusion,
            action_head=candidate.action_head,
            chunk_size=candidate.chunk_size,
            image_size=int(candidate.image_size),
            state_dim=int(candidate.state_dim),
            action_dim=int(actions.shape[1]),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(candidate.lr))
        sample_count = int(actions.shape[0])
        batch = max(1, min(int(train_budget.batch_size), sample_count))
        limit = int(train_budget.max_train_steps)
        wall_limit = float(train_budget.wall_seconds_limit)
        losses: list[float] = []
        first_loss = float("nan")
        model.train()
        for step in range(limit):
            index = torch.randint(0, sample_count, (batch,))
            batch_images = tuple(images[key][index].to(device) for key in CAMERAS) if use_images else None
            prediction = model(
                batch_images,
                states[index].to(device),
                previous[index].to(device),
                steps[index].to(device),
            )
            loss = F.l1_loss(prediction, chunk_targets[index].to(device))
            if step == 0:
                first_loss = float(loss.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if wall_limit > 0 and (time.perf_counter() - started) > wall_limit:
                break
        model.eval()
        with torch.no_grad():
            index = torch.arange(sample_count)
            batch_images = tuple(images[key][index].to(device) for key in CAMERAS) if use_images else None
            prediction = model(batch_images, states.to(device), previous.to(device), steps.to(device))
            offline = F.l1_loss(prediction, chunk_targets.to(device)).item()
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, output / MODEL_FILE)
        spec = {
            "format": FORMAT,
            "created_at": now_iso(),
            "candidate": candidate.to_dict(),
            "architecture": model.architecture(),
            "cameras": list(CAMERAS),
            "data_image_size": int(payload["front"].shape[1]),
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
            "state_mean": payload["state"].mean(axis=0).tolist(),
            "state_std": np.where(payload["state"].std(axis=0) < 1e-4, 1e-4, payload["state"].std(axis=0)).tolist(),
            "device": device,
            "train_steps": len(losses),
            "batch_size": batch,
            "first_loss": first_loss,
            "final_loss": losses[-1] if losses else float("nan"),
            "loss_last_10_mean": float(np.mean(losses[-10:])) if losses else float("nan"),
            "offline_normalized_mae": float(offline),
            "num_frames": sample_count,
            "num_episodes": len(_episode_bounds(seeds)),
            "torch_version": str(torch.__version__),
            "wall_seconds": round(time.perf_counter() - started, 3),
        }
        _write_json(output / SPEC_FILE, spec)
        _write_json(output / TRAIN_LOG_FILE, {"losses": losses, "steps": len(losses), "batch_size": batch})
        metrics = {
            "first_loss": first_loss,
            "final_loss": losses[-1] if losses else float("nan"),
            "loss_last_10_mean": float(np.mean(losses[-10:])) if losses else float("nan"),
            "offline_normalized_mae": float(offline),
            "train_steps": float(len(losses)),
            "device": 0.0 if device == "cpu" else 1.0,
            "num_frames": float(sample_count),
            "num_episodes": float(len(_episode_bounds(seeds))),
        }
        return TrainResult(
            status=STATUS_OK,
            train_steps=len(losses),
            metrics=metrics,
            artifact=str((output / MODEL_FILE).resolve()),
            wall_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return TrainResult(
            status=STATUS_FAILED,
            wall_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _load_model(checkpoint_dir: str | Path, device: str) -> tuple[BC2Model, dict[str, Any]]:
    directory = Path(checkpoint_dir)
    model_path = directory / MODEL_FILE
    spec = _read_json(directory / SPEC_FILE)
    model = BC2Model(**dict(spec["architecture"]))
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, spec


class BC2Policy:
    name = "a3opt_bc2"

    def __init__(self, checkpoint_dir: str | Path, device: str | None = "cpu", execution_horizon: int = EXECUTION_HORIZON) -> None:
        self.device = _resolve_device(device)
        self.model, self.spec = _load_model(checkpoint_dir, self.device)
        self.action_mode = "joint_position"
        self.execution_horizon = max(1, int(execution_horizon))
        self.action_mean = np.asarray(self.spec["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(self.spec["action_std"], dtype=np.float32)
        self.state_mean = np.asarray(self.spec["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(self.spec["state_std"], dtype=np.float32)
        self.cameras = tuple(self.spec.get("cameras", CAMERAS))
        self.data_image_size = int(self.spec.get("data_image_size", DATA_IMAGE_SIZE))
        self._queue: list[np.ndarray] = []
        self._previous = np.zeros(ACTION_DIM, dtype=np.float32)
        self._step = 0

    @property
    def finished(self) -> bool:
        return False

    def reset(self, context: Any | None = None) -> None:
        self._queue.clear()
        self._previous = np.zeros(ACTION_DIM, dtype=np.float32)
        self._step = 0

    def close(self) -> None:
        self._queue.clear()

    def act(self, observation: Mapping[str, Any], task: str = "") -> np.ndarray:
        if not self._queue:
            self._queue = self._predict(observation)
        action = self._queue.pop(0)
        self._previous = ((np.asarray(action, dtype=np.float32) - self.action_mean) / self.action_std).astype(np.float32)
        self._step += 1
        return action

    def predict_chunk(self, observation: Mapping[str, Any]) -> np.ndarray:
        return self._chunk(observation)

    def _chunk(self, observation: Mapping[str, Any]) -> np.ndarray:
        state = np.asarray(observation[STATE], dtype=np.float32)
        state_normalized = (state - self.state_mean) / self.state_std
        state_tensor = torch.from_numpy(state_normalized[None]).to(self.device)
        previous_tensor = torch.from_numpy(self._previous[None]).to(self.device)
        fraction = torch.tensor([[min(self._step / STEP_SCALE, 1.0)]], dtype=torch.float32, device=self.device)
        images: tuple[torch.Tensor, ...] | None = None
        if self.model.use_images:
            frames = []
            for camera in self.cameras:
                key = FRONT if camera == "front" else (LEFT if camera == "left_wrist" else RIGHT)
                frame = downsample(observation[key], self.data_image_size)
                frames.append(torch.from_numpy(frame[None]).to(self.device))
            images = tuple(frames)
        with torch.no_grad():
            chunk = self.model(images, state_tensor, previous_tensor, fraction)[0]
        normalized = chunk.detach().cpu().numpy().astype(np.float32)
        actions = normalized * self.action_std[None, :] + self.action_mean[None, :]
        actions = np.nan_to_num(actions, nan=0.0, posinf=4.0, neginf=-4.0)
        actions = np.clip(actions, -4.0, 4.0)
        return np.ascontiguousarray(actions)

    def _predict(self, observation: Mapping[str, Any]) -> list[np.ndarray]:
        chunk = self._chunk(observation)
        return [np.ascontiguousarray(row) for row in chunk[: self.execution_horizon]]
class EnsemblePolicy:
    name = "a3opt_bc2_ensemble_avg"

    def __init__(
        self,
        checkpoint_dirs: Sequence[str | Path],
        device: str | None = "cpu",
        execution_horizon: int = EXECUTION_HORIZON,
    ) -> None:
        self.policies = [BC2Policy(path, device=device, execution_horizon=execution_horizon) for path in checkpoint_dirs]
        self.action_mode = "joint_position"
        self.execution_horizon = int(execution_horizon)
        self._queue: list[np.ndarray] = []

    @property
    def finished(self) -> bool:
        return False

    def reset(self, context: Any | None = None) -> None:
        for policy in self.policies:
            policy.reset(context)
        self._queue.clear()

    def close(self) -> None:
        for policy in self.policies:
            policy.close()
        self._queue.clear()

    def act(self, observation: Mapping[str, Any], task: str = "") -> np.ndarray:
        if not self._queue:
            chunks = []
            for policy in self.policies:
                chunks.append(policy.predict_chunk(observation))
            averaged = np.mean(np.stack(chunks, axis=0), axis=0)
            averaged = np.nan_to_num(averaged, nan=0.0, posinf=4.0, neginf=-4.0)
            averaged = np.clip(averaged, -4.0, 4.0)
            self._queue = [np.ascontiguousarray(row) for row in averaged[: self.execution_horizon]]
        action = self._queue.pop(0)
        reference = self.policies[0]
        normalized = ((np.asarray(action, dtype=np.float32) - reference.action_mean) / reference.action_std).astype(np.float32)
        for policy in self.policies:
            policy._previous = normalized.copy()
            policy._step += 1
        return action


class TemporalEnsemblePolicy:
    name = "a3opt_bc2_ensemble"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str | None = "cpu",
        execution_horizon: int = EXECUTION_HORIZON,
        coefficient: float = 0.01,
    ) -> None:
        self.base = BC2Policy(checkpoint_dir, device=device, execution_horizon=execution_horizon)
        self.action_mode = "joint_position"
        self.coefficient = float(coefficient)
        self._pending: dict[int, list[tuple[float, np.ndarray]]] = {}
        self._step = 0

    @property
    def finished(self) -> bool:
        return False

    def reset(self, context: Any | None = None) -> None:
        self.base.reset(context)
        self._pending.clear()
        self._step = 0

    def close(self) -> None:
        self.base.close()
        self._pending.clear()

    def act(self, observation: Mapping[str, Any], task: str = "") -> np.ndarray:
        self.base._step = self._step
        chunk = self.base.predict_chunk(observation)
        for offset, row in enumerate(chunk):
            weight = float(np.exp(-self.coefficient * offset))
            self._pending.setdefault(self._step + offset, []).append((weight, row))
        entries = self._pending.pop(self._step, [])
        if entries:
            total = sum(weight for weight, _ in entries)
            action = sum(weight * row for weight, row in entries) / max(total, 1e-8)
        else:
            action = chunk[0]
        action = np.asarray(action, dtype=np.float64)
        self.base._previous = ((action.astype(np.float32) - self.base.action_mean) / self.base.action_std).astype(np.float32)
        self._step += 1
        return np.ascontiguousarray(action)


def evaluate_checkpoint(
    candidate: CandidateSpec,
    checkpoint_dir: str | Path,
    seeds: Sequence[int],
    max_steps: int,
    out_dir: str | Path,
    cfg: BC2Config | None = None,
) -> EvalResult:
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    settings = cfg or BC2Config()
    started = time.perf_counter()
    result = EvalResult(status=STATUS_FAILED, max_score=MAX_SCORE_2X5)
    details: list[dict[str, Any]] = []
    policy: BC2Policy | None = None
    try:
        benchmark = make_cookie_benchmark(task="2x5", profile="basic", source_column="1", max_steps=int(max_steps))
        policy = BC2Policy(checkpoint_dir, device=settings.device, execution_horizon=settings.execution_horizon)
        env = benchmark.create_env(render_cameras=True)
        rows: list[EpisodeRow] = []
        try:
            for index, seed in enumerate(seeds):
                score = benchmark.run_episode(policy=policy, seed=int(seed), episode_idx=index + 1, env=env)
                rows.append(
                    EpisodeRow(
                        seed=int(seed),
                        score=float(score.score),
                        success=bool(score.success),
                        steps=int(score.steps),
                    )
                )
                details.append(score.to_dict())
        finally:
            env.close()
        result.episodes = rows
        result.status = STATUS_OK
    except Exception as exc:
        result.status = STATUS_FAILED
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        if policy is not None:
            policy.close()
    result.wall_seconds = time.perf_counter() - started
    payload = result.to_dict()
    payload["candidate"] = candidate.to_dict()
    payload["max_steps"] = int(max_steps)
    payload["seeds"] = [int(seed) for seed in seeds]
    payload["checkpoint"] = str(Path(checkpoint_dir))
    payload["details"] = details
    try:
        _write_json(Path(out_dir) / "eval.json", payload)
    except OSError:
        pass
    return result


class LocalSimV2Backend:
    name = "local_sim_v2"

    def __init__(
        self,
        config: BC2Config | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        base = config if isinstance(config, BC2Config) else BC2Config(**dict(config or {}))
        for key, value in overrides.items():
            if value is not None and hasattr(base, key):
                setattr(base, key, value)
        self.config = base

    def available(self) -> tuple[bool, str]:
        missing = [
            name
            for name in ("torch", "mujoco", "a3_dual_arm_sim")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            return False, f"missing python packages: {', '.join(missing)}"
        if not Path(self.config.dataset_path).is_file():
            return False, f"missing dataset: {self.config.dataset_path}; run scripts/record_multicam.py"
        return True, f"torch {torch.__version__}, multicamera dataset present"

    def experiment_dir(self, candidate: CandidateSpec) -> Path:
        safe = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in candidate.candidate_id
        )
        return Path(self.config.experiment_root) / (safe or "candidate")

    def ensure_dataset(self) -> Path:
        dataset = Path(self.config.dataset_path)
        if dataset.is_file():
            return dataset
        command = [
            sys.executable,
            str(self.config.record_script),
            "--dataset",
            str(dataset),
            "--episodes",
            str(len(self.config.record_seeds)),
            "--seed-start",
            str(min(self.config.record_seeds)),
            "--image-size",
            str(DATA_IMAGE_SIZE),
        ]
        environment = dict(self.config.record_env) or None
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True, env=environment)
        return dataset

    def run(
        self,
        candidate: CandidateSpec,
        train_budget: ExperimentBudget,
        eval_seeds: Sequence[int],
        max_eval_steps: int,
    ) -> ExperimentOutcome:
        experiment_dir = self.experiment_dir(candidate)
        checkpoint_dir = experiment_dir / "checkpoint"
        eval_dir = experiment_dir / "eval"
        train = TrainResult(status=STATUS_FAILED, error="training was not started")
        evaluation = EvalResult(status=STATUS_FAILED, error="evaluation was not started")
        error = ""
        try:
            dataset = self.ensure_dataset()
            train = train_candidate(candidate, dataset, train_budget, checkpoint_dir, config=self.config)
            if train.status != STATUS_OK:
                raise RuntimeError(train.error or "training failed")
            evaluation = evaluate_checkpoint(candidate, checkpoint_dir, eval_seeds, int(max_eval_steps), eval_dir, cfg=self.config)
            if evaluation.status != STATUS_OK:
                raise RuntimeError(evaluation.error or "evaluation failed")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if evaluation.status != STATUS_OK:
                evaluation.status = STATUS_FAILED
                evaluation.error = error
        outcome = ExperimentOutcome(
            candidate=candidate,
            train=train,
            eval=evaluation,
            status=STATUS_FAILED if error or train.status != STATUS_OK or evaluation.status != STATUS_OK else STATUS_OK,
            error=error,
        )
        try:
            _write_json(experiment_dir / "outcome.json", outcome.to_dict())
        except OSError:
            pass
        return outcome


__all__ = [
    "BC2Config",
    "BC2Model",
    "BC2Policy",
    "LocalSimV2Backend",
    "EnsemblePolicy",
    "TemporalEnsemblePolicy",
    "downsample",
    "evaluate_checkpoint",
    "train_candidate",
]
