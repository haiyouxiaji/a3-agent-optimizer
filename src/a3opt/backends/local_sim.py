from __future__ import annotations

import importlib.util
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
DEFAULT_DATASET_PATH = PROJECT_ROOT / "artifacts" / "datasets" / "expert_2x5_basic_6ep.npz"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "artifacts" / "local_sim"
DEFAULT_SEEDS = (1000, 1001, 1002, 1003, 1004, 1005)
DEFAULT_TASK = "2x5"
DEFAULT_PROFILE = "basic"
DEFAULT_SOURCE_COLUMN = "1"
FRONT_IMAGE = "observation.images.front"
STATE = "observation.state"
JOINT_ACTION_DIM = 16
DATA_IMAGE_SIZE = 64
MODEL_FILE = "model.pt"
SPEC_FILE = "spec.json"
TRAIN_LOG_FILE = "train_log.json"
FORMAT = "a3opt-bc-v1"
HIDDEN_SIZE = 128

DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_path": str(DEFAULT_DATASET_PATH),
    "seeds": list(DEFAULT_SEEDS),
    "task": DEFAULT_TASK,
    "profile": DEFAULT_PROFILE,
    "source_column": DEFAULT_SOURCE_COLUMN,
    "collect_max_steps": 600,
    "image_size": DATA_IMAGE_SIZE,
    "device": "cpu",
    "experiment_root": str(DEFAULT_EXPERIMENT_ROOT),
    "randomize_boxes": True,
    "randomize_cookies": True,
}

DEFAULT_RECORD_CONFIG: dict[str, Any] = {
    "dataset_path": str(DEFAULT_DATASET_PATH),
    "seeds": list(DEFAULT_SEEDS),
    "task": DEFAULT_TASK,
    "profile": DEFAULT_PROFILE,
    "source_column": DEFAULT_SOURCE_COLUMN,
    "expert_policy": "same_column",
    "max_steps": 600,
    "image_size": DATA_IMAGE_SIZE,
    "randomize_boxes": True,
    "randomize_cookies": True,
}


def _write_json(path: Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_device(requested: str | None = None) -> str:
    if requested is not None and str(requested).startswith("cuda"):
        return str(requested) if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def downsample_image(image: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"expected HxWxC image, got shape {array.shape}")
    target = int(size)
    if target < 1:
        raise ValueError("image size must be positive")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    height, width = int(array.shape[0]), int(array.shape[1])
    if height == target and width == target:
        return np.ascontiguousarray(array)
    if height % target == 0 and width % target == 0:
        factor_y = height // target
        factor_x = width // target
        blocks = array.reshape(target, factor_y, target, factor_x, array.shape[2])
        return blocks.mean(axis=(1, 3)).astype(np.uint8)
    rows = np.minimum((np.arange(target) * height) // target, height - 1)
    cols = np.minimum((np.arange(target) * width) // target, width - 1)
    return np.ascontiguousarray(array[rows][:, cols])


def _prepare_images(images: torch.Tensor, image_size: int) -> torch.Tensor:
    tensor = images.float().div(255.0)
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    if int(tensor.shape[-2]) != int(image_size) or int(tensor.shape[-1]) != int(image_size):
        tensor = F.interpolate(
            tensor, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False
        )
    return tensor.contiguous()


class BCModel(nn.Module):
    def __init__(
        self,
        encoder: str = "tiny_cnn",
        fusion: str = "concat",
        action_head: str = "chunk",
        chunk_size: int = 20,
        image_size: int = DATA_IMAGE_SIZE,
        state_dim: int = JOINT_ACTION_DIM,
        action_dim: int = JOINT_ACTION_DIM,
        hidden: int = HIDDEN_SIZE,
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
        self.use_image = self.encoder == "tiny_cnn" and self.image_size > 0
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        if self.use_image:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.hidden),
                nn.ReLU(),
            )
        head_input = self.hidden
        if self.use_image and self.fusion == "film":
            self.film = nn.Linear(self.hidden, 2 * self.hidden)
        elif self.use_image:
            head_input = 2 * self.hidden
        self.head = nn.Sequential(
            nn.Linear(head_input, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.chunk_len * self.action_dim),
        )

    def architecture(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder,
            "fusion": self.fusion,
            "action_head": self.action_head,
            "chunk_size": self.chunk_len,
            "image_size": self.image_size,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden": self.hidden,
        }

    def forward(self, image: torch.Tensor | None, state: torch.Tensor) -> torch.Tensor:
        state_feature = self.state_encoder(state)
        if self.use_image:
            if image is None:
                raise ValueError("image tensor is required for the tiny_cnn encoder")
            image_feature = self.image_encoder(_prepare_images(image, self.image_size))
            if self.fusion == "film":
                modulation = self.film(state_feature)
                gamma, beta = modulation.chunk(2, dim=-1)
                feature = image_feature * (1.0 + torch.tanh(gamma)) + beta
            else:
                feature = torch.cat([image_feature, state_feature], dim=-1)
        else:
            feature = state_feature
        return self.head(feature).reshape(-1, self.chunk_len, self.action_dim)


class FrontRecorder:
    def __init__(self, image_size: int = DATA_IMAGE_SIZE) -> None:
        self.image_size = int(image_size)
        self.episodes: list[dict[str, Any]] = []
        self._front: list[np.ndarray] = []
        self._state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._seeds: list[int] = []
        self._current: dict[str, Any] | None = None
        self._episode_metadata: dict[str, Any] = {}

    def start_episode(self, context: Any, controller_type: str = "") -> None:
        self._current = {
            "seed": int(context.seed),
            "task": str(context.task),
            "action_mode": str(context.action_mode),
            "controller": str(controller_type),
            "frames": 0,
        }
        self._episode_metadata = {}

    def set_episode_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        if not isinstance(metadata, Mapping):
            return
        keys = ("profile", "source_column", "seed", "task_instruction", "prompt_style")
        self._episode_metadata = {key: metadata[key] for key in keys if key in metadata}

    def add_frame(self, observation: Mapping[str, Any], action: np.ndarray) -> None:
        if self._current is None:
            raise RuntimeError("start_episode must be called before add_frame")
        self._front.append(downsample_image(observation[FRONT_IMAGE], self.image_size))
        self._state.append(np.asarray(observation[STATE], dtype=np.float32).copy())
        self._action.append(np.asarray(action, dtype=np.float32).copy())
        self._seeds.append(int(self._current["seed"]))
        self._current["frames"] = int(self._current["frames"]) + 1

    def finish_episode(self, success: bool | None = None) -> None:
        if self._current is None:
            return
        episode = dict(self._current)
        episode["success"] = None if success is None else bool(success)
        episode["metadata"] = dict(self._episode_metadata)
        self.episodes.append(episode)
        self._current = None

    def discard_episode(self) -> None:
        if self._current is None:
            return
        frames = int(self._current["frames"])
        if frames > 0:
            del self._front[-frames:]
            del self._state[-frames:]
            del self._action[-frames:]
            del self._seeds[-frames:]
        self._current = None

    def close(self) -> None:
        if self._current is not None:
            self.discard_episode()

    def export(self) -> dict[str, np.ndarray]:
        count = len(self._front)
        if count == 0:
            return {
                "front": np.zeros((0, self.image_size, self.image_size, 3), dtype=np.uint8),
                "state": np.zeros((0, JOINT_ACTION_DIM), dtype=np.float32),
                "action": np.zeros((0, JOINT_ACTION_DIM), dtype=np.float32),
                "seeds": np.zeros((0,), dtype=np.int32),
            }
        return {
            "front": np.stack(self._front, axis=0).astype(np.uint8),
            "state": np.stack(self._state, axis=0).astype(np.float32),
            "action": np.stack(self._action, axis=0).astype(np.float32),
            "seeds": np.asarray(self._seeds, dtype=np.int32),
        }

    def episode_frames(self) -> list[dict[str, Any]]:
        return [
            {
                "seed": int(episode["seed"]),
                "frames": int(episode["frames"]),
                "success": episode["success"],
                "controller": episode["controller"],
                "metadata": episode["metadata"],
            }
            for episode in self.episodes
        ]


def _load_dataset(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(Path(dataset_path)) as data:
        front = np.asarray(data["front"], dtype=np.uint8)
        state = np.asarray(data["state"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        seeds = np.asarray(data["seeds"], dtype=np.int64)
    if front.ndim != 4 or front.shape[-1] != 3:
        raise ValueError(f"front must be NxHxWx3, got {front.shape}")
    if state.ndim != 2 or state.shape != action.shape:
        raise ValueError(f"state and action must share a 2D shape, got {state.shape} and {action.shape}")
    if front.shape[0] != state.shape[0] or seeds.shape[0] != state.shape[0]:
        raise ValueError("front, state and seeds must describe the same number of frames")
    return front, state, action, seeds


def _episode_bounds(seeds: np.ndarray) -> list[tuple[int, int]]:
    if seeds.size == 0:
        return []
    changes = np.flatnonzero(seeds[1:] != seeds[:-1]) + 1
    starts = np.concatenate((np.array([0]), changes))
    ends = np.concatenate((changes, np.array([int(seeds.size)])))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _chunk_targets(action: np.ndarray, seeds: np.ndarray, chunk_len: int) -> np.ndarray:
    targets = np.empty((action.shape[0], int(chunk_len), action.shape[1]), dtype=np.float32)
    for start, end in _episode_bounds(seeds):
        segment = action[start:end]
        for offset in range(segment.shape[0]):
            available = min(int(chunk_len), segment.shape[0] - offset)
            targets[start + offset, :available] = segment[offset : offset + available]
            if available < int(chunk_len):
                targets[start + offset, available:] = segment[-1]
    return targets


def train_candidate(
    candidate: CandidateSpec,
    dataset_path: str | Path,
    train_budget: ExperimentBudget,
    out_dir: str | Path,
) -> TrainResult:
    started = time.perf_counter()
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        front, state, action, seeds = _load_dataset(dataset_path)
        num_frames = int(front.shape[0])
        if num_frames == 0:
            raise ValueError("dataset contains no frames")
        steps_target = int(train_budget.max_train_steps)
        if steps_target < 1:
            raise ValueError("train_budget.max_train_steps must be positive")
        batch_size = min(max(1, int(train_budget.batch_size)), num_frames)
        device = _resolve_device()
        torch.manual_seed(int(candidate.seed))
        np.random.seed(int(candidate.seed))
        model = BCModel(
            encoder=candidate.encoder,
            fusion=candidate.fusion,
            action_head=candidate.action_head,
            chunk_size=candidate.chunk_size,
            image_size=candidate.image_size,
            state_dim=int(candidate.state_dim),
            action_dim=int(action.shape[1]),
            hidden=HIDDEN_SIZE,
        ).to(device)
        chunk_targets = torch.from_numpy(_chunk_targets(action, seeds, model.chunk_len))
        images = torch.from_numpy(front)
        states = torch.from_numpy(state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(candidate.lr))
        wall_limit = float(train_budget.wall_seconds_limit)
        losses: list[float] = []
        first_loss = float("nan")
        model.train()
        for step in range(steps_target):
            index = torch.randint(0, num_frames, (batch_size,))
            prediction = model(images[index].to(device), states[index].to(device))
            loss = F.mse_loss(prediction, chunk_targets[index].to(device))
            if step == 0:
                first_loss = float(loss.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if wall_limit > 0 and (time.perf_counter() - started) > wall_limit:
                break
        model.eval()
        train_steps = len(losses)
        final_loss = losses[-1] if losses else float("nan")
        last_window = float(np.mean(losses[-10:])) if losses else float("nan")
        episode_count = len(_episode_bounds(seeds))
        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            out / MODEL_FILE,
        )
        spec = {
            "format": FORMAT,
            "created_at": now_iso(),
            "candidate": candidate.to_dict(),
            "architecture": model.architecture(),
            "data_image_size": int(front.shape[1]),
            "device": device,
            "train_steps": train_steps,
            "batch_size": batch_size,
            "first_loss": first_loss,
            "final_loss": final_loss,
            "loss_last_10_mean": last_window,
            "num_frames": num_frames,
            "num_episodes": episode_count,
            "torch_version": str(torch.__version__),
            "wall_seconds": round(time.perf_counter() - started, 3),
        }
        _write_json(out / SPEC_FILE, spec)
        _write_json(
            out / TRAIN_LOG_FILE,
            {
                "losses": losses,
                "steps": train_steps,
                "batch_size": batch_size,
                "dataset_path": str(Path(dataset_path).resolve()),
            },
        )
        metrics = {
            "first_loss": first_loss,
            "final_loss": final_loss,
            "loss_last_10_mean": last_window,
            "train_steps": float(train_steps),
            "device": 0.0 if device == "cpu" else 1.0,
            "num_frames": float(num_frames),
            "num_episodes": float(episode_count),
        }
        return TrainResult(
            status=STATUS_OK,
            train_steps=train_steps,
            metrics=metrics,
            artifact=str((out / MODEL_FILE).resolve()),
            wall_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return TrainResult(
            status=STATUS_FAILED,
            wall_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _load_model(checkpoint_dir: str | Path, device: str = "cpu") -> tuple[BCModel, dict[str, Any]]:
    directory = Path(checkpoint_dir)
    model_path = (directory / MODEL_FILE) if directory.is_dir() else directory
    if not model_path.is_file():
        raise FileNotFoundError(f"missing checkpoint model: {model_path}")
    spec_path = model_path.parent / SPEC_FILE
    if not spec_path.is_file():
        raise FileNotFoundError(f"missing checkpoint spec: {spec_path}")
    spec = _read_json(spec_path)
    architecture = spec.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("checkpoint spec is missing the architecture block")
    model = BCModel(**dict(architecture))
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, spec


class BCPolicy:
    name = "a3opt_bc"

    def __init__(self, checkpoint_dir: str | Path, device: str | None = "cpu") -> None:
        self.checkpoint_dir = str(checkpoint_dir)
        self.device = _resolve_device(device)
        self.model, self.spec = _load_model(checkpoint_dir, self.device)
        self.action_mode = "joint_position"
        self._queue: list[np.ndarray] = []

    @property
    def finished(self) -> bool:
        return False

    def reset(self, context: Any | None = None) -> None:
        self._queue.clear()

    def close(self) -> None:
        self._queue.clear()

    def act(self, observation: Mapping[str, Any], task: str = "") -> np.ndarray:
        if not self._queue:
            self._queue = self._predict(observation)
        return self._queue.pop(0)

    def _predict(self, observation: Mapping[str, Any]) -> list[np.ndarray]:
        state_vector = np.asarray(observation[STATE], dtype=np.float32).reshape(1, -1)
        state_tensor = torch.from_numpy(state_vector).to(self.device)
        image_tensor: torch.Tensor | None = None
        if self.model.use_image:
            data_size = int(self.spec.get("data_image_size", DATA_IMAGE_SIZE))
            front = downsample_image(observation[FRONT_IMAGE], data_size)
            image_tensor = torch.from_numpy(front[None]).to(self.device)
        with torch.no_grad():
            chunk = self.model(image_tensor, state_tensor)[0]
        actions = np.asarray(chunk.detach().cpu().numpy(), dtype=np.float64)
        actions = np.nan_to_num(actions, nan=0.0, posinf=4.0, neginf=-4.0)
        actions = np.clip(actions, -4.0, 4.0)
        return [np.ascontiguousarray(row) for row in actions]


def load_policy(checkpoint_dir: str | Path, device: str | None = "cpu") -> BCPolicy:
    return BCPolicy(checkpoint_dir, device=device)


def record_dataset(
    cfg: Mapping[str, Any] | None = None,
    on_episode: Callable[[int, int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    settings = {**DEFAULT_RECORD_CONFIG, **dict(cfg or {})}
    dataset_path = Path(settings["dataset_path"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in settings["seeds"]]
    if not seeds:
        raise ValueError("at least one seed is required")
    recorder = FrontRecorder(image_size=int(settings["image_size"]))
    benchmark = make_cookie_benchmark(
        task=str(settings["task"]),
        profile=str(settings["profile"]),
        source_column=str(settings["source_column"]),
        max_steps=int(settings["max_steps"]),
        randomize_boxes=bool(settings["randomize_boxes"]),
        randomize_cookies=bool(settings["randomize_cookies"]),
    )
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    env = benchmark.create_env(render_cameras=True)
    try:
        for index, seed in enumerate(seeds):
            score = benchmark.run_episode(
                policy=str(settings["expert_policy"]),
                seed=seed,
                episode_idx=index + 1,
                recorder=recorder,
                env=env,
            )
            report = {
                "seed": seed,
                "score": float(score.score),
                "success": bool(score.success),
                "steps": int(score.steps),
                "wall_seconds": float(score.wall_seconds),
                "source_column": score.source_column,
                "failure_reason": score.failure_reason,
            }
            reports.append(report)
            if on_episode is not None:
                on_episode(index, len(seeds), seed, report)
    finally:
        env.close()
        recorder.close()
    arrays = recorder.export()
    if arrays["front"].shape[0] == 0:
        raise RuntimeError("no frames were recorded; every episode was discarded")
    front = arrays["front"]
    state = arrays["state"]
    action = arrays["action"]
    seed_array = arrays["seeds"]
    np.savez_compressed(dataset_path, front=front, state=state, action=action, seeds=seed_array)
    recorded_seeds = sorted({int(value) for value in seed_array.tolist()})
    meta = {
        "format": "a3opt-expert-npz-v1",
        "created_at": now_iso(),
        "dataset_path": str(dataset_path.resolve()),
        "task": str(settings["task"]),
        "profile": str(settings["profile"]),
        "source_column": str(settings["source_column"]),
        "expert_policy": str(settings["expert_policy"]),
        "image_size": int(settings["image_size"]),
        "episodes": len(recorded_seeds),
        "seeds": seeds,
        "recorded_seeds": recorded_seeds,
        "frames": int(front.shape[0]),
        "episode_frames": recorder.episode_frames(),
        "episode_reports": reports,
        "benchmark": {
            "max_steps": int(settings["max_steps"]),
            "randomize_boxes": bool(settings["randomize_boxes"]),
            "randomize_cookies": bool(settings["randomize_cookies"]),
        },
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(dataset_path.with_suffix(".meta.json"), meta)
    return meta


def evaluate_checkpoint(
    candidate: CandidateSpec,
    checkpoint_dir: str | Path,
    seeds: Sequence[int],
    max_steps: int,
    out_dir: str | Path,
    cfg: Mapping[str, Any] | None = None,
) -> EvalResult:
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    settings = {**DEFAULT_CONFIG, **dict(cfg or {})}
    started = time.perf_counter()
    result = EvalResult(status=STATUS_FAILED, max_score=MAX_SCORE_2X5)
    details: list[dict[str, Any]] = []
    policy: BCPolicy | None = None
    try:
        benchmark = make_cookie_benchmark(
            task=str(settings["task"]),
            profile=str(settings["profile"]),
            source_column=str(settings["source_column"]),
            max_steps=int(max_steps),
        )
        policy = load_policy(checkpoint_dir, device=str(settings.get("device", "cpu")))
        env = benchmark.create_env(render_cameras=True)
        rows: list[EpisodeRow] = []
        try:
            for index, seed in enumerate(seeds):
                score = benchmark.run_episode(
                    policy=policy,
                    seed=int(seed),
                    episode_idx=index + 1,
                    env=env,
                )
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


class LocalSimBackend:
    name = "local_sim"

    def __init__(self, config: Mapping[str, Any] | None = None, **overrides: Any) -> None:
        merged = dict(DEFAULT_CONFIG)
        if config is not None:
            merged.update(dict(config))
        merged.update({key: value for key, value in overrides.items() if value is not None})
        self.config = merged

    def available(self) -> tuple[bool, str]:
        missing = [
            name
            for name in ("torch", "mujoco", "a3_dual_arm_sim")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            return False, f"missing python packages: {', '.join(missing)}"
        return True, f"torch {torch.__version__}, mujoco and a3_dual_arm_sim importable"

    def experiment_dir(self, candidate: CandidateSpec) -> Path:
        safe = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in candidate.candidate_id
        )
        return Path(self.config["experiment_root"]) / (safe or "candidate")

    def ensure_dataset(self) -> Path:
        dataset_path = Path(self.config["dataset_path"])
        if not dataset_path.is_file():
            record_dataset(
                {
                    "dataset_path": str(dataset_path),
                    "seeds": list(self.config["seeds"]),
                    "task": str(self.config["task"]),
                    "profile": str(self.config["profile"]),
                    "source_column": str(self.config["source_column"]),
                    "max_steps": int(self.config["collect_max_steps"]),
                    "image_size": int(self.config["image_size"]),
                    "randomize_boxes": bool(self.config["randomize_boxes"]),
                    "randomize_cookies": bool(self.config["randomize_cookies"]),
                }
            )
        return dataset_path

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
            dataset_path = self.ensure_dataset()
            train = train_candidate(candidate, dataset_path, train_budget, checkpoint_dir)
            if train.status != STATUS_OK:
                raise RuntimeError(train.error or "training failed")
            evaluation = evaluate_checkpoint(
                candidate,
                checkpoint_dir,
                eval_seeds,
                int(max_eval_steps),
                eval_dir,
                cfg=self.config,
            )
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
    "BCModel",
    "BCPolicy",
    "FrontRecorder",
    "LocalSimBackend",
    "downsample_image",
    "evaluate_checkpoint",
    "load_policy",
    "record_dataset",
    "train_candidate",
]
