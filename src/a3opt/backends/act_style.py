from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from a3opt.backends.bc2 import (
    CAMERAS,
    DEFAULT_SEEDS,
    DATA_IMAGE_SIZE,
    FRONT,
    LEFT,
    RIGHT,
    STATE,
    STEP_SCALE,
    _chunk_targets,
    _episode_bounds,
    _history,
    _load_dataset,
    _step_fraction,
    downsample,
)
from a3opt.contracts import now_iso

ACT_FORMAT = "a3opt-act-style-v1"
DEFAULT_DATASET = Path("artifacts/datasets/expert_2x5_multicam_60ep.npz")
DEFAULT_OUTPUT = Path("artifacts/act_style")
MODEL_FILE = "model.pt"
SPEC_FILE = "spec.json"
STATE_DIM = 16
ACTION_DIM = 16


@dataclass
class ACTStyleConfig:
    dataset_path: Path = DEFAULT_DATASET
    output_dir: Path = DEFAULT_OUTPUT
    steps: int = 12000
    batch_size: int = 32
    lr: float = 1e-4
    chunk_size: int = 50
    execution_horizon: int = 8
    d_model: int = 256
    layers: int = 4
    heads: int = 8
    latent_dim: int = 32
    kl_weight: float = 10.0
    image_size: int = 128
    seed: int = 0
    device: str = "auto"
    pretrained_backbone: bool = False
    save_freq: int = 0


def resolve_device(requested: str | None) -> str:
    if requested is None or requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


class ACTStyleModel(nn.Module):
    def __init__(self, config: ACTStyleConfig) -> None:
        super().__init__()
        from torchvision import models

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if config.pretrained_backbone else None
        backbone = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.config = config
        self.camera_count = len(CAMERAS)
        self.chunk_size = int(config.chunk_size)
        self.action_dim = ACTION_DIM
        d = int(config.d_model)
        self.image_proj = nn.Linear(512 * self.camera_count, d)
        self.state_proj = nn.Sequential(nn.Linear(STATE_DIM * 2 + 1, d), nn.ReLU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, 3, d))
        self.latent_token = nn.Parameter(torch.zeros(1, 1, d))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(config.heads),
            dim_feedforward=4 * d,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(config.layers))
        self.action_encoder = nn.Sequential(
            nn.Linear(self.chunk_size * self.action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, int(config.latent_dim))
        self.logvar = nn.Linear(256, int(config.latent_dim))
        self.latent_proj = nn.Linear(int(config.latent_dim), d)
        self.query = nn.Parameter(torch.zeros(1, self.chunk_size, d))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=int(config.heads),
            dim_feedforward=4 * d,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(config.layers))
        self.head = nn.Linear(d, self.action_dim)

    def forward(
        self,
        images: Sequence[torch.Tensor],
        state: torch.Tensor,
        previous_action: torch.Tensor,
        step_fraction: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = []
        for image in images:
            batch = image.shape[0]
            tensor = image.float().div(255.0).permute(0, 3, 1, 2)
            features.append(self.backbone(tensor).flatten(1))
        image_token = self.image_proj(torch.cat(features, dim=-1)).unsqueeze(1)
        state_token = self.state_proj(torch.cat([state, previous_action, step_fraction], dim=-1)).unsqueeze(1)
        batch = state.shape[0]
        if target is not None:
            encoded = self.action_encoder(target.flatten(1))
            mu = self.mu(encoded)
            logvar = self.logvar(encoded)
            if self.training:
                noise = torch.randn_like(mu)
                latent = mu + noise * torch.exp(0.5 * logvar)
            else:
                latent = mu
            kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        else:
            latent = torch.zeros(batch, self.config.latent_dim, device=state.device)
            kl = torch.zeros((), device=state.device)
        latent_token = self.latent_proj(latent).unsqueeze(1) + self.latent_token
        tokens = torch.cat([image_token, state_token, latent_token], dim=1) + self.pos
        memory = self.encoder(tokens)
        queries = self.query.expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        prediction = self.head(decoded)
        return prediction, kl


def train_act_style(config: ACTStyleConfig) -> dict[str, Any]:
    started = time.perf_counter()
    device = resolve_device(config.device)
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    payload = _load_dataset(Path(config.dataset_path))
    actions = payload["action"]
    seeds = payload["seeds"]
    action_mean = actions.mean(axis=0)
    action_std = np.where(actions.std(axis=0) < 1e-4, 1e-4, actions.std(axis=0))
    normalized = (actions - action_mean) / action_std
    state_mean = payload["state"].mean(axis=0)
    state_std = np.where(payload["state"].std(axis=0) < 1e-4, 1e-4, payload["state"].std(axis=0))
    chunk_targets = torch.from_numpy(
        _chunk_targets(normalized, seeds, config.execution_horizon if config.chunk_size == 1 else config.chunk_size)
    )
    states = torch.from_numpy(((payload["state"] - state_mean) / state_std).astype(np.float32))
    previous = torch.from_numpy(_history(normalized, seeds).astype(np.float32))
    fraction = torch.from_numpy(_step_fraction(seeds).astype(np.float32))
    images = [torch.from_numpy(payload[key]) for key in CAMERAS]
    sample_count = int(states.shape[0])
    model = ACTStyleModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=1e-4)
    batch = max(1, min(int(config.batch_size), sample_count))
    losses: list[float] = []
    first_loss = float("nan")
    model.train()
    for step in range(int(config.steps)):
        index = torch.randint(0, sample_count, (batch,))
        batch_images = tuple(images[cam][index].to(device) for cam in range(len(CAMERAS)))
        prediction, kl = model(
            batch_images,
            states[index].to(device),
            previous[index].to(device),
            fraction[index].to(device),
            chunk_targets[index].to(device),
        )
        l1 = torch.nn.functional.l1_loss(prediction, chunk_targets[index].to(device))
        loss = l1 + float(config.kl_weight) * kl
        if step == 0:
            first_loss = float(l1.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(l1.detach().cpu()))
        if config.save_freq and step > 0 and step % int(config.save_freq) == 0:
            torch.save(
                {
                    "format": ACT_FORMAT,
                    "state_dict": model.state_dict(),
                    "config": config_repr(config),
                    "action_mean": action_mean.tolist(),
                    "action_std": action_std.tolist(),
                    "state_mean": state_mean.tolist(),
                    "state_std": state_std.tolist(),
                },
                Path(config.output_dir) / f"model_step{step:06d}.pt",
            )
    model.eval()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": ACT_FORMAT,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "config": config_repr(config),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "created_at": now_iso(),
        "dataset_path": str(Path(config.dataset_path).resolve()),
        "train_steps": len(losses),
        "first_loss": first_loss,
        "final_loss": losses[-1] if losses else float("nan"),
        "loss_last_10_mean": float(np.mean(losses[-10:])) if losses else float("nan"),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "torch_version": str(torch.__version__),
        "device": device,
    }
    torch.save(checkpoint, output / MODEL_FILE)
    (output / SPEC_FILE).write_text(json.dumps({k: v for k, v in checkpoint.items() if k != "state_dict"}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {key: checkpoint[key] for key in ("train_steps", "first_loss", "final_loss", "loss_last_10_mean", "wall_seconds", "device", "torch_version")}


def config_repr(config: ACTStyleConfig) -> dict[str, Any]:
    return {
        "chunk_size": int(config.chunk_size),
        "execution_horizon": int(config.execution_horizon),
        "d_model": int(config.d_model),
        "layers": int(config.layers),
        "heads": int(config.heads),
        "latent_dim": int(config.latent_dim),
        "image_size": int(config.image_size),
        "pretrained_backbone": bool(config.pretrained_backbone),
    }


class ACTStylePolicy:
    name = "a3opt_act_style"

    def __init__(self, checkpoint_file: str | Path, device: str | None = "cpu") -> None:
        self.device = resolve_device(device)
        checkpoint = torch.load(Path(checkpoint_file), map_location="cpu", weights_only=False)
        if checkpoint.get("format") != ACT_FORMAT:
            raise ValueError(f"not an act-style checkpoint: {checkpoint_file}")
        inner = dict(checkpoint["config"])
        self.config = ACTStyleConfig(
            chunk_size=inner["chunk_size"],
            execution_horizon=inner["execution_horizon"],
            d_model=inner["d_model"],
            layers=inner["layers"],
            heads=inner["heads"],
            latent_dim=inner["latent_dim"],
            image_size=inner["image_size"],
            pretrained_backbone=inner["pretrained_backbone"],
        )
        self.model = ACTStyleModel(self.config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)
        self.state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
        self.action_mode = "joint_position"
        self.execution_horizon = int(self.config.execution_horizon)
        self._queue: list[np.ndarray] = []
        self._previous = np.zeros(ACTION_DIM, dtype=np.float32)
        self._step = 0

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

    def _predict(self, observation: Mapping[str, Any]) -> list[np.ndarray]:
        state = np.asarray(observation[STATE], dtype=np.float32)
        state_tensor = torch.from_numpy(((state - self.state_mean) / self.state_std)[None]).to(self.device)
        previous = torch.from_numpy(self._previous[None]).to(self.device)
        fraction = torch.tensor([[min(self._step / STEP_SCALE, 1.0)]], dtype=torch.float32, device=self.device)
        frames = []
        for camera in CAMERAS:
            key = FRONT if camera == "front" else (LEFT if camera == "left_wrist" else RIGHT)
            frame = downsample(observation[key], int(self.config.image_size))
            frames.append(torch.from_numpy(frame[None]).to(self.device))
        with torch.no_grad():
            prediction, _ = self.model(tuple(frames), state_tensor, previous, fraction, None)
        chunk = prediction[0].detach().cpu().numpy().astype(np.float32)
        actions = chunk * self.action_std[None, :] + self.action_mean[None, :]
        actions = np.nan_to_num(actions, nan=0.0, posinf=4.0, neginf=-4.0)
        actions = np.clip(actions, -4.0, 4.0)
        return [np.ascontiguousarray(row) for row in actions[: self.execution_horizon]]


__all__ = [
    "ACT_FORMAT",
    "ACTStyleConfig",
    "ACTStyleModel",
    "ACTStylePolicy",
    "train_act_style",
]
