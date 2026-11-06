from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from a3opt.backends.act_style import ACT_FORMAT, ACTStyleConfig, ACTStylePolicy, train_act_style


def write_synthetic(path, frames: int = 24, size: int = 128):
    rng = np.random.default_rng(0)
    np.savez_compressed(
        path,
        front=rng.integers(0, 256, (frames, size, size, 3), dtype=np.uint8),
        left_wrist=rng.integers(0, 256, (frames, size, size, 3), dtype=np.uint8),
        right_wrist=rng.integers(0, 256, (frames, size, size, 3), dtype=np.uint8),
        state=rng.normal(size=(frames, 16)).astype(np.float32),
        action=rng.normal(scale=0.2, size=(frames, 16)).astype(np.float32),
        seeds=np.asarray([7] * 12 + [8] * 12, dtype=np.int32),
    )


@pytest.fixture()
def checkpoint(tmp_path):
    dataset = tmp_path / "synthetic.npz"
    write_synthetic(dataset)
    config = ACTStyleConfig(
        dataset_path=dataset,
        output_dir=tmp_path / "checkpoint",
        steps=2,
        batch_size=2,
        chunk_size=4,
        execution_horizon=2,
        d_model=64,
        layers=1,
        heads=4,
        latent_dim=8,
        image_size=128,
        device="cpu",
    )
    report = train_act_style(config)
    assert report["train_steps"] == 2
    assert (tmp_path / "checkpoint" / "model.pt").is_file()
    return tmp_path / "checkpoint" / "model.pt"


def test_policy_returns_joint_targets(checkpoint):
    rng = np.random.default_rng(1)
    policy = ACTStylePolicy(checkpoint, device="cpu")
    policy.reset(None)
    observation = {
        "observation.images.front": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation.images.left_wrist": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation.images.right_wrist": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation.state": rng.normal(size=(16,)).astype(np.float32),
    }
    action = policy.act(observation, "")
    assert action.shape == (16,)
    assert np.all(np.isfinite(action))


def test_checkpoint_format(checkpoint):
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format"] == ACT_FORMAT
    assert payload["train_steps"] == 2
    assert "state_dict" in payload
