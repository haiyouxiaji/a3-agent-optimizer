from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from a3opt.backends.bc2 import BC2Config, BC2Policy, downsample, train_candidate
from a3opt.backends.policy_plugin import create_policy
from a3opt.contracts import CandidateSpec, ExperimentBudget

FRAMES = 24


def write_synthetic(path: Path, size: int = 128) -> Path:
    rng = np.random.default_rng(0)
    np.savez_compressed(
        path,
        front=rng.integers(0, 256, (FRAMES, size, size, 3), dtype=np.uint8),
        left_wrist=rng.integers(0, 256, (FRAMES, size, size, 3), dtype=np.uint8),
        right_wrist=rng.integers(0, 256, (FRAMES, size, size, 3), dtype=np.uint8),
        state=rng.normal(size=(FRAMES, 16)).astype(np.float32),
        action=rng.normal(scale=0.2, size=(FRAMES, 16)).astype(np.float32),
        seeds=np.asarray([7] * 12 + [8] * 12, dtype=np.int32),
    )
    return path


def make_candidate() -> CandidateSpec:
    return CandidateSpec(
        candidate_id="synthetic-bc2",
        encoder="tiny_cnn",
        fusion="concat",
        action_head="chunk",
        chunk_size=4,
        lr=1e-3,
        image_size=64,
        state_dim=16,
        seed=0,
    )


@pytest.fixture()
def checkpoint(tmp_path) -> Path:
    dataset = write_synthetic(tmp_path / "synthetic.npz")
    result = train_candidate(
        make_candidate(),
        dataset,
        ExperimentBudget(max_train_steps=4, batch_size=4),
        tmp_path / "checkpoint",
        config=BC2Config(dataset_path=dataset, experiment_root=tmp_path),
    )
    assert result.status == "ok"
    assert (tmp_path / "checkpoint" / "model.pt").is_file()
    assert (tmp_path / "checkpoint" / "spec.json").is_file()
    return tmp_path / "checkpoint"


def observation(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "observation.images.front": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation.images.left_wrist": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation.images.right_wrist": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation.state": rng.normal(size=(16,)).astype(np.float32),
    }


def test_policy_predicts_joint_targets(checkpoint) -> None:
    rng = np.random.default_rng(1)
    policy = BC2Policy(checkpoint, device="cpu", execution_horizon=2)
    policy.reset(None)
    action = policy.act(observation(rng), "")
    assert action.shape == (16,)
    assert np.all(np.isfinite(action))


def test_policy_plugin_detects_bc2_checkpoint(checkpoint, monkeypatch) -> None:
    monkeypatch.setenv("A3OPT_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("A3OPT_DEVICE", "cpu")
    policy = create_policy()
    assert isinstance(policy, BC2Policy)
    assert policy.action_mode == "joint_position"


def test_downsample_block_mean() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:128] = 255
    small = downsample(image, 128)
    assert small.shape == (128, 128, 3)
    assert small.dtype == np.uint8
