from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from a3opt.backends.local_sim import (
    BCPolicy,
    LocalSimBackend,
    downsample_image,
    evaluate_checkpoint,
    train_candidate,
)
from a3opt.backends.policy_plugin import create_policy
from a3opt.contracts import CandidateSpec, ExperimentBudget

FRAME_SEEDS = [7] * 12 + [8] * 8


def make_candidate(candidate_id: str, **overrides):
    defaults = {
        "encoder": "tiny_cnn",
        "fusion": "concat",
        "action_head": "chunk",
        "chunk_size": 2,
        "lr": 1e-3,
        "image_size": 64,
        "seed": 0,
    }
    defaults.update(overrides)
    return CandidateSpec(candidate_id=candidate_id, **defaults)


def write_synthetic_dataset(path: Path, frames: int = 20, image_size: int = 64) -> Path:
    rng = np.random.default_rng(0)
    front = rng.integers(0, 256, size=(frames, image_size, image_size, 3), dtype=np.uint8)
    state = rng.normal(0.0, 0.2, size=(frames, 16)).astype(np.float32)
    action = rng.normal(0.0, 0.1, size=(frames, 16)).astype(np.float32)
    seeds = np.asarray(FRAME_SEEDS[:frames], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, front=front, state=state, action=action, seeds=seeds)
    return path


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    return write_synthetic_dataset(tmp_path / "synthetic.npz")


def train_synthetic(candidate: CandidateSpec, dataset: Path, out_dir: Path, steps: int = 4):
    return train_candidate(
        candidate,
        dataset,
        ExperimentBudget(max_train_steps=steps, batch_size=4),
        out_dir,
    )


def make_observation() -> dict[str, np.ndarray]:
    return {
        "observation.images.front": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.state": np.zeros(16, dtype=np.float32),
    }


def test_train_candidate_writes_checkpoint(synthetic_dataset: Path, tmp_path: Path) -> None:
    candidate = make_candidate("tiny_cnn-concat-chunk-k2")
    result = train_synthetic(candidate, synthetic_dataset, tmp_path / "ckpt")
    assert result.status == "ok"
    assert result.train_steps == 4
    artifact = Path(result.artifact)
    assert artifact.is_file()
    assert artifact.name == "model.pt"
    assert (artifact.parent / "spec.json").is_file()
    assert (artifact.parent / "train_log.json").is_file()
    for key in ("first_loss", "final_loss", "train_steps", "device"):
        assert key in result.metrics
        assert math.isfinite(result.metrics[key])
    assert result.metrics["final_loss"] >= 0.0
    spec = json.loads((artifact.parent / "spec.json").read_text(encoding="utf-8"))
    assert spec["architecture"]["encoder"] == "tiny_cnn"
    assert spec["architecture"]["chunk_size"] == 2
    assert spec["num_frames"] == 20
    assert spec["num_episodes"] == 2
    payload = result.to_dict()
    assert math.isfinite(payload["metrics"]["final_loss"])
    assert payload["wall_seconds"] >= 0.0


def test_train_candidate_state_mlp_without_images(synthetic_dataset: Path, tmp_path: Path) -> None:
    candidate = make_candidate(
        "state_mlp-concat-chunk-k1",
        encoder="state_mlp",
        action_head="chunk",
        chunk_size=1,
        image_size=0,
        seed=1,
    )
    result = train_synthetic(candidate, synthetic_dataset, tmp_path / "ckpt", steps=3)
    assert result.status == "ok"
    assert result.train_steps == 3
    assert Path(result.artifact).is_file()
    spec = json.loads((Path(result.artifact).parent / "spec.json").read_text(encoding="utf-8"))
    assert spec["architecture"]["encoder"] == "state_mlp"
    assert spec["architecture"]["chunk_size"] == 1


def test_saved_policy_predicts_joint_targets(synthetic_dataset: Path, tmp_path: Path) -> None:
    candidate = make_candidate("tiny_cnn-film-chunk-k3", fusion="film", chunk_size=3)
    result = train_synthetic(candidate, synthetic_dataset, tmp_path / "ckpt")
    assert result.status == "ok"
    policy = BCPolicy(Path(result.artifact).parent)
    assert policy.action_mode == "joint_position"
    assert policy.finished is False
    policy.reset(None)
    observation = make_observation()
    actions = [policy.act(observation, "transfer 10 cookies into target box") for _ in range(7)]
    assert len(actions) == 7
    for action in actions:
        assert action.shape == (16,)
        assert np.all(np.isfinite(action))
        assert np.all(np.abs(action) <= 4.0)
    policy.close()


def test_policy_plugin_requires_and_loads_checkpoint(
    synthetic_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("A3OPT_CHECKPOINT", raising=False)
    with pytest.raises(RuntimeError):
        create_policy()
    candidate = make_candidate("plugin-check")
    result = train_synthetic(candidate, synthetic_dataset, tmp_path / "ckpt")
    assert result.status == "ok"
    monkeypatch.setenv("A3OPT_CHECKPOINT", str(Path(result.artifact).parent))
    policy = create_policy()
    assert isinstance(policy, BCPolicy)
    assert policy.action_mode == "joint_position"
    policy.reset(None)
    assert policy.act(make_observation()).shape == (16,)
    policy.close()


def test_downsample_image_uses_block_mean() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[0:4, 0:4] = 200
    small = downsample_image(image, 64)
    assert small.shape == (64, 64, 3)
    assert small.dtype == np.uint8
    assert int(small[0, 0, 0]) == 200
    assert int(small[1, 1, 0]) == 0


def test_local_sim_backend_contract(tmp_path: Path) -> None:
    backend = LocalSimBackend(
        {
            "dataset_path": str(tmp_path / "missing.npz"),
            "experiment_root": str(tmp_path / "experiments"),
        }
    )
    assert backend.name == "local_sim"
    available, reason = backend.available()
    assert isinstance(available, bool)
    assert isinstance(reason, str)
    assert backend.config["seeds"] == [1000, 1001, 1002, 1003, 1004, 1005]


@pytest.mark.skipif(
    os.environ.get("A3OPT_RUN_SIM_TESTS") != "1",
    reason="set A3OPT_RUN_SIM_TESTS=1 to run the real MuJoCo evaluation path",
)
def test_real_sim_evaluation_path(synthetic_dataset: Path, tmp_path: Path) -> None:
    candidate = make_candidate("sim-eval-check", action_head="chunk", chunk_size=1)
    result = train_synthetic(candidate, synthetic_dataset, tmp_path / "ckpt", steps=2)
    assert result.status == "ok"
    evaluation = evaluate_checkpoint(
        candidate,
        Path(result.artifact).parent,
        [2100],
        3,
        tmp_path / "eval",
    )
    assert evaluation.status == "ok", evaluation.error
    assert len(evaluation.episodes) == 1
    row = evaluation.episodes[0]
    assert row.seed == 2100
    assert 0.0 <= row.score <= 10.0
    assert 1 <= row.steps <= 3
    assert (tmp_path / "eval" / "eval.json").is_file()
