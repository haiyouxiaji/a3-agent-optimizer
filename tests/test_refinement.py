from __future__ import annotations

import numpy as np
import pytest

from a3opt.backends.policy_plugin import create_policy
from a3opt.backends.trajectory import FRONT, fit_trajectory, layout_features
from a3opt.refinement_eval import DEVELOPMENT_SEEDS, summarize, validate_rows
from a3opt.sampling import choose_layouts


def _row(seed: int, score: int) -> dict:
    return {"seed": seed, "score": score, "max_score": 10, "steps": 410, "source_column": 1,
            "success": score == 10, "cookies_in_target": score, "cookies_in_source": 80 - score,
            "layout_features": [seed - DEVELOPMENT_SEEDS[0], 0, 0, 0, 0]}


def test_benchmark_validation_rejects_duplicates_and_invalid_success() -> None:
    rows = [_row(2000, 10), _row(2001, 5)]
    metrics = summarize(rows, (2000, 2001))
    assert metrics["mean_score"] == 7.5
    assert metrics["success_rate"] == 0.5
    with pytest.raises(ValueError, match="evaluation episodes"):
        validate_rows([rows[0], rows[0]], (2000, 2001))
    with pytest.raises(ValueError, match="success label"):
        validate_rows([{**rows[0], "cookies_in_source": 69}], (2000,))


def test_sampling_uses_failures_without_reusing_development_seeds() -> None:
    datasets = np.array([[2, 0, 0, 0, 0], [3, 0, 0, 0, 0]])
    layouts = {1200: [5, 0, 0, 0, 0], 1201: [0, 0, 0, 0, 0]}
    feedback = [_row(seed, 0 if seed == 2005 else 10) for seed in DEVELOPMENT_SEEDS]
    choice = choose_layouts(layouts, datasets, feedback, 1, method="feedback", random_seed=0)
    assert choice[0]["seed"] == 1200
    with pytest.raises(ValueError, match="training pool"):
        choose_layouts({2000: [0] * 5}, datasets, feedback, 1, method="feedback", random_seed=0)


def test_trajectory_checkpoint_matches_the_policy_plugin_protocol(tmp_path, monkeypatch) -> None:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[44:64, 72:105] = (220, 160, 20)
    frame[65:86, 18:39] = (115, 132, 154)
    feature = layout_features(frame)
    actions = np.zeros((10, 16), dtype=float)
    actions[[0, 1, 5, 6], 7] = 1
    checkpoint = tmp_path / "trajectory.npz"
    fit_trajectory(checkpoint, np.array([feature]), [actions], np.array([1000]))
    monkeypatch.setenv("A3OPT_CHECKPOINT", str(checkpoint))
    policy = create_policy()
    policy.reset(object())
    result = policy.act({FRONT: frame}, "transfer cookies")
    assert result.shape == (16,)
    assert result[7] == 1
    policy.close()
