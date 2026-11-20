import numpy as np
import pytest

from a3opt import placement_wrist_rgb


def test_wrist_policy_uses_the_configured_task_slots(monkeypatch):
    pytest.importorskip("a3_dual_arm_sim")
    monkeypatch.setattr(placement_wrist_rgb.PlacementPolicy, "__init__", lambda *args: None)
    policy = placement_wrist_rgb.WristPlacementPolicy(None, 4, .75)
    np.testing.assert_allclose(policy.goals, [[.028, 0.], [-.028, 0.]])


def test_wrist_fit_admits_only_the_axis_with_improved_held_out_error():
    rows = []
    for seed in range(sum(placement_wrist_rgb.SPLIT_SIZES)):
        signal = np.sin(seed)
        rows.append({"column": seed % 4 + 1, "samples": [
            {"batch": batch, "sensors": [0.] * 6, "wrist": [signal], "offset": [.003 * signal, .001]}
            for batch in range(2)]})
    model, audit = placement_wrist_rgb.fit(rows)
    assert audit["active"] == [True, False]
    assert model["deadband"][0] >= .0015
