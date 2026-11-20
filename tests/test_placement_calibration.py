import numpy as np
import pytest

from a3opt import placement_calibration as calibration


def test_pose_correction_respects_bounds_and_disabled_axes():
    pose = np.array([.1, .1, .2, .2, .01])
    updates = np.array([1., -1., .001, -1., 1.])
    blend = np.array([1., .5, 0., 1., .5])
    expected = [.003, -.0015, 0., -.003, .010]
    np.testing.assert_allclose(calibration.bounded_pose(pose, pose + updates, blend) - pose, expected)


def test_calibration_audit_rejects_part_of_the_fit_without_refitting(monkeypatch):
    monkeypatch.setattr(calibration.advanced_trajectory, "predict_pose", lambda *_: np.zeros(5))
    signal = np.tile(np.linspace(-1., 1., 12), 5)
    labels = signal[:, None] * [.002, .001, .001, -.001, .01]
    dataset = {"outline": signal[:, None], "silhouette": signal[:, None], "wrist": -signal[:, None], "labels": labels}
    models, initial = calibration.fit(dataset, {}, 36, 12)
    assert set(initial["audit"]) == {"source_xy", "target_xy_yaw"}
    assert np.all(models["blend"] == 1.)

    changed = labels.copy()
    changed[48:, :2] = 0.
    revised, audit = calibration.fit({**dataset, "labels": changed}, {}, 36, 12)
    assert audit["winner"] == initial["winner"]
    np.testing.assert_array_equal(revised["weights"], models["weights"])
    np.testing.assert_array_equal(revised["blend"], [0., 0., 1., 1., 1.])


def test_pair_audit_allows_bounded_plan_change_but_rejects_changed_reset():
    old = {"seed": 12, "column": 2, "profile": "advanced", "suite": "random", "strict": False,
           "task_prompt": "source column 2", "initial_pose": [0.] * 5, "estimated_pose": [0.] * 5,
           "cameras": {}, "light_diffuse": [], "color": "gold", "requested_score": 5, "score": 5,
           "success": False, "safety_reason": None, "action_sha256": "old", "applied_action_sha256": "old"}
    new = {**old, "estimated_pose": [.001, 0., 0., 0., 0.], "requested_score": 10, "score": 10,
           "success": True, "action_sha256": "new"}
    model = (calibration.Calibration("checkpoint", "digest"),)
    assert calibration.compare([old], [new], model)["paired_count_delta"] == 5
    with pytest.raises(ValueError, match="outside the locked contract"):
        calibration.compare([old], [{**new, "initial_pose": [1.] * 5}], model)
    with pytest.raises(ValueError, match="unchanged pose"):
        calibration.compare([old], [{**new, "estimated_pose": old["estimated_pose"]}], model)
