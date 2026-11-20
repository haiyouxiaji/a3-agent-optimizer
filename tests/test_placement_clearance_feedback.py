import numpy as np

from a3opt import placement_clearance_feedback as feedback
from a3opt.backends.adaptive_phase import PLACE


def test_wall_label_is_signed_for_the_batch_and_excludes_scoring_tolerance():
    cookie = {"abs_vertical_axis": 1., "bottom_above_floor_m": .05,
              "footprint_margin_m": [.002, .01], "xyz_m": [.035, 0., .1]}
    snapshot = {"batches": [[cookie] * 5, [cookie] * 5]}
    np.testing.assert_allclose(feedback.clearance(snapshot, 0, .004), -.002)
    np.testing.assert_allclose(feedback.clearance(snapshot, 1, .004), .068)
    snapshot["batches"][0][0] = {**cookie, "bottom_above_floor_m": .02}
    assert feedback.clearance(snapshot, 0, .004) is None


def test_sensor_holdout_counts_unsupported_triggers_as_errors():
    samples = [dict(margin_m=-.002, estimated_m=-.001)] * 4 + [dict(margin_m=.01, estimated_m=.01)] * 4
    unsupported = dict(margin_m=None, estimated_m=-.001)
    baseline = feedback.metrics([*samples, unsupported])
    assert baseline["unsupported_triggers"] == 1
    assert feedback.improves_sensor(baseline, feedback.metrics([*samples, {**unsupported, "estimated_m": .01}]))
    assert not feedback.improves_sensor(baseline, feedback.metrics([*samples, unsupported]))


def test_feedback_abstains_without_wrist_and_keeps_incumbent_descent(monkeypatch):
    controller = feedback.FeedbackPolicy.__new__(feedback.FeedbackPolicy)
    model = feedback.WallModel("wrist", (0.,) * 67, (1.,) * 67, (0.,) * 67, -.001)
    controller.feedback = {0: feedback.Feedback(0, "inset4-hover12", model)}
    controller.profiles = {(1, 0): "vertical"}
    controller.actions = [[], []]
    controller.cursor, controller.retry_remaining = (0, PLACE, 0), 0
    calls = []
    monkeypatch.setattr(feedback, "task_column", int)
    monkeypatch.setattr(feedback, "readings", lambda *args: np.zeros(17))
    monkeypatch.setitem(feedback.IMAGE_ENCODERS, "wrist", lambda frame: np.zeros(50))
    monkeypatch.setattr(feedback.descent, "adjust", lambda phases, xyz, name: calls.append(name))
    monkeypatch.setattr(feedback.descent.DescentPolicy, "act", lambda self, obs, task: "reference")
    monkeypatch.setattr(feedback.PlacementPolicy, "act", lambda self, obs, task: "corrected")
    monkeypatch.setattr(controller, "shift_placement", lambda batch, shift: calls.append((batch, tuple(shift))))
    frame = {"observation.eef_pose": np.zeros(7), feedback.wrist.WRIST: np.ones((8, 8, 3))}
    assert controller.camera_names == ("left_wrist",)
    assert controller.act({}, "1") == "reference"
    assert controller.act(frame, "2") == "reference"
    assert controller.act(frame, "1") == "corrected"
    controller.retry_remaining = 1
    assert controller.act(frame, "1") == "reference"
    assert calls == ["vertical", (0, (-.004, 0., .012))]


def test_camera_schedule_requests_front_before_first_action():
    controller = feedback.FeedbackPolicy.__new__(feedback.FeedbackPolicy)
    controller.actions = None
    assert controller.camera_names == ("front",)
