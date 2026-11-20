import numpy as np

from a3opt.backends.adaptive_phase import PLACE, TRANSFER
from a3opt.backends.advanced_phase import SAMPLES
from a3opt.release_outcome_agent import OutcomeReleasePolicy, fit_rule, validate_rule


def test_routing_requires_reproducible_improvement_over_fixed_arms():
    features = np.zeros((24, 9))
    features[:12, 7], features[12:, 7] = -1, 1
    labels = np.tile([[7, 0, 0], [9, 0, 0], [9, 0, 0]], (24, 1, 1))
    labels[:12, 1] = [10, 1, 0]
    labels[12:, 2] = [10, 1, 0]
    learned = fit_rule(features, labels)
    assert (learned["signal"], learned["left"], learned["right"]) == (7, 1, 2)
    validation = validate_rule(features, labels)
    assert validation["eligible"]
    assert validation["routed_objective"] == [24, 240, 0]
    assert not validate_rule(np.zeros_like(features), labels)["eligible"]


def test_routing_rejects_count_regression_and_does_not_bypass_safety():
    features = np.zeros((24, 9))
    features[:12, 7], features[12:, 7] = -1, 1
    labels = np.tile([[9, 0, 0], [9, 0, 0], [10, 1, 1]], (24, 1, 1))
    labels[:3, 1] = [10, 1, 0]
    labels[3:4, 1] = [3, 0, 0]
    labels[12:, 1] = [3, 0, 0]
    assert fit_rule(features, labels)["signal"] is None
    assert not validate_rule(features, labels)["eligible"]


def test_route_changes_only_first_placement_and_only_in_focused_column(monkeypatch):
    policy = OutcomeReleasePolicy.__new__(OutcomeReleasePolicy)
    policy.column = 1
    policy.rule = {"signal": 7, "threshold": 0, "left": 0, "right": 2, "fallback": 0}
    policy.calibration = None
    policy.calibrations = {2: {"center": np.zeros(9), "scale": np.ones(9), "weights": np.zeros(9), "bias": .004}}
    calls = []

    def capture(controller, observation):
        return np.full(9, observation["signal"])

    def shift(controller):
        calls.append(controller.active_arm)
        controller.actions[0][PLACE][:, 0] += controller.release_shift_m

    monkeypatch.setattr("a3opt.release_outcome_agent.observed_release_features", capture)
    monkeypatch.setattr("a3opt.release_feedback_agent.observed_release_features", capture)
    monkeypatch.setattr(OutcomeReleasePolicy, "apply_shift", shift)
    for task, signal, expected in (("transfer 10 cookies into target box", 1., .004),
                                   ("transfer 10 cookies from source column 2 into target box", 1., 0.),
                                   ("transfer 10 cookies into target box", np.nan, 0.)):
        policy.reset()
        policy.actions = [[np.zeros((2, 16)) for _ in SAMPLES] for _ in range(2)]
        policy.cursor = (0, TRANSFER, 1)
        assert policy.act({"signal": signal}, task).sum() == 0
        assert policy.act({"signal": signal}, task)[0] == expected
        assert not np.any(policy.actions[1])
    assert calls == ["feedback"]
