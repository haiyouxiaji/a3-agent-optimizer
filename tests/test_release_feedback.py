import numpy as np

from a3opt.backends.adaptive_phase import PLACE, TRANSFER
from a3opt.backends.advanced_phase import SAMPLES
from a3opt.release_feedback_agent import ReleaseFeedbackPolicy, predict_shift, regress


def test_observations_only_change_commands_at_the_first_descent(monkeypatch):
    policy = ReleaseFeedbackPolicy.__new__(ReleaseFeedbackPolicy)
    policy.reset()
    policy.actions = [[np.zeros((2, 16)) for _ in SAMPLES] for _ in range(2)]
    policy.cursor = (0, TRANSFER, 1)
    policy.calibration = {"center": np.zeros(9), "scale": np.ones(9), "weights": np.zeros(9),
                          "bias": .004, "column": 1}
    observations = []

    def capture(controller, observation):
        observations.append(observation["time"])
        return np.zeros(9)

    def shift(controller):
        controller.actions[0][PLACE][:, 0] += controller.release_shift_m

    monkeypatch.setattr("a3opt.release_feedback_agent.observed_release_features", capture)
    monkeypatch.setattr(ReleaseFeedbackPolicy, "apply_shift", shift)
    task = "transfer 10 cookies into target box"
    assert policy.act({"time": 1}, task).sum() == 0
    assert observations == []
    assert policy.act({"time": 2}, task)[0] == .004
    assert policy.act({"time": 3}, task)[0] == .004
    assert observations == [2]
    assert not np.any(policy.actions[1])


def test_initial_feature_model_cannot_use_runtime_signals():
    features = np.random.default_rng(10).normal(size=(16, 9))
    labels = .003 * features[:, 7]
    model = regress(features, labels, (5, 6), 1., 2)
    changed = features.copy()
    changed[:, 7:] += 99
    np.testing.assert_array_equal(predict_shift(model, features), predict_shift(model, changed))
