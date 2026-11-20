import numpy as np

from a3opt import placement_release_hold as hold
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER


class CartesianSolver:
    ranges = np.tile([-1., 1.], (7, 1))

    def shift(self, position, displacement):
        return position + np.r_[displacement, np.zeros(4)]


def test_release_hold_is_inactive_before_the_first_plan(monkeypatch):
    policy = hold.ReleaseHoldPolicy.__new__(hold.ReleaseHoldPolicy)
    policy.release_hold = hold.ReleaseHold("planned-18")
    policy.actions = None
    monkeypatch.setattr(hold.descent.DescentPolicy, "act", lambda self, observation, task: "initial action")
    assert policy.act({}, "") == "initial action"


def test_replan_holds_the_arm_during_opening_and_preserves_the_grippers(monkeypatch):
    monkeypatch.setattr(hold.descent, "tool_xyz", lambda solver, position: position[:3].copy())
    baseline = [np.zeros((5, 16)) for _ in range(RETRACT + 1)]
    baseline[TRANSFER][-1, 2] = .070
    baseline[OPEN][:, 7] = np.linspace(.2, .5, 5)
    baseline[RETRACT][:, 2] = np.linspace(0., .070, 5)
    actions = [phase.copy() for phase in baseline]
    assert hold.replan(actions, np.array([.020, -.005, .070]), 0., "outward-24", CartesianSolver())
    expected = np.array([.008, -.005, .024, 0., 0., 0., 0.])
    np.testing.assert_allclose(actions[PLACE][-1, :7], expected)
    np.testing.assert_allclose(actions[OPEN][:, :7], np.tile(expected, (5, 1)))
    np.testing.assert_allclose(actions[RETRACT][0, :7], expected)
    np.testing.assert_array_equal(actions[RETRACT][-1], baseline[RETRACT][-1])
    for actual, original in zip(actions, baseline, strict=True):
        np.testing.assert_array_equal(actual[:, 7:], original[:, 7:])


def test_replan_leaves_baseline_intact_when_the_hold_pose_is_unreachable(monkeypatch):
    monkeypatch.setattr(hold.descent, "tool_xyz", lambda solver, position: np.zeros(3))
    baseline = [np.zeros((5, 16)) for _ in range(RETRACT + 1)]
    actions = [phase.copy() for phase in baseline]
    assert not hold.replan(actions, np.array([0., 0., .06]), 0., "planned-18", CartesianSolver())
    np.testing.assert_array_equal(actions, baseline)


def test_a_release_hold_that_harms_a_control_cannot_be_selected():
    def terminal(count):
        return {"counted_ids": list(range(count)), "success": count == 10, "safety_reason": None, "wrong_source": 0}

    def trial(seed, reference, planned, anchored):
        return {"seed": seed, "trials": {"reference": terminal(reference), "planned-18": terminal(planned),
                                       "anchored-24": terminal(anchored), "outward-24": terminal(reference)}}

    rows = [trial(20, 5, 10, 7), trial(21, 5, 10, 7), trial(22, 10, 9, 10)]
    selected, metrics = hold.fit(rows, [22])
    assert metrics["planned-18"]["gain"] == 9 and not metrics["planned-18"]["admitted"]
    assert selected == (hold.ReleaseHold("anchored-24"),)
