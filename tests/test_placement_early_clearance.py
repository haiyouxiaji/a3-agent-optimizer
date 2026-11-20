from types import SimpleNamespace

import numpy as np
import pytest

from a3opt import placement_clearance_refine as refine, placement_early_clearance as clearance


def test_probe_can_read_the_initial_observation_before_actions_exist():
    controller = clearance.EarlyClearancePolicy.__new__(clearance.EarlyClearancePolicy)
    controller.clearance = clearance.Clearance(1, "raise14")
    controller.actions = None

    assert controller.early_signals({}, "") is None


@pytest.mark.parametrize("recipe", clearance.RECIPES)
def test_clearance_preserves_gripper_actions_and_fixed_phase_lengths(monkeypatch, recipe):
    solver = SimpleNamespace(ranges=np.asarray([[-1., 1.]] * 7))
    monkeypatch.setattr(clearance.descent, "tool_xyz", lambda solver, joints: joints[:3].copy())
    monkeypatch.setattr(clearance.recovery, "joint_target", lambda solver, initial, target: np.r_[target, initial[3:]])
    phases = [np.zeros((32, 16)) for _ in range(clearance.RETRACT + 1)]
    phases[clearance.PLACE][:, 2] = np.linspace(.10, .02, 32)
    phases[clearance.OPEN][:, 7] = np.linspace(.01, .09, 32)
    original = [phase.copy() for phase in phases]
    frame = 11
    measured = phases[clearance.PLACE][frame - 1, :3].copy()

    assert clearance.replan(phases, frame, measured, np.array([1., 0.]), recipe, solver)

    rise, inset = clearance.RECIPES[recipe]
    assert np.allclose(phases[clearance.OPEN][:, :3], [-inset, 0., .02 + rise])
    assert np.array_equal(phases[clearance.PLACE][:frame], original[clearance.PLACE][:frame])
    assert np.array_equal(phases[clearance.RETRACT][-1], original[clearance.RETRACT][-1])
    assert all(before.shape == after.shape and np.array_equal(before[:, 7:], after[:, 7:])
               for before, after in zip(original, phases, strict=True))


def test_guard_solves_repeated_clearance_gain_and_rejects_unseen_harm():
    def row(signal, gain):
        return {"features": [signal, *([0.] * (len(refine.SIGNALS) - 1))], "gain": gain,
                "complete_delta": int(gain > 0), "unsafe": False}

    groups = {name: [row(.002, 5), row(0., -4)] for name in ("first", "second", "third")}

    fitted, metrics = refine.fit(groups, 1, "raise14")
    assert metrics["admitted"] and all(fold["effect"]["gain"] == 5 for fold in metrics["heldout"].values())
    assert fitted[0].activates(row(.002, 0)["features"])
    assert not fitted[0].activates(row(0., 0)["features"])

    groups["third"].append(row(.002, -4))
    fitted, metrics = refine.fit(groups, 1, "raise14")
    assert not fitted and not metrics["admitted"]
    assert metrics["heldout"]["third"]["effect"]["harmed"] == 1
