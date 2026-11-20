import numpy as np

from a3opt import placement_contact_feedback as contact
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT


def test_guard_displacement_keeps_executed_descent_and_gripper_commands(monkeypatch):
    class Kinematics:
        @staticmethod
        def shift(joints, offset):
            return joints + np.r_[offset, np.zeros(4)]

    controller = contact.ContactPolicy.__new__(contact.ContactPolicy)
    controller.pose = np.zeros(5)
    controller.actions = [[np.zeros((4, 16)) for _ in range(RETRACT + 1)] for _ in range(2)]
    phases = controller.actions[1]
    phases[OPEN][:, 7] = .4
    monkeypatch.setattr(contact, "nominal_kinematics", Kinematics)
    contact.apply_correction(controller, 1, 2, "inset4-raise12")
    np.testing.assert_array_equal(phases[PLACE][:2], 0.)
    np.testing.assert_allclose(phases[PLACE][-1, :3], [.004, 0., .012])
    np.testing.assert_array_equal(phases[OPEN][:, 7], .4)
    np.testing.assert_allclose(phases[RETRACT][-1, :7], 0.)


def test_contact_gate_keeps_reference_before_initializing_actions():
    controller = contact.ContactPolicy.__new__(contact.ContactPolicy)
    controller.actions = None
    assert controller.pending("unused task") is None
