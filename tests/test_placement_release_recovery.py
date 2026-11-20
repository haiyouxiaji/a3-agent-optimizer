from types import SimpleNamespace

import numpy as np

from a3opt import placement_release_recovery as recovery
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER


def test_recovery_uses_the_available_budget_and_keeps_the_completed_prefix(monkeypatch):
    solver = SimpleNamespace(shift=lambda position, delta: position + np.r_[delta, np.zeros(4)],
                             ranges=np.tile([-1., 1.], (7, 1)))
    monkeypatch.setattr(recovery, "nominal_kinematics", lambda: solver)
    monkeypatch.setattr(recovery.descent, "tool_xyz", lambda _, position: position[:3].copy())
    phases = [np.zeros((8, 16)) for _ in range(RETRACT + 1)]
    phases[TRANSFER][-1, 2] = .075
    phases[PLACE][:, 2] = np.linspace(.070, 0., 8)
    phases[OPEN][:, 7] = np.linspace(.01, .25, 8)
    original = [phase.copy() for phase in phases]
    policy = SimpleNamespace(actions=[None, phases], cursor=(1, PLACE, 6), pose=np.zeros(5), step=490)
    observed = {"observation.eef_pose": [0., 0., .020], "observation.state": [0., 0., .020, 0., 0., 0., 0.]}
    assert not recovery.apply_recovery(policy, observed, "raise-then-outward")
    np.testing.assert_array_equal(phases, original)
    policy.step = 410
    assert recovery.apply_recovery(policy, observed, "raise-then-outward")
    assert len(phases[PLACE]) == len(original[PLACE]) + recovery.EXTRA_STEPS
    np.testing.assert_array_equal(phases[PLACE][:6], original[PLACE][:6])
    np.testing.assert_array_equal(phases[OPEN][:, 7:], original[OPEN][:, 7:])
    np.testing.assert_allclose(phases[OPEN][:, :3], np.tile([-.004, 0., .026], (8, 1)))


def test_inserting_recovery_steps_cannot_trigger_a_second_recovery(monkeypatch):
    policy = recovery.RecoveryPolicy.__new__(recovery.RecoveryPolicy)
    policy.recovery = recovery.Recovery("raise-in-place")
    policy.recovery_checked = policy.probe = False
    policy.retry_remaining = 0
    policy.transfer_force = np.zeros(7)
    original = np.zeros((8, 16))
    phases = {PLACE: original}
    policy.actions = [None, phases]
    policy.cursor = (1, PLACE, 6)
    monkeypatch.setattr(recovery, "task_column", int)
    monkeypatch.setattr(recovery, "nominal_kinematics", object)
    monkeypatch.setattr(recovery.descent, "tool_xyz", lambda _, joints: joints[:3])
    monkeypatch.setattr(recovery.descent.DescentPolicy, "act", lambda *_: "action")
    monkeypatch.setattr(recovery, "apply_recovery", lambda *_: phases.update({PLACE: np.zeros((16, 16))}))
    observed = {"observation.force": np.zeros(17), "observation.eef_pose": np.zeros(3)}
    assert policy.act(observed, "2") == "action"
    policy.cursor = (1, PLACE, 12)
    assert policy.signal(observed, "2") is None
