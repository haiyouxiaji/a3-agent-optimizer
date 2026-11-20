import numpy as np

from a3opt import placement_clearance_agent as clearance
from a3opt import placement_descent_agent as descent
from a3opt import placement_descent_compose as composition
from a3opt import placement_descent_refine as refinement
from a3opt import placement_servo_agent as servo
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT


def test_alignment_finishes_before_landing_without_changing_gripper_or_duration(monkeypatch):
    class Solver:
        def shift(self, joints, displacement):
            changed = joints.copy()
            changed[:3] += displacement
            return changed

    monkeypatch.setattr(descent, "nominal_kinematics", Solver)
    monkeypatch.setattr(descent, "tool_xyz", lambda solver, joints: joints[:3].copy())
    phases = [np.zeros((10, 16)) for _ in range(RETRACT + 1)]
    phases[PLACE][:, 0] = np.linspace(0., .006, 10)
    phases[PLACE][:, 2] = np.linspace(.08, .02, 10)
    phases[OPEN][:, 0] = .006
    phases[OPEN][:, 7] = np.linspace(.28, .5, 10)
    original = [phase.copy() for phase in phases]

    descent.adjust(phases, np.zeros(3), "align-early")

    np.testing.assert_allclose(phases[PLACE][4:, 0], .006)
    np.testing.assert_array_equal(phases[PLACE][:, 2:], original[PLACE][:, 2:])
    np.testing.assert_array_equal(phases[OPEN][:, 7:], original[OPEN][:, 7:])
    np.testing.assert_array_equal(phases[RETRACT][-1], original[RETRACT][-1])
    assert [len(phase) for phase in phases] == [len(phase) for phase in original]


def test_descent_selection_requires_two_independent_gains_and_no_major_drop():
    def receipt(seed, old, changed):
        options = {key: {"counted_ids": list(range(changed if key == "vertical/0" else old)),
                         "success": False, "wrong_source": 0, "safety_reason": None} for key in descent.interventions()}
        return {"seed": seed, "column": 2, "options": options}

    failure = receipt(1, 5, 10)
    controls = [receipt(10, 10, 10), receipt(11, 10, 10)]
    assert not descent.fit([failure, *controls])[0]
    training = [failure, receipt(2, 5, 9), *controls]
    assert descent.fit(training)[0] == (descent.Descent(2, 0, "vertical"),)
    assert not descent.fit([*training, receipt(3, 10, 5)])[0]


def test_focused_descent_counts_each_scene_once_when_shortlisting_profiles():
    def receipt(seed, unsafe=False):
        options = {"reference": {"counted_ids": [1], "success": False}}
        for scope in ("0", "1", "0+1"):
            options[f"vertical/{scope}"] = {"counted_ids": [1, 2], "success": False,
                                            "safety_reason": None, "wrong_source": int(unsafe)}
        return {"seed": seed, "options": options}

    assert refinement.choose_profile([receipt(10), receipt(11, unsafe=True)])[0] is None
    assert refinement.choose_profile([receipt(10), receipt(11)])[0] == "vertical"


def test_composition_skips_equivalent_development_winner_before_a_lock_exists(monkeypatch):
    model = (descent.Descent(1, 0, "vertical"), descent.Descent(2, 0, "vertical"))
    rows = {arm: [{rule: arm for rule in composition.RULE_NAMES}] for arm in ("reference", "selected")}
    stats = {"paired_count_delta": 1, "complete": 1, "reference_complete": 1,
             "major_losses": 0, "wrong_source": 0, "safety_stops": 0}
    monkeypatch.setattr(composition.placement_search, "compare", lambda *args: stats)

    assert composition.equivalent_confirmation(model[::-1], [("source", model, rows)]) == "source"
    assert composition.equivalent_confirmation(model[:1], [("source", model, rows)]) is None
    stats["major_losses"] = 1
    assert composition.equivalent_confirmation(model, [("source", model, rows)]) is None


def test_online_servo_residual_has_deadband_direction_and_norm_bound():
    anchor = np.array([.3, -.1])
    np.testing.assert_array_equal(servo.horizontal_residual(anchor, anchor + .0004, .85), np.zeros(2))
    np.testing.assert_array_equal(servo.horizontal_residual(anchor, np.array([.3, np.nan]), .85), np.zeros(2))
    update = servo.horizontal_residual(anchor, anchor + np.array([.020, -.010]), .85)
    assert update[0] < 0 < update[1]
    np.testing.assert_allclose(np.linalg.norm(update), servo.MAX_RESIDUAL_M)


def test_clearance_preserves_gripper_and_retreat_endpoint(monkeypatch):
    class Solver:
        ranges = np.array([[-1., 1.]] * 7)

    monkeypatch.setattr(clearance.release_residual, "nominal_kinematics", Solver)
    monkeypatch.setattr(clearance, "shift_command", lambda solver, joints, height, anchor, weight: joints + .01)
    phases = [np.zeros((5, 16)) for _ in range(RETRACT + 1)]
    phases[OPEN][:, 7] = np.linspace(.3, .5, 5)
    previous = np.asarray(phases).copy()

    clearance.adjust(phases, np.zeros(7), "clearance-4mm", .76)

    np.testing.assert_array_equal(np.asarray(phases)[:, :, 7:], previous[:, :, 7:])
    np.testing.assert_array_equal(phases[RETRACT][-1], previous[RETRACT, -1])
    np.testing.assert_allclose(phases[OPEN][:, :7], .01)


def test_calibrated_orientation_rotation_is_world_signed_and_bounded():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_allclose(clearance.orientation_error(np.eye(3), rotation), [0., 0., clearance.MAX_ROTATION_RAD])
