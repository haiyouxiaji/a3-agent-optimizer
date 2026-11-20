import numpy as np

from a3opt import placement_wall_clearance_agent as clearance
from a3opt.backends.adaptive_phase import PLACE


def test_wall_clearance_uses_opposite_insets_and_preserves_the_first_descent(monkeypatch):
    controller = clearance.WallClearancePolicy.__new__(clearance.WallClearancePolicy)
    controller.wall_clearance = clearance.WallClearance("inset4-hover12", (.030, .030))
    controller.actions = [[], []]
    controller.pose = np.zeros(5)
    controller.profiles = {(1, 0): "vertical"}
    calls = []
    monkeypatch.setattr(clearance, "task_column", int)
    monkeypatch.setattr(clearance.guard, "planned_x", lambda pose, phases, stage: .032 if phases is controller.actions[0] else -.032)
    monkeypatch.setattr(clearance.descent, "adjust", lambda phases, xyz, name: calls.append(name))
    monkeypatch.setattr(clearance.descent.DescentPolicy, "act", lambda self, obs, task: "reference")
    monkeypatch.setattr(clearance.PlacementPolicy, "act", lambda self, obs, task: "corrected")
    monkeypatch.setattr(controller, "shift_placement", lambda batch, shift: calls.append((batch, tuple(shift))))
    controller.cursor = (0, PLACE, 0)
    assert controller.act({"observation.eef_pose": np.zeros(7)}, "1") == "corrected"
    controller.cursor = (1, PLACE, 0)
    assert controller.act({}, "1") == "corrected"
    assert controller.act({}, "2") == "reference"
    assert calls == ["vertical", (0, (-.004, 0., .012)), (1, (.004, 0., .012))]
