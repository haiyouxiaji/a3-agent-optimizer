import numpy as np
import pytest

from a3opt import placement_first_hover as first, placement_hover_guard as guard, placement_routed_agent as routed
from a3opt.backends.adaptive_phase import PLACE


RECORDS = [[2, 1, "rise-20mm-in"], [4, 0, "rise20-x-minus", .043]]


def test_routed_model_requires_disjoint_columns_and_bound_stage():
    assert [item.record() for item in routed.decode(RECORDS)] == RECORDS
    for invalid in ([[2, 1, "rise-20mm-in"], [2, 0, "rise20-x-minus", .043]],
                    [[2, 0, "rise-20mm-in"], RECORDS[1]], RECORDS[::-1], RECORDS[:1]):
        with pytest.raises(ValueError):
            routed.decode(invalid)


def test_routed_policy_activates_each_residual_only_in_its_column_and_batch(monkeypatch):
    controller = routed.RoutedPlacementPolicy.__new__(routed.RoutedPlacementPolicy)
    controller.late_hover, controller.projection = routed.decode(RECORDS)
    controller.actions = [[], []]
    controller.pose = np.zeros(5)
    controller.first_hover = {4: controller.projection.recipe}
    controller.profiles = {}
    calls = []
    for module in (routed, routed.projection, first):
        monkeypatch.setattr(module, "task_column", int)
    monkeypatch.setattr(guard, "planned_x", lambda pose, actions, stage: .045)
    monkeypatch.setattr(first.descent.DescentPolicy, "act", lambda self, obs, task: "reference")
    monkeypatch.setattr(first.PlacementPolicy, "act", lambda self, obs, task: "adjusted")
    monkeypatch.setattr(controller, "shift_placement", lambda batch, shift: calls.append((batch, tuple(shift))))

    controller.cursor = (0, PLACE, 0)
    assert controller.act({}, "2") == "reference"
    assert controller.act({}, "4") == "adjusted"
    controller.cursor = (1, PLACE, 0)
    assert controller.act({}, "4") == "reference"
    assert controller.act({}, "3") == "reference"
    controller.act({}, "2")
    controller.cursor = (1, PLACE, 1)
    controller.act({}, "2")
    assert calls == [(0, (-.004, 0., .020)), (1, (-.005, 0., .020))]


def test_routing_fit_preserves_the_hard_gates_on_each_endpoint():
    good = {"paired_count_delta": .25, "complete": 7, "reference_complete": 7, "major_losses": 0,
            "safety_stops": 0, "wrong_source": 0}
    names = ("late", "early")
    proposals = dict(zip(names, routed.decode(RECORDS), strict=True))
    panels = {name: [{"metrics": {"default": good, "strict-500": good}}] for name in names}
    assert routed.fit(panels, proposals)[0] == RECORDS
    panels["early"][0]["metrics"]["strict-500"] = {**good, "complete": 6}
    frozen, evidence = routed.fit(panels, proposals)
    assert not frozen and evidence["late"]["kept"] and not evidence["early"]["kept"]
