import pytest

from a3opt import placement_release_router as router


def terminal(count):
    return {"counted_ids": list(range(count)), "success": count == 10, "safety_reason": None, "wrong_source": 0}


def recorded_row(reading, original, recovery, stationary):
    return {"readings": {"transfer_x": reading, "descent_x": 0.},
            "recovery": {"trials": {"features": [0.] * 4,
                                    "outcomes": {"reference": terminal(original), "raise-in-place": terminal(recovery)}}},
            "hold": {"trials": {"planned-18": terminal(recovery), "anchored-24": terminal(recovery),
                                "outward-24": terminal(stationary)}}}


def test_routing_improves_held_out_cases_without_erasing_a_later_recovery(monkeypatch):
    fallback = router.recovery.Recovery("raise-in-place")
    support_sizes = []

    def select_fallback(panels):
        support_sizes.append(len(panels))
        return fallback

    monkeypatch.setattr(router.recovery, "select_gate", select_fallback)
    panels = {str(index): [recorded_row(.010, 3, 4, 8), recorded_row(.002, 4, 10, 9)] for index in range(4)}
    selected, report = router.fit(panels)
    actual, = selected
    assert support_sizes == [3, 3, 3, 3, 4]
    assert actual.gate.recipe == "outward-24"
    assert actual.gate.activates({"transfer_x": .010})
    assert not actual.gate.activates({"transfer_x": .002})
    assert not actual.gate.activates({"transfer_x": float("inf")})
    assert router.decode(actual.record()) == actual
    assert report["fold_gain"] == 16
    assert report["always_hold_gain"]["outward-24"] == 12


def test_join_does_not_use_counterfactuals_from_different_reference_physics():
    recovery = {"seed": 5, "continuation": {"physics_sha256": "one", "command_sha256": "late"}}
    hold = {"seed": 5, "continuation": {"physics_sha256": "two", "command_sha256": "early"}}
    with pytest.raises(ValueError, match="counterfactual reference endpoints differ"):
        router.build_panels({"panel": [recovery]}, {5: hold}, {}, None)
