from collections import Counter

import numpy as np
import pytest

from a3opt import placement_first_hover as first, placement_hover_agent as hover
from a3opt import placement_hover_guard as guard
from a3opt import placement_hover_guard_refine as refine
from a3opt.backends.adaptive_phase import PLACE


def test_hover_seed_splits_preserve_focused_development_and_balanced_confirmation():
    def choose_column(suite, seed):
        assert suite == "random"
        return (seed + seed // 4) % 4 + 1

    splits = hover.schedule(140000, choose_column)
    assert sum(map(len, splits.values())) == len(hover.scheduled_seeds(splits))
    assert {choose_column("random", seed) for seed in splits["scout"] + splits["development"]} == {2}
    assert Counter(choose_column("random", seed) for seed in splits["confirmation"]) == {1: 16, 2: 16, 3: 16, 4: 16}
    with pytest.raises(ValueError, match="boundary"):
        hover.schedule(140001, choose_column)


def test_hover_changes_only_second_batch_descent_for_active_column(monkeypatch):
    policy = hover.HoverPolicy.__new__(hover.HoverPolicy)
    policy.hover = hover.Hover("rise-20mm")
    changes = []
    monkeypatch.setattr(hover, "task_column", int)
    baseline_action = np.zeros(16)
    monkeypatch.setattr(hover.placement_descent_agent.DescentPolicy, "act", lambda self, observation, task: baseline_action)
    monkeypatch.setattr(policy, "shift_placement", lambda batch, offset: changes.append((batch, offset.tolist())))

    policy.actions = None
    assert not hasattr(policy, "cursor")
    assert policy.act({}, "2") is baseline_action

    policy.actions = []
    for cursor, task in ((None, "2"), ((0, PLACE, 0), "2"), ((1, PLACE, 3), "2"), ((1, PLACE, 0), "3"), ((1, PLACE, 0), "2")):
        policy.cursor = cursor
        policy.act({}, task)
    assert changes == [(1, [0., 0., .020])]


def test_hover_selects_tilt_losses_after_transfer_and_preserves_scored_controls():
    def scene(seed, score, transferred, tilt, complete=False):
        landed = [{"scored": index < score - 5, "abs_vertical_axis": tilt} for index in range(transferred)]
        return {"strict-500": {"seed": seed, "column": 2, "score": score, "requested_score": score, "success": complete,
                               "safety_reason": None, "phase_geometry": [{}, {
                                   "transfer": {"batches": [[], [{"abs_vertical_axis": 1.}] * transferred]},
                                   "descent": {"batches": [[], landed]}}], "physical_terminal": {"batches": [[], landed]}}}

    plan = hover.plan_cases([scene(10, 5, 0, 1.), scene(11, 7, 5, .85), scene(12, 7, 5, .99),
                             scene(13, 10, 5, 1.), scene(14, 10, 5, 1., complete=True), scene(15, 6, 5, .89)])
    assert plan == {"failures": [15, 11], "controls": [14, 13]}

    def outcome(score):
        return {"counted_ids": np.arange(score).tolist(), "success": score == 10, "safety_reason": None, "wrong_source": 0}

    dataset = [{"seed": seed, "trials": {"reference": outcome(score), **{name: outcome(score) for name in hover.OFFSETS}}}
               for seed, score in ((15, 6), (11, 7), (14, 10), (13, 10))]
    for row in dataset[:2]:
        row["trials"].update({"rise-12mm": outcome(10), "rise-20mm": outcome(9)})
    dataset[2]["trials"]["rise-12mm"] = outcome(9)
    fitted = hover.fit(dataset, plan["controls"])
    assert fitted["selected"] == "rise-20mm"
    assert not fitted["candidates"]["rise-12mm"]["eligible"]


def test_first_hover_preserves_the_existing_descent_and_runs_it_once(monkeypatch):
    policy = first.FirstHoverPolicy.__new__(first.FirstHoverPolicy)
    policy.first_hover = {1: "rise20"}
    policy.profiles = {(1, 0): "vertical"}
    policy.actions = [[], []]
    policy.cursor = (0, PLACE, 0)
    changes = []
    monkeypatch.setattr(first, "task_column", int)
    monkeypatch.setattr(first.descent, "adjust", lambda phases, measured, profile: changes.append(("descent", profile)))
    monkeypatch.setattr(first.descent.DescentPolicy, "act", lambda self, observation, task: changes.append(("reference", task)))
    monkeypatch.setattr(first.PlacementPolicy, "act", lambda self, observation, task: changes.append(("act", task)))
    monkeypatch.setattr(policy, "shift_placement", lambda batch, offset: changes.append((batch, offset.tolist())))

    policy.act({"observation.eef_pose": np.zeros(7)}, "1")
    policy.act({}, "2")
    policy.cursor = (1, PLACE, 0)
    policy.act({}, "1")
    assert changes == [("descent", "vertical"), (0, [0., 0., .020]), ("act", "1"), ("reference", "2"), ("reference", "1")]


def test_hover_gate_only_changes_the_first_descent_above_the_fitted_cutoff(monkeypatch):
    policy = guard.GatedHoverPolicy.__new__(guard.GatedHoverPolicy)
    policy.gate = guard.HoverGate(4, "rise20-x-minus", "transfer_x", .043)
    policy.actions = [[], []]
    policy.pose = np.zeros(5)
    monkeypatch.setattr(guard, "task_column", int)
    monkeypatch.setattr(first.FirstHoverPolicy, "act", lambda self, observation, task: "hover")
    monkeypatch.setattr(first.descent.DescentPolicy, "act", lambda self, observation, task: "reference")
    policy.cursor = (0, PLACE, 0)
    monkeypatch.setattr(guard, "planned_x", lambda pose, actions, stage: .042)
    assert policy.act({}, "4") == "reference"
    monkeypatch.setattr(guard, "planned_x", lambda pose, actions, stage: .043)
    assert policy.act({}, "4") == "hover"


def test_refined_hover_gate_rejects_a_transfer_only_false_positive(monkeypatch):
    policy = refine.PersistentProjectionPolicy.__new__(refine.PersistentProjectionPolicy)
    policy.projection = refine.PersistentProjection(4, "rise20-x-minus", .043)
    policy.actions = [[], []]
    policy.pose = np.zeros(5)
    policy.cursor = (0, PLACE, 0)
    monkeypatch.setattr(refine, "task_column", int)
    monkeypatch.setattr(first.FirstHoverPolicy, "act", lambda self, observation, task: "hover")
    monkeypatch.setattr(first.descent.DescentPolicy, "act", lambda self, observation, task: "reference")

    monkeypatch.setattr(guard, "planned_x", lambda pose, actions, stage: .042 if stage == PLACE else .047)
    assert policy.act({}, "4") == "reference"
    monkeypatch.setattr(guard, "planned_x", lambda pose, actions, stage: .043)
    assert policy.act({}, "4") == "hover"
