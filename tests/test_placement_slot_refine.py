import numpy as np

from a3opt import placement_compose as composed, placement_hover_agent as hover
from a3opt import placement_slot_refine as row
from a3opt.backends.adaptive_phase import PLACE


def test_row_alignment_uses_bounded_first_batch_shift_only(monkeypatch):
    policy = row.RowPolicy.__new__(row.RowPolicy)
    model = row.RowAlignment((0.,) * 7, (1.,) * 7, (0.,) * 7, .05)
    policy.row_parameters = model.parameters()
    policy.pose = np.zeros(5)
    changes = []
    monkeypatch.setattr(row, "release_features", lambda pose, phases: np.zeros(7))
    monkeypatch.setattr(row, "task_column", int)
    monkeypatch.setattr(row.placement_descent_agent.DescentPolicy, "act", lambda self, observation, task: observation)
    monkeypatch.setattr(policy, "shift_placement", lambda batch, offset: changes.append((batch, offset.tolist())))

    policy.actions = None
    policy.act({}, "2")
    policy.actions = [[], []]
    for cursor, task in (((0, PLACE, 0), "3"), ((1, PLACE, 0), "2"), ((0, PLACE, 1), "2"), ((0, PLACE, 0), "2")):
        policy.cursor = cursor
        policy.act({}, task)
    assert changes == [(0, [0., .012, 0.])]


def test_row_alignment_requires_strict_gain_and_count_preservation_in_full_episodes():
    clean = {"mean": 9.2, "paired_count_delta": 0., "paired_count_ci95": [0., 0.], "complete": 9,
             "reference_complete": 7, "paired_success_rate_ci95": [0.05, 0.20],
             "major_losses": 0, "wrong_source": 0, "safety_stops": 0}
    assert row.supports_alignment({"default": clean, "strict-500": clean}, confirmation=True)
    assert not row.supports_alignment({"default": clean, "strict-500": {**clean, "complete": 7}})
    assert not row.supports_alignment({"default": {**clean, "paired_count_delta": -.1}, "strict-500": clean})
    assert not row.supports_alignment({"default": {**clean, "paired_count_ci95": [-.1, .1]},
                                       "strict-500": clean}, confirmation=True)


def test_composed_policy_applies_each_component_at_its_batch(monkeypatch):
    policy = composed.ComposedPolicy.__new__(composed.ComposedPolicy)
    alignment = row.RowAlignment((0.,) * 7, (1.,) * 7, (0.,) * 7, .006)
    policy.row_parameters = alignment.parameters()
    policy.hover = hover.Hover("rise-20mm-in")
    policy.pose = np.zeros(5)
    changes = []
    monkeypatch.setattr(row, "release_features", lambda pose, phases: np.zeros(7))
    monkeypatch.setattr(row, "task_column", int)
    monkeypatch.setattr(composed, "task_column", int)
    monkeypatch.setattr(row.placement_descent_agent.DescentPolicy, "act", lambda self, observation, task: observation)
    monkeypatch.setattr(policy, "shift_placement", lambda batch, offset: changes.append((batch, offset.tolist())))

    policy.actions = None
    policy.act({}, "2")
    policy.actions = [[], []]
    for cursor, task in (((0, PLACE, 0), "3"), ((1, PLACE, 0), "4"), ((0, PLACE, 0), "2"), ((1, PLACE, 0), "2")):
        policy.cursor = cursor
        policy.act({}, task)
    assert changes == [(0, [0., .006, 0.]), (1, [-.005, 0., .020])]
