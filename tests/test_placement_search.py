import numpy as np
import pytest

from a3opt import placement_regrasp_agent, placement_sim, placement_visual_feedback, placement_visual_geometry
from a3opt.backends.adaptive_phase import CLOSE, LIFT, OPEN, PLACE, RETRACT, TRANSFER
from a3opt.placement_contact import ContactMonitor, ContactPolicy
from a3opt.placement_phase_agent import PhaseTiming, PhaseTimingPolicy, adjust, fit as fit_phase_timing
from a3opt.placement_phase_compose import screen
from a3opt.placement_search import best_option, branch_stats, compare
from a3opt.placement_tail import OFFSETS, fit_column


def test_placement_residual_preserves_transfer_and_rejoins_retract(monkeypatch):
    class Kinematics:
        def shift(self, command, displacement):
            return command + np.pad(displacement, (0, 4))

    monkeypatch.setattr(placement_sim, "nominal_kinematics", Kinematics)
    policy = object.__new__(placement_sim.PlacementPolicy)
    policy.pose = np.zeros(5)
    policy.actions = [[np.ones((4, 16)) for _ in range(RETRACT + 1)] for _ in range(2)]
    policy.shift_placement(1, np.array([-.003, .0, .008]))
    assert (policy.actions[0][PLACE] == 1).all()
    assert (policy.actions[1][TRANSFER] == 1).all()
    assert (policy.actions[1][OPEN][:, 7:] == 1).all()
    assert policy.actions[1][PLACE][-1, :3] == pytest.approx([.997, 1., 1.008])
    assert policy.actions[1][OPEN][-1, :3] == pytest.approx([.997, 1., 1.008])
    assert (policy.actions[1][RETRACT][-1] == 1).all()


def test_training_selection_rejects_an_option_with_a_major_loss():
    def rollout(count):
        return {"counted_ids": list(range(count)), "filled_slots": count, "safety_reason": None, "source_remaining": 70}

    branches = [{"outcomes": {"reference": rollout(old), "small": rollout(low), "large": rollout(high)}}
                for old, low, high in [(5, 7, 10), (10, 10, 6), (5, 5, 10)]]
    stats = {option: branch_stats(branches, option) for option in ("reference", "small", "large")}
    assert stats["large"]["count_delta"] == 6
    assert stats["large"]["major_losses"] == 1
    assert best_option(stats) == "small"
    assert best_option(stats, minimum_gains=2) == "reference"


def test_placement_activation_reads_projected_tool_position(monkeypatch):
    policy = object.__new__(placement_sim.PlacementPolicy)
    policy.pose = np.array([0., 0., .05, .1, np.pi / 2])
    policy.actions = [[], []]
    policy.cursor = (0, PLACE, 0)
    item = placement_sim.Correction(1, 0, (-.003, 0., .008), .03)
    policy.corrections = {(1, 0): item}
    applied = []
    policy.shift_placement = lambda batch, offset: applied.append((batch, offset.tolist()))
    monkeypatch.setattr(placement_sim.ReleaseTimingPolicy, "act", lambda self, observation, task: np.zeros(16))
    policy.act({"observation.eef_pose": [.05, .12, 0.]}, "transfer 10 cookies into target box")
    assert applied == []
    policy.act({"observation.eef_pose": [.05, .14, 0.]}, "transfer 10 cookies into target box")
    assert applied == [(0, [-.003, 0., .008])]


def test_paired_evaluation_rejects_a_mismatched_transfer_prefix():
    matching = {"seed": 12, "column": 1, "profile": "advanced", "suite": "random", "strict": True,
                "initial_pose": [], "estimated_pose": [], "cameras": {}, "light_diffuse": [], "color": ""}
    reference = {**matching, "phase_geometry": [{"transfer": {"prefix_sha256": ["old"]}}, {}]}
    selected = {**matching, "phase_geometry": [{"transfer": {"prefix_sha256": ["new"]}}, {}]}
    model = (placement_sim.Correction(1, 0, (0., 0., .008)),)
    with pytest.raises(ValueError, match="shared physical prefix"):
        compare([reference], [selected], model)


def test_tail_fit_scores_the_same_second_action_on_the_actual_first_batch_state():
    def rollout(count, alternate, first_corrected=False):
        def terminal(result):
            return {"counted_ids": list(range(result)), "source_remaining": 70, "safety_reason": None}

        return {"score": count, "wrong_source": 0, "continuation": terminal(count),
                "fork": {"tool_x_m": .03, "first_corrected": first_corrected,
                         "outcomes": {name: {"physics": terminal(alternate if name == "out3" else count),
                                              "wrong_source": 0} for name in OFFSETS}}}

    pairs = [{"reference": rollout(5, 0), "anchor": rollout(5, 10, True)},
             {"reference": rollout(6, 1), "anchor": rollout(6, 10, True)},
             {"reference": rollout(10, 10), "anchor": None}]
    first = placement_sim.Correction(2, 0, (-.003, 0., .008), .02)
    chosen, report = fit_column(2, pairs, first)
    assert chosen == (first, placement_sim.Correction(2, 1, OFFSETS["out3"]))
    assert report["selected"] == [item.record() for item in chosen]


def test_contact_policy_handles_initial_frame_and_requires_sustained_late_force(monkeypatch):
    policy = object.__new__(ContactPolicy)
    policy.actions = None
    monkeypatch.setattr(placement_sim.PlacementPolicy, "act", lambda self, observation, task: np.zeros(16))
    assert not policy.act({}).any()

    monitor = ContactMonitor()
    for frame, load in ((0, 0.), (1, 10.), (2, 10.), (3, 0.), (4, 0.), (5, 0.), (6, 10.)):
        assert monitor.update({"observation.force": [load, 0., 0.]}, frame, 8) == 0.
    assert monitor.update({"observation.force": [6., 0., 0.]}, 7, 8) == 6.
def test_phase_timing_delegates_first_action_without_a_cursor(monkeypatch):
    policy = object.__new__(PhaseTimingPolicy)
    policy.actions = None
    expected = np.arange(16)
    monkeypatch.setattr(placement_sim.PlacementPolicy, "act", lambda self, observation, task: expected)
    assert policy.act({}) is expected


def test_phase_timing_preserves_waypoint_bounds_and_budget():
    from types import SimpleNamespace

    descent = np.tile(np.linspace(0.1, 0.2, 12)[:, None], (1, 16))
    opening = np.tile(np.linspace(0.2, 0.5, 10)[:, None], (1, 16))
    phases = [np.zeros((1, 16)) for _ in range(OPEN + 1)]
    phases[PLACE], phases[OPEN] = descent.copy(), opening.copy()
    policy = SimpleNamespace(actions=[phases], step=478)
    adjust(policy, 0, "soft-landing")
    assert len(phases[PLACE]) == len(descent)
    np.testing.assert_array_equal(phases[PLACE][[0, -1]], descent[[0, -1]])
    assert phases[PLACE][5, 0] > descent[5, 0]
    adjust(policy, 0, "settle-before-open")
    assert len(phases[PLACE]) == len(descent)
    policy.step -= 4
    adjust(policy, 0, "settle-before-open")
    np.testing.assert_array_equal(phases[PLACE][-4:], np.repeat(descent[-1:], 4, axis=0))
    adjust(policy, 0, "early-unload")
    assert phases[PLACE][-1, 7] > descent[-1, 7]
    np.testing.assert_array_equal(phases[OPEN][:, :7], opening[:, :7])
    assert phases[OPEN][-1, 7] == opening[-1, 7]


def test_phase_timing_requires_repeated_gains_and_a_preserved_control():
    def terminal(count):
        return {"counted_ids": list(range(count)), "success": count == 10, "wrong_source": 0, "safety_reason": None}

    def sample(original, alternate):
        control = terminal(original)
        return {"batch": 1, "column": 2, "control": control,
                "trials": {"reference": control, "soft-landing": terminal(alternate),
                           "early-unload": control, "settle-before-open": control}}

    failure = sample(5, 10)
    assert fit_phase_timing([failure, sample(10, 10)])[0] == ()
    assert fit_phase_timing([failure, sample(6, 9), sample(10, 5)])[0] == ()
    assert fit_phase_timing([failure, sample(6, 9), sample(10, 10)])[0] == (PhaseTiming(2, 1, "soft-landing"),)


def placement_terminal(count, wrong_source=0):
    return {"counted_ids": list(range(count)), "success": count == 10, "wrong_source": wrong_source, "safety_reason": None}


def test_composition_counts_distinct_training_scenes():
    control = placement_terminal(8)
    trials = {"soft-landing": placement_terminal(9), "early-unload": control, "settle-before-open": control}
    first = {"seed": 99, "column": 2, "batch": 0, "trials": trials, "control": control}
    second = {**first, "batch": 1}
    assert not any(entry["admitted"] for entry in screen([first, second]))
    scores = [entry for entry in screen([first, {**second, "seed": 100}]) if entry["admitted"]]
    assert [(entry["recipe"], entry["gains"]) for entry in scores] == [("soft-landing", [99, 100])]


def test_regrasp_does_not_replace_healthy_first_or_second_grasps(monkeypatch):
    policy_type = placement_regrasp_agent.RegraspPolicy
    policy = object.__new__(policy_type)
    policy.actions = [[np.zeros((1, 16)) for _ in range(OPEN + 1)] for _ in range(2)]
    policy.regrasp = {2: placement_regrasp_agent.Regrasp(2, .14, "nudge-left")}
    calls = []
    monkeypatch.setattr(placement_sim.PlacementPolicy, "act", lambda self, observation, task: None)
    monkeypatch.setattr(placement_regrasp_agent, "insert_retry", lambda policy, obs, recipe: calls.append(recipe))
    monkeypatch.setattr(placement_regrasp_agent, "grip_features", lambda obs, command: [0., obs["gap"]])
    task = "Transfer 10 cookies from source column 2 into the target box."
    policy.cursor = (0, LIFT, 0)
    policy.act({"gap": .22}, task)
    policy.cursor = (1, LIFT, 0)
    policy.act({"gap": .03}, task)
    assert not calls
    policy.act({"gap": .22}, task)
    assert calls == ["nudge-left"]


def test_regrasp_respects_remaining_steps_and_preserves_the_other_arm(monkeypatch):
    from types import SimpleNamespace

    class Kinematics:
        def shift(self, command, displacement):
            return command + np.pad(displacement, (0, 4))

    monkeypatch.setattr(placement_regrasp_agent, "nominal_kinematics", Kinematics)
    phases = [np.full((10, 16), .3) for _ in range(OPEN + 1)]
    policy = SimpleNamespace(actions=[[], phases], step=470)
    measured = {"observation.state": np.full(16, .5)}
    placement_regrasp_agent.insert_retry(policy, measured, "nudge-left")
    assert phases[LIFT].shape == (10, 16)
    policy.step = 400
    placement_regrasp_agent.insert_retry(policy, measured, "nudge-left")
    assert phases[LIFT].shape == (22, 16)
    np.testing.assert_array_equal(phases[LIFT][:, 8:], .3)
    np.testing.assert_array_equal(phases[CLOSE], .3)


def test_regrasp_selects_only_repeated_safe_healthy_separated_gains():
    original = placement_terminal(5)
    trial = placement_terminal(7)
    healthy = {"seed": 9, "column": 2, "gap": .027, "lifted": 5, "score": 10, "trials": None, "original": None}
    defaults = {recipe: original for recipe in placement_regrasp_agent.RECIPES}
    blocked = {"seed": 10, "column": 2, "gap": .24, "lifted": 0, "score": 5, "original": original,
               "trials": {**defaults, "nudge-right": trial}}
    assert not placement_regrasp_agent.fit([healthy, blocked], 2)[0]
    assert placement_regrasp_agent.fit([healthy, blocked, {**blocked, "seed": 11}], 2)[0] == (
        placement_regrasp_agent.Regrasp(2, .18, "nudge-right"),)
    bad = {**blocked, "seed": 11, "trials": {**defaults, "nudge-right": placement_terminal(7, wrong_source=1)}}
    assert not placement_regrasp_agent.fit([healthy, blocked, bad], 2)[0]


def test_visual_gate_declines_overlapping_observations():
    rows = [{"chroma_change": .16, "post_first_count": 1}]
    rows += [{"chroma_change": value, "post_first_count": 5} for value in (.12, .14, .18, .20)]
    assert placement_visual_feedback.threshold(rows) is None
    rows[0]["chroma_change"] = .08
    assert placement_visual_feedback.threshold(rows) == pytest.approx(.10)


def test_visual_offset_requires_crossfit_gains_in_different_columns():
    measurements = ((.02, 0), (.03, 1), (.04, 0), (.12, 5), (.14, 5), (.16, 5), (.18, 5), (.20, 5))
    rows, reference = [], {}
    for seed, (chroma, count) in enumerate(measurements):
        rows.append({"seed": seed, "column": 3 if seed == 1 else 2, "chroma_change": chroma, "post_first_count": count})
        baseline = {**placement_terminal(count), "source_remaining": 70, "filled_slots": count}
        options = {recipe: {"physics": baseline, "wrong_source": 0} for recipe in OFFSETS}
        if seed < 2:
            options["out3"] = {"physics": {**baseline, "counted_ids": list(range(count + 4))}, "wrong_source": 0}
        reference[seed] = {"fork": {"outcomes": options}}

    model, report = placement_visual_feedback.fit(rows, reference)
    assert report["recipe"] == "out3"
    assert len(model) == 4
    rows[1]["column"] = 2
    assert not placement_visual_feedback.fit(rows, reference)[0]


def test_visual_geometry_keeps_spatial_information_at_equal_chroma():
    frames = np.zeros((3, placement_visual_geometry.WIDTH, placement_visual_geometry.WIDTH))
    frames[1, :24, :24] = .8
    frames[2, 24:, 24:] = .8
    first = placement_visual_geometry.describe(frames[0], frames[1])
    second = placement_visual_geometry.describe(frames[0], frames[2])
    assert first.mean() == pytest.approx(second.mean())
    assert not np.array_equal(first, second)
    rows = [{"features": pattern, "label": label} for pattern, label in ((first, 1), (second, 0))] * 2
    fitted = placement_visual_geometry.regressor(rows)
    assert placement_visual_geometry.prediction(first, fitted) > .5
    assert placement_visual_geometry.prediction(second, fitted) < .5
    assert placement_visual_geometry.regressor(rows[:3]) is None
