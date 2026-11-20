import numpy as np
import pytest

from a3opt.adaptive_agent import seed_schedule
from a3opt.adaptive_audit import _matches
from a3opt.adaptive_fit import fit_grip, select_columns
from a3opt.adaptive_grid import summarize_cell
from a3opt.backends.adaptive_phase import (
    CLOSE, FEATURES, LIFT, MAX_RETRY_STEPS, OPEN, PLACE, SHORT_PROMPT, AdaptivePhasePolicy, Recipe, task_column,
)
from a3opt.backends.advanced_phase import SAMPLES, AdvancedPhasePolicy
from a3opt.backends.release_residual import MAX_JOINT_CORRECTION, MAX_OFFSET, ReleaseResidualPolicy, nominal_kinematics
from a3opt.backends.second_batch import ReleaseRecipe, SecondBatchPolicy
from a3opt.backends.slot_alignment import FEATURE_COUNT, SlotAlignmentPolicy, predict_shift
from a3opt.branch_agent import dominant_branch
from a3opt.release_agent import _fit
from a3opt.second_batch_agent import seed_schedule as second_batch_schedule
from a3opt.second_batch_fit import fit_release, improvement, paired_comparison
from a3opt.slot_alignment_fit import fit_alignment
from a3opt.slot_guard_agent import learn_bounds


def model_fixture() -> dict:
    rng = np.random.default_rng(17)
    return {"format": AdvancedPhasePolicy.name, "columns": np.ones(10), "batches": np.repeat([0, 1], 5),
            "poses": rng.normal(0, .012, (10, 5)), "traces": rng.normal(.4, .5, (10, sum(SAMPLES), 16)),
            "lengths": rng.integers(10, 31, size=(10, len(SAMPLES))),
            "phase_penalties": np.full((2, len(SAMPLES)), .006)}


def test_neutral_recipe_exactly_reproduces_original_actions() -> None:
    model = model_fixture()
    baseline = AdvancedPhasePolicy.__new__(AdvancedPhasePolicy)
    baseline.model = model
    adaptive = AdaptivePhasePolicy(model, (Recipe(),) * 4)
    adaptive.pose = np.zeros(5)
    phases = adaptive._phase_actions(1, Recipe())
    planned = np.concatenate([phase for batch in phases for phase in batch])
    np.testing.assert_array_equal(planned, baseline._action_plan(1, adaptive.pose))


def test_short_prompt_and_task_type() -> None:
    assert task_column(SHORT_PROMPT) == 1
    assert task_column("Transfer 10 cookies from source column 3 into the target box in two batches of five.") == 3
    with pytest.raises(ValueError):
        task_column("Transfer 20 cookies from source column 3 into the target box.")


def test_second_batch_release_preserves_first_batch_and_transfer(monkeypatch) -> None:
    class Kinematics:
        def shift(self, joints, offset):
            return joints + np.pad(offset, (0, 4))

    monkeypatch.setattr("a3opt.backends.second_batch.nominal_kinematics", Kinematics)
    reference = AdaptivePhasePolicy(model_fixture(), (Recipe(joint_placement=True),) * 4)
    selected = SecondBatchPolicy(reference, 1, ReleaseRecipe((.002, -.001, .006), open_hold=6))
    untouched = SecondBatchPolicy(reference, 2, selected.release)
    for policy in (reference, selected, untouched):
        policy.pose = np.array([.001, -.001, .002, -.002, .03])
    baseline = reference._phase_actions(1, reference.recipes[0])
    candidate = selected._phase_actions(1, selected.recipes[0])
    inactive = untouched._phase_actions(1, untouched.recipes[0])
    for original, altered in zip(baseline[0] + baseline[1][:PLACE], candidate[0] + candidate[1][:PLACE], strict=True):
        np.testing.assert_array_equal(altered, original)
    for original, altered in zip(baseline[0] + baseline[1], inactive[0] + inactive[1], strict=True):
        np.testing.assert_array_equal(altered, original)
    assert not np.array_equal(candidate[1][PLACE], baseline[1][PLACE])
    assert len(candidate[1][OPEN]) == len(baseline[1][OPEN]) + 6
    np.testing.assert_array_equal(candidate[1][OPEN][-6:], np.repeat(candidate[1][OPEN][-7:-6], 6, axis=0))
    np.testing.assert_array_equal(candidate[1][-1][-1], baseline[1][-1][-1])


def second_batch_row(seed: int, *, success: bool = False, contact: bool = True) -> dict:
    first = [{"id": index, "scored": True} for index in range(5)]
    second = [{"id": index, "scored": success, "target_xyz_m": [.018, 0., .023],
               "pad_contacts": [True, contact]} for index in range(5, 10)]
    events = {"before_descent": {"step": 400, "batches": [first, second]},
              "before_open": {"step": 430, "batches": [first, second]},
              "terminal": {"step": 500, "batches": [first, second]}}
    score = 10 if success else 5
    return {"seed": seed, "column": 1, "suite": "1", "profile": "advanced", "prefix_steps": 400,
            "prefix_sha256": "prefix", "action_sha256": "action", "score": score, "requested_score": score,
            "safety_reason": None, "success": success, "initial_pose": [], "estimated_pose": [], "cameras": {},
            "light_diffuse": [], "color": "gold", "batches": [{}, {"retained_through_transfer": 5}],
            "placement_events": events, "second_batch_goal_m": [.028, 0., .020]}


def test_second_batch_fit_requires_two_sided_contact_and_two_labels() -> None:
    rows = [second_batch_row(72000), second_batch_row(72001, contact=False)]
    proposals, dataset = fit_release(rows)
    assert list(proposals) == ["open-hold", "clearance"]
    assert len(dataset["samples"]) == len(dataset["rejected"]) == 1
    proposals, dataset = fit_release([*rows, second_batch_row(72002)])
    assert proposals["alignment"].offset_m == pytest.approx((.010, 0., -.003))
    assert {sample["seed"] for sample in dataset["samples"]} == {72000, 72002}


def test_second_batch_pair_checks_prefix_and_safety() -> None:
    rows = [second_batch_row(72000), second_batch_row(72001)]
    selected = [second_batch_row(72000, success=True), second_batch_row(72001)]
    assert improvement(paired_comparison(rows, selected))
    assert not improvement(paired_comparison(rows, [selected[0], {**selected[1], "safety_reason": "contact"}]))
    with pytest.raises(ValueError, match="pre-placement actions differ"):
        paired_comparison(rows, [{**selected[0], "prefix_sha256": "different"}, selected[1]])


def test_second_batch_scout_training_development_and_confirmation_are_disjoint() -> None:
    schedule = second_batch_schedule(72000)
    groups = [*schedule["scout"].values(), schedule["training"], schedule["development"], *schedule["confirmation"].values()]
    assert len(set().union(*groups)) == sum(map(len, groups))
    with pytest.raises(ValueError, match="multiple of four"):
        second_batch_schedule(72001)


def test_slot_alignment_learns_signed_shift_from_held_out_seed_labels() -> None:
    slots = np.column_stack((np.full(5, .028), np.linspace(-.012, .012, 5)))
    records = []
    for seed, offset in enumerate(np.r_[np.linspace(-.010, -.007, 6), np.linspace(.007, .010, 6)], 73600):
        cookies = [{"id": index, "scored": True, "target_xyz_m": [slot[0], slot[1] + offset, .018]}
                   for index, slot in enumerate(slots)]
        records.append({"seed": seed, "column": 2, "score": 5, "requested_score": 5, "safety_reason": None,
                        "first_after_retract": {"batches": [cookies, []], "step": 220}, "front_slot_xy_m": slots.tolist(),
                        "first_release_features": [0., 0., 0., 0., 0., .1 + offset, .1 + offset], "action_sha256": "test"})
    model, dataset = fit_alignment(records, 2)
    assert dataset["validation"]["baseline"]["rows_within_y_tolerance"] == 0
    assert dataset["validation"]["selected"]["rows_within_y_tolerance"] == len(records)
    assert predict_shift(model, np.array(records[0]["first_release_features"])) > .005
    assert predict_shift(model, np.array(records[-1]["first_release_features"])) < -.005


def test_slot_alignment_keeps_first_transport_and_second_plan_exact(monkeypatch) -> None:
    class Kinematics:
        def shift(self, joints, offset):
            return joints + np.pad(offset, (0, 4))

    monkeypatch.setattr("a3opt.backends.slot_alignment.nominal_kinematics", Kinematics)
    monkeypatch.setattr("a3opt.backends.slot_alignment.release_features", lambda *_: np.zeros(FEATURE_COUNT))
    original = AdaptivePhasePolicy(model_fixture(), (Recipe(),) * 4)
    aligned = SlotAlignmentPolicy(original, {"center": np.zeros(FEATURE_COUNT), "scale": np.ones(FEATURE_COUNT),
                                            "weights": np.zeros(FEATURE_COUNT), "column": 1, "bias": .006})
    original.pose = aligned.pose = np.zeros(5)
    baseline = original._phase_actions(1, original.recipes[0])
    candidate = aligned._phase_actions(1, aligned.recipes[0])
    for before, after in zip(baseline[0][:PLACE] + baseline[1], candidate[0][:PLACE] + candidate[1], strict=True):
        np.testing.assert_array_equal(before, after)
    assert not np.array_equal(baseline[0][OPEN], candidate[0][OPEN])
    np.testing.assert_array_equal(baseline[0][-1][-1], candidate[0][-1][-1])


def test_slot_guard_excludes_score_loss_and_safety_violations() -> None:
    samples = [{"seed": 74800, "magnitude_m": .0032, "score_delta": 0, "success_delta": 0, "violation": True},
               {"seed": 74801, "magnitude_m": .0055, "score_delta": 0, "success_delta": 1, "violation": False},
               {"seed": 74802, "magnitude_m": .0072, "score_delta": 0, "success_delta": 1, "violation": False},
               {"seed": 74803, "magnitude_m": .0101, "score_delta": -2, "success_delta": 0, "violation": False}]
    choice = learn_bounds(samples)
    assert choice["training_seeds_active"] == [74801, 74802]
    assert choice["bounds_m"] == pytest.approx([.004, .010])
    assert learn_bounds(samples[:-2]) is None


def test_grip_retry_cost_is_limited_and_skipped_near_deadline() -> None:
    model = model_fixture()
    recipe = Recipe(retry_probability=.55, close_bias=.03)
    detector = {"intercept": -10., "center": np.zeros(FEATURES), "scale": np.ones(FEATURES),
                "coefficients": np.zeros(FEATURES)}
    policy = AdaptivePhasePolicy(model, (recipe,) * 4, detector)
    policy.pose = np.zeros(5)
    policy.recipe = recipe
    policy.actions = policy._phase_actions(1, recipe)
    close = policy.actions[0][CLOSE][-1, 7]
    policy.cursor = (0, LIFT, 0)
    observation = {"observation.state": np.zeros(16), "observation.velocity": np.zeros(16),
                   "observation.force": np.zeros(18)}
    for _ in range(MAX_RETRY_STEPS):
        action = policy.act(observation)
        assert policy.stage == CLOSE
    assert action[7] == pytest.approx(max(close - .01, 0.))
    policy.act(observation)
    assert policy.stage == LIFT and policy.retries == [1, 0]
    policy.retries = [0, 0]
    policy.step = 499
    policy.cursor = (0, LIFT, 0)
    policy.act(observation)
    assert policy.retries == [0, 0]


def test_grip_gate_uses_grouped_training_and_handles_single_class() -> None:
    samples = [{"seed": 52000 + scene, "features": [float(scene % 2)] * FEATURES,
                "complete_grasp": bool(scene % 2)} for scene in range(16) for _ in range(2)]
    _, validation = fit_grip(samples)
    assert validation["scenes"] == 16 and validation["samples"] == 32
    assert validation["selected_gate"]["false_alarms"] == 0
    _, validation = fit_grip([sample for sample in samples if sample["complete_grasp"]])
    assert validation["retry_probability"] == 0


def test_split_seeds_never_overlap() -> None:
    schedule = seed_schedule(52000)
    splits = [*map(set, (schedule[name] for name in ("scout", "focus", "focus_extra"))),
              *map(set, schedule["development"].values()), *map(set, schedule["confirmation"].values())]
    assert sum(map(len, splits)) == len(set.union(*splits))


def test_promotion_requires_no_new_safety_or_wrong_column_credits() -> None:
    baseline = [{"suite": "random", "seed": 52100 + column, "column": column, "score": 5, "requested_score": 5,
                 "success": False, "safety_reason": None} for column in range(1, 5)]
    selected = [dict(row) for row in baseline]
    selected[0].update(score=9, requested_score=8)
    selected[1].update(score=9, requested_score=9, safety_reason="safety")
    selected[2].update(score=7, requested_score=7)
    assert select_columns({"baseline": baseline, "candidate": selected}) == ["baseline", "baseline", "candidate", "baseline"]


def test_release_kinematics_uses_the_clipped_pose_and_bounded_correction() -> None:
    pytest.importorskip("a3_dual_arm_sim")
    pytest.importorskip("mujoco")
    solver = nominal_kinematics()
    command = np.array([1.402, 1.425, -.606, -1.379, -1.318, .596, .162])

    def position(joints):
        solver.data.qpos[solver.positions] = joints
        solver.mujoco.mj_kinematics(solver.model, solver.data)
        return solver.data.site_xpos[solver.site].copy()

    displacement = np.array([-.008, -.010, -.014])
    shifted = solver.shift(command, displacement)
    np.testing.assert_allclose(position(shifted) - position(command), displacement, atol=.003)
    command[1] = solver.ranges[1, 1] + .5
    clipped = np.clip(command, *solver.ranges.T)
    shifted = solver.shift(command, displacement)
    np.testing.assert_array_equal(shifted, solver.shift(clipped, displacement))
    assert np.max(np.abs(shifted - clipped)) <= MAX_JOINT_CORRECTION + 1e-12


def test_release_shift_preserves_actions_outside_the_selected_column() -> None:
    model = model_fixture()
    baseline = AdaptivePhasePolicy(model, (Recipe(),) * 4)
    residual = ReleaseResidualPolicy(model, 3, np.array([0., 0., -.015]))
    baseline.pose = residual.pose = np.zeros(5)
    expected = np.concatenate([stage for batch in baseline._phase_actions(1, Recipe()) for stage in batch])
    actual = np.concatenate([stage for batch in residual._phase_actions(1, Recipe()) for stage in batch])
    np.testing.assert_array_equal(actual, expected)


def test_release_fit_uses_only_safe_labelled_training_rollouts() -> None:
    rows = [
        {"seed": seed, "applied_action_sha256": str(seed), "batches": [{"release_label": {"error": error}}, {}],
         "safety_reason": safety}
        for seed, error, safety in ((64000, [.003, .07, -.04], None), (64001, [.005, .01, -.02], None),
                                    (64002, [.002, 0., 0.], "safety"))
    ]
    rows.append({"seed": 64003, "batches": [{}, {}], "safety_reason": None})
    offset, samples = _fit(rows)
    assert [sample["seed"] for sample in samples] == [64000, 64001]
    np.testing.assert_allclose(offset, [.004, MAX_OFFSET[1], -MAX_OFFSET[2]])


def test_archived_replay_requires_all_original_fields() -> None:
    original = {"batches": [{"lifted_requested": 5}, {}], "action_sha256": "abc"}
    newer = {"batches": [{"lifted_requested": 5, "retained_through_transfer": 5}, {}], "action_sha256": "abc"}
    assert _matches(original, newer)
    assert not _matches(original, {**newer, "action_sha256": "changed"})
    assert not _matches(original, {"action_sha256": "abc"})


def test_branch_fit_keeps_other_columns_and_second_batch() -> None:
    model = model_fixture()
    model["seeds"] = np.arange(10)
    model["columns"][-1] = 4
    endpoint = sum(SAMPLES[:7]) - 1
    model["traces"][:5, endpoint, 6] = [-1.3, .02, .08, .12, .18]
    model["traces"][8, endpoint, 6] = -1.3
    filtered, diagnosis = dominant_branch(model, 1)
    assert diagnosis["removed_seeds"] == [0]
    assert [len(cluster["training_seeds"]) for cluster in diagnosis["clusters"]] == [1, 4]
    np.testing.assert_array_equal(filtered["traces"], model["traces"][1:])
    np.testing.assert_array_equal(filtered["columns"], model["columns"][1:])


def test_course_grid_reports_rejected_instructions_without_assigning_a_score() -> None:
    timeout = {"score": 0, "requested_score": 0, "success": False, "failure_reason": "max_steps_exceeded",
               "safety_reason": None}
    rejected = {"score": None, "requested_score": None, "success": None,
                "failure_reason": "unsupported_task_instruction"}
    assert summarize_cell([timeout], 20)["mean_score"] == 0
    cell = summarize_cell([timeout, rejected], 20)
    assert (cell["episodes"], cell["scored_episodes"], cell["mean_score"], cell["successes"]) == (2, 1, None, None)
    assert cell["termination"] == {"max_steps_exceeded": 1, "unsupported_task_instruction": 1}
