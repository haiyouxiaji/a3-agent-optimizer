import numpy as np
import pytest

from a3opt.adaptive_agent import seed_schedule
from a3opt.adaptive_audit import _matches
from a3opt.adaptive_fit import fit_grip, select_columns
from a3opt.adaptive_grid import summarize_cell
from a3opt.backends.adaptive_phase import (
    CLOSE, FEATURES, LIFT, MAX_RETRY_STEPS, SHORT_PROMPT, AdaptivePhasePolicy, Recipe, task_column,
)
from a3opt.backends.advanced_phase import SAMPLES, AdvancedPhasePolicy
from a3opt.backends.release_residual import MAX_JOINT_CORRECTION, MAX_OFFSET, ReleaseResidualPolicy, nominal_kinematics
from a3opt.branch_agent import dominant_branch
from a3opt.release_agent import _fit


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
