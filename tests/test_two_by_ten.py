from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from a3opt.advanced_agent import _npz
from a3opt.backends.advanced_phase import SAMPLES
from a3opt.backends.advanced_trajectory import FRONT, POSE_SCALE
from a3opt.backends.two_by_ten import SOURCE_COLUMN, TwoByTenPolicy, active_batches, instruction
from a3opt.refinement_eval import sha256
from a3opt.two_by_ten_agent import Budget, choose_recipe, feedback_plan, initial_plan, uniform_plan
from a3opt.two_by_ten_sim import TransferTrace, benchmark, check_evaluation, source_columns


def plan_class():
    return pytest.importorskip("a3_dual_arm_sim.tasks.cookie_2x10_plan").Cookie2x10Plan


def test_parses_all_native_instructions_and_rejects_modified_count():
    plan_type = plan_class()
    for count in range(10):
        for column in range(1, 5):
            assert instruction(plan_type(count, count).prompt(column)) == (column, count)
    prompt = plan_type(3, 3).prompt(1).replace("then 7", "then 8", 1)
    with pytest.raises(ValueError, match="inconsistent"):
        instruction(prompt)


def test_policy_does_not_execute_zero_grasp_phases(tmp_path):
    plan_type = plan_class()
    path = tmp_path / "fixture.npz"
    traces = np.zeros((2, 4, sum(SAMPLES), 16))
    for batch in range(4):
        traces[:, batch, :, 0] = batch + 1
    _npz(path, format=TwoByTenPolicy.name, recipe="mean", counts=np.zeros(2), traces=traces,
         lengths=np.ones((2, 4, len(SAMPLES))), poses=np.zeros((2, len(POSE_SCALE))))
    policy = TwoByTenPolicy(path)
    assert active_batches(0) == (1, 3)
    policy.act({}, plan_type(0, 0).prompt(SOURCE_COLUMN))
    assert policy.actions[:, 0].tolist() == [2] * len(SAMPLES) + [4] * len(SAMPLES)


def test_trace_uses_executed_batch_at_expert_transitions():
    trace = TransferTrace()
    expert = SimpleNamespace(batch_index=0, phase=SimpleNamespace(name="SELECT_COOKIE"))

    def skip_zero(observation, task):
        expert.batch_index = 1
        expert.phase = SimpleNamespace(name="APPROACH")
        return np.zeros(16)

    trace.capture_expert_action(expert, skip_zero, {}, "")
    assert trace.expert_step == (1, "SELECT_COOKIE")

    def finish_batch(observation, task):
        expert.batch_index = 2
        expert.phase = SimpleNamespace(name="SELECT_COOKIE")
        return np.zeros(16)

    expert.phase = SimpleNamespace(name="VERIFY_RELEASE")
    trace.capture_expert_action(expert, finish_batch, {}, "")
    assert trace.expert_step == (1, "VERIFY_RELEASE")


def test_equal_sampling_budgets_and_split_isolation():
    plan_type = plan_class()
    budget = Budget(initial_repeats=5, feedback_attempts=150)
    initial = initial_plan(budget)
    arms = [initial, feedback_plan(budget, [{**row, "accepted": False} for row in initial], []), uniform_plan(budget)]
    assert [len(plan) for plan in arms] == [150, 150, 150]
    partitions = [{row["seed"] for row in plan} for plan in arms]
    for split in ("development", "confirmation"):
        seeds = budget.evaluation_seeds(split)
        assert Counter(plan_type.from_seed("random", seed).first_grasp for seed in seeds) == dict.fromkeys(range(10), 2)
        partitions.append(set(seeds))
    assert len(set.union(*partitions)) == sum(map(len, partitions))
    assert {row["reason"] for row in arms[1]} == {"missing_demonstrations", "coverage_only"}


def test_gate_cannot_default_to_an_unsafe_or_wrong_source_checkpoint():
    valid = {"wrong_source_episodes": 0, "safety_stops": 0}
    unsafe = {**valid, "safety_stops": 1}
    wrong_column = {**valid, "wrong_source_episodes": 1}
    comparison = {"reference": unsafe, "candidate": wrong_column, "mean_paired_gain": 1., "success_delta": 0}
    assert choose_recipe(comparison) is None
    assert choose_recipe({**comparison, "candidate": valid}) == "visual"
    assert choose_recipe({**comparison, "reference": valid}) == "mean"


def test_eval_receipt_rejects_wrong_cookie_identity_and_wrong_cutoff(tmp_path):
    plan_type = plan_class()
    seed = 82800
    checkpoint = tmp_path / "fixture.npz"
    _npz(checkpoint, fixture=0)
    plan = plan_type.from_seed("random", seed)
    source_ids = np.flatnonzero(source_columns() == SOURCE_COLUMN - 1)
    other_id = np.flatnonzero(source_columns() != SOURCE_COLUMN - 1)[0]
    snapshot = {"steps": 500, "score": 1, "requested_score": 1, "peak_count": 1,
                "by_column": [1, 0, 0, 0], "target_cookie_ids": [int(source_ids[0])],
                "target_column_counts": [1, 0], "source_remaining": 79, "hold_steps": 0,
                "success": False, "safety_reason": None, "action_sha256": "0" * 64, "applied_action_sha256": "1" * 64}
    row = {**snapshot, "steps": 3200, "task": "2x10", "profile": "advanced", "seed": seed,
           "first_grasp": plan.first_grasp, "column": SOURCE_COLUMN, "task_prompt": plan.prompt(SOURCE_COLUMN),
           "checkpoint_sha256": sha256(checkpoint), "at_500": snapshot}
    job = checkpoint, "advanced", seed
    check_evaluation(job, row)
    with pytest.raises(ValueError, match="identities"):
        check_evaluation(job, {**row, "target_cookie_ids": [int(other_id)]})
    with pytest.raises(ValueError, match="500-step snapshot"):
        check_evaluation(job, {**row, "at_500": {**snapshot, "steps": 499}})


def test_upstream_plugin_receives_a_rendered_front_frame(tmp_path, monkeypatch):
    plan_class()
    pytest.importorskip("mujoco")
    pytest.importorskip("cv2")
    checkpoint = tmp_path / "fixture.npz"
    _npz(checkpoint, format=TwoByTenPolicy.name, recipe="visual")
    monkeypatch.setenv("A3OPT_2X10_CHECKPOINT", str(checkpoint))
    observed = []

    def act(policy, observation, task):
        assert instruction(task) == (SOURCE_COLUMN, 0)
        observed.append(observation[FRONT])
        return observation["observation.state"]

    monkeypatch.setattr(TwoByTenPolicy, "act", act)
    result = benchmark("advanced", first_grasp=0, max_steps=1).run_episode("a3opt.backends.two_by_ten:create_policy", 2)
    assert result.steps == len(observed) == 1
    assert np.ptp(observed[0]) > 0
