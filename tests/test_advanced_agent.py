from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import pytest

import a3opt.advanced_agent as agent
from a3opt.backends.advanced_trajectory import AdvancedTrajectoryPolicy, BOUNDARIES, PHASE_SAMPLES, align_batch
from a3opt.backends.advanced_phase import AdvancedPhasePolicy, BOUNDARIES as FINE_BOUNDARIES, SAMPLES, align_phases
from a3opt.refinement_eval import read_json, sha256


def test_budget_reserves_disjoint_training_and_balanced_confirmation_seeds():
    budget = agent.AdvancedBudget()
    budget.validate()
    groups = [set(budget.seeds("train"))] + [set(budget.seeds(split, suite))
                                           for split in ("development", "confirmation") for suite in agent.SUITES]
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert budget.seeds("confirmation", "random")[0] % 4 == 0
    assert len(budget.seeds("confirmation", "random")) == 20
    with pytest.raises(ValueError):
        agent.AdvancedBudget(calibration=80, rounds=4).validate()


def test_phase_alignment_and_policy_use_distinct_column_demos(tmp_path):
    phases = np.array(["APPROACH", *BOUNDARIES, "VERIFY_RELEASE"])
    first, lengths = align_batch(np.ones((len(phases), 16)) * .15, phases)
    second, _ = align_batch(np.ones((len(phases), 16)) * .45, phases)
    assert sum(lengths) == len(phases) and len(first) == sum(PHASE_SAMPLES)
    checkpoint = tmp_path / "synthetic.npz"
    np.savez(checkpoint, format=AdvancedTrajectoryPolicy.name, poses=np.zeros((2, 5)), traces=[first, second],
             lengths=[lengths, lengths], columns=[1, 4], batches=[0, 0], neighbors=12, ridge=.02, mode="factored")
    policy = AdvancedTrajectoryPolicy(checkpoint)
    assert np.allclose(policy._action_plan(1, np.zeros(5)), .15)
    assert np.allclose(policy._action_plan(4, np.zeros(5)), .45)
    with pytest.raises(ValueError, match="no training demonstrations"):
        policy._action_plan(2, np.zeros(5))
    with pytest.raises(ValueError, match="missing expert phase"):
        align_batch(np.zeros((2, 16)), np.array(["APPROACH", "RETRACT"]))


@pytest.mark.parametrize("policy_type", (AdvancedTrajectoryPolicy, AdvancedPhasePolicy))
def test_upstream_policy_plugin_loads_the_advanced_checkpoint(tmp_path, monkeypatch, policy_type):
    from a3opt.backends.policy_plugin import CHECKPOINT_ENV, create_policy

    checkpoint = tmp_path / "advanced.npz"
    np.savez(checkpoint, format=policy_type.name)
    monkeypatch.setenv(CHECKPOINT_ENV, str(checkpoint))
    policy = create_policy()
    assert isinstance(policy, policy_type)
    assert policy.camera_names == ("front",)
    policy.close()


def test_fine_alignment_keeps_preclose_descent_and_grip_targets_separate():
    phases = np.repeat(("APPROACH", *FINE_BOUNDARIES), 2)
    actions = np.repeat(np.arange(len(phases), dtype=float)[:, None], 16, axis=1)
    trace, lengths = align_phases(actions, phases)
    assert lengths.tolist() == [2] * len(SAMPLES)
    for index, endpoint in enumerate(np.cumsum(SAMPLES) - 1):
        assert trace[endpoint, 7] == 2 * index + 1


def test_wrong_column_counts_only_toward_upstream_score():
    row = {"column": 4, "score": 10, "requested_score": 0, "success": True,
           "estimated_pose": [0] * 5, "initial_pose": [0] * 5}
    metric = agent.summarize([row])
    assert metric["mean_score"] == 10
    assert metric["mean_requested_score"] == 0
    assert metric["full_successes"] == 0


def test_agent_freezes_once_and_resumes_with_same_training_records(tmp_path, monkeypatch):
    class ImmediatePool:
        def __init__(self, max_workers):
            self.workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, function, *args):
            result = Future()
            result.set_result(function(*args))
            return result

    budget = agent.AdvancedBudget(calibration=48, initial_attempts=20, rounds=1)
    expert_calls = []
    eval_calls = []

    def calibrate(pool, path, seeds):
        agent._npz(path, seeds=seeds)
        return {}, {seed: (np.zeros(5), "golden") for seed in seeds}

    def collect(seed, column, folder):
        assert seed in budget.seeds("train")
        expert_calls.append(seed)
        path = folder / f"{column}-{seed}.npz"
        agent._npz(path, pose=np.zeros(5), seed=seed, column=column,
                   trace_0=np.zeros((sum(PHASE_SAMPLES), 16)), lengths_0=np.ones(len(PHASE_SAMPLES)))
        return {"seed": seed, "column": column, "batches": [0], "shard": path.name,
                "sha256": sha256(path), "color": "golden", "pose": [0] * 5}

    def evaluate(pool, checkpoint, root, run_budget, split):
        eval_calls.append((checkpoint, split))
        if split == "confirmation":
            lock = read_json(tmp_path / "confirmation-lock.json")
            assert lock["checkpoint_sha256"] == sha256(checkpoint)
            assert checkpoint.name == "policy.npz"
        result = {suite: {"metrics": {"mean_requested_score": 6.0, "full_successes": 1}, "rows": []}
                  for suite in agent.SUITES}
        agent.write_json(root / "results.json", result)
        return result

    monkeypatch.setattr(agent, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(agent, "_calibrate", calibrate)
    monkeypatch.setattr(agent, "collect_expert", collect)
    monkeypatch.setattr(agent, "_evaluate", evaluate)
    monkeypatch.setattr(agent, "simulator_identity", lambda: {"sim_revision": "mock"})
    outcome = agent.run_agent(tmp_path, budget, 1)
    assert outcome["milestone_met"] and not outcome["target_met"]
    assert [split for _, split in eval_calls] == ["development", "development", "confirmation"]
    assert len(set(expert_calls)) == 20
    assert agent.run_agent(tmp_path, budget, 1) == outcome
    assert len(expert_calls) == 20
