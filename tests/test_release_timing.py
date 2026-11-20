import numpy as np
import pytest

from a3opt.backends.adaptive_phase import AdaptivePhasePolicy
from a3opt.release_timing_agent import STAGES, ReleaseTimingPolicy, compare, passes
from a3opt.release_timing_random_eval import MatchedPolicy


def test_timing_preserves_waypoint_endpoints_and_other_phases(monkeypatch):
    original = [[np.arange(20 * 16).reshape(20, 16) + batch * 100 + stage for stage in range(8)] for batch in range(2)]
    monkeypatch.setattr(AdaptivePhasePolicy, "_phase_actions", lambda *_: [list(batch) for batch in original])
    policy = ReleaseTimingPolicy.__new__(ReleaseTimingPolicy)
    policy.timing_column, policy.timing_scale = 4, .75

    for column in (2, 4):
        plan = policy._phase_actions(column, None)
        for batch, phases in enumerate(plan):
            for stage, actions in enumerate(phases):
                original_actions = original[batch][stage]
                if column == 4 and batch == 1 and stage in STAGES:
                    assert len(actions) == 15
                    np.testing.assert_array_equal(actions[[0, -1]], original_actions[[0, -1]])
                else:
                    np.testing.assert_array_equal(actions, original_actions)


def test_timing_requires_a_matching_first_batch_and_rejects_major_loss():
    def receipt(seed, count, success, open_step):
        return {"seed": seed, "profile": "advanced", "suite": "4", "column": 4, "task_prompt": "column 4",
                "initial_pose": [], "color": "golden", "cameras": {}, "light_diffuse": [],
                "first_placement_prefix": {"steps": 210, "applied": str(seed)}, "score": count,
                "requested_score": count, "success": success, "safety_reason": None, "second_open_step": open_step}

    baseline = [receipt(100, 10, False, 495), receipt(101, 10, True, 472), receipt(102, 6, False, 495)]
    candidate = [receipt(100, 10, True, 465), receipt(101, 6, False, 442), receipt(102, 10, True, 465)]
    contrast = compare({"1.0": baseline, "0.75": candidate})["0.75"]
    assert contrast["paired_open_step_advance_median"] == 30
    assert contrast["mean"] == contrast["reference_mean"]
    assert contrast["complete"] > contrast["reference_complete"]
    assert contrast["major_losses"] == 1
    assert not passes(contrast)

    candidate[0]["first_placement_prefix"] = {"steps": 209, "applied": "changed"}
    with pytest.raises(ValueError, match="paired policies differ"):
        compare({"1.0": baseline, "0.75": candidate})


def test_shared_rollout_stops_when_shadow_policy_disagrees(monkeypatch):
    class Reference:
        def act(self, observation, task):
            return np.zeros(16)

    policy = MatchedPolicy.__new__(MatchedPolicy)
    policy.reference = Reference()
    policy.matched_steps = 0
    monkeypatch.setattr(ReleaseTimingPolicy, "act", lambda self, observation, task: np.full(16, observation["command"]))

    np.testing.assert_array_equal(policy.act({"command": 0}), np.zeros(16))
    with pytest.raises(ValueError, match="differs from its reference"):
        policy.act({"command": 1})
    assert policy.matched_steps == 1
