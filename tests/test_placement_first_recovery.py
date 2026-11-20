from types import SimpleNamespace

import numpy as np
import pytest

from a3opt import placement_first_recovery as first


@pytest.mark.parametrize("recipe", first.recovery.RECIPES)
def test_recovering_during_first_descent_preserves_the_second_batch_start(monkeypatch, recipe):
    solver = SimpleNamespace(ranges=np.asarray([[-1., 1.]] * 7))
    monkeypatch.setattr(first, "nominal_kinematics", lambda: solver)
    monkeypatch.setattr(first.descent, "tool_xyz", lambda solver, joints: joints[:3].copy())
    monkeypatch.setattr(first.recovery, "joint_target", lambda solver, initial, target: np.r_[target, initial[3:]])
    batches = [[np.zeros((32, 16)) for _ in range(first.RETRACT + 1)] for _ in range(2)]
    batches[0][first.TRANSFER][-1, 2] = .10
    batches[0][first.PLACE][:, 2] = np.linspace(.10, .02, 32)
    batches[0][first.OPEN][:, 7] = np.linspace(.01, .09, 32)
    frame = 24
    controller = SimpleNamespace(actions=batches, cursor=(0, first.PLACE, frame), pose=np.zeros(5))
    original = [[stage.copy() for stage in batch] for batch in batches]
    joints = batches[0][first.PLACE][frame - 1, :7].copy()
    observation = {"observation.eef_pose": joints[:3], "observation.state": joints}

    assert first.raise_for_release(controller, observation, recipe)

    assert all(before.shape == after.shape for before, after in zip(original[0], batches[0], strict=True))
    assert np.array_equal(original[0][first.PLACE][:frame], batches[0][first.PLACE][:frame])
    assert np.array_equal(original[0][first.OPEN][:, 7:], batches[0][first.OPEN][:, 7:])
    assert np.array_equal(original[0][first.RETRACT][-1], batches[0][first.RETRACT][-1])
    assert all(np.array_equal(before, after) for before, after in zip(original[1], batches[1], strict=True))
    assert np.all(batches[0][first.OPEN][:, 2] > joints[2])
