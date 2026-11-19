from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER, AdaptivePhasePolicy, Recipe


MAX_OFFSET = np.array([.018, .024, .023])
MAX_JOINT_CORRECTION = .16


def nominal_model_path() -> Path:
    from a3_dual_arm_sim.paths import resource_root

    return resource_root() / "models" / "a3_cookie_transfer.xml"


class LeftArmKinematics:
    def __init__(self, path: Path):
        import mujoco
        from a3_dual_arm_sim.contracts import LEFT_JOINTS

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site("L_eef").id
        joints = [self.model.joint(name).id for name in LEFT_JOINTS]
        self.positions = self.model.jnt_qposadr[joints]
        self.dofs = self.model.jnt_dofadr[joints]
        self.ranges = self.model.jnt_range[joints]
        self.jac_position = np.zeros((3, self.model.nv))
        self.jac_rotation = np.zeros_like(self.jac_position)

    def shift(self, command: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        low, high = self.ranges.T
        start = np.clip(command, low, high)
        self.data.qpos[self.positions] = start
        self.mujoco.mj_kinematics(self.model, self.data)
        self.mujoco.mj_comPos(self.model, self.data)
        self.mujoco.mj_jacSite(self.model, self.data, self.jac_position, self.jac_rotation, self.site)
        jacobian = np.vstack((self.jac_position[:, self.dofs], .10 * self.jac_rotation[:, self.dofs]))
        correction = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .004**2 * np.eye(6), np.r_[displacement, 0., 0., 0.])
        norm = np.max(np.abs(correction))
        if norm > MAX_JOINT_CORRECTION:
            correction *= MAX_JOINT_CORRECTION / norm
        return np.clip(start + correction, low, high)


@lru_cache(maxsize=1)
def nominal_kinematics() -> LeftArmKinematics:
    return LeftArmKinematics(nominal_model_path())


def _ramp(values: np.ndarray) -> np.ndarray:
    fraction = np.clip(values, 0, 1)
    return fraction * fraction * (3 - 2 * fraction)


class ReleaseResidualPolicy(AdaptivePhasePolicy):
    name = "a3opt_release_residual"

    def __init__(self, model: dict, column: int, offset: np.ndarray):
        if column not in range(5) or offset.shape != (3,) or not np.all(np.abs(offset) <= MAX_OFFSET):
            raise ValueError("invalid release correction")
        self.release_column = column
        self.release_offset = offset
        super().__init__(model, (Recipe(),) * 4)

    def _phase_actions(self, column: int, recipe: Recipe) -> list[list[np.ndarray]]:
        plan = super()._phase_actions(column, recipe)
        if column != self.release_column:
            return plan
        coherent = super()._phase_actions(column, Recipe(joint_placement=True))[0]
        old_retract = plan[0][RETRACT][-1]
        if np.any(self.release_offset):
            self._apply_offset(coherent)
        plan[0] = coherent
        approach = plan[1][0]
        length = min(len(approach), 3)
        blend = np.linspace(1., 0., length + 1)[1:, None]
        approach[:length] += blend * (coherent[RETRACT][-1] - old_retract)
        return plan

    def _apply_offset(self, phases: list[np.ndarray]) -> None:
        angle = self.pose[4]
        cosine, sine = np.cos(angle), np.sin(angle)
        horizontal = np.array([[cosine, -sine], [sine, cosine]]) @ self.release_offset[:2]
        solver = nominal_kinematics()
        for stage in (TRANSFER, PLACE, OPEN, RETRACT):
            actions = phases[stage]
            progress = np.linspace(0., 1., len(actions))
            if stage == TRANSFER:
                lateral = _ramp((progress - .5) * 2)
                vertical = np.zeros_like(progress)
            elif stage == PLACE:
                lateral, vertical = np.ones_like(progress), _ramp(progress)
            elif stage == OPEN:
                lateral = vertical = np.ones_like(progress)
            else:
                lateral = vertical = 1 - _ramp(progress)
            for action, xy_scale, z_scale in zip(actions, lateral, vertical, strict=True):
                displacement = np.r_[horizontal * xy_scale, self.release_offset[2] * z_scale]
                if np.any(displacement):
                    action[:7] = solver.shift(action[:7], displacement)


def load_checkpoint(path: Path) -> ReleaseResidualPolicy:
    with np.load(path, allow_pickle=False) as archive:
        model = dict(archive)
    return ReleaseResidualPolicy(model, int(model.pop("release_column")), model.pop("release_offset"))


def upstream_policy() -> ReleaseResidualPolicy:
    return load_checkpoint(Path(os.environ["A3OPT_RELEASE_CHECKPOINT"]))
