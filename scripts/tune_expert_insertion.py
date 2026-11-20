from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from a3opt import advanced_sim, placement_benchmark as benchmark, placement_expert_landing as landing
from a3opt import placement_search, placement_servo_agent as servo, placement_source_guard as guard
from a3opt.advanced_agent import _audit_expert, _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, task_column
from a3opt.backends.advanced_phase import align_phases
from a3opt.backends.advanced_trajectory import POSE_SCALE
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES

from audit_placement_confirmation import scheduled_seeds
from confirm_expert_grasp import audit, measurements, resume
from confirm_source_guard import compare, inputs
from tune_landing_path import no_regression


POINTS = 9
GRID = np.linspace(0., 1., POINTS)


@dataclass(frozen=True)
class Insertion:
    column: int
    batch: int
    curve: tuple
    orientation: tuple

    def record(self):
        return [self.column, self.batch, [list(point) for point in self.curve], list(self.orientation)]


def read_tool(solver, joints):
    solver.data.qpos[solver.positions] = joints
    solver.mujoco.mj_kinematics(solver.model, solver.data)
    return solver.data.site_xpos[solver.site].copy(), Rotation.from_matrix(solver.data.site_xmat[solver.site].reshape(3, 3))


def read_demonstrations(attempts, folder, known, anchor):
    solver = nominal_kinematics()
    rows = []
    for attempt in attempts:
        if not attempt["success"] or attempt["score"] != 10 or attempt["batches"] != [0, 1]:
            continue
        seed = attempt["seed"]
        with np.load(folder / attempt["shard"], allow_pickle=False) as shard:
            pose = shard["pose"]
            if int(shard["seed"]) != seed or int(shard["column"]) != anchor.column or not np.allclose(
                    pose, known[seed]["initial_pose"], rtol=0., atol=1e-10):
                raise ValueError("expert insertion did not reproduce the selected layout")
            stages = shard[f"phases_{anchor.batch}"]
            actions = shard[f"actions_{anchor.batch}"]
            align_phases(actions, stages)
            samples = actions[np.isin(stages, ("DESCEND_TO_PLACE", "OPEN"))]
            descent_count = np.count_nonzero(stages == "DESCEND_TO_PLACE")
            samples = samples[:descent_count + 1]
        coordinates, rotations = zip(*(read_tool(solver, action[:7]) for action in samples), strict=True)
        local = np.asarray([landing.local_point(point, pose) for point in coordinates])
        descent = local[0, 2] - local[-1, 2]
        if descent < .025:
            continue
        fitted = np.column_stack([np.interp(GRID, np.linspace(0., 1., len(local)), local[:, axis]) for axis in range(3)])
        straight = fitted[0, :2] + GRID[:, None] * (fitted[-1, :2] - fitted[0, :2])
        relative = Rotation.from_rotvec([0., 0., -pose[4]]) * rotations[-1]
        curve = np.column_stack([fitted[:, :2] - straight, (fitted[0, 2] - fitted[:, 2]) / descent])
        rows.append({"seed": seed, "curve": curve.tolist(), "orientation_xyzw": relative.as_quat().tolist(),
                     "shard_sha256": attempt["sha256"]})
    return rows


def fit(rows, anchor):
    curve = np.median([row["curve"] for row in rows], axis=0)
    curve[:, :2] = np.clip(curve[:, :2], -.008, .008)
    curve[:, 2] = np.maximum.accumulate(np.clip(curve[:, 2], 0., 1.))
    curve[0], curve[-1] = (0., 0., 0.), (0., 0., 1.)
    values = tuple(tuple(map(float, point)) for point in curve)
    quaternion = Rotation.from_quat([row["orientation_xyzw"] for row in rows]).mean().as_quat()
    return ((Insertion(anchor.column, anchor.batch, values, ()),),
            (Insertion(anchor.column, anchor.batch, values, tuple(map(float, quaternion))),))


def pose_command(solver, initial, target, orientation):
    joints = initial.copy()
    for _ in range(5):
        position, rotation = read_tool(solver, joints)
        error = target - position
        twist = (orientation * rotation.inv()).as_rotvec() if orientation is not None else np.zeros(3)
        if np.linalg.norm(error) <= .0015 and np.linalg.norm(twist) <= .025:
            return joints
        solver.mujoco.mj_comPos(solver.model, solver.data)
        solver.mujoco.mj_jacSite(solver.model, solver.data, solver.jac_position, solver.jac_rotation, solver.site)
        jacobian = np.vstack((solver.jac_position[:, solver.dofs], .10 * solver.jac_rotation[:, solver.dofs]))
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .004**2 * np.eye(6), np.r_[error, .10 * twist])
        joints = np.clip(joints + delta, *solver.ranges.T)
        if not np.isfinite(joints).all() or np.max(np.abs(joints - initial)) > .18:
            break
    return None


def replan(phases, frame, observation, pose, anchor, model):
    updated = [phase.copy() for phase in phases]
    if not landing.adopt_anchor(updated, frame, observation, pose, anchor):
        return None
    lowering = updated[PLACE]
    solver = nominal_kinematics()
    start, initial_rotation = read_tool(solver, lowering[frame - 1, :7])
    goal, _ = read_tool(solver, lowering[-1, :7])
    if model.orientation:
        expert_rotation = Rotation.from_rotvec([0., 0., pose[4]]) * Rotation.from_quat(model.orientation)
        rotation_delta = (expert_rotation * initial_rotation.inv()).as_rotvec()
        size = np.linalg.norm(rotation_delta)
        if size > .12:
            rotation_delta *= .12 / size
    curve = np.asarray(model.curve)
    basis = landing.tray_basis(pose)
    for action, progress in zip(lowering[frame:], np.arange(1, len(lowering) - frame + 1) / (len(lowering) - frame), strict=True):
        offset_x, offset_y, height = (np.interp(progress, GRID, curve[:, axis]) for axis in range(3))
        shift = np.r_[basis @ [offset_x, offset_y], 0.]
        weight = np.array([_ramp(progress), _ramp(progress), height])
        target = start + weight * (goal - start) + shift
        orientation = Rotation.from_rotvec(_ramp(progress) * rotation_delta) * initial_rotation if model.orientation else None
        command = pose_command(solver, action[:7], target, orientation)
        if command is None:
            return None
        action[:7] = command
    final = lowering[-1, :7]
    retreat = updated[RETRACT]
    weights = 1. - _ramp(np.linspace(0., 1., len(retreat)))[:, None]
    retreat[:, :7] = np.clip(retreat[:, :7] + weights * (final - retreat[0, :7]), *solver.ranges.T)
    updated[OPEN][:, :7] = final
    return updated


class InsertionPolicy(guard.SourceGuardPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, baseline=(), **options):
        models = [item for item in corrections if isinstance(item, Insertion)]
        if len(models) > 1:
            raise ValueError("expected one learned insertion")
        self.insertion = models[0] if models else None
        super().__init__(checkpoint, focus, scale, (*baseline, *(item for item in corrections if not isinstance(item, Insertion))), **options)
        if self.insertion and (self.landing is None or (self.insertion.column, self.insertion.batch) != (self.landing.column, self.landing.batch)):
            raise ValueError("learned insertion requires the frozen anchor for the same column and batch")

    def act(self, observation, task=""):
        item = self.insertion
        if (item and self.actions is not None and self.cursor is not None and not self.retry_remaining
                and self.cursor[:2] == (item.batch, PLACE) and task_column(task) == item.column):
            phases = self.actions[item.batch]
            frame = max(1, len(phases[PLACE]) // 4)
            if self.cursor[2] == frame:
                adjusted = replan(phases, frame, observation, self.pose, self.landing, item)
                if adjusted is not None:
                    self.actions[item.batch] = adjusted
                    anchor, self.landing = self.landing, None
                    try:
                        return super().act(observation, task)
                    finally:
                        self.landing = anchor
        return super().act(observation, task)


def sample(rows, anchor):
    failed, controls = [], []
    for row in rows:
        if row["column"] != anchor.column or row["safety_reason"] or row["score"] != row["requested_score"]:
            continue
        lost = servo.lost_during_descent({**row, "terminal": row["physical_terminal"]}, anchor.batch)
        if lost:
            failed.append((-lost, row["requested_score"], row["seed"], row))
        elif row["success"]:
            controls.append(row)
    failures = [item[-1] for item in sorted(failed)[:3]]
    chosen = []
    for failure in failures[:len(controls)]:
        pose = np.asarray(failure["estimated_pose"])
        nearest = min(controls, key=lambda row: np.linalg.norm((np.asarray(row["estimated_pose"]) - pose) / POSE_SCALE))
        chosen.append(nearest)
        controls.remove(nearest)
    return failures[:len(chosen)] + chosen


def run(source, parent, gate, grasp, output, seed_block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, timing, constructor, baseline, lock, _, _ = inputs(source, parent)
    anchor = next(item for item in baseline if isinstance(item, landing.Landing))
    panels = {
        "source": (source / "confirmation/rollouts/results.json", "selected"),
        "gate": (gate / "development/results.json", "reference"),
        "parent": (parent / "confirmation/rollouts/results.json", "selected"),
        "grasp": (grasp / "confirmation/rollouts/results.json", "reference"),
    }
    sampled = {name: sample([row["strict-500"] for row in read_json(path)[arm]], anchor)
               for name, (path, arm) in panels.items()}
    training_rows = [row for group in sampled.values() for row in group]
    known = {row["seed"]: row for row in training_rows}
    if len(known) != len(training_rows) or len(known) < 8:
        raise ValueError("expert insertion requires distinct failure and control layouts")
    development = [seed for seed in range(seed_block, seed_block + 48) if choose_column("random", seed) == anchor.column]
    confirmation = list(range(seed_block + 600, seed_block + 728))
    reserved = {*development, *confirmation}
    protocols = [*output.parent.glob("*/protocol.json"), *output.parent.glob("*/replication-plan.json")]
    if any(reserved & scheduled_seeds(read_json(path).get("schedule", {})) for path in protocols if path.parent != output):
        raise ValueError("expert insertion requires fresh evaluation layouts")
    protocol = output / "protocol.json"
    _record_once(protocol, {"environment": simulator_identity(), "rules": RULES, "strict_rules": STRICT_RULES,
                 "source_lock_sha256": sha256(source / "confirmation-lock.json"), "checkpoint_sha256": sha256(checkpoint),
                 "code_sha256": sha256(Path(__file__)), "collection_sha256": sha256(Path(advanced_sim.__file__)),
                 "history_sha256": {name: sha256(path) for name, (path, _) in panels.items()},
                 "sampling": {name: [row["seed"] for row in group] for name, group in sampled.items()},
                 "scope": [anchor.column, anchor.batch], "schedule": {"training": list(known), "development": development,
                                                                      "confirmation": confirmation}})
    baseline_type = partial(InsertionPolicy, baseline=baseline, **constructor.keywords)
    candidate_type = partial(InsertionPolicy, **constructor.keywords)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        folder = output / "training" / "shards"
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Agent selected {len(known)} new expert insertion demonstrations for column {anchor.column}, batch {anchor.batch + 1}", flush=True)
        attempts = _jobs(pool, advanced_sim.collect_expert, [(seed, anchor.column, folder) for seed in known],
                         output / "training" / "attempts", lambda job: str(job[0]), _audit_expert)
        dataset = output / "training" / "dataset.json"
        rows = read_demonstrations(attempts, folder, known, anchor)
        _record_once(dataset, {"attempts": attempts, "rows": rows})
        if len(rows) < 6:
            return {"status": "reference_retained_insufficient_expert_insertions", "usable_demonstrations": len(rows)}
        options = dict(zip(("expert-position", "expert-pose"), fit(rows, anchor), strict=True))
        print(f"Agent fitted {len(rows)} expert demonstrations and is comparing both curves against the reference", flush=True)
        paired = benchmark.evaluate(pool, output / "training" / "trials", checkpoint, timing, list(known),
                                    {"reference": (), **options}, policy_type=baseline_type)
        scores = {name: measurements({"reference": paired["reference"], "selected": paired[name]}, model, compare_rows=compare)
                  for name, model in options.items()}
        eligible = [name for name, metrics in scores.items() if all(placement_search.passes(row) for row in metrics.values())]
        chosen = max(eligible, key=lambda name: scores[name]["strict-500"]["paired_count_delta"], default=None)
        fit_path = output / "fit.json"
        _record_once(fit_path, {"dataset_sha256": sha256(dataset), "curves": {name: model[0].record() for name, model in options.items()},
                                "training": scores, "selected": chosen})
        if chosen is None:
            return {"status": "reference_retained_after_expert_insertion_fit", "training": scores}
        model = options[chosen]
        print(f"Agent selected {chosen} and is evaluating {len(development)} new development pairs", flush=True)
        paired = benchmark.evaluate(pool, output / "development", checkpoint, timing, development,
                                    {"reference": (), "selected": model}, policy_type=baseline_type)
        metrics = measurements(paired, model, compare_rows=compare)
        selection = output / "development-selection.json"
        _record_once(selection, metrics)
        if not all(no_regression(row) and row["paired_count_delta"] > 0 for row in metrics.values()):
            return {"status": "reference_retained_after_expert_insertion_development", "development": metrics}
        corrections = (*baseline, *model)
        frozen = {**lock, "seeds": confirmation, "corrections": [item.record() for item in corrections],
                  "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)}
        _record_once(output / "confirmation-lock.json", frozen)
        parity_seeds = sorted({development[0], metrics["default"]["changes"][0]["seed"]})
        parity = benchmark.evaluate(pool, output / "calibration", checkpoint, timing, parity_seeds,
                                    {"reference": (), "selected": corrections}, policy_type=candidate_type)
    if not (output / "result.json").is_file():
        resume(output, checkpoint, timing, candidate_type, corrections, frozen, parity, metrics, workers,
               compare_rows=compare, parity_seeds=parity_seeds)
    return audit(output, checkpoint, timing, corrections, frozen, metrics, compare_rows=compare,
                 retained_status="reference_retained_after_expert_insertion_confirmation")


def main():
    parser = argparse.ArgumentParser(description="Collect expert rollouts and learn a tray-relative insertion trajectory.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--grasp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=177000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    source, parent, gate, grasp, output = (path.resolve() for path in (args.source, args.parent, args.gate, args.grasp, args.out))
    if output in (source, parent, gate, grasp) or args.seed_block < 0 or args.seed_block % 4 or args.workers < 1:
        parser.error("expected separate output, an aligned seed block, and a positive worker budget")
    with _run_lock(output):
        result = run(source, parent, gate, grasp, output, args.seed_block, args.workers)
        if "confirmation" not in result:
            _record_once(output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
