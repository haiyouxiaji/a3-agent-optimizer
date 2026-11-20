from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark as benchmark, placement_descent_agent as descent
from a3opt import placement_expert_landing as landing, placement_first_hover as first
from a3opt import placement_hover_agent as hover, placement_release_recovery as recovery, placement_search
from a3opt.advanced_agent import _audit_expert, _record_once
from a3opt.backends.adaptive_phase import CLOSE, DESCEND, LIFT, PRE_CLOSE, task_column
from a3opt.backends.advanced_phase import align_phases
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


BATCH = 1
LIMIT = np.array([.006, .006, .003])
MIN_GAIN_M = .0005


@dataclass(frozen=True)
class Grasp:
    column: int
    anchor: tuple[float, ...]
    axes: tuple[bool, ...]

    def __post_init__(self):
        if (self.column not in range(1, 5) or len(self.anchor) != 3 or not np.isfinite(self.anchor).all()
                or len(self.axes) != 3 or not any(self.axes) or not all(isinstance(active, bool) for active in self.axes)):
            raise ValueError("invalid learned grasp position")

    @property
    def batch(self):
        return BATCH

    def record(self):
        return [self.column, self.batch, list(self.anchor), list(self.axes)]


def source_anchor(pose):
    return np.r_[pose[:2], 0.]


def adjust_grasp(phases, pose, grasp):
    solver = nominal_kinematics()
    final = phases[CLOSE][-1, :7]
    nominal = descent.tool_xyz(solver, final)
    shift = np.clip(source_anchor(pose) + grasp.anchor - nominal, -LIMIT, LIMIT) * grasp.axes
    if not np.isfinite(shift).all() or recovery.joint_target(solver, final, nominal + shift) is None:
        return False
    for stage in (PRE_CLOSE, DESCEND, CLOSE, LIFT):
        actions = phases[stage]
        progress = np.linspace(0., 1., len(actions))
        weights = np.ones((len(actions), 3))
        if stage == PRE_CLOSE:
            weights[:, :2] = _ramp(progress)[:, None]
            weights[:, 2] = 0.
        elif stage == DESCEND:
            weights[:, 2] = _ramp(progress)
        elif stage == LIFT:
            weights[:] = 1. - _ramp(2. * progress)[:, None]
        for action, offset in zip(actions, weights * shift, strict=True):
            if np.any(offset):
                action[:7] = solver.shift(action[:7], offset)
    return True


class ExpertGraspPolicy(landing.ExpertLandingPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent, late_recovery, release_anchor):
        if len(corrections) > 1:
            raise ValueError("expected one second-batch grasp position")
        self.grasp = corrections[0] if corrections else None
        super().__init__(checkpoint, focus, scale, release_anchor, incumbent=incumbent, late_recovery=late_recovery)

    def act(self, observation, task=""):
        if (self.grasp is not None and self.actions is not None and self.cursor == (BATCH, PRE_CLOSE, 0)
                and not self.retry_remaining and task_column(task) == self.grasp.column):
            adjust_grasp(self.actions[BATCH], self.pose, self.grasp)
        return super().act(observation, task)


def count_missing_lifts(rows):
    misses = {}
    for episode in rows:
        row = episode["strict-500"]
        if (not row["safety_reason"] and row["score"] == row["requested_score"] < 10
                and "lift" in row["phase_geometry"][BATCH]):
            lifted = len(row["phase_geometry"][BATCH]["lift"]["batches"][BATCH])
            if lifted < 5:
                misses[row["seed"]] = min(5 - lifted, 10 - row["requested_score"])
    return misses


def build_dataset(source, checkpoint, timing, column, data):
    controller = descent.DescentPolicy(checkpoint, *timing)
    solver = nominal_kinematics()
    attempts = {row["seed"]: row for row in data["attempts"]}
    shards = source / "training" / "shards"
    examples = []
    for row in data["rows"]:
        seed = row["seed"]
        attempt = attempts[seed]
        _audit_expert((seed, column, shards), attempt)
        pose = np.asarray(row["estimated_pose"])
        with np.load(shards / attempt["shard"], allow_pickle=False) as shard:
            actions, phases = shard[f"actions_{BATCH}"], shard[f"phases_{BATCH}"]
            align_phases(actions, phases)
            last = np.flatnonzero(phases == "CLOSE")[-1]
            expert = descent.tool_xyz(solver, actions[last, :7])
        controller.pose = pose
        plan = controller._phase_actions(column, controller.recipes[column - 1])
        nominal = descent.tool_xyz(solver, plan[BATCH][CLOSE][-1, :7])
        origin = source_anchor(pose)
        examples.append({"seed": seed, "shard_sha256": attempt["sha256"], "estimated_pose": pose.tolist(),
                         "expert_grasp_m": (expert - origin).tolist(), "nominal_grasp_m": (nominal - origin).tolist()})
    return examples


def fit(rows, column):
    if len(rows) < 6 or len({row["seed"] for row in rows}) != len(rows):
        return (), {"observations": len(rows), "admitted": False}
    expert = np.asarray([row["expert_grasp_m"] for row in rows])
    nominal = np.asarray([row["nominal_grasp_m"] for row in rows])
    heldout = np.vstack([np.median(np.delete(expert, index, axis=0), axis=0) for index in range(len(rows))])
    predicted = nominal + np.clip(heldout - nominal, -LIMIT, LIMIT)
    reference_error = np.mean(abs(nominal - expert), axis=0)
    candidate_error = np.mean(abs(predicted - expert), axis=0)
    axes = tuple(map(bool, reference_error - candidate_error > MIN_GAIN_M))
    model = (Grasp(column, tuple(np.median(expert, axis=0)), axes),) if any(axes) else ()
    return model, {"observations": len(rows), "admitted": bool(model), "mean_absolute_error_m": {
        "nominal": reference_error.tolist(), "median": candidate_error.tolist()}}


def compare(reference, selected, model):
    if any(old["phase_geometry"][0] != new["phase_geometry"][0] for old, new in zip(reference, selected, strict=True)):
        raise ValueError("second-batch grasp correction changed the first batch")
    return placement_search.compare(reference, selected, model, prefix_batch=0)


def run(source, root, block, workers, fit_only=False):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected an aligned seed block and a positive worker budget")
    source_protocol = source / "protocol.json"
    lock_path = source / "confirmation-lock.json"
    parent_fit = source / "fit.json"
    parent_selection = source / "development-selection.json"
    data_path = source / "training" / "dataset.json"
    parent, lock, frozen_fit, scores, data = map(read_json, (source_protocol, lock_path, parent_fit, parent_selection, data_path))
    checkpoint, timing = Path(lock["checkpoint"]), tuple(lock["timing"])
    if (parent["environment"] != simulator_identity() or parent["rules"] != RULES or parent["strict_rules"] != STRICT_RULES
            or parent["code_sha256"] != sha256(Path(landing.__file__)) or parent["dependency_sha256"] != landing.dependencies()
            or any(lock[key] != parent[key] for key in ("incumbent", "timing", "late_recovery", "checkpoint_sha256"))
            or sha256(checkpoint) != lock["checkpoint_sha256"] or frozen_fit["dataset_sha256"] != sha256(data_path)
            or frozen_fit["corrections"] != lock["corrections"] or not all(placement_search.passes(row) for row in scores.values())
            or any(lock[key] != sha256(path) for key, path in (("protocol_sha256", source_protocol), ("fit_sha256", parent_fit),
                                                               ("selection_sha256", parent_selection)))):
        raise ValueError("grasp learning requires the frozen parent, scored development, and audited demonstrations")
    column, batch = parent["scope"]
    if batch != 0 or len(frozen_fit["corrections"]) != 1 or frozen_fit["corrections"][0][:2] != [column, batch]:
        raise ValueError("the grasp learner requires a first-batch release model for the same column")
    release_anchor = tuple(landing.Landing(item_column, batch, tuple(anchor), tuple(center), tuple(map(tuple, coefficients)), tuple(axes))
                           for item_column, batch, anchor, center, coefficients, axes in frozen_fit["corrections"])
    parent_development = source / "development" / "results.json"
    paired = read_json(parent_development)
    if hover.pair_stats(paired, release_anchor) != scores:
        raise ValueError("the recorded parent development differs from its score report")
    for seed, episode in zip(parent["schedule"]["development"], paired["selected"], strict=True):
        benchmark.check((checkpoint, *timing, seed, release_anchor), episode)
    misses = count_missing_lifts(paired["selected"])
    if not misses:
        return {"status": "reference_retained_without_second_batch_lift_losses"}
    development = [seed for seed in range(block, block + 64) if choose_column("random", seed) == column]
    confirmation = list(range(block + 400, block + 464))
    archives = [*root.parent.glob("*/protocol.json"), *root.parent.glob("*/replication-plan.json"),
                *checkpoint.parent.parent.glob("*/*protocol.json")]
    if any(set(development + confirmation) & hover.scheduled_seeds(read_json(path).get("schedule", {}))
           for path in archives if path.parent != root):
        raise ValueError("grasp learning requires new development and confirmation scenes")
    dependencies = {**landing.dependencies(), landing.__name__: sha256(Path(landing.__file__))}
    protocol = root / "protocol.json"
    _record_once(protocol, {"environment": parent["environment"], "rules": RULES, "strict_rules": STRICT_RULES,
                 "checkpoint_sha256": sha256(checkpoint), "code_sha256": sha256(Path(__file__)), "dependencies": dependencies,
                 "parent_lock_sha256": sha256(lock_path), "parent_development_sha256": sha256(parent_development),
                 "parent_dataset_sha256": sha256(data_path), "second_batch_missing_lifts": {str(key): value for key, value in misses.items()},
                 "scope": [column, BATCH], "schedule": {"training": parent["schedule"]["training"],
                                                        "development": development, "confirmation": confirmation}})
    dataset = root / "training" / "dataset.json"
    rows = build_dataset(source, checkpoint, timing, column, data)
    _record_once(dataset, {"rows": rows})
    fitted, metrics = fit(rows, column)
    records = [item.record() for item in fitted]
    fit_path = root / "fit.json"
    _record_once(fit_path, {**metrics, "dataset_sha256": sha256(dataset), "corrections": records})
    if not fitted or fit_only:
        return {"status": "expert_grasp_fitted" if fitted else "reference_retained_after_expert_grasp_fit", "fit": metrics}
    incumbent = tuple(descent.Descent(*item) for item in lock["incumbent"])
    late_recovery = tuple(recovery.Recovery(*item[2:]) for item in lock["late_recovery"])
    policy_type = partial(ExpertGraspPolicy, incumbent=incumbent, late_recovery=late_recovery, release_anchor=release_anchor)
    print(f"Agent diagnosed {sum(misses.values())} missing second-batch lifts and will evaluate {len(development)} fresh pairs", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        trials = benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                    {"reference": (), "selected": fitted}, policy_type=policy_type)
        dev = {rule: compare([row[rule] for row in trials["reference"]], [row[rule] for row in trials["selected"]], fitted)
               for rule in ("default", "strict-500")}
        selection = root / "development-selection.json"
        _record_once(selection, dev)
        if not all(placement_search.passes(row) for row in dev.values()):
            return {"status": "reference_retained_after_expert_grasp_development", "development": dev, "corrections": records}
        _record_once(root / "confirmation-lock.json", {"parent_lock_sha256": sha256(lock_path), "checkpoint_sha256": sha256(checkpoint),
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection),
                     "seeds": confirmation, "corrections": records})
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                  fitted, confirmation[:4], policy_type=policy_type, compare_rows=compare)
    if any(row["scored_pairs_by_column"] != dict.fromkeys(map(str, range(1, 5)), 16) for row in final.values()):
        raise ValueError("confirmation requires four equally represented columns")
    status = "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_expert_grasp_confirmation"
    return {"status": status, "confirmation": final, "development": dev, "corrections": records,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Learn the second-batch grasp position from the expert demonstrations.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=169200)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(args.source.resolve(), root, args.seed_block, args.workers, args.fit_only)
        if not args.fit_only:
            _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
