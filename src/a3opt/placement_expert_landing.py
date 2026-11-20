from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import advanced_sim, placement_benchmark as benchmark, placement_descent_agent as descent
from a3opt import placement_first_hover as first, placement_first_recovery as history, placement_hover_agent as hover
from a3opt import placement_release_recovery as recovery, placement_routed_agent as routed, placement_search
from a3opt import placement_servo_agent as servo
from a3opt.advanced_agent import _audit_expert, _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER, task_column
from a3opt.backends.advanced_phase import align_phases
from a3opt.backends.advanced_trajectory import POSE_SCALE
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


LIMIT = np.array([.009, .009, .014])
MIN_GAIN_M = .0005


@dataclass(frozen=True)
class Landing:
    column: int
    batch: int
    anchor: tuple[float, ...]
    center: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    axes: tuple[bool, ...]

    def __post_init__(self):
        if (self.column not in range(1, 5) or self.batch not in (0, 1) or len(self.anchor) != 3
                or len(self.center) != len(POSE_SCALE) or np.shape(self.coefficients) != (len(POSE_SCALE), 3)
                or not np.isfinite(np.r_[self.anchor, self.center, np.ravel(self.coefficients)]).all()
                or len(self.axes) != 3 or not any(self.axes) or not all(isinstance(active, bool) for active in self.axes)):
            raise ValueError("invalid learned release anchor")

    def record(self):
        return [self.column, self.batch, list(self.anchor), list(self.center),
                [list(row) for row in self.coefficients], list(self.axes)]

    def target(self, pose):
        return np.asarray(self.anchor) + ((pose - self.center) / POSE_SCALE) @ np.asarray(self.coefficients)


def tray_basis(pose):
    cosine, sine = np.cos(pose[4]), np.sin(pose[4])
    return np.array([[cosine, -sine], [sine, cosine]])


def local_point(point, pose):
    return np.r_[(point[:2] - pose[2:4]) @ tray_basis(pose), point[2]]


def adopt_anchor(phases, frame, observation, pose, model):
    solver = nominal_kinematics()
    lowering, opening, retreat = (phases[stage] for stage in (PLACE, OPEN, RETRACT))
    remaining = len(lowering) - frame
    measured = np.asarray(observation["observation.eef_pose"][:3])
    if remaining < 6 or not np.isfinite(np.r_[pose, measured]).all():
        return False
    origin = lowering[-1, :7].copy()
    planned = descent.tool_xyz(solver, origin)
    shift = np.clip(model.target(pose) - local_point(planned, pose), -LIMIT, LIMIT) * model.axes
    target = planned + np.r_[tray_basis(pose) @ shift[:2], shift[2]]
    if not 0. <= measured[2] - target[2] <= .085:
        return False
    hold = recovery.joint_target(solver, origin, target)
    if hold is None:
        return False
    prior = lowering[frame - 1, :7].copy()
    lowering[frame:, :7] = prior + _ramp(np.arange(1, remaining + 1) / remaining)[:, None] * (hold - prior)
    opening[:, :7] = hold
    if len(retreat) > 1:
        fade = 1. - _ramp(np.linspace(0., 1., len(retreat)))[:, None]
        retreat[:, :7] = np.clip(retreat[:, :7] + fade * (hold - retreat[0, :7]), *solver.ranges.T)
    return True


class ExpertLandingPolicy(recovery.RecoveryPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent, late_recovery):
        if len(corrections) > 1:
            raise ValueError("expected one learned anchor")
        self.landing = corrections[0] if corrections else None
        super().__init__(checkpoint, focus, scale, late_recovery, incumbent=incumbent)

    def act(self, observation, task=""):
        model = self.landing
        if (model is not None and self.actions is not None and self.cursor is not None and not self.retry_remaining
                and self.cursor[:2] == (model.batch, PLACE) and task_column(task) == model.column):
            frame = max(1, len(self.actions[model.batch][PLACE]) // 4)
            if self.cursor[2] == frame:
                adopt_anchor(self.actions[model.batch], frame, observation, self.pose, model)
        return super().act(observation, task)


def archive_rows(root, checkpoint, timing, common, late_recovery):
    protocol = read_json(root / "protocol.json")
    replication = read_json(root / "replication-protocol.json")
    initial = read_json(root / "confirmation-audit.json")
    repeated = read_json(root / "replication-result.json")
    if (any(protocol[key] != value for key, value in common.items()) or replication["environment"] != common["environment"]
            or replication["source_lock_sha256"] != sha256(root / "confirmation-lock.json")
            or replication["checkpoint_sha256"] != sha256(checkpoint)
            or replication["evaluator_sha256"] != sha256(Path(benchmark.__file__))
            or replication["corrections"] != [item.record() for item in late_recovery]):
        raise ValueError("release recovery archives differ from the frozen reference")
    locks = {"confirmation": (protocol["schedule"]["confirmation"], "confirmation.json.gz", initial["evaluation_sha256"]),
             "replication": (replication["schedule"]["replication"], "replication-rollouts.json.gz", repeated["replication_sha256"])}
    panels, fingerprints = {}, {}
    for label, (seeds, name, expected) in locks.items():
        raw = gzip.decompress((root / name).read_bytes())
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{label} archive differs from its recorded digest")
        pairs = json.loads(raw)
        for seed, entry in zip(seeds, pairs["selected"], strict=True):
            benchmark.check((checkpoint, *timing, seed, late_recovery), entry)
        measured = hover.pair_stats(pairs, late_recovery)
        reported = initial["confirmation"] if label == "confirmation" else repeated["replication"]
        if any(row != {key: reported[rule][key] for key in row} for rule, row in measured.items()):
            raise ValueError(f"{label} scores do not reproduce")
        panels[label] = [entry["strict-500"] for entry in pairs["selected"]]
        fingerprints[name] = expected
    return panels, fingerprints


def sample_plan(panels):
    choices = []
    for column in range(1, 5):
        for batch in (0, 1):
            strata = {}
            missing = 0
            for name, rows in panels.items():
                observed = [row for row in rows if row["column"] == column and not row["safety_reason"]
                            and row["score"] == row["requested_score"] and "transfer" in row["phase_geometry"][batch]]
                losses = sorted((-servo.lost_during_descent({**row, "terminal": row["physical_terminal"]}, batch), row["seed"])
                                for row in observed)
                cases = [(loss, seed) for loss, seed in losses if loss < 0]
                controls = sorted(row["seed"] for row in observed if row["success"])
                if cases and controls:
                    missing -= sum(loss for loss, _ in cases)
                    strata[name] = {"failures": [seed for _, seed in cases[:4]], "controls": controls[:3]}
            if len(strata) == len(panels):
                choices.append((missing, -column, -batch, strata))
    _, column, batch, plan = max(choices, key=lambda item: item[:3])
    return -column, -batch, plan


def training_rows(attempts, shards, known, checkpoint, timing, batch):
    solver = nominal_kinematics()
    controller = descent.DescentPolicy(checkpoint, *timing)
    data, rejected = [], []
    for attempt in attempts:
        seed, column = attempt["seed"], attempt["column"]
        if not attempt["success"] or attempt["score"] != 10 or attempt["batches"] != [0, 1]:
            rejected.append(seed)
            continue
        archived = known[seed]
        predicted = np.asarray(archived["estimated_pose"])
        with np.load(shards / attempt["shard"], allow_pickle=False) as shard:
            if (int(shard["seed"]), int(shard["column"])) != (seed, column) or not np.allclose(
                    shard["pose"], archived["initial_pose"], rtol=0., atol=1e-10):
                raise ValueError("expert demonstration did not reproduce the target layout")
            actions, phases = shard[f"actions_{batch}"], shard[f"phases_{batch}"]
            align_phases(actions, phases)
            index = int(np.flatnonzero(phases == "OPEN")[0])
            expert = descent.tool_xyz(solver, actions[index, :7])
        controller.pose = predicted
        plan = controller._phase_actions(column, controller.recipes[column - 1])
        command = descent.tool_xyz(solver, plan[batch][PLACE][-1, :7])
        data.append({"seed": seed, "column": column, "batch": batch, "estimated_pose": predicted.tolist(),
                     "release_anchor_m": local_point(expert, predicted).tolist(),
                     "nominal_anchor_m": local_point(command, predicted).tolist(), "shard_sha256": attempt["sha256"]})
    return data, rejected


def regress(rows, affine):
    pose = np.asarray([row["estimated_pose"] for row in rows])
    labels = np.asarray([row["release_anchor_m"] for row in rows])
    center = pose.mean(axis=0)
    scaled = (pose - center) / POSE_SCALE
    if affine:
        anchor = labels.mean(axis=0)
        coeff = np.linalg.solve(scaled.T @ scaled + .8 * np.eye(len(POSE_SCALE)), scaled.T @ (labels - anchor))
    else:
        anchor, coeff = np.median(labels, axis=0), np.zeros((len(POSE_SCALE), 3))
    return anchor, center, coeff


def fit(rows, column, batch):
    if len(rows) < 6:
        return (), {"observations": len(rows), "admitted": False}
    errors = np.zeros((3, 3))
    for held in rows:
        training = [row for row in rows if row["seed"] != held["seed"]]
        actual = np.asarray(held["release_anchor_m"])
        nominal = np.asarray(held["nominal_anchor_m"])
        pose = np.asarray(held["estimated_pose"])
        errors[0] += abs(nominal - actual)
        for index, affine in enumerate((False, True), start=1):
            anchor, center, coeff = regress(training, affine)
            corrected = nominal + np.clip(anchor + ((pose - center) / POSE_SCALE) @ coeff - nominal, -LIMIT, LIMIT)
            errors[index] += abs(corrected - actual)
    errors /= len(rows)
    choice = np.argmin(errors[1:], axis=0)
    active = errors[0] - errors[1 + choice, np.arange(3)] > MIN_GAIN_M
    median, ridge = (regress(rows, affine) for affine in (False, True))
    anchor = [float((median, ridge)[index][0][axis]) for axis, index in enumerate(choice)]
    coeff = np.column_stack([(median, ridge)[index][2][:, axis] for axis, index in enumerate(choice)])
    model = (Landing(column, batch, tuple(anchor), tuple(median[1]), tuple(map(tuple, coeff)), tuple(map(bool, active))),) if active.any() else ()
    return model, {"observations": len(rows), "admitted": bool(model), "mean_absolute_error_m": {
        name: errors[index].tolist() for index, name in enumerate(("nominal", "median", "ridge"))},
        "fit_by_axis": [("median", "ridge")[index] if enabled else "reference" for index, enabled in zip(choice, active, strict=True)]}


def dependencies():
    modules = (advanced_sim, benchmark, descent, history, recovery, servo, placement_search)
    return {module.__name__: sha256(Path(module.__file__)) for module in modules}


def run(root, release, recovery_archive, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected an aligned seed block and a positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint, timing = release / frozen["checkpoint"], tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*item) for item in frozen["corrections"])
    common = {"environment": simulator_identity(), "incumbent": frozen["corrections"], "timing": list(timing),
              "rules": RULES, "strict_rules": STRICT_RULES}
    if (common["environment"]["sim_revision"] != frozen["sim_revision"] or sha256(checkpoint) != frozen["checkpoint_sha256"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("expert fitting requires the frozen published checkpoint")
    late_recovery = history.frozen_recovery(recovery_archive, checkpoint, timing, incumbent, common)
    panels, fingerprints = archive_rows(recovery_archive, checkpoint, timing, common, late_recovery)
    known = {row["seed"]: row for group in panels.values() for row in group}
    column, batch, plan = sample_plan(panels)
    training = [seed for group in plan.values() for seed in group["failures"] + group["controls"]]
    development = [seed for seed in range(block, block + 64) if choose_column("random", seed) == column]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    archives = [*root.parent.glob("*/protocol.json"), *release.glob("*-protocol.json"), *recovery_archive.glob("*protocol.json")]
    if (len(known) != sum(map(len, panels.values())) or len(training) != len(set(training)) or reserved & set(known)
            or any(reserved & hover.scheduled_seeds(read_json(path).get("schedule", {})) for path in archives if path.parent != root)):
        raise ValueError("expert fitting requires fresh development and confirmation scenes")
    protocol = root / "protocol.json"
    fallback = [item.record() for item in late_recovery]
    _record_once(protocol, {**common, "code_sha256": sha256(Path(__file__)), "dependency_sha256": dependencies(),
                 "reference_sha256": sha256(manifest), "checkpoint_sha256": sha256(checkpoint), "late_recovery": fallback,
                 "source_sha256": fingerprints, "scope": [column, batch], "sampling": plan,
                 "schedule": {"training": training, "development": development, "confirmation": confirmation},
                 "confirmation_requires": routed.REQUIREMENTS})
    policy_type = partial(ExpertLandingPolicy, incumbent=incumbent, late_recovery=late_recovery)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent selected column {column}, batch {batch + 1} and is collecting {len(training)} expert demonstrations", flush=True)
        shards = root / "training" / "shards"
        shards.mkdir(parents=True, exist_ok=True)
        attempts = _jobs(pool, advanced_sim.collect_expert, [(seed, column, shards) for seed in training],
                         root / "training" / "attempts", lambda job: str(job[0]), _audit_expert)
        rows, rejected = training_rows(attempts, shards, known, checkpoint, timing, batch)
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"rows": rows, "attempts": attempts, "rejected": rejected})
        model, metrics = fit(rows, column, batch)
        fit_path = root / "fit.json"
        records = [item.record() for item in model]
        _record_once(fit_path, {**metrics, "corrections": records, "dataset_sha256": sha256(dataset)})
        if not model:
            return {"status": "reference_retained_after_expert_landing_fit", "fit": metrics}
        print(f"Agent selected a learned release position for {len(development)} unseen development pairs", flush=True)
        paired = benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                    {"reference": (), "selected": model}, policy_type=policy_type)
        statistics = hover.pair_stats(paired, model)
        selection = root / "development-selection.json"
        _record_once(selection, statistics)
        if not all(placement_search.passes(row) for row in statistics.values()):
            return {"status": "reference_retained_after_expert_landing_development", "development": statistics, "corrections": records}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "late_recovery": fallback,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection),
                     "seeds": confirmation, "corrections": records})
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                   confirmation[:4], policy_type=policy_type)
    if any(row["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for row in final.values()):
        raise ValueError("confirmation requires four-column balance")
    status = "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_expert_landing_confirmation"
    return {"status": status, "confirmation": final, "development": statistics, "corrections": records,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Fit a box-relative release anchor from independently collected expert attempts.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--recovery-archive", type=Path, default=Path("evidence/release-recovery-20261121"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=167200)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.recovery_archive.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
