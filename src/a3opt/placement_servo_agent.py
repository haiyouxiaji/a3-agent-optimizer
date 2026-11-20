from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_descent_agent as descent, placement_search
from a3opt.advanced_agent import _record_once
from a3opt.backends import release_residual
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, task_column
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.placement_sim import PlacementPolicy
from a3opt.refinement_eval import read_json, sha256


MODES = {"vertical": (0., 0.), "servo": (.85, 0.), "servo-raised": (.85, .004)}
RULE_NAMES = ("default", "strict-500")
MAX_RESIDUAL_M = .003
DEADBAND_M = .0005
FRESH_PER_COLUMN = 2
UPRIGHT_ALIGNMENT = float(np.cos(np.deg2rad(15.)))


@dataclass(frozen=True)
class Servo:
    column: int
    batch: int
    mode: str

    def __post_init__(self):
        if self.column not in range(1, 5) or self.batch not in (0, 1) or self.mode not in MODES:
            raise ValueError("invalid placement servo")

    def record(self):
        return [self.column, self.batch, self.mode]


def horizontal_residual(anchor, observed, gain):
    error = anchor - observed
    if not np.all(np.isfinite(error)):
        return np.zeros(2)
    residual = gain * np.sign(error) * np.maximum(0., np.abs(error) - DEADBAND_M)
    length = np.linalg.norm(residual)
    return residual * min(1., MAX_RESIDUAL_M / max(length, MAX_RESIDUAL_M))


class ServoPolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.servo_modes = {(item.column, item.batch): MODES[item.mode] for item in corrections}
        if len(self.servo_modes) != len(corrections):
            raise ValueError("duplicate placement servo")
        super().__init__(checkpoint, focus, scale)

    def reset(self, context=None):
        super().reset(context)
        self.anchor = None
        self.residual = np.zeros(2)

    def act(self, observation, task=""):
        cursor = self.cursor if self.actions is not None else None
        mode = None
        if cursor is not None and cursor[1] in (PLACE, OPEN, RETRACT):
            batch, phase, frame = cursor
            mode = self.servo_modes.get((task_column(task), batch))
            if mode is not None and phase == PLACE and frame == 0:
                self.anchor = np.asarray(observation["observation.eef_pose"][:2], dtype=float).copy()
                self.residual = np.zeros(2)
                descent.adjust(self.actions[batch], observation["observation.eef_pose"], "vertical")
                if mode[1]:
                    self.shift_placement(batch, np.array([0., 0., mode[1]]))
        action = super().act(observation, task)
        if mode is None or not mode[0]:
            return action
        if phase == RETRACT:
            progress = (frame + 1) / len(self.actions[batch][RETRACT])
            residual = self.residual * (1. - release_residual._ramp(progress))
        else:
            observed = np.asarray(observation["observation.eef_pose"][:2], dtype=float)
            target = horizontal_residual(self.anchor, observed, mode[0])
            self.residual = .5 * (self.residual + target)
            residual = self.residual
        if np.any(residual):
            action[:7] = release_residual.nominal_kinematics().shift(action[:7], np.r_[residual, 0.])
        return action


def lost_during_descent(row, batch):
    stages = row["phase_geometry"][batch]
    if not all(stage in stages for stage in ("transfer", "descent")):
        return 0
    held = {item["id"] for item in stages["transfer"]["batches"][batch]
            if item["abs_vertical_axis"] >= UPRIGHT_ALIGNMENT and item["bottom_above_floor_m"] >= .035}
    tilted = {item["id"] for item in stages["descent"]["batches"][batch] if item["abs_vertical_axis"] < UPRIGHT_ALIGNMENT}
    return len((held & tilted) - set(row["terminal"]["counted_ids"]))


def acquisition_plan(rows):
    groups = {}
    for column in range(1, 5):
        members = [row for row in rows if row["column"] == column and not row["safety_reason"]]
        losses = [[(lost_during_descent(row, batch), row["seed"]) for row in members] for batch in range(2)]
        batch = max(range(2), key=lambda index: sum(loss for loss, _ in losses[index]))
        cases = sorted(((-loss, seed) for loss, seed in losses[batch] if loss))
        controls = sorted(row["seed"] for row in members if row["score"] == 10 and "transfer" in row["phase_geometry"][batch])
        if cases and len(controls) >= 2:
            groups[column] = {"batch": batch, "lost_cookies": -sum(loss for loss, _ in cases),
                              "targeted_seeds": [seed for _, seed in cases[:3]] + controls[:3]}
    columns = sorted(groups, key=lambda column: (-groups[column]["lost_cookies"], column))[:2]
    return {column: groups[column] for column in columns}


def paired(rows, name, model, *, control="reference", column=None):
    return {rule: placement_search.compare(
        [row[rule] for row in rows[control] if column is None or row[rule]["column"] == column],
        [row[rule] for row in rows[name] if column is None or row[rule]["column"] == column], model)
        for rule in RULE_NAMES}


def fit(plan, rows):
    selected, evidence = [], {}
    for column, entry in plan.items():
        eligible, report = {}, {}
        for mode in ("servo", "servo-raised"):
            model = (Servo(column, entry["batch"], mode),)
            stats = paired(rows, mode, model, column=column)
            ablation = paired(rows, mode, model, control="vertical", column=column)
            report[mode] = {"against_reference": stats, "against_open_loop_vertical": ablation}
            if all(placement_search.passes(result) and sum(change["delta"] > 0 for change in result["changes"]) >= 2
                   for result in stats.values()) and all(
                    result["paired_count_delta"] >= 0 and result["major_losses"] == 0
                    and result["complete"] >= result["reference_complete"] for result in ablation.values()):
                eligible[mode] = (min(result["paired_count_delta"] for result in stats.values()), stats["strict-500"]["complete"])
        winner = max(eligible, key=eligible.get, default=None)
        if winner:
            selected.append(Servo(column, entry["batch"], winner))
        evidence[str(column)] = {"winner": winner, "comparisons": report}
    return tuple(selected), evidence


def run(root, reference, timing_run, sources, parity, workers, block):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block % 4 or not sources or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("placement servo requires verified scoring and a balanced reserved panel")
    development = list(range(block + 100, block + 120))
    confirmation = list(range(block + 200, block + 240))
    fresh_pool = list(range(block, block + 4 * FRESH_PER_COLUMN))
    schedule = {"scout": {"random": []}, "training": fresh_pool, "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, schedule)
    data = [placement_search.reuse_training(source, checkpoint, timing, identity) for source in sources]
    history = [row for rows, _ in data for row in rows]
    prior_seeds = {row["seed"] for row in history}
    if len(prior_seeds) != len(history) or prior_seeds.intersection(range(block, block + 240)):
        raise ValueError("placement servo requires distinct training and evaluation scenes")
    plan = acquisition_plan(history)
    schedule["training"] = []
    for column, entry in plan.items():
        fresh = [seed for seed in fresh_pool if choose_column("random", seed) == column]
        entry["new_training_seeds"] = fresh
        schedule["training"].extend(entry["targeted_seeds"] + fresh)
    reference_setup(reference, timing_run, schedule)
    _record_once(root / "protocol.json", {**identity, "schedule": schedule, "sampling": {str(key): value for key, value in plan.items()},
                 "training_sources": [evidence for _, evidence in data], "modes": {key: list(value) for key, value in MODES.items()},
                 "residual_limit_m": MAX_RESIDUAL_M, "deadband_m": DEADBAND_M, "target_mean_strictly_above": placement_search.TARGET,
                 "parity_sha256": sha256(parity), "code_sha256": sha256(Path(__file__)),
                 "descent_code_sha256": sha256(Path(descent.__file__)), "ik_code_sha256": sha256(Path(release_residual.__file__)),
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__))})
    if not plan:
        return {"status": "reference_retained_without_training_support"}
    print(f"Reserved {len(schedule['training'])} placement-servo training scenes", flush=True)
    models = {"reference": (), **{mode: tuple(Servo(column, entry["batch"], mode) for column, entry in plan.items()) for mode in MODES}}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = placement_benchmark.evaluate(pool, root / "training", checkpoint, timing, schedule["training"], models, policy_type=ServoPolicy)
        selected, evidence = fit(plan, rows)
        fit_path = root / "fit.json"
        _record_once(fit_path, {"columns": evidence, "corrections": placement_search.encode(selected),
                                "dataset_sha256": sha256(root / "training" / "results.json")})
        print(f"Completed training comparisons and selected {placement_search.encode(selected)}", flush=True)
        if not selected:
            return {"status": "reference_retained_after_training"}
        rows = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                            {"reference": (), "selected": selected}, policy_type=ServoPolicy)
        checked = paired(rows, "selected", selected)
        selection = root / "development-selection.json"
        _record_once(selection, checked)
        if not all(placement_search.passes(stats) for stats in checked.values()):
            return {"status": "reference_retained_after_development", "development": checked}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": placement_search.encode(selected),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(fit_path),
                     "selection_sha256": sha256(selection)})
        confirmed = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, selected,
                                                      confirmation[:4], policy_type=ServoPolicy)
    primary = confirmed["default"]
    accepted = primary["mean"] > placement_search.TARGET and all(placement_search.passes(stats) for stats in confirmed.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation", "development": checked,
            "confirmation": confirmed, "active_policy_above_target": accepted,
            "target_lower_bound": primary["mean_ci95"][0] > placement_search.TARGET}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_SERVO_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("placement servo requires the frozen reference checkpoint")
    return ServoPolicy(checkpoint, *lock["timing"], tuple(Servo(*item) for item in lock["corrections"]))


def main():
    parser = argparse.ArgumentParser(description="Learn bounded online XY feedback from paired advanced placement trajectories.")
    for name in ("out", "reference", "timing", "parity"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed-block", type=int, default=134000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), [path.resolve() for path in args.source],
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
