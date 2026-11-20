from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_descent_agent as descent, placement_search, placement_wrist_rgb as wrist
from a3opt.adaptive_sim import phase_model
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, task_column
from a3opt.backends.advanced_phase import SAMPLES
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.placement_servo_agent import paired
from a3opt.placement_sim import PlacementPolicy
from a3opt.refinement_eval import read_json, sha256


FINISH_AT_FRACTION = .40
MAX_COMMAND_SHIFT = .009
MIN_TRAINING_BIAS = .002


@dataclass(frozen=True)
class Landing:
    column: int
    batch: int
    target_y_m: float

    def __post_init__(self):
        if self.column not in range(1, 5) or self.batch not in (0, 1) or not abs(self.target_y_m) <= .02:
            raise ValueError("invalid box-relative landing target")

    def record(self):
        return [self.column, self.batch, self.target_y_m]


def retarget(phases, observed, box, target_y_m):
    rotation = wrist.basis(box[4])
    solver = nominal_kinematics()
    center_xy = box[2:4]
    open_xy = descent.tool_xyz(solver, phases[OPEN][0, :7])[:2]
    previous_open_end = descent.tool_xyz(solver, phases[OPEN][-1, :7])[:2]
    original = (open_xy - center_xy) @ rotation
    goal = center_xy + rotation @ [original[0], target_y_m]
    shift = np.clip((goal - open_xy) @ rotation, -MAX_COMMAND_SHIFT, MAX_COMMAND_SHIFT)
    goal = open_xy + rotation @ shift
    retreat_shift = np.clip((goal - previous_open_end) @ rotation, -MAX_COMMAND_SHIFT, MAX_COMMAND_SHIFT)
    start = np.asarray(observed[:2], dtype=float)
    for phase in (PLACE, OPEN, RETRACT):
        actions = phases[phase]
        for frame, action in enumerate(actions, start=1):
            progress = frame / len(actions)
            if phase == RETRACT:
                correction = retreat_shift * (1. - _ramp(progress))
            else:
                fraction = _ramp(progress / FINISH_AT_FRACTION) if phase == PLACE else 1.
                xyz = descent.tool_xyz(solver, action[:7])
                target = start + fraction * (goal - start)
                correction_xy = (target - xyz[:2]) @ rotation
                correction = np.clip(correction_xy, -MAX_COMMAND_SHIFT, MAX_COMMAND_SHIFT)
            if np.any(correction):
                action[:7] = solver.shift(action[:7], np.r_[rotation @ correction, 0.])


class LandingPolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.landing = {(item.column, item.batch): item.target_y_m for item in corrections}
        if len(self.landing) != len(corrections):
            raise ValueError("duplicate box-relative landing target")
        super().__init__(checkpoint, focus, scale)

    def act(self, observation, task=""):
        if self.actions is not None and self.cursor is not None and self.cursor[1:] == (PLACE, 0):
            batch = self.cursor[0]
            key = task_column(task), batch
            if key in self.landing:
                retarget(self.actions[batch], observation["observation.eef_pose"], self.pose, self.landing[key])
        return super().act(observation, task)


def plan(checkpoint, rows):
    expert = phase_model(checkpoint)
    open_frame = sum(SAMPLES[:OPEN])
    solver = nominal_kinematics()
    translated = [descent.tool_xyz(solver, trace[open_frame, :7])[:2] - pose[2:4]
                  for trace, pose in zip(expert["traces"], expert["poses"], strict=True)]
    local_y = np.asarray([(delta @ wrist.basis(pose[4]))[1] for delta, pose in zip(translated, expert["poses"], strict=True)])
    plans = {}
    for column in range(1, 5):
        candidates = {}
        for batch in range(2):
            expert_rows = local_y[(expert["columns"] == column) & (expert["batches"] == batch)]
            if len(expert_rows) < 5:
                continue
            target = float(np.median(expert_rows))
            sensors = [(row, np.asarray(sample["sensors"])) for row in rows if row["column"] == column and len(row["samples"]) == 2
                       for sample in row["samples"] if sample["batch"] == batch]
            if len(sensors) < 5:
                continue
            errors = [(row["seed"], row["score"], float(values[1] * .04 + (.06 * values[2:4] @ wrist.basis(values[-1] * .08))[1] - target))
                      for row, values in sensors]
            bias = float(np.median([error for _, _, error in errors]))
            failures = sorted((score, -abs(error), seed) for seed, score, error in errors if score < 10)
            controls = sorted((-abs(error), seed) for seed, score, error in errors if score == 10)
            if abs(bias) >= MIN_TRAINING_BIAS and len(failures) >= 2 and len(controls) >= 2:
                candidates[batch] = {"bias_m": bias, "target_y_m": target, "failure_count": len(failures),
                                     "expert_samples": len(expert_rows), "wrist_samples": len(sensors),
                                     "seeds": [seed for _, _, seed in failures[:3]] + [seed for _, seed in controls[:3]]}
        if candidates:
            batch = max(candidates, key=lambda index: abs(candidates[index]["bias_m"]))
            plans[column] = {"batch": batch, **candidates[batch]}
    return plans


def run(root, reference, timing_run, acquisition, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("box-relative landing requires a balanced schedule and verified evaluation")
    development = list(range(block, block + 20))
    confirmation = list(range(block + 100, block + 140))
    protocol_path = acquisition / "protocol.json"
    source = read_json(protocol_path)
    reserved = source["schedule"]
    training = reserved["training"]
    schedule = {"scout": {"random": []}, "training": training, "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, schedule)
    if (set(development + confirmation).intersection(reserved["development"] + reserved["confirmation"]["random"])
            or any(source[key] != identity[key] for key in ("environment", "reference_sha256", "timing", "strict_rules"))
            or sha256(Path(wrist.__file__)) != source["code_sha256"]):
        raise ValueError("landing training requires audited, compatible wrist receipts and disjoint panels")
    dataset = acquisition / "training" / "dataset.json"
    rows = read_json(dataset)["rows"]
    dataset_hash = sha256(dataset)
    if [row["seed"] for row in rows] != training or dataset_hash != read_json(acquisition / "fit.json")["dataset_sha256"]:
        raise ValueError("wrist receipt list differs from the frozen training collection")
    for row in rows:
        wrist.audit((checkpoint, *timing, acquisition / "training" / "frames", row["seed"]), row)
    sampled = plan(checkpoint, rows)
    seeds = [seed for entry in sampled.values() for seed in entry["seeds"]]
    _record_once(root / "protocol.json", {**identity, "schedule": schedule, "source_dataset_sha256": dataset_hash,
                 "source_protocol_sha256": sha256(protocol_path), "code_sha256": sha256(Path(__file__)),
                 "source_wrist_code_sha256": sha256(Path(wrist.__file__)),
                 "descent_code_sha256": sha256(Path(descent.__file__)), "parity_sha256": sha256(parity),
                 "training_plan": {str(key): entry for key, entry in sampled.items()}, "replay_seeds": seeds,
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__)), "target_mean_strictly_above": placement_search.TARGET})
    if not sampled:
        return {"status": "reference_retained_without_training_support"}
    model = tuple(Landing(column, entry["batch"], entry["target_y_m"]) for column, entry in sampled.items())
    print(f"Reserved {len(seeds)} geometric landing training pairs for columns {sorted(sampled)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        data = placement_benchmark.evaluate(pool, root / "training", checkpoint, timing, seeds,
                                           {"reference": (), "target": model}, policy_type=LandingPolicy)
        stored = {row["seed"]: row for row in rows}
        for item in data["reference"]:
            strict = item["strict-500"]
            original = stored[strict["seed"]]
            if strict["action_sha256"] != original["action_sha256"] or strict["score"] != original["score"]:
                raise ValueError("geometric landing baseline did not reproduce its training acquisition")
        measures = {str(item.column): paired(data, "target", (item,), column=item.column) for item in model}
        selected = tuple(item for item in model if all(placement_search.passes(entry) and
                         sum(change["delta"] > 0 for change in entry["changes"]) >= 2 for entry in measures[str(item.column)].values()))
        fit_path = root / "fit.json"
        _record_once(fit_path, {"training": measures, "corrections": placement_search.encode(selected),
                                "dataset_sha256": sha256(root / "training" / "results.json")})
        print(f"Audited {len(seeds)} landing pairs; selected {placement_search.encode(selected)}", flush=True)
        if not selected:
            return {"status": "reference_retained_after_training"}
        rollouts = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                {"reference": (), "selected": selected}, policy_type=LandingPolicy)
        checked = paired(rollouts, "selected", selected)
        selection = root / "development-selection.json"
        _record_once(selection, checked)
        if not all(placement_search.passes(stats) for stats in checked.values()):
            return {"status": "reference_retained_after_development", "development": checked}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": placement_search.encode(selected),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(fit_path),
                     "selection_sha256": sha256(selection)})
        confirmed = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, selected,
                                                      confirmation[:4], policy_type=LandingPolicy)
    primary = confirmed["default"]
    accepted = primary["mean"] > placement_search.TARGET and all(placement_search.passes(stats) for stats in confirmed.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation", "development": checked,
            "confirmation": confirmed, "active_policy_above_target": accepted,
            "target_lower_bound": primary["mean_ci95"][0] > placement_search.TARGET}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_LANDING_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("box-relative landing requires the frozen reference checkpoint")
    return LandingPolicy(checkpoint, *lock["timing"], tuple(Landing(*item) for item in lock["corrections"]))


def main():
    parser = argparse.ArgumentParser(description="Learn a box-relative placement path from expert and public-observation training receipts.")
    for name in ("out", "reference", "timing", "parity", "acquisition"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed-block", type=int, default=137000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.acquisition.resolve(), args.parity.resolve(),
                     args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
