from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark as benchmark, placement_descent_agent as descent
from a3opt import placement_expert_landing as landing, placement_release_recovery as recovery
from a3opt import placement_search, placement_servo_agent as servo, placement_source_guard as guard
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import PLACE, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES

from audit_placement_confirmation import scheduled_seeds
from confirm_expert_grasp import audit, measurements, resume
from confirm_source_guard import compare, inputs
from replicate_source_guard import RETAINED


RECIPES = ("preserve-curve", "align-before-lowering")


@dataclass(frozen=True)
class LandingPath:
    column: int
    batch: int
    recipe: str

    def __post_init__(self):
        if self.column not in range(1, 5) or self.batch not in (0, 1) or self.recipe not in RECIPES:
            raise ValueError("invalid learned landing path")

    def record(self):
        return [self.column, self.batch, self.recipe]


def plan_path(phases, frame, observation, pose, anchor, recipe):
    proposed = [phase.copy() for phase in phases]
    if not landing.adopt_anchor(proposed, frame, observation, pose, anchor):
        return None
    lowering = proposed[PLACE]
    original = phases[PLACE]
    hold = lowering[-1, :7].copy()
    count = len(lowering) - frame
    solver = nominal_kinematics()
    if recipe == "preserve-curve":
        blend = _ramp(np.arange(1, count + 1) / count)[:, None]
        lowering[frame:, :7] = np.clip(original[frame:, :7] + blend * (hold - original[-1, :7]), *solver.ranges.T)
    else:
        if count < 12:
            return None
        start = original[frame - 1, :7].copy()
        waypoint = descent.tool_xyz(solver, hold)
        waypoint[2] = descent.tool_xyz(solver, start)[2]
        alignment = recovery.joint_target(solver, start, waypoint)
        if alignment is None:
            return None
        align_steps = count // 3
        segments = []
        for first, last, steps in ((start, alignment, align_steps), (alignment, hold, count - align_steps)):
            weight = _ramp(np.arange(1, steps + 1) / steps)[:, None]
            segments.append(first + weight * (last - first))
        lowering[frame:, :7] = np.concatenate(segments)
    return proposed


class LandingPathPolicy(guard.SourceGuardPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent, late_recovery, baseline=()):
        self.path = next((item for item in corrections if isinstance(item, LandingPath)), None)
        corrections = (*baseline, *(item for item in corrections if not isinstance(item, LandingPath)))
        super().__init__(checkpoint, focus, scale, corrections, incumbent=incumbent, late_recovery=late_recovery)
        if self.path and (self.landing is None or (self.path.column, self.path.batch) != (self.landing.column, self.landing.batch)):
            raise ValueError("a landing path requires the frozen anchor for the same column and batch")

    def act(self, observation, task=""):
        if (self.path and self.actions is not None and self.cursor is not None and not self.retry_remaining
                and self.cursor[:2] == (self.path.batch, PLACE) and task_column(task) == self.path.column):
            phases = self.actions[self.path.batch]
            frame = max(1, len(phases[PLACE]) // 4)
            if self.cursor[2] == frame:
                updated = plan_path(phases, frame, observation, self.pose, self.landing, self.path.recipe)
                if updated is not None:
                    self.actions[self.path.batch] = updated
                    original = self.landing
                    self.landing = None
                    try:
                        return super().act(observation, task)
                    finally:
                        self.landing = original
        return super().act(observation, task)


def sample(panel, anchor):
    failures, controls = [], []
    for reference, episode in zip(panel["reference"], panel["selected"], strict=True):
        row = episode["strict-500"]
        if (row["column"] != anchor.column or row["safety_reason"] or row["score"] != row["requested_score"]
                or "descent" not in row["phase_geometry"][anchor.batch]):
            continue
        loss = servo.lost_during_descent({**row, "terminal": row["physical_terminal"]}, anchor.batch)
        gain = row["requested_score"] - reference["strict-500"]["requested_score"]
        if loss:
            failures.append((gain, -loss, row["seed"], episode))
        elif row["success"]:
            controls.append((-gain, row["seed"], episode))
    return [item[-1] for item in (*sorted(failures)[:3], *sorted(controls)[:3])]


def no_regression(stats):
    return (stats["paired_count_delta"] >= 0 and stats["complete"] >= stats["reference_complete"]
            and not (stats["major_losses"] or stats["safety_stops"] or stats["wrong_source"]))


def run(source, parent, output, seed_block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, timing, constructor, models, lock, _, selection = inputs(source, parent)
    audit(source, checkpoint, timing, models, lock, selection, compare_rows=compare, retained_status=RETAINED)
    anchor = next(item for item in models if isinstance(item, landing.Landing))
    receipt_paths = {name: root / "confirmation/rollouts/results.json" for name, root in (("parent", parent), ("source", source))}
    groups = {name: sample(read_json(path), anchor) for name, path in receipt_paths.items()}
    training = {name: [row["default"]["seed"] for row in rows] for name, rows in groups.items()}
    training_seeds = [seed for seeds in training.values() for seed in seeds]
    if any(len(seeds) < 3 for seeds in training.values()) or len(training_seeds) != len(set(training_seeds)):
        raise ValueError("landing tuning needs distinct failure and control receipts from both panels")
    development = [seed for seed in range(seed_block, seed_block + 48) if choose_column("random", seed) == anchor.column]
    confirmation = list(range(seed_block + 600, seed_block + 664))
    reserved = {*development, *confirmation}
    earlier = [*output.parent.glob("*/protocol.json"), *output.parent.glob("*/replication-plan.json")]
    if any(reserved & scheduled_seeds(read_json(path).get("schedule", {})) for path in earlier if path.parent != output):
        raise ValueError("landing tuning needs previously unused development and confirmation seeds")
    protocol = output / "protocol.json"
    options = {recipe: (LandingPath(anchor.column, anchor.batch, recipe),) for recipe in RECIPES}
    _record_once(protocol, {"environment": simulator_identity(), "rules": RULES, "strict_rules": STRICT_RULES,
                 "source_lock_sha256": sha256(source / "confirmation-lock.json"), "controller_sha256": sha256(checkpoint),
                 "code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(benchmark.__file__)),
                 "history_sha256": {name: sha256(path) for name, path in receipt_paths.items()},
                 "scope": [anchor.column, anchor.batch], "recipes": list(options),
                 "schedule": {"training": training, "development": development, "confirmation": confirmation}})
    baseline_type = partial(LandingPathPolicy, baseline=models, **constructor.keywords)
    confirmation_type = partial(LandingPathPolicy, **constructor.keywords)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent selected column {anchor.column}, batch {anchor.batch + 1} and {len(training_seeds)} training scenes", flush=True)
        trials = {}
        for name, seeds in training.items():
            folder = output / "training" / name
            outcomes = benchmark.evaluate(pool, folder, checkpoint, timing, seeds, options, policy_type=baseline_type)
            trials[name] = {recipe: measurements({"reference": groups[name], "selected": rows}, options[recipe], compare_rows=compare)
                            for recipe, rows in outcomes.items()}
        eligible = [recipe for recipe in options if all(no_regression(stats) for group in trials.values() for stats in group[recipe].values())]
        gains = {recipe: sum(group[recipe]["default"]["paired_count_delta"] * len(training[name]) for name, group in trials.items())
                 for recipe in eligible}
        choice = max(gains, key=gains.get, default=None)
        if choice is not None and gains[choice] <= 0:
            choice = None
        fit_path = output / "fit.json"
        _record_once(fit_path, {"trials": trials, "training_receipt_sha256": {
            name: sha256(output / "training" / name / "results.json") for name in groups}, "choice": choice})
        if choice is None:
            return {"status": "reference_retained_after_landing_path_fit", "training": trials}
        model = options[choice]
        print(f"Agent selected {choice} and is evaluating {len(development)} new development pairs", flush=True)
        paired = benchmark.evaluate(pool, output / "development", checkpoint, timing, development,
                                    {"reference": (), "selected": model}, policy_type=baseline_type)
        metrics = measurements(paired, model, compare_rows=compare)
        selection_path = output / "development-selection.json"
        _record_once(selection_path, metrics)
        if not all(placement_search.passes(stats) for stats in metrics.values()):
            return {"status": "reference_retained_after_landing_path_development", "development": metrics, "recipe": choice}
        corrections = (*models, *model)
        records = [item.record() for item in corrections]
        frozen = {**lock, "seeds": confirmation, "corrections": records, "protocol_sha256": sha256(protocol),
                  "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection_path)}
        _record_once(output / "confirmation-lock.json", frozen)
        parity_seeds = sorted({development[0], metrics["default"]["changes"][0]["seed"]})
        parity = benchmark.evaluate(pool, output / "calibration", checkpoint, timing, parity_seeds,
                                    {"reference": (), "selected": corrections}, policy_type=confirmation_type)
    if not (output / "result.json").is_file():
        resume(output, checkpoint, timing, confirmation_type, corrections, frozen, parity, metrics, workers,
               compare_rows=compare, parity_seeds=parity_seeds)
    return audit(output, checkpoint, timing, corrections, frozen, metrics, compare_rows=compare,
                 retained_status="reference_retained_after_landing_path_confirmation")


def main():
    parser = argparse.ArgumentParser(description="Learn how to approach a frozen release anchor using paired failure feedback.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=173000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    source, parent, output = args.source.resolve(), args.parent.resolve(), args.out.resolve()
    if output in (source, parent) or args.seed_block < 0 or args.seed_block % 4 or args.workers < 1:
        parser.error("expected separate output, an aligned seed block, and a positive worker budget")
    with _run_lock(output):
        result = run(source, parent, output, args.seed_block, args.workers)
        if "confirmation" not in result:
            _record_once(output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
