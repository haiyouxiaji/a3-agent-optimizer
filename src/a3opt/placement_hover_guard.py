from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_compose, placement_descent_agent as descent
from a3opt import placement_first_hover as first, placement_hover_agent as hover, placement_search
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import PLACE, TRANSFER, task_column
from a3opt.backends.release_residual import nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import projected_tool_x
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


FEATURES = {"transfer_x": TRANSFER, "descent_x": PLACE}
RULE_NAMES = ("default", "strict-500")
NEAR_EDGE_CASES = 8
DEVELOPMENT_CASES = 16
CONFIRMATION_CASES = 64


@dataclass(frozen=True)
class HoverGate:
    column: int
    recipe: str
    feature: str
    cutoff_m: float
    batch = first.BATCH

    def __post_init__(self):
        first.FirstHover(self.column, self.recipe)
        if self.feature not in FEATURES or not np.isfinite(self.cutoff_m):
            raise ValueError("invalid first-batch hover activation rule")

    def record(self):
        return [self.column, self.batch, self.recipe, self.feature, self.cutoff_m]


def planned_x(pose, actions, stage):
    tool = descent.tool_xyz(nominal_kinematics(), actions[stage][-1, :7])
    return projected_tool_x(tool, pose)


class GatedHoverPolicy(first.FirstHoverPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        if len(corrections) > 1:
            raise ValueError("expected one guarded first-batch recipe")
        self.gate = corrections[0] if corrections else None
        super().__init__(checkpoint, focus, scale, corrections, incumbent=incumbent)

    def act(self, observation, task=""):
        if (self.gate is not None and self.actions is not None and self.cursor == (first.BATCH, PLACE, 0)
                and task_column(task) == self.gate.column):
            value = planned_x(self.pose, self.actions[first.BATCH], FEATURES[self.gate.feature])
            if value < self.gate.cutoff_m:
                return descent.DescentPolicy.act(self, observation, task)
        return super().act(observation, task)


def label(row):
    stages = row["phase_geometry"][first.BATCH]
    transfer = stages.get("transfer", {}).get("batches", [[]])[first.BATCH]
    if len(transfer) != 5 or min(cookie["abs_vertical_axis"] for cookie in transfer) < .98:
        return False
    end = stages.get("retract", {}).get("batches", [[]])[first.BATCH]
    return sum(cookie["scored"] for cookie in end) < 5 and any(
        cookie["abs_vertical_axis"] < .97 for stage in ("descent", "open")
        for cookie in stages.get(stage, {}).get("batches", [[]])[first.BATCH])


def features(checkpoint, timing, policy_type, column, rows):
    policy = policy_type(checkpoint, *timing)
    output = []
    for entry in rows:
        if entry["column"] != column or entry["safety_reason"] or entry["score"] != entry["requested_score"]:
            continue
        policy.pose = np.asarray(entry["estimated_pose"])
        actions = policy._phase_actions(column, policy.recipes[column - 1])[first.BATCH]
        output.append({"seed": entry["seed"], "tilt_loss": label(entry),
                       **{name: float(planned_x(policy.pose, actions, stage)) for name, stage in FEATURES.items()}})
    return output


def acquisition(features_by_scene, used):
    remaining = [row for row in features_by_scene if row["seed"] not in used]
    ordered = sorted(remaining, key=lambda row: (-max(row[name] for name in FEATURES), row["seed"]))
    return [row["seed"] for value in (True, False) for row in
            [entry for entry in ordered if entry["tilt_loss"] == value][:NEAR_EDGE_CASES]]


def paired_changes(reference, selected):
    changes = {}
    for seed in sorted(reference):
        controls, trials = reference[seed], selected[seed]
        delta = [trials[rule]["requested_score"] - controls[rule]["requested_score"] for rule in RULE_NAMES]
        complete = [int(trials[rule]["success"]) - int(controls[rule]["success"]) for rule in RULE_NAMES]
        unsafe = any(trials[rule]["safety_reason"] or trials[rule]["score"] != trials[rule]["requested_score"] for rule in RULE_NAMES)
        changes[seed] = {"delta": delta, "complete": complete, "unsafe": bool(unsafe)}
    return changes


def fit(points, outcomes, item):
    trials = []
    for feature in FEATURES:
        thresholds = [float(value / 2000) for value in range(2 * int(1000 * min(row[feature] for row in points)),
                      2 * int(1000 * max(row[feature] for row in points)) + 1)]
        for cutoff in thresholds:
            active = [row for row in points if row[feature] >= cutoff]
            sampled = [outcomes[row["seed"]] for row in active if row["seed"] in outcomes]
            gains = sum(min(row["delta"]) > 0 for row in sampled)
            losses = sum(min(row["delta"] + row["complete"]) < 0 or row["unsafe"] for row in sampled)
            detected = sum(row["tilt_loss"] for row in active)
            controls = len(active) - detected
            eligible = gains >= 2 and not losses and detected >= 3 and detected >= 2 * controls
            trials.append({"feature": feature, "cutoff_m": cutoff, "paired_gains": gains, "paired_losses": losses,
                           "paired_effect": sum(min(row["delta"]) for row in sampled), "detected_tilts": detected,
                           "other_activations": controls, "eligible": eligible})
    best = max((row for row in trials if row["eligible"]), default=None,
               key=lambda row: (row["paired_effect"], row["paired_gains"], row["detected_tilts"], -row["other_activations"], row["cutoff_m"]))
    model = () if best is None else (HoverGate(item.column, item.recipe, best["feature"], best["cutoff_m"]),)
    return model, trials


def run(root, release, sources, previous, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected aligned seeds and a positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint = release / frozen["checkpoint"]
    checkpoint_hash = sha256(checkpoint)
    timing = tuple(frozen["timing"])
    environment = simulator_identity()
    incumbent = tuple(descent.Descent(*entry) for entry in frozen["corrections"])
    common = {"environment": environment, "timing": list(timing), "incumbent": frozen["corrections"],
              "rules": RULES, "strict_rules": STRICT_RULES, "reference_sha256": sha256(manifest)}
    if (checkpoint_hash != frozen["checkpoint_sha256"] or environment["sim_revision"] != frozen["sim_revision"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("the installed controller differs from the published reference")
    prior_protocol, prior_fit, source_hashes = placement_compose.source_refinement(previous, first, common)
    records = prior_fit["corrections"]
    if len(records) != 1 or records[0][1] != first.BATCH:
        raise ValueError("this experiment expects a single first-batch candidate")
    item = first.FirstHover(records[0][0], records[0][2])
    earlier = [seed for seed in prior_protocol["schedule"]["development"] if choose_column("random", seed) == item.column]
    paired, prior_rollouts = placement_compose.source_rollouts(previous, Path("development"), earlier, checkpoint, timing, (item,))
    history, hashes = first.historical_rows(release, sources, frozen, common, checkpoint, timing, incumbent)
    rows = history + [entry["strict-500"] for entry in paired["reference"]]
    known = {entry["seed"]: entry for entry in rows}
    if len(known) != len(rows):
        raise ValueError("guard fitting requires unique historical seeds")
    base_policy = partial(first.FirstHoverPolicy, incumbent=incumbent)
    points = features(checkpoint, timing, base_policy, item.column, rows)
    training = acquisition(points, set(earlier))
    development = [seed for seed in range(block, block + 4 * DEVELOPMENT_CASES) if choose_column("random", seed) == item.column]
    confirmation = list(range(block + 400, block + 400 + CONFIRMATION_CASES))
    reserved = set(development + confirmation)
    paths = [*(source / "protocol.json" for source in sources), *(root.parent.glob("*/protocol.json"))]
    if reserved & set(known) or any(path.parent != root and reserved & hover.scheduled_seeds(
            read_json(path).get("schedule", {})) for path in paths):
        raise ValueError("guard evaluation requires previously unused seeds")
    protocol_path = root / "protocol.json"
    _record_once(protocol_path, {**common, "profile": "advanced", "code_sha256": sha256(Path(__file__)),
                 "first_hover_code_sha256": sha256(Path(first.__file__)), "evaluator_sha256": frozen["evaluator_sha256"],
                 "schedule": {"training": sorted(row["seed"] for row in points), "training_replays": training,
                              "development": development, "confirmation": confirmation}, "historical_rollouts": hashes,
                 "previous": {**source_hashes, "rollout_sha256": prior_rollouts}, "unconditional": item.record(),
                 "confirmation_requires": ["course_mean_above_9", "positive_lower_bound_on_paired_gain_both_rules",
                                           "no_major_losses", "no_safety_or_wrong_source", "no_success_count_regression"]})
    print(f"Full-episode training on {len(training)} selected historical scenes and {len(earlier)} completed pairs", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        training_data = placement_benchmark.evaluate(pool, root / "training" / "rollouts", checkpoint, timing, training,
                                                    {"reference": (), "selected": (item,)}, policy_type=base_policy)
        shared_metrics = hover.pair_stats(training_data, (item,))
        for entry in training_data["reference"]:
            replay = entry["strict-500"]
            if any(replay[key] != known[replay["seed"]][key] for key in hover.MATCH_FIELDS):
                raise ValueError("training comparison does not reproduce the historical reference")
        controls, trials = ({entry["default"]["seed"]: entry for entry in paired[name] + training_data[name]} for name in ("reference", "selected"))
        outcome = paired_changes(controls, trials)
        dataset_path = root / "training" / "dataset.json"
        _record_once(dataset_path, {"points": points, "paired_outcomes": {str(seed): values for seed, values in outcome.items()},
                      "replay_metrics": shared_metrics, "rollouts_sha256": sha256(root / "training" / "rollouts" / "results.json")})
        model, options = fit(points, outcome, item)
        fit_path = root / "fit.json"
        corrections = [entry.record() for entry in model]
        _record_once(fit_path, {"corrections": corrections, "options": options, "dataset_sha256": sha256(dataset_path)})
        if not model:
            return {"status": "reference_retained_after_training"}
        print(f"Frozen fitted guard {corrections}; evaluating {len(development)} fresh development pairs", flush=True)
        policy_type = partial(GatedHoverPolicy, incumbent=incumbent)
        evaluated = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                  {"reference": (), "selected": model}, policy_type=policy_type)
        progress = hover.pair_stats(evaluated, model)
        selection = root / "development-selection.json"
        _record_once(selection, progress)
        if not all(placement_search.passes(stats) for stats in progress.values()):
            return {"status": "reference_retained_after_development", "development": progress}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                     "timing": list(timing), "incumbent": frozen["corrections"], "corrections": corrections, "seeds": confirmation,
                     "protocol_sha256": sha256(protocol_path), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)})
        print(f"Guarded candidate frozen before {len(confirmation)} fresh four-column confirmation pairs", flush=True)
        confirmed = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                       model, confirmation[:4], policy_type=policy_type)
    balance = Counter({str(column): CONFIRMATION_CASES // 4 for column in range(1, 5)})
    if any(Counter(row["scored_pairs_by_column"]) != balance for row in confirmed.values()):
        raise ValueError("guard confirmation is not balanced across all four source columns")
    return {"status": "candidate_promoted" if first.supports_confirmation(confirmed) else "reference_retained_after_confirmation",
            "development": progress, "confirmation": confirmed}


def main():
    parser = argparse.ArgumentParser(description="Fit a first-batch hover activation threshold using full simulator episodes.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--first-hover", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=145000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    folder = args.out.resolve()
    with _run_lock(folder):
        result = run(folder, args.release.resolve(), [source.resolve() for source in args.source], args.first_hover.resolve(),
                     args.seed_block, args.workers)
        _record_once(folder / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
