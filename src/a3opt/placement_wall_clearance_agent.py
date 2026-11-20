from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_compose, placement_descent_agent as descent
from a3opt import placement_first_hover as first, placement_hover_agent as hover, placement_hover_guard as guard
from a3opt import placement_routed_agent as routed, placement_search
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import PLACE, TRANSFER, task_column
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementPolicy
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


COLUMN = 1
OUTWARD = (1, -1)
RECIPES = {"inset4": (.004, 0.), "hover12": (0., .012), "inset4-hover12": (.004, .012)}
MAX_FAILURES = 5
MAX_CONTROLS = 3
DEVELOPMENT_CASES = 16
CONFIRMATION_CASES = 64


@dataclass(frozen=True)
class WallClearance:
    recipe: str
    cutoffs_m: tuple[float, float]
    column = COLUMN
    batch = 0

    def __post_init__(self):
        if self.recipe not in RECIPES or len(self.cutoffs_m) != 2 or not np.isfinite(self.cutoffs_m).all():
            raise ValueError("invalid wall clearance recipe")

    def record(self):
        return [self.column, self.recipe, list(self.cutoffs_m)]


class WallClearancePolicy(descent.DescentPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        if len(corrections) > 1:
            raise ValueError("expected one wall clearance policy")
        self.wall_clearance = corrections[0] if corrections else None
        super().__init__(checkpoint, focus, scale, incumbent)

    def act(self, observation, task=""):
        correction = self.wall_clearance
        if (correction is not None and self.actions is not None and self.cursor is not None
                and self.cursor[1:] == (PLACE, 0) and task_column(task) == COLUMN):
            batch = self.cursor[0]
            phases = self.actions[batch]
            projected = OUTWARD[batch] * guard.planned_x(self.pose, phases, TRANSFER)
            if projected >= correction.cutoffs_m[batch]:
                prior = self.profiles.get((COLUMN, batch))
                if prior is not None:
                    descent.adjust(phases, observation["observation.eef_pose"], prior)
                inset, rise = RECIPES[correction.recipe]
                self.shift_placement(batch, np.array([-OUTWARD[batch] * inset, 0., rise]))
                return PlacementPolicy.act(self, observation, task)
        return super().act(observation, task)


def labels(rows, checkpoint, timing, incumbent):
    controller = WallClearancePolicy(checkpoint, *timing, incumbent=incumbent)
    examples = []
    for row in rows:
        if row["column"] != COLUMN or row["safety_reason"] or row["score"] != row["requested_score"]:
            continue
        controller.pose = np.asarray(row["estimated_pose"])
        actions = controller._phase_actions(COLUMN, controller.recipes[COLUMN - 1])
        for batch, stages in enumerate(row["phase_geometry"]):
            moved = stages.get("transfer", {}).get("batches", [[], []])[batch]
            if len(moved) != 5 or min(cookie["abs_vertical_axis"] for cookie in moved) < .98:
                continue
            upright = min((cookie["abs_vertical_axis"] for name in ("descent", "open", "retract")
                           for cookie in stages.get(name, {}).get("batches", [[], []])[batch]), default=1.)
            lost = max(0, 5 - sum(cookie["scored"] for cookie in row["physical_terminal"]["batches"][batch]))
            examples.append({"seed": row["seed"], "batch": batch,
                             "projection_m": OUTWARD[batch] * guard.planned_x(controller.pose, actions[batch], TRANSFER),
                             "margin_m": min(cookie["footprint_margin_m"][0] for cookie in moved),
                             "lost": lost if upright < .97 else 0, "complete": row["success"]})
    return examples


def screen(examples):
    results = []
    for batch in range(2):
        samples = [example for example in examples if example["batch"] == batch]
        candidates = []
        if not samples:
            return ()
        for threshold in sorted({sample["projection_m"] for sample in samples if sample["lost"] > 0}):
            active = [sample for sample in samples if sample["projection_m"] >= threshold]
            positives = sum(sample["lost"] > 0 for sample in active)
            controls = sum(sample["complete"] for sample in active)
            if positives >= 2 and controls and positives >= controls:
                candidates.append({"cutoff_m": threshold, "failures": positives, "controls": controls,
                                   "lost": sum(sample["lost"] for sample in active), "active": len(active)})
        choice = max(candidates, key=lambda row: (row["lost"], -row["controls"], -row["active"], row["cutoff_m"]), default=None)
        if choice is None:
            return ()
        results.append(choice)
    return results


def sample_plan(examples, thresholds):
    failures, controls = {}, {}
    for row in examples:
        limit = thresholds[row["batch"]]
        if row["projection_m"] < limit:
            continue
        seed = row["seed"]
        if row["lost"]:
            failures[seed] = failures.get(seed, 0) + row["lost"]
        elif row["complete"]:
            controls[seed] = min(controls.get(seed, float("inf")), row["margin_m"])
    return {"failures": sorted(failures, key=lambda seed: (-failures[seed], seed))[:MAX_FAILURES],
            "controls": sorted(controls, key=lambda seed: (controls[seed], seed))[:MAX_CONTROLS]}


def read_training_source(source, common, checkpoint, timing):
    protocol_path = source / "protocol.json"
    protocol = read_json(protocol_path)
    folder = source / "confirmation" / "rollouts"
    seeds = protocol["schedule"]["replication"]["random"]
    selected = routed.decode(protocol["model"])
    trials, digest = placement_compose.source_rollouts(source, Path("confirmation/rollouts"), seeds, checkpoint, timing, selected)
    stored = read_json(source / "result.json")
    original = hover.pair_stats(trials, selected)
    if (any(protocol[key] != value for key, value in common.items())
            or protocol["reference_sha256"] != sha256(checkpoint)
            or protocol["evaluator_sha256"] != sha256(Path(placement_benchmark.__file__))
            or read_json(source / "confirmation" / "audit.json")["matched_native_replays"] != 8
            or stored["replication_results_sha256"] != digest
            or any(stored["replication"][rule][key] != value for rule, stats in original.items() for key, value in stats.items())
            or read_json(folder / "request.json")["seeds"] != seeds):
        raise ValueError("wall clearance training source does not match its paired receipts")
    return [row["strict-500"] for row in trials["reference"]], {"protocol_sha256": sha256(protocol_path), "rollout_sha256": digest}


def run(root, release, source, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected a balanced seed block and a positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint = release / frozen["checkpoint"]
    timing = tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*item) for item in frozen["corrections"])
    common = {"environment": simulator_identity(), "incumbent": frozen["corrections"], "timing": list(timing),
              "rules": RULES, "strict_rules": STRICT_RULES}
    if (common["environment"]["sim_revision"] != frozen["sim_revision"] or sha256(checkpoint) != frozen["checkpoint_sha256"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("wall clearance requires the frozen published controller")
    observed, source_hash = read_training_source(source, common, checkpoint, timing)
    prior, archive = first.historical_rows(release, (), frozen, common, checkpoint, timing, incumbent)
    rows = observed + prior
    examples = labels(rows, checkpoint, timing, incumbent)
    thresholds = screen(examples)
    diagnostic = root / "training" / "labels.json"
    _record_once(diagnostic, {"examples": examples, "threshold_search": thresholds, "source": source_hash, "archive": archive})
    if not thresholds:
        return {"status": "reference_retained_without_clearance_signal"}
    limits = tuple(row["cutoff_m"] for row in thresholds)
    plan = sample_plan(examples, limits)
    probes = plan["failures"] + plan["controls"]
    development = [seed for seed in range(block, block + 4 * DEVELOPMENT_CASES) if choose_column("random", seed) == COLUMN]
    confirmation = list(range(block + 400, block + 400 + CONFIRMATION_CASES))
    history = [source / "protocol.json", *(root.parent.glob("*/protocol.json"))]
    reserved = set(development + confirmation)
    if (reserved & {row["seed"] for row in rows} or any(path.parent != root and reserved & hover.scheduled_seeds(
            read_json(path).get("schedule", {})) for path in history)):
        raise ValueError("clearance development and confirmation require unseen scenes")
    protocol_path = root / "protocol.json"
    _record_once(protocol_path, {**common, "profile": "advanced", "reference_sha256": sha256(manifest),
                 "evaluator_sha256": frozen["evaluator_sha256"], "code_sha256": sha256(Path(__file__)),
                 "guard_code_sha256": sha256(Path(guard.__file__)), "labels_sha256": sha256(diagnostic), "plan": plan,
                 "schedule": {"training": sorted({row["seed"] for row in rows}), "development": development, "confirmation": confirmation},
                 "recipes": {name: list(values) for name, values in RECIPES.items()}, "confirmation_requires": routed.REQUIREMENTS})
    if min(map(len, plan.values())) < 2:
        return {"status": "reference_retained_without_training_support"}
    models = {name: (WallClearance(name, limits),) for name in RECIPES}
    policy_type = partial(WallClearancePolicy, incumbent=incumbent)
    print(f"Agent selected {len(probes)} placement failure and control scenes for full-episode training", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pairs = placement_benchmark.evaluate(pool, root / "training" / "rollouts", checkpoint, timing, probes,
                                             {"reference": (), **models}, policy_type=policy_type)
        expected = {row["seed"]: row for row in rows}
        for pair in pairs["reference"]:
            receipt = pair["strict-500"]
            if any(receipt[key] != expected[receipt["seed"]][key] for key in hover.MATCH_FIELDS):
                raise ValueError("the training reference did not replay its archived episode")
        summaries, candidates = {}, []
        for name, model in models.items():
            metrics = hover.pair_stats({"reference": pairs["reference"], "selected": pairs[name]}, model)
            eligible = all(placement_search.passes(stats) and len(stats["changes"]) >= 2
                           and all(change["delta"] > 0 for change in stats["changes"]) for stats in metrics.values())
            summaries[name] = {"metrics": metrics, "eligible": eligible}
            if eligible:
                candidates.append(name)
        chosen = max(candidates, key=lambda name: (summaries[name]["metrics"]["default"]["mean"], -RECIPES[name][1]), default=None)
        fit = root / "fit.json"
        model = models[chosen] if chosen else ()
        corrections = [item.record() for item in model]
        _record_once(fit, {"candidate": chosen, "corrections": corrections, "training": summaries,
                           "dataset_sha256": sha256(root / "training" / "rollouts" / "results.json")})
        if not model:
            return {"status": "reference_retained_after_training"}
        print(f"Agent selected {chosen} for {len(development)} new development pairs", flush=True)
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                              {"reference": (), "selected": model}, policy_type=policy_type)
        metrics = hover.pair_stats(paired, model)
        selection = root / "development-selection.json"
        _record_once(selection, metrics)
        if not all(placement_search.passes(stats) for stats in metrics.values()):
            return {"status": "reference_retained_after_development", "development": metrics}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "corrections": corrections, "seeds": confirmation,
                     "protocol_sha256": sha256(protocol_path), "fit_sha256": sha256(fit), "selection_sha256": sha256(selection)})
        print(f"Candidate frozen before {len(confirmation)} new four-column pairs", flush=True)
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                   model, confirmation[:4], policy_type=policy_type)
    balance = {str(column): CONFIRMATION_CASES // 4 for column in range(1, 5)}
    if any(stats["scored_pairs_by_column"] != balance for stats in final.values()):
        raise ValueError("confirmation requires four-column balance")
    return {"status": "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_confirmation",
            "development": metrics, "confirmation": final}


def main():
    parser = argparse.ArgumentParser(description="Learn a bounded release residual near target-box walls.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=149000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.source.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
