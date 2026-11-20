from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_contact_feedback as contact, placement_descent_agent as descent
from a3opt import placement_first_hover as first, placement_hover_agent as hover, placement_routed_agent as routed
from a3opt import placement_search, placement_wall_clearance_agent as wall
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementPolicy
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


RECIPES = {"align-then-lower": (.35, 0., 0.), "align-then-hover": (.35, .010, 0.),
           "inset-then-hover": (.45, .014, .004)}
MAX_SHIFT = np.array([.012, .012, .028])


@dataclass(frozen=True)
class StagedEntry:
    column: int
    batch: int
    recipe: str

    def __post_init__(self):
        if self.column not in range(1, 5) or self.batch not in range(2) or self.recipe not in RECIPES:
            raise ValueError("invalid staged descent correction")

    def record(self):
        return [self.column, self.batch, self.recipe]


def entry_path(start, target, progress, alignment_fraction):
    horizontal = _ramp(progress / alignment_fraction)
    vertical = _ramp((progress - alignment_fraction) / (1. - alignment_fraction))
    return start + np.array([horizontal, horizontal, vertical]) * (target - start)


def replan(policy, observation, item):
    fraction, height, inset = RECIPES[item.recipe]
    phases = policy.actions[item.batch]
    solver = nominal_kinematics()
    start = np.asarray(observation["observation.eef_pose"][:3], dtype=float)
    if not np.isfinite(start).all():
        return False
    yaw = policy.pose[4]
    inward = -wall.OUTWARD[item.batch] * inset
    offset = np.array([inward * np.cos(yaw), inward * np.sin(yaw), height])
    target = descent.tool_xyz(solver, phases[PLACE][-1, :7]) + offset
    for index, action in enumerate(phases[PLACE], start=1):
        progress = index / len(phases[PLACE])
        position = descent.tool_xyz(solver, action[:7])
        displacement = entry_path(start, target, progress, fraction) - position
        action[:7] = solver.shift(action[:7], np.clip(displacement, -MAX_SHIFT, MAX_SHIFT))
    for action in phases[OPEN]:
        displacement = target - descent.tool_xyz(solver, action[:7])
        action[:7] = solver.shift(action[:7], np.clip(displacement, -MAX_SHIFT, MAX_SHIFT))
    for index, action in enumerate(phases[RETRACT], start=1):
        weight = 1. - _ramp(index / len(phases[RETRACT]))
        if weight and np.any(offset):
            action[:7] = solver.shift(action[:7], weight * offset)
    return True


class StagedEntryPolicy(descent.DescentPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        self.staged = {(item.column, item.batch): item for item in corrections}
        if len(self.staged) != len(corrections):
            raise ValueError("duplicate staged descent correction")
        super().__init__(checkpoint, focus, scale, incumbent)

    def act(self, observation, task=""):
        if self.actions is not None and self.cursor is not None and self.cursor[1:] == (PLACE, 0) and not self.retry_remaining:
            item = self.staged.get((task_column(task), self.cursor[0]))
            if item is not None and replan(self, observation, item):
                return PlacementPolicy.act(self, observation, task)
        return super().act(observation, task)


def training_history(release, source, frozen, common, checkpoint, timing, incumbent):
    published, fingerprints = first.historical_rows(release, (), frozen, common, checkpoint, timing, incumbent)
    pairs = []
    for filename in fingerprints:
        pairs.extend(json.loads(gzip.decompress((release / filename).read_bytes()))["selected"])
    if [pair["strict-500"] for pair in pairs] != published:
        raise ValueError("published paired rows differ from the verified training history")
    if source is not None:
        earlier, fingerprints["routed_replication"] = wall.read_training_source(source, common, checkpoint, timing)
        routed_rows = read_json(source / "confirmation/rollouts/results.json")["reference"]
        if [pair["strict-500"] for pair in routed_rows] != earlier:
            raise ValueError("routed paired rows differ from the verified training history")
        pairs.extend(routed_rows)
    indexed = {pair["strict-500"]["seed"]: pair for pair in pairs}
    if len(indexed) != len(pairs):
        raise ValueError("duplicate archived reference seed")
    return indexed, fingerprints


def choose(metrics):
    eligible = [name for name, pair in metrics.items() if all(placement_search.passes(stats)
        and sum(change["delta"] > 0 for change in stats["changes"]) >= 2
        and all(change["delta"] >= -1 for change in stats["changes"]) for stats in pair.values())]
    return max(eligible, key=lambda name: (metrics[name]["strict-500"]["mean"], -RECIPES[name][1]), default=None)


def run(root, release, source, block, workers):
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
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("staged descent requires the frozen published controller")
    previous, sources = training_history(release, source, frozen, common, checkpoint, timing, incumbent)
    plan = contact.sample_plan([pair["strict-500"] for pair in previous.values()])
    columns = {item["column"] for item in plan}
    development = [seed for seed in range(block, block + 64) if choose_column("random", seed) in columns]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    protocols = list(root.parent.glob("*/protocol.json"))
    if source is not None:
        protocols.append(source / "protocol.json")
    if reserved & set(previous) or any(path.parent != root and reserved & hover.scheduled_seeds(
            read_json(path).get("schedule", {})) for path in protocols):
        raise ValueError("staged descent development and confirmation require unused seeds")
    training = [seed for group in plan for seed in group["failures"] + group["controls"]]
    protocol = root / "protocol.json"
    _record_once(protocol, {**common, "reference_sha256": sha256(manifest), "sources": sources, "sampling": plan,
                 "schedule": {"training": training, "development": development, "confirmation": confirmation},
                 "code_sha256": sha256(Path(__file__)), "recipes": {name: list(values) for name, values in RECIPES.items()},
                 "confirmation_requires": routed.REQUIREMENTS})
    if not plan:
        return {"status": "reference_retained_without_descent_cases"}
    policy_type = partial(StagedEntryPolicy, incumbent=incumbent)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        decisions, admitted, groups = {}, [], {}
        print(f"Agent testing staged descent on {len(training)} historical failures and controls", flush=True)
        for group in plan:
            column, batch = group["column"], group["batch"]
            seeds = group["failures"] + group["controls"]
            models = {name: (StagedEntry(column, batch, name),) for name in RECIPES}
            folder = root / "training" / str(column)
            trials = placement_benchmark.evaluate(pool, folder, checkpoint, timing, seeds, models, policy_type=policy_type)
            reference = [previous[seed] for seed in seeds]
            metrics = {name: hover.pair_stats({"reference": reference, "selected": trials[name]}, model)
                       for name, model in models.items()}
            recipe = choose(metrics)
            decisions[str(column)] = {"selected": recipe, "metrics": metrics}
            groups[str(column)] = {"seeds": seeds, "reference": reference, "trials_sha256": sha256(folder / "results.json")}
            if recipe is not None:
                admitted.extend(models[recipe])
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"groups": groups})
        model = tuple(admitted)
        records = [item.record() for item in model]
        fit = root / "fit.json"
        _record_once(fit, {"corrections": records, "decisions": decisions, "dataset_sha256": sha256(dataset)})
        if not model:
            return {"status": "reference_retained_after_staged_training", "decisions": decisions}
        development = [seed for seed in development if choose_column("random", seed) in {item.column for item in model}]
        print(f"Agent selected {len(model)} staged descents for {len(development)} unseen pairs", flush=True)
        measurements = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                    {"reference": (), "selected": model}, policy_type=policy_type)
        scores = hover.pair_stats(measurements, model)
        selection = root / "development-selection.json"
        _record_once(selection, scores)
        if not all(placement_search.passes(stats) for stats in scores.values()):
            return {"status": "reference_retained_after_staged_development", "development": scores, "corrections": records}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "seeds": confirmation, "corrections": records,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit), "selection_sha256": sha256(selection)})
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                   model, confirmation[:4], policy_type=policy_type)
    if any(stats["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for stats in final.values()):
        raise ValueError("confirmation requires four-column balance")
    return {"status": "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_staged_confirmation",
            "development": scores, "confirmation": final, "corrections": records,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Train placement paths that align above the tray before lowering.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=154000)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.source.resolve() if args.source else None, args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
