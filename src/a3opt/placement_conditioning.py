from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from a3opt import placement_benchmark, placement_sim
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import APPROACH, PLACE, Recipe
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.placement_search import MAJOR_LOSS, TARGET, compare, confirm_combined, encode, passes, reuse_training
from a3opt.refinement_eval import read_json, sha256


@dataclass(frozen=True)
class Conditioning:
    column: int
    batch: int

    def __post_init__(self):
        if self.column not in range(1, 5) or self.batch not in (0, 1):
            raise ValueError("invalid joint placement intervention")

    def record(self):
        return [self.column, self.batch]


class ConditioningPolicy(placement_sim.PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.conditioning = frozenset((item.column, item.batch) for item in corrections)
        if len(self.conditioning) != len(corrections):
            raise ValueError("duplicate joint placement intervention")
        super().__init__(checkpoint, focus, scale)

    def _phase_actions(self, column: int, recipe: Recipe):
        original = super()._phase_actions(column, recipe)
        batches = {batch for batch in range(2) if (column, batch) in self.conditioning}
        if recipe.joint_placement or not batches:
            return original
        alternate = super()._phase_actions(column, replace(recipe, joint_placement=True))
        for batch in batches:
            original[batch][PLACE:] = alternate[batch][PLACE:]
        if 0 in batches:
            original[1][APPROACH] = alternate[1][APPROACH]
        return original


def train_models(pool, folder, checkpoint, timing, rows):
    models, comparisons = [], {}
    for column in range(1, 5):
        controls = [row for row in rows if row["column"] == column and not row["safety_reason"]]
        if len(controls) < 4 or sum(row["score"] < 10 for row in controls) < 2:
            continue
        seeds = [row["seed"] for row in controls]
        variants = {"first": (Conditioning(column, 0),), "second": (Conditioning(column, 1),),
                    "both": (Conditioning(column, 0), Conditioning(column, 1))}
        observed = placement_benchmark.evaluate(pool, folder / str(column), checkpoint, timing, seeds, variants,
                                                policy_type=ConditioningPolicy)
        scores = {}
        for name, model in variants.items():
            changes, violations = [], 0
            for control, combined in zip(controls, observed[name], strict=True):
                row = combined["strict-500"]
                batch = model[0].batch
                before = control["phase_geometry"][batch].get("transfer", {}).get("prefix_sha256")
                after = row["phase_geometry"][batch].get("transfer", {}).get("prefix_sha256")
                if row["seed"] != control["seed"] or before != after:
                    raise ValueError("joint placement comparison changed the recorded transfer prefix")
                changes.append(row["requested_score"] - control["score"])
                violations += bool(row["safety_reason"] or row["requested_score"] != row["score"])
            scores[name] = {"gain": sum(changes), "gains": sum(delta > 0 for delta in changes),
                            "major_losses": sum(delta <= -MAJOR_LOSS for delta in changes), "violations": violations}
        admitted = [name for name, score in scores.items() if score["gains"] >= 2 and score["gain"] > 0
                    and not (score["major_losses"] or score["violations"])]
        selected = max(admitted, key=lambda name: (scores[name]["gain"], -len(variants[name]))) if admitted else None
        if selected:
            models.extend(variants[selected])
        comparisons[str(column)] = {"seeds": seeds, "scores": scores, "selected": selected}
    return tuple(models), comparisons


def run(root, reference, timing_run, sources, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and native evaluation audit are required")
    development = list(range(block, block + 12))
    confirmation = list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    stored = [reuse_training(source, checkpoint, timing, identity) for source in sources]
    rows = [row for group, _ in stored for row in group]
    splits["training"] = [row["seed"] for row in rows]
    if len(set(splits["training"])) != len(rows) or set(splits["training"]).intersection(development + confirmation):
        raise ValueError("conditioning requires disjoint training and evaluation seeds")
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "sources": [src for _, src in stored],
                 "training_code_sha256": sha256(Path(__file__)), "native_parity_sha256": sha256(parity),
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__)), "target": TARGET})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        model, fit = train_models(pool, root / "training", checkpoint, timing, rows)
        _record_once(root / "fit.json", {"corrections": encode(model), "columns": fit})
        print(f"Fitted placement conditioning with {len(model)} phase interventions", flush=True)
        if not model:
            return {"status": "reference_retained_after_training"}
        outcomes = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                {"reference": (), "selected": model}, policy_type=ConditioningPolicy)
        scores = {rule: compare([row[rule] for row in outcomes["reference"]], [row[rule] for row in outcomes["selected"]], model)
                  for rule in ("default", "strict-500")}
        _record_once(root / "development-selection.json", scores)
        if not all(passes(stats) for stats in scores.values()):
            return {"status": "reference_retained_after_development", "development": scores}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        verified = confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                    confirmation[:4], policy_type=ConditioningPolicy)
    confirmed = all(passes(stats) for stats in verified.values())
    return {"status": "candidate_promoted" if confirmed else "reference_retained_after_confirmation",
            "active_policy_above_target": confirmed and verified["default"]["mean"] > TARGET,
            "development": scores, "confirmation": verified}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_CONDITIONING_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("joint placement requires the frozen checkpoint")
    return ConditioningPolicy(checkpoint, *lock["timing"], tuple(Conditioning(*item) for item in lock["corrections"]))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate source-aware placement trajectory conditioning.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-block", type=int, default=115000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), [source.resolve() for source in args.source],
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
