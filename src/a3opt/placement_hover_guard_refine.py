from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_compose, placement_descent_agent as descent, placement_search
from a3opt import placement_first_hover as first, placement_hover_agent as hover, placement_hover_guard as guard
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import PLACE, TRANSFER, task_column
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


@dataclass(frozen=True)
class PersistentProjection:
    column: int
    recipe: str
    cutoff_m: float
    batch = first.BATCH

    def __post_init__(self):
        first.FirstHover(self.column, self.recipe)
        if not np.isfinite(self.cutoff_m):
            raise ValueError("invalid projection cutoff")

    def record(self):
        return [self.column, self.batch, self.recipe, self.cutoff_m]


class PersistentProjectionPolicy(first.FirstHoverPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        if len(corrections) > 1:
            raise ValueError("expected one projection cutoff")
        self.projection = corrections[0] if corrections else None
        super().__init__(checkpoint, focus, scale, corrections, incumbent=incumbent)

    def act(self, observation, task=""):
        gate = self.projection
        if gate is not None and self.actions is not None and self.cursor == (first.BATCH, PLACE, 0) and task_column(task) == gate.column:
            if min(guard.planned_x(self.pose, self.actions[first.BATCH], stage) for stage in (TRANSFER, PLACE)) < gate.cutoff_m:
                return descent.DescentPolicy.act(self, observation, task)
        return super().act(observation, task)


def fit(points, outcomes, prior):
    candidates = []
    high = max(min(row[name] for name in guard.FEATURES) for row in points)
    start, stop = int(np.ceil(prior.cutoff_m * 2000)), int(np.floor(high * 2000))
    for index in range(start, stop + 1):
        cutoff = index / 2000
        active = [row for row in points if min(row[name] for name in guard.FEATURES) >= cutoff]
        measured = [outcomes[row["seed"]] for row in active if row["seed"] in outcomes]
        gain = sum(min(row["delta"]) > 0 for row in measured)
        adverse = sum(row["unsafe"] or min(row["delta"] + row["complete"]) < 0 for row in measured)
        tilts = sum(row["tilt_loss"] for row in active)
        candidates.append({"cutoff_m": cutoff, "improved": gain, "adverse": adverse, "tilts": tilts,
                           "exposed": len(active), "gain": sum(min(row["delta"]) for row in measured),
                           "eligible": gain >= 3 and not adverse and tilts >= 2 * (len(active) - tilts)})
    choice = max((row for row in candidates if row["eligible"]), default=None,
                 key=lambda row: (row["gain"], row["improved"], -row["exposed"], row["cutoff_m"]))
    return (() if choice is None else (PersistentProjection(prior.column, prior.recipe, choice["cutoff_m"]),)), candidates


def run(root, release, previous, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected aligned seeds and a positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint = release / frozen["checkpoint"]
    timing = tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*entry) for entry in frozen["corrections"])
    environment = simulator_identity()
    checkpoint_hash = sha256(checkpoint)
    if (checkpoint_hash != frozen["checkpoint_sha256"] or environment["sim_revision"] != frozen["sim_revision"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("the installed controller differs from the published reference")
    common = {"environment": environment, "timing": list(timing), "incumbent": frozen["corrections"],
              "rules": RULES, "strict_rules": STRICT_RULES, "reference_sha256": sha256(manifest)}
    earlier, fitted, fingerprints = placement_compose.source_refinement(previous, guard, common)
    column, batch, recipe, feature, cutoff = fitted["corrections"][0]
    prior = guard.HoverGate(column, recipe, feature, cutoff)
    if len(fitted["corrections"]) != 1 or batch != first.BATCH or earlier["first_hover_code_sha256"] != sha256(Path(first.__file__)):
        raise ValueError("the earlier guard does not match its frozen controller")
    feedback, feedback_hash = placement_compose.source_rollouts(previous, Path("development"), earlier["schedule"]["development"],
                                                                checkpoint, timing, (prior,))
    if hover.pair_stats(feedback, (prior,)) != read_json(previous / "result.json")["development"]:
        raise ValueError("feedback differs from the earlier development decision")
    from_previous = read_json(previous / "training" / "dataset.json")
    points = from_previous["points"]
    outcomes = {int(seed): item for seed, item in from_previous["paired_outcomes"].items()}
    base = partial(first.FirstHoverPolicy, incumbent=incumbent)
    new_points = guard.features(checkpoint, timing, base, column, [entry["strict-500"] for entry in feedback["reference"]])
    points.extend(new_points)
    if len({entry["seed"] for entry in points}) != len(points):
        raise ValueError("projection fit requires disjoint training sources")
    activated = {entry["seed"] for entry in new_points if entry[prior.feature] >= prior.cutoff_m}
    arms = ({entry["default"]["seed"]: entry for entry in feedback[name] if entry["default"]["seed"] in activated}
            for name in ("reference", "selected"))
    outcomes.update(guard.paired_changes(*arms))
    development = [seed for seed in range(block, block + 4 * guard.DEVELOPMENT_CASES) if choose_column("random", seed) == column]
    confirmation = list(range(block + 400, block + 400 + guard.CONFIRMATION_CASES))
    history = [previous / "protocol.json", *(path for path in root.parent.glob("*/protocol.json") if path.parent != root)]
    reserved = set(development + confirmation)
    if reserved & {entry["seed"] for entry in points} or any(reserved & hover.scheduled_seeds(read_json(path).get("schedule", {})) for path in history):
        raise ValueError("projection evaluation requires fresh seeds")
    protocol = root / "protocol.json"
    _record_once(protocol, {**common, "profile": "advanced", "schedule": {"training": sorted(entry["seed"] for entry in points),
         "development": development, "confirmation": confirmation}, "previous": {**fingerprints, "feedback_sha256": feedback_hash},
         "evaluator_sha256": frozen["evaluator_sha256"], "code_sha256": sha256(Path(__file__)),
         "first_hover_code_sha256": sha256(Path(first.__file__)), "guard_code_sha256": sha256(Path(guard.__file__)),
         "confirmation_requires": earlier["confirmation_requires"]})
    dataset_path = root / "training" / "dataset.json"
    _record_once(dataset_path, {"points": points, "outcomes": {str(seed): entry for seed, entry in outcomes.items()},
                                "feedback_activated_seeds": sorted(activated)})
    model, search = fit(points, outcomes, prior)
    fit_path = root / "fit.json"
    corrections = [entry.record() for entry in model]
    _record_once(fit_path, {"corrections": corrections, "search": search, "dataset_sha256": sha256(dataset_path)})
    if not model:
        return {"status": "reference_retained_after_training"}
    print(f"Fitted {corrections}; evaluating {len(development)} fresh development pairs", flush=True)
    policy_type = partial(PersistentProjectionPolicy, incumbent=incumbent)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        measured = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                 {"reference": (), "selected": model}, policy_type=policy_type)
        progress = hover.pair_stats(measured, model)
        selection = root / "development-selection.json"
        _record_once(selection, progress)
        if not all(placement_search.passes(stats) for stats in progress.values()):
            return {"status": "reference_retained_after_development", "development": progress}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                     "timing": list(timing), "incumbent": frozen["corrections"], "corrections": corrections, "seeds": confirmation,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)})
        print(f"Candidate frozen before {len(confirmation)} fresh four-column confirmation pairs", flush=True)
        confirmed = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                       model, confirmation[:4], policy_type=policy_type)
    balance = Counter({str(source): guard.CONFIRMATION_CASES // 4 for source in range(1, 5)})
    if any(Counter(stats["scored_pairs_by_column"]) != balance for stats in confirmed.values()):
        raise ValueError("confirmation is not balanced across all four source columns")
    return {"status": "candidate_promoted" if first.supports_confirmation(confirmed) else "reference_retained_after_confirmation",
            "development": progress, "confirmation": confirmed}


def main():
    parser = argparse.ArgumentParser(description="Fit a guard that also checks the planned descent endpoint.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=146000)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.previous.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
