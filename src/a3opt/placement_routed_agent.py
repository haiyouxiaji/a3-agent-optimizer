from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_compose, placement_descent_agent as descent, placement_search
from a3opt import placement_first_hover as first, placement_hover_agent as hover, placement_hover_guard as guard
from a3opt import placement_hover_guard_refine as projection
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import PLACE, task_column
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


CONFIRMATION_CASES = 64
REQUIREMENTS = ["course_mean_above_9", "positive_lower_bound_on_paired_gain_both_rules",
                "no_major_losses", "no_safety_or_wrong_source", "no_success_count_regression"]


def decode(records):
    if len(records) != 2 or tuple(map(len, records)) != (3, 4):
        raise ValueError("expected a second-batch hover and a first-batch projection gate")
    late, early = records
    model = hover.Hover(late[2]), projection.PersistentProjection(early[0], early[2], early[3])
    if model[0].column == model[1].column or [item.record() for item in model] != list(records):
        raise ValueError("expected two placement residuals on distinct source columns")
    return model


class RoutedPlacementPolicy(projection.PersistentProjectionPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        model = decode([item.record() for item in corrections]) if corrections else ()
        self.late_hover = model[0] if model else None
        super().__init__(checkpoint, focus, scale, model[1:], incumbent=incumbent)

    def act(self, observation, task=""):
        if (self.late_hover is not None and self.actions is not None and self.cursor == (hover.BATCH, PLACE, 0)
                and task_column(task) == hover.COLUMN):
            self.shift_placement(hover.BATCH, np.asarray(hover.OFFSETS[self.late_hover.recipe]))
        return super().act(observation, task)


def dependencies():
    return {module.__name__: sha256(Path(module.__file__)) for module in (hover, first, guard, projection, placement_search)}


def source_panel(root, stage, seeds, checkpoint, timing, model):
    path = Path("confirmation/rollouts") if stage == "confirmation" else Path("development")
    pairs, digest = placement_compose.source_rollouts(root, path, seeds, checkpoint, timing, model)
    measured = hover.pair_stats(pairs, model)
    expected = read_json(root / "result.json")[stage] if stage == "confirmation" else read_json(root / "development-selection.json")
    if measured != expected:
        raise ValueError(f"paired source scores differ from the saved decision for {root.name}/{stage}")
    return {"stage": stage, "seeds": seeds, "rollout_sha256": digest, "metrics": measured}


def fit(panels, proposals):
    evidence = {}
    for name, model in proposals.items():
        passes = bool(panels[name]) and all(all(placement_search.passes(panel["metrics"][rule])
                                               for rule in ("default", "strict-500")) for panel in panels[name])
        evidence[name] = {"correction": model.record(), "kept": passes}
    selected = [item["correction"] for item in evidence.values() if item["kept"]]
    return ([item.record() for item in decode(selected)] if len(selected) == len(proposals) else []), evidence


def run(root, release, late, early, block, workers, *, fit_only=False):
    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected a four-column seed boundary and a positive worker budget")
    manifest = release / "controller.json"
    reference = read_json(manifest)
    checkpoint = release / reference["checkpoint"]
    timing = tuple(reference["timing"])
    environment = simulator_identity()
    incumbent = tuple(descent.Descent(*entry) for entry in reference["corrections"])
    if (sha256(checkpoint) != reference["checkpoint_sha256"] or environment["sim_revision"] != reference["sim_revision"]
            or sha256(Path(descent.__file__)) != reference["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != reference["evaluator_sha256"]):
        raise ValueError("routing requires the unchanged published reference")
    common = {"reference_sha256": sha256(manifest), "environment": environment, "timing": list(timing),
              "incumbent": reference["corrections"], "rules": RULES, "strict_rules": STRICT_RULES}
    late_protocol, late_fit, late_hashes = placement_compose.source_refinement(late, hover, common)
    early_protocol, early_fit, early_hashes = placement_compose.source_refinement(early, projection, common)
    late_model, early_model = decode([[hover.COLUMN, hover.BATCH, late_fit["selected"]], *early_fit["corrections"]])
    if (late_protocol["selection_code_sha256"] != sha256(Path(placement_search.__file__))
            or early_protocol["first_hover_code_sha256"] != sha256(Path(first.__file__))
            or early_protocol["guard_code_sha256"] != sha256(Path(guard.__file__))
            or any(item.column in (late_model.column, early_model.column) for item in incumbent)):
        raise ValueError("the routed residuals do not match their frozen dependencies")
    sources = {"late-hover": late_hashes, "early-projection": early_hashes}
    late_seeds = late_protocol["schedule"]
    early_seeds = early_protocol["schedule"]
    late_lock, early_lock = (read_json(path / "confirmation-lock.json") for path in (late, early))
    for path, lock, fingerprints, item, schedule in ((late, late_lock, late_hashes, late_model, late_seeds),
                                                      (early, early_lock, early_hashes, early_model, early_seeds)):
        if (lock["corrections"] != [item.record()] or lock["seeds"] != schedule["confirmation"]
                or lock["checkpoint_sha256"] != reference["checkpoint_sha256"] or lock["timing"] != list(timing)
                or lock["incumbent"] != reference["corrections"]
                or lock["protocol_sha256"] != fingerprints["protocol_sha256"] or lock["fit_sha256"] != fingerprints["fit_sha256"]
                or lock["selection_sha256"] != sha256(path / "development-selection.json")):
            raise ValueError(f"source confirmation lock disagrees with its fit: {path}")
    if read_json(late / "confirmation" / "audit.json")["matched_native_replays"] != 8:
        raise ValueError("late hover requires matching native score receipts")
    panels = {"late-hover": [source_panel(late, stage, late_seeds[stage], checkpoint, timing, (late_model,))
                              for stage in ("development", "confirmation")],
              "early-projection": [source_panel(early, "development", early_seeds["development"], checkpoint, timing, (early_model,))]}
    training = [seed for groups in panels.values() for panel in groups for seed in panel["seeds"]]
    if len(training) != len(set(training)):
        raise ValueError("routed fitting panels overlap")
    seeds = list(range(block, block + CONFIRMATION_CASES))
    history = (release / "confirmation-protocol.json", release / "replication-protocol.json", late / "protocol.json",
               early / "protocol.json", *(path for path in root.parent.glob("*/protocol.json") if path.parent != root))
    if set(seeds) & set(training) or any(set(seeds) & hover.scheduled_seeds(read_json(path).get("schedule", {})) for path in history):
        raise ValueError("routed confirmation overlaps a reserved or previously used seed")
    protocol = root / "protocol.json"
    _record_once(protocol, {**common, "profile": "advanced", "schedule": {"training": training, "confirmation": seeds},
                            "sources": sources, "confirmation_requires": REQUIREMENTS,
                            "evaluator_sha256": reference["evaluator_sha256"], "code_sha256": sha256(Path(__file__)),
                            "dependency_sha256": dependencies()})
    dataset = root / "training" / "dataset.json"
    _record_once(dataset, {"panels": panels, "sources": sources})
    corrections, evidence = fit(panels, {"late-hover": late_model, "early-projection": early_model})
    fit_path = root / "fit.json"
    _record_once(fit_path, {"corrections": corrections, "dataset_sha256": sha256(dataset)})
    selection = root / "development-selection.json"
    _record_once(selection, evidence)
    if not corrections:
        return {"status": "reference_retained_after_training"}
    _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": reference["checkpoint_sha256"],
                 "timing": list(timing), "incumbent": reference["corrections"], "corrections": corrections, "seeds": seeds,
                 "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)})
    if fit_only:
        return {"status": "candidate_frozen"}
    print(f"Selected {corrections} from verified historical pairs; running {len(seeds)} fresh four-column pairs", flush=True)
    policy_type = partial(RoutedPlacementPolicy, incumbent=incumbent)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        confirmation = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, seeds, decode(corrections),
                                                          seeds[:4], policy_type=policy_type)
    balance = {str(column): len(seeds) // 4 for column in range(1, 5)}
    if any(stats["scored_pairs_by_column"] != balance for stats in confirmation.values()):
        raise ValueError("confirmation is not balanced across all four source columns")
    return {"status": "candidate_promoted" if first.supports_confirmation(confirmation) else "reference_retained_after_confirmation",
            "confirmation": confirmation}


def main():
    parser = argparse.ArgumentParser(description="Fit and confirm placement routing from two disjoint column-specific updates.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--late", type=Path, required=True)
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=147400)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.late.resolve(), args.early.resolve(), args.seed_block, args.workers,
                     fit_only=args.fit_only)
        if result["status"] != "candidate_frozen":
            _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
