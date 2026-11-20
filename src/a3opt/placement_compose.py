from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_descent_agent, placement_hover_agent as hover
from a3opt import placement_search, placement_slot_refine as row
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import PLACE, task_column
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


CONFIRMATION_SCENES = 64


class ComposedPolicy(row.RowPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        if corrections and (len(corrections) != 2 or not isinstance(corrections[0], row.RowAlignment)
                            or not isinstance(corrections[1], hover.Hover)):
            raise ValueError("expected the fitted row and hover corrections in execution order")
        super().__init__(checkpoint, focus, scale, corrections[:1], incumbent=incumbent)
        self.hover = corrections[1] if corrections else None

    def act(self, observation, task=""):
        if (self.hover is not None and self.actions is not None and self.cursor == (hover.BATCH, PLACE, 0)
                and task_column(task) == hover.COLUMN):
            self.shift_placement(hover.BATCH, np.asarray(hover.OFFSETS[self.hover.recipe]))
        return super().act(observation, task)


def decode(records):
    _, _, center, scale, weights, bias = records[0]
    model = (row.RowAlignment(tuple(center), tuple(scale), tuple(weights), bias), hover.Hover(records[1][2]))
    if [entry.record() for entry in model] != records:
        raise ValueError("invalid composed placement records")
    return model


def stable_gain(confirmation):
    return (confirmation["default"]["mean"] > placement_search.TARGET
            and confirmation["strict-500"]["complete"] > confirmation["strict-500"]["reference_complete"]
            and all(placement_search.passes(stats) and stats["paired_count_ci95"][0] > 0 for stats in confirmation.values()))


def source_rollouts(root, path, seeds, checkpoint, timing, model):
    rollout_path = root / path / "results.json"
    paired = read_json(rollout_path)
    for arm, corrections in (("reference", ()), ("selected", model)):
        for seed, episode in zip(seeds, paired[arm], strict=True):
            receipt = root / path / "episodes" / arm / f"{seed}.json"
            if read_json(receipt) != episode:
                raise ValueError(f"source aggregate disagrees with individual receipt {receipt}")
            placement_benchmark.check((checkpoint, *timing, seed, corrections), episode)
    return paired, sha256(rollout_path)


def source_refinement(root, module, common):
    protocol_path, fit_path = root / "protocol.json", root / "fit.json"
    protocol, fit = read_json(protocol_path), read_json(fit_path)
    if (any(protocol.get(key) != value for key, value in common.items())
            or protocol["code_sha256"] != sha256(Path(module.__file__))
            or fit["dataset_sha256"] != sha256(root / "training" / "dataset.json")):
        raise ValueError(f"the source refinement differs from its frozen evidence: {root}")
    return protocol, fit, {"protocol_sha256": sha256(protocol_path), "fit_sha256": sha256(fit_path)}


def run(root, release, hover_run, row_run, block, workers):
    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected an aligned confirmation block and a positive worker budget")
    released_path = release / "controller.json"
    released = read_json(released_path)
    checkpoint = release / released["checkpoint"]
    timing = tuple(released["timing"])
    incumbent = tuple(placement_descent_agent.Descent(*entry) for entry in released["corrections"])
    environment = simulator_identity()
    checkpoint_hash = sha256(checkpoint)
    if (checkpoint_hash != released["checkpoint_sha256"] or environment["sim_revision"] != released["sim_revision"]
            or sha256(Path(placement_descent_agent.__file__)) != released["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != released["evaluator_sha256"]
            or any(item.column == row.COLUMN for item in incumbent)):
        raise ValueError("composition requires the unchanged published reference and independent active column")
    common = {"reference_sha256": sha256(released_path), "environment": environment, "rules": RULES, "strict_rules": STRICT_RULES,
              "incumbent": released["corrections"], "timing": list(timing)}
    hover_source, hover_fit, hover_hashes = source_refinement(hover_run, hover, common)
    row_source, row_fit, row_hashes = source_refinement(row_run, row, common)
    lock_path = hover_run / "confirmation-lock.json"
    lock = read_json(lock_path)
    candidate = hover.Hover(hover_fit["selected"])
    if (lock["corrections"] != [candidate.record()] or lock["protocol_sha256"] != hover_hashes["protocol_sha256"]
            or lock["fit_sha256"] != hover_hashes["fit_sha256"] or lock["seeds"] != hover_source["schedule"]["confirmation"]
            or read_json(hover_run / "confirmation" / "audit.json")["matched_native_replays"] != 8):
        raise ValueError("hover source was not frozen and replayed")
    model = decode([*row_fit["corrections"], candidate.record()])
    hovered, hover_rollout_hash = source_rollouts(hover_run, Path("confirmation/rollouts"), lock["seeds"], checkpoint, timing, model[1:])
    prior_hover = hover.pair_stats(hovered, model[1:])
    development = row_source["schedule"]["development"]
    aligned, row_rollout_hash = source_rollouts(row_run, Path("development"), development, checkpoint, timing, model[:1])
    prior_row = hover.pair_stats(aligned, model[:1])
    if prior_hover != read_json(hover_run / "result.json")["confirmation"] or prior_row != read_json(row_run / "result.json")["development"]:
        raise ValueError("source metrics disagree with their original paired receipts")
    if (not all(placement_search.passes(stats) and stats["paired_count_ci95"][0] > 0 for stats in prior_hover.values())
            or prior_row["strict-500"]["complete"] <= prior_row["strict-500"]["reference_complete"]
            or any(stats["major_losses"] or stats["wrong_source"] or stats["safety_stops"] for stats in prior_row.values())):
        return {"status": "reference_retained_without_complementary_evidence"}
    seeds = list(range(block, block + CONFIRMATION_SCENES))
    history = [hover_run / "protocol.json", row_run / "protocol.json", release / "confirmation-protocol.json", release / "replication-protocol.json",
               *(path for path in root.parent.glob("*/protocol.json") if path.parent != root)]
    for path in history:
        if set(seeds) & hover.scheduled_seeds(read_json(path).get("schedule", {})):
            raise ValueError(f"new confirmation panel overlaps {path}")
    protocol = root / "protocol.json"
    _record_once(protocol, {**common, "profile": "advanced", "schedule": {"development": development, "confirmation": seeds},
                            "evaluator_sha256": released["evaluator_sha256"], "code_sha256": sha256(Path(__file__)),
                            "row_code_sha256": row_source["code_sha256"], "hover_code_sha256": hover_source["code_sha256"],
                            "confirmation_requires": ["course_mean_above_9", "positive_lower_bound_on_paired_gain_both_rules",
                                "strict_success_count_gain", "course_success_count_preserved", "no_major_losses_or_safety_or_wrong_source"]})
    fit_path = root / "fit.json"
    corrections = [entry.record() for entry in model]
    _record_once(fit_path, {"corrections": corrections, "hover": {**hover_hashes, "rollout_sha256": hover_rollout_hash,
                  "confirmation_lock_sha256": sha256(lock_path)}, "row": {**row_hashes, "rollout_sha256": row_rollout_hash}})
    policy_type = partial(ComposedPolicy, incumbent=incumbent)
    print(f"Testing the combined controller on {len(development)} previously used development scenes", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        composed = placement_benchmark.evaluate(pool, root / "interaction", checkpoint, timing, development,
                                                  {"selected": model}, policy_type=policy_type)["selected"]
        progress = hover.pair_stats({"reference": aligned["reference"], "selected": composed}, model)
        selection = root / "development-selection.json"
        _record_once(selection, progress)
        if not row.supports_alignment(progress) or not all(placement_search.passes(stats) for stats in progress.values()):
            return {"status": "reference_retained_after_development", "development": progress}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                     "timing": list(timing), "incumbent": released["corrections"], "corrections": corrections, "seeds": seeds,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)})
        print(f"Composed candidate frozen before {len(seeds)} fresh four-column confirmation pairs", flush=True)
        measured = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, seeds,
                                                      model, seeds[:4], policy_type=policy_type)
    expected = Counter({str(column): CONFIRMATION_SCENES // 4 for column in range(1, 5)})
    if any(Counter(stats["scored_pairs_by_column"]) != expected for stats in measured.values()):
        raise ValueError("composed confirmation is not balanced across all four columns")
    return {"status": "candidate_promoted" if stable_gain(measured) else "reference_retained_after_confirmation",
            "development": progress, "confirmation": measured, "target_lower_bound": measured["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Test whether fitted first-row alignment and later hover release improve the same controller.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    for name in ("hover", "row", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--confirmation-start", type=int, default=143400)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.hover.resolve(), args.row.resolve(), args.confirmation_start, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
