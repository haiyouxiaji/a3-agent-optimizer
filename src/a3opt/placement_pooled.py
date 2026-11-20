from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt import placement_benchmark, placement_tail
from a3opt.advanced_agent import _record_once
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.placement_search import TARGET, compare, confirm_combined, encode, passes
from a3opt.placement_sim import Correction, PlacementPolicy
from a3opt.refinement_eval import read_json, sha256


def fit(pairs: list[dict]) -> tuple[tuple[Correction, ...], dict]:
    reference = [pair["reference"] for pair in pairs]
    options = {}
    for name, offset in placement_tail.OFFSETS.items():
        if name == "reference":
            continue
        gains, slot_deltas, violations = [], [], 0
        for row in reference:
            if row["fork"] is None:
                continue
            outcomes = row["fork"]["outcomes"]
            baseline, selected = (outcomes[key]["physics"] for key in ("reference", name))
            gains.append((row["column"], len(selected["counted_ids"]) - len(baseline["counted_ids"])))
            slot_deltas.append(selected["filled_slots"] - baseline["filled_slots"])
            violations += bool(outcomes[name]["wrong_source"] or selected["safety_reason"]
                               or selected["source_remaining"] != baseline["source_remaining"])
        options[name] = {
            "count_gain": sum(delta for _, delta in gains),
            "slot_gain": sum(slot_deltas),
            "worst_count_change": min((delta for _, delta in gains), default=0),
            "worst_slot_change": min(slot_deltas, default=0),
            "improved_columns": sorted({column for column, delta in gains if delta > 0}),
            "violations": violations,
            "motion_mm": 1000 * sum(abs(value) for value in offset),
        }
    eligible = [name for name, stats in options.items() if stats["count_gain"] > 0
                and len(stats["improved_columns"]) >= 2 and not stats["violations"]
                and stats["worst_count_change"] >= 0 and stats["worst_slot_change"] >= -1 and stats["slot_gain"] >= 0]
    winner = max(eligible, key=lambda name: (options[name]["count_gain"], options[name]["slot_gain"],
                                             -options[name]["motion_mm"]), default=None)
    model = tuple(Correction(column, 1, placement_tail.OFFSETS[winner]) for column in range(1, 5)) if winner else ()
    return model, {"options": options, "selected": winner}


def data(source, identity, checkpoint, timing, evaluation):
    from a3_dual_arm_sim.sim.randomization import choose_column

    protocol_path = source / "protocol.json"
    protocol = read_json(protocol_path)
    schedule = protocol["schedule"]
    seeds = schedule["training"]
    reserved = seeds + schedule["development"] + schedule["confirmation"]["random"]
    if (any(protocol[key] != identity[key] for key in ("environment", "reference_sha256", "timing", "strict_rules"))
            or protocol["training_code_sha256"] != sha256(Path(placement_tail.__file__))
            or protocol["offsets_m"] != {name: list(offset) for name, offset in placement_tail.OFFSETS.items()}
            or len(seeds) != len(set(seeds)) or set(reserved).intersection(evaluation)):
        raise ValueError("pooled placement requires compatible, disjoint training receipts")
    if Counter(choose_column("random", seed) for seed in seeds) != {column: 6 for column in range(1, 5)}:
        raise ValueError("pooled placement expects six locked training seeds in each source column")
    anchor = tuple(Correction(column, batch, tuple(xyz), *limits) for column, batch, xyz, *limits in protocol["anchor"])
    dataset = source / "training" / "dataset.json"
    pairs = read_json(dataset)["pairs"]
    if len(pairs) != len(seeds):
        raise ValueError("pooled placement requires the complete training acquisition")
    for seed, pair in zip(seeds, pairs, strict=True):
        placement_tail.audit_pair((checkpoint, *timing, seed, anchor), pair)
    return pairs, {"seeds": seeds, "source_protocol_sha256": sha256(protocol_path), "dataset_sha256": sha256(dataset)}

def run(root, reference, timing_run, source, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and verified joint evaluator are required")
    development = list(range(block, block + 16))
    confirmation = list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    pairs, evidence = data(source, identity, checkpoint, timing, development + confirmation)
    splits["training"] = evidence["seeds"]
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "training_source": evidence,
                 "code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "parity_sha256": sha256(parity), "target_mean_strictly_above": TARGET})
    model, stats = fit(pairs)
    _record_once(root / "fit.json", {**stats, "corrections": encode(model), "training_source": evidence})
    if not model:
        return {"status": "reference_retained_after_training"}
    print(f"Pooled {len(pairs)} training receipts and selected {stats['selected']}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outputs = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                               {"reference": (), "selected": model})
        comparison = {rule: compare([row[rule] for row in outputs["reference"]], [row[rule] for row in outputs["selected"]], model)
                      for rule in ("default", "strict-500")}
        _record_once(root / "development-selection.json", comparison)
        if not all(passes(stats) for stats in comparison.values()):
            return {"status": "reference_retained_after_development", "development": comparison}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        print(f"Frozen {stats['selected']} for {len(confirmation)} confirmation pairs", flush=True)
        checked = confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model, confirmation[:4])
    primary = checked["default"]
    above = primary["mean"] > TARGET
    accepted = above and all(passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation",
            "target_point_estimate": above, "target_lower_bound": primary["mean_ci95"][0] > TARGET,
            "active_policy_above_target": accepted, "development": comparison, "confirmation": checked}


def main():
    parser = argparse.ArgumentParser(description="Fit shared second-batch corrections from audited placement trajectories.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed-block", type=int, default=118000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.source.resolve(), args.parity.resolve(),
                     args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
