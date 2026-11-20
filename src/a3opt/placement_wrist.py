from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt import placement_benchmark, placement_orientation
from a3opt.advanced_agent import _record_once
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.placement_hybrid import decode, development_pairs
from a3opt.placement_orientation import ANGLES, Orientation, OrientationPolicy
from a3opt.placement_search import TARGET, branch_stats, compare, confirm_combined, encode, passes
from a3opt.refinement_eval import read_json, sha256


class WristPolicy(OrientationPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        orientations = tuple(item for item in corrections if isinstance(item, Orientation))
        super().__init__(checkpoint, focus, scale, orientations)
        offsets = [item for item in corrections if not isinstance(item, Orientation)]
        self.corrections = {(item.column, item.batch): item for item in offsets}
        if len(self.corrections) != len(offsets):
            raise ValueError("duplicate wrist-policy placement offset")


def fit(rows):
    trials, candidates = {}, {}
    for name, angle in ANGLES.items():
        if name == "reference":
            continue
        groups = {}
        for column in range(1, 5):
            forks = [row["trials"][0] for row in rows if row["column"] == column and row["trials"][0] is not None]
            groups[column] = branch_stats(forks, name)
        safe = [column for column, stats in groups.items() if stats["pairs"] >= 3 and stats["improved"]
                and stats["slot_delta"] >= 0 and not (stats["worse"] or stats["safety_stops"] or stats["source_count_changes"])]
        trials[name] = {"groups": {str(column): stats for column, stats in groups.items()}, "columns": safe}
        if len(safe) >= 2:
            candidates[name] = (sum(groups[column]["count_delta"] for column in safe),
                                tuple(Orientation(column, 0, angle) for column in safe))
    winner = max(candidates, key=lambda name: candidates[name][0], default=None)
    return candidates[winner][1] if winner else (), {"selected": winner, "trials": trials}


def run(root, reference, timing_run, source, parent, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and verified joint evaluator are required")
    parent_protocol_path, parent_fit_path = parent / "protocol.json", parent / "fit.json"
    parent_protocol, pooled_fit = read_json(parent_protocol_path), read_json(parent_fit_path)
    source_protocol_path, dataset = source / "protocol.json", source / "training" / "dataset.json"
    source_protocol = read_json(source_protocol_path)
    training = source_protocol["schedule"]["training"]
    development = parent_protocol["schedule"]["development"]
    confirmation = list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": training, "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    rows = read_json(dataset)["rows"]
    previous_confirmation = (parent_protocol["schedule"]["confirmation"]["random"]
                             + source_protocol["schedule"]["confirmation"]["random"])
    if (len(training) != len(set(training)) or [row["seed"] for row in rows] != training
            or set(training).intersection(development + confirmation) or set(confirmation).intersection(development + previous_confirmation)
            or any(protocol[key] != identity[key] for protocol in (source_protocol, parent_protocol)
                   for key in ("environment", "reference_sha256", "timing", "strict_rules"))
            or source_protocol["training_code_sha256"] != sha256(Path(placement_orientation.__file__))
            or parent_protocol["evaluator_sha256"] != sha256(Path(placement_benchmark.__file__))):
        raise ValueError("wrist intervention requires compatible training, development and independent confirmation")
    for row in rows:
        placement_orientation.audit((checkpoint, *timing, row["seed"]), row)
    angles, stats = fit(rows)
    pooled = decode(pooled_fit["corrections"])
    model = (*angles, *pooled)
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "training_dataset_sha256": sha256(dataset),
                 "orientation_protocol_sha256": sha256(source_protocol_path), "parent_protocol_sha256": sha256(parent_protocol_path),
                 "parent_fit_sha256": sha256(parent_fit_path), "training_code_sha256": sha256(Path(__file__)),
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__)), "parity_sha256": sha256(parity),
                 "target_mean_strictly_above": TARGET})
    _record_once(root / "fit.json", {**stats, "corrections": encode(model), "wrist": encode(angles)})
    if not pooled or not angles:
        return {"status": "reference_retained_after_training"}
    print(f"Selected first-batch wrist correction in {len(angles)} columns", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        observed = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                {"selected": model}, policy_type=WristPolicy)
        parents, parent_hash = development_pairs(parent, parent_protocol, checkpoint, timing, development, pooled, timeout=1800)
        scores = {name: {rule: compare([row[rule] for row in parents[arm]], [row[rule] for row in observed["selected"]], model)
                         for rule in ("default", "strict-500")} for name, arm in (("reference", "reference"), ("pooled", "selected"))}
        _record_once(root / "development-selection.json", {"scores": scores, "parent_results_sha256": parent_hash})
        if not all(passes(stats) for group in scores.values() for stats in group.values()):
            return {"status": "reference_retained_after_development", "development": scores}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model, confirmation[:4],
                                   policy_type=WristPolicy)
    primary = checked["default"]
    above = primary["mean"] > TARGET
    accepted = above and all(passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation",
            "target_point_estimate": above, "target_lower_bound": primary["mean_ci95"][0] > TARGET,
            "active_policy_above_target": accepted, "development": scores, "confirmation": checked}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_WRIST_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("wrist policy requires the frozen checkpoint")
    records = lock["corrections"]
    angles = tuple(Orientation(*item) for item in records if isinstance(item[2], (float, int)))
    offsets = decode(item for item in records if isinstance(item[2], list))
    return WristPolicy(checkpoint, *lock["timing"], (*angles, *offsets))


def main():
    parser = argparse.ArgumentParser(description="Fit source-column wrist corrections and verify full paired rollouts.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-block", type=int, default=120000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.source.resolve(), args.parent.resolve(),
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
