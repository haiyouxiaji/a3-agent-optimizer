from __future__ import annotations

import argparse
import json
from pathlib import Path

from a3opt import placement_search, placement_source_guard as guard
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES

from confirm_expert_grasp import audit, resume, verify_receipts


def compare(reference, selected, models):
    corrected = {item.column for item in models if isinstance(item, guard.SourceGuard)}
    if any(old["phase_geometry"][0] != new["phase_geometry"][0]
           for old, new in zip(reference, selected, strict=True) if new["column"] in corrected):
        raise ValueError("source separation changed the first batch")
    return placement_search.compare(reference, selected, models, prefix_batch=0)


def inputs(root, parent):
    protocol_path = root / "protocol.json"
    lock_path = root / "confirmation-lock.json"
    fit_path = root / "fit.json"
    selection_path = root / "development-selection.json"
    dataset = root / "training" / "dataset.json"
    protocol, lock, fit, selection = map(read_json, (protocol_path, lock_path, fit_path, selection_path))
    checkpoint, timing, constructor, anchors, controls, parent_lock, lock_hash, rows_hash = guard.load_parent(parent)
    schedule = protocol["schedule"]
    training, development, confirmation = (schedule[name] for name in ("training", "development", "confirmation"))
    scope, failures = guard.scope_of(controls.values())
    incumbent_keys = ("incumbent", "late_recovery", "timing", "checkpoint_sha256")
    if (protocol["parent_lock_sha256"] != lock_hash or protocol["parent_rollouts_sha256"] != rows_hash
            or protocol["environment"] != simulator_identity() or (protocol["rules"], protocol["strict_rules"]) != (RULES, STRICT_RULES)
            or protocol["code_sha256"] != sha256(Path(guard.__file__)) or protocol["scope"] != list(scope)
            or protocol["failures"] != failures or any(protocol[key] != parent_lock[key] or lock[key] != parent_lock[key] for key in incumbent_keys)
            or len(set(training + development + confirmation)) != len(training + development + confirmation)
            or confirmation != lock["seeds"] or set(development + confirmation) & controls.keys()
            or any(lock[key] != sha256(path) for key, path in (("protocol_sha256", protocol_path), ("fit_sha256", fit_path),
                                                             ("selection_sha256", selection_path), ("checkpoint_sha256", checkpoint)))
            or fit["dataset_sha256"] != sha256(dataset)):
        raise ValueError("source separation differs from its frozen protocol")
    options = tuple(guard.SourceGuard(*scope, name) for name in protocol["options"])
    rows = read_json(dataset)["rows"]
    for seed, row in zip(training, rows, strict=True):
        if row != read_json(root / "training" / "episodes" / f"{seed}.json"):
            raise ValueError("source separation training receipt differs from its recorded episode")
        guard.audit_trial((checkpoint, timing, constructor, options, seed), row, archive=controls)
    scores = {option.recipe: guard.assess((row["reference"], row["candidates"][option.recipe]) for row in rows) for option in options}
    choice = max((option for option in options if guard.eligible(scores[option.recipe])),
                 key=lambda option: (scores[option.recipe]["net_gain"], -guard.OPTIONS[option.recipe][0]), default=None)
    if choice is None or scores != fit["training"] or choice.record() != fit["guard"]:
        raise ValueError("source-separation selection differs from the training trials")
    models = (*anchors, choice)
    if lock["corrections"] != [item.record() for item in models] or len({item.column for item in models}) != len(models):
        raise ValueError("source separation overlaps the learned release scope or differs from the lock")
    dev_path = root / "development"
    paired = read_json(dev_path / "results.json")
    verify_receipts(dev_path, checkpoint, timing, development, (choice,), paired)
    if selection != guard.assess(zip(paired["reference"], paired["selected"], strict=True)) or not guard.eligible(selection):
        raise ValueError("source separation did not pass its new development episodes")
    return checkpoint, timing, constructor, models, lock, paired, selection


def calibration(parent, models, source_pairs):
    landing = read_json(parent / "development" / "results.json")
    original = read_json(parent / "development-selection.json")
    gain = original["default"]["changes"][0]["seed"]
    seeds = [*[row["default"]["seed"] for row in landing["reference"][:2]], gain,
             *[row["default"]["seed"] for row in source_pairs["reference"][:3]]]
    seeds = list(dict.fromkeys(seeds))
    records = [item.record() for item in models]
    expected = {"reference": [], "selected": []}
    for arm, entries in expected.items():
        by_seed = {pair["default"]["seed"]: pair for pair in (*landing[arm], *source_pairs[arm])}
        for seed in seeds:
            entries.append({rule: {**row, "corrections": records if arm == "selected" else []}
                            for rule, row in by_seed[seed].items()})
    return expected, seeds


def main():
    parser = argparse.ArgumentParser(description="Complete and audit the frozen release and source-separation controller.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    root, parent = args.run.resolve(), args.parent.resolve()
    with _run_lock(root):
        checkpoint, timing, constructor, models, lock, development, selection = inputs(root, parent)
        if args.resume and not (root / "result.json").is_file():
            expected, seeds = calibration(parent, models, development)
            resume(root, checkpoint, timing, constructor, models, lock, expected, selection, args.workers,
                   compare_rows=compare, parity_seeds=seeds)
        result = audit(root, checkpoint, timing, models, lock, selection, compare_rows=compare,
                       retained_status="reference_retained_after_source_guard_confirmation")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
