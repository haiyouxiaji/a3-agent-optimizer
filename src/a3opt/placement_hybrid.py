from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt import placement_benchmark, placement_pooled
from a3opt.advanced_agent import _record_once
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.placement_search import TARGET, compare, confirm_combined, encode, passes
from a3opt.placement_sim import Correction
from a3opt.refinement_eval import read_json, sha256


def decode(records):
    return tuple(Correction(column, batch, tuple(xyz), *limits) for column, batch, xyz, *limits in records)


def fit(pairs, anchor, second_option):
    stats, model = {}, []
    for item in anchor:
        group = [pair for pair in pairs if pair["reference"]["column"] == item.column and pair["anchor"] is not None]
        deltas, slot_deltas, violations = [], [], 0
        for pair in group:
            reference, corrected = (pair[name]["fork"] for name in ("reference", "anchor"))
            if reference is None or corrected is None:
                violations += 1
                continue
            before, after = (fork["outcomes"][second_option] for fork in (reference, corrected))
            original, candidate = before["physics"], after["physics"]
            deltas.append(len(candidate["counted_ids"]) - len(original["counted_ids"]))
            slot_deltas.append(candidate["filled_slots"] - original["filled_slots"])
            violations += bool(after["wrong_source"] or candidate["safety_reason"]
                               or candidate["source_remaining"] != original["source_remaining"])
        admit = (len(deltas) >= 3 and sum(deltas) > 0 and min(deltas) >= 0 and min(slot_deltas) >= -1
                 and sum(slot_deltas) >= 0 and not violations)
        stats[str(item.column)] = {"active_scenes": len(group), "paired_deltas": deltas, "paired_slot_deltas": slot_deltas,
                                   "violations": violations, "admitted": admit}
        if admit:
            model.append(item)
    return (tuple(model) if len(model) >= 2 else ()), stats


def development_pairs(parent, parent_protocol, checkpoint, timing, development, pooled, timeout):
    request_path = parent / "development" / "request.json"
    path = parent / "development" / "results.json"
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("parent development receipts did not become available within the reserved budget")
        time.sleep(10)
    request, outputs = read_json(request_path), read_json(path)
    if (request["checkpoint_sha256"] != sha256(checkpoint) or request["timing"] != list(timing)
            or request["seeds"] != development or request["models"] != {"reference": [], "selected": encode(pooled)}
            or request["rules"] != [parent_protocol["rules"], parent_protocol["strict_rules"]]
            or set(outputs) != {"reference", "selected"}):
        raise ValueError("parent development does not match the predeclared setup")
    for name, model in (("reference", ()), ("selected", pooled)):
        for seed, row in zip(development, outputs[name], strict=True):
            placement_benchmark.check((checkpoint, *timing, seed, model), row)
    return outputs, sha256(path)


def run(root, reference, timing_run, source, parent, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and verified joint evaluator are required")
    parent_protocol_path, parent_fit_path = parent / "protocol.json", parent / "fit.json"
    parent_protocol, pooled_fit = read_json(parent_protocol_path), read_json(parent_fit_path)
    if pooled_fit["selected"] is None:
        return {"status": "reference_retained_after_training"}
    development = parent_protocol["schedule"]["development"]
    confirmation = list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    train, evidence = placement_pooled.data(source, identity, checkpoint, timing, development + confirmation)
    if (pooled_fit["training_source"] != evidence or parent_protocol["training_source"] != evidence
            or set(confirmation).intersection(development + parent_protocol["schedule"]["confirmation"]["random"])
            or parent_protocol["evaluator_sha256"] != sha256(Path(placement_benchmark.__file__))):
        raise ValueError("the composed policy requires disjoint confirmation and compatible parent training")
    splits["training"] = evidence["seeds"]
    anchor = decode(read_json(source / "protocol.json")["anchor"])
    pooled = decode(pooled_fit["corrections"])
    first, stats = fit(train, anchor, pooled_fit["selected"])
    model = (*first, *pooled) if first else ()
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "tail_training": evidence,
                 "shared_development_protocol_sha256": sha256(parent_protocol_path), "parent_fit_sha256": sha256(parent_fit_path),
                 "training_code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "parity_sha256": sha256(parity), "target_mean_strictly_above": TARGET})
    _record_once(root / "fit.json", {"columns": stats, "first": encode(first), "corrections": encode(model)})
    if not model:
        return {"status": "reference_retained_after_training"}
    print(f"Selected first-batch repair in {len(first)} columns alongside the shared second-batch change", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        observed = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                {"selected": model})
        parents, parent_results_hash = development_pairs(parent, parent_protocol, checkpoint, timing, development,
                                                          pooled, timeout=1800)
        scores = {name: {rule: compare([row[rule] for row in parents[arm]], [row[rule] for row in observed["selected"]], model)
                         for rule in ("default", "strict-500")} for name, arm in (("reference", "reference"), ("pooled", "selected"))}
        _record_once(root / "development-selection.json", {"scores": scores, "parent_results_sha256": parent_results_hash})
        if not all(passes(stats) for results in scores.values() for stats in results.values()):
            return {"status": "reference_retained_after_development", "development": scores}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model, confirmation[:4])
    primary = checked["default"]
    above = primary["mean"] > TARGET
    accepted = above and all(passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation",
            "target_point_estimate": above, "target_lower_bound": primary["mean_ci95"][0] > TARGET,
            "active_policy_above_target": accepted, "development": scores, "confirmation": checked}


def main():
    parser = argparse.ArgumentParser(description="Fit a two-batch intervention from replays of the physical first-batch state.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-block", type=int, default=119000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.source.resolve(), args.parent.resolve(),
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
