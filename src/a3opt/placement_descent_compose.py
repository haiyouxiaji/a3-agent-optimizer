from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt import placement_benchmark, placement_descent_agent as descent, placement_search
from a3opt.advanced_agent import _record_once
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.refinement_eval import read_json, sha256


RULE_NAMES = ("default", "strict-500")


def finished_development(source):
    deadline = time.monotonic() + 1200
    while not (source / "development-selection.json").is_file():
        if (source / "result.json").is_file():
            return False
        if time.monotonic() > deadline:
            raise TimeoutError(f"waiting for the reserved development runs: {source}")
        time.sleep(5.)
    return True


def audit_source(source, checkpoint, timing, protocol, seeds):
    directory = source / "development"
    fit, request = read_json(source / "fit.json"), read_json(directory / "request.json")
    model = tuple(descent.Descent(*item) for item in fit["corrections"])
    if (not model or request["checkpoint_sha256"] != sha256(checkpoint) or request["timing"] != list(timing)
            or request["seeds"] != seeds or request["models"] != {"reference": [], "selected": fit["corrections"]}
            or protocol["evaluator_sha256"] != sha256(Path(placement_benchmark.__file__))):
        raise ValueError("descent composition requires frozen, compatible development receipts")
    results_path = directory / "results.json"
    results = read_json(results_path)
    for name, corrections in (("reference", ()), ("selected", model)):
        for seed, row in zip(seeds, results[name], strict=True):
            placement_benchmark.check((checkpoint, *timing, seed, corrections), row)
    return model, results, sha256(results_path)


def from_development(sources):
    report, selected = {}, []
    for column in range(1, 5):
        candidates, evidence = [], {}
        for name, model, rows in sources:
            local = tuple(item for item in model if item.column == column)
            if not local:
                continue
            stats = {}
            for rule in RULE_NAMES:
                control, trial = ([row[rule] for row in rows[arm] if row[rule]["column"] == column]
                                  for arm in ("reference", "selected"))
                stats[rule] = placement_search.compare(control, trial, local)
            evidence[name] = stats
            if all(placement_search.passes(entry) for entry in stats.values()):
                rank = (min(entry["paired_count_delta"] for entry in stats.values()),
                        stats["strict-500"]["complete"] - stats["strict-500"]["reference_complete"], -len(local))
                candidates.append((rank, name, local))
        winner = max(candidates, key=lambda item: item[0], default=None)
        if winner:
            selected.extend(winner[2])
        report[str(column)] = {"selection": winner[1] if winner else None, "development": evidence}
    return tuple(selected), report


def equivalent_confirmation(selected, sources):
    for source, model, rows in sources:
        if set(model) == set(selected) and all(placement_search.passes(placement_search.compare(
                [row[rule] for row in rows["reference"]], [row[rule] for row in rows["selected"]], model))
                for rule in RULE_NAMES):
            return source
    return None


def run(root, reference, timing_run, sources, parity, workers, block):
    if block % 4 or len(sources) < 2 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("descent composition needs verified evaluation and two independent development panels")
    confirmation = list(range(block, block + 40))
    schedule = {"scout": {"random": []}, "training": [], "development": [], "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, schedule)
    records, used_development = [], set()
    for source in sources:
        path = source / "protocol.json"
        protocol = read_json(path)
        prior = protocol["schedule"]
        seeds = prior["development"]
        if (set(seeds).intersection(used_development)
                or set(prior["training"] + seeds + prior["confirmation"]["random"]).intersection(confirmation)
                or any(protocol[key] != identity[key] for key in ("environment", "reference_sha256", "timing", "strict_rules"))
                or protocol.get("policy_code_sha256", protocol["code_sha256"]) != sha256(Path(descent.__file__))):
            raise ValueError("descent composition requires matching controllers and disjoint panels")
        used_development.update(seeds)
        records.append((source, protocol, sha256(path), seeds))
    _record_once(root / "protocol.json", {**identity, "schedule": schedule, "code_sha256": sha256(Path(__file__)),
                 "policy_code_sha256": sha256(Path(descent.__file__)), "parity_sha256": sha256(parity),
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "target_mean_strictly_above": placement_search.TARGET,
                 "development_sources": {source.name: {"protocol_sha256": digest, "seeds": seeds}
                                         for source, _, digest, seeds in records}})
    available, hashes = [], {}
    for source, protocol, _, seeds in records:
        if finished_development(source):
            model, output, digest = audit_source(source, checkpoint, timing, protocol, seeds)
            available.append((source.name, model, output))
            hashes[source.name] = digest
    selected, report = from_development(available)
    encoded = placement_search.encode(selected)
    _record_once(root / "development-selection.json", {"columns": report, "source_results_sha256": hashes, "corrections": encoded})
    if not selected:
        return {"status": "reference_retained_without_safe_column"}
    fit_path = root / "fit.json"
    _record_once(fit_path, {"corrections": encoded, "selection_sha256": sha256(root / "development-selection.json")})
    equivalent = equivalent_confirmation(selected, available)
    if equivalent is not None:
        return {"status": "equivalent_candidate_in_confirmation", "equivalent_run": equivalent}
    _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                 "timing": list(timing), "seeds": confirmation, "corrections": encoded,
                 "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(fit_path),
                 "selection_sha256": sha256(root / "development-selection.json")})
    print(f"Frozen descent placement for columns {sorted({item.column for item in selected})}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        checked = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, selected,
                                                    confirmation[:4], policy_type=descent.DescentPolicy)
    primary = checked["default"]
    accepted = primary["mean"] > placement_search.TARGET and all(placement_search.passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation", "confirmation": checked,
            "active_policy_above_target": accepted, "target_lower_bound": primary["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Compose independently acting source-column policies using held-out development outcomes.")
    for name in ("out", "reference", "timing", "parity"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed-block", type=int, default=133000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), [source.resolve() for source in args.source],
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
