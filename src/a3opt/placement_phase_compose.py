from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt import placement_benchmark, placement_phase_agent, placement_search
from a3opt.advanced_agent import _record_once
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.refinement_eval import read_json, sha256


def screen(rows):
    report = []
    for column in sorted({row["column"] for row in rows}):
        group = [row for row in rows if row["column"] == column and row["trials"]]
        if {row["batch"] for row in group} != {0, 1}:
            continue
        for recipe in placement_phase_agent.RECIPES[1:]:
            gains, losses, violations = set(), set(), 0
            for row in group:
                original, trial = row["control"], row["trials"][recipe]
                change = len(trial["counted_ids"]) - len(original["counted_ids"])
                if change > 0:
                    gains.add(row["seed"])
                elif change < 0:
                    losses.add(row["seed"])
                violations += bool(trial["safety_reason"] or trial["wrong_source"] or (original["success"] and not trial["success"]))
            report.append({"column": column, "recipe": recipe, "gains": sorted(gains), "losses": sorted(losses),
                           "violations": violations, "admitted": len(gains) >= 2 and not (losses or violations)})
    return report


def select(rows, model):
    matched = {}
    for rule in ("default", "strict-500"):
        reference, selected = ([row[rule] for row in rows[side]] for side in ("reference", "selected"))
        matched[rule] = placement_search.compare(reference, selected, model)
    return matched


def locked_probes(acquisition, checkpoint, timing, identity):
    protocol_path = acquisition / "protocol.json"
    dataset_path = acquisition / "training" / "dataset.json"
    protocol, rows = read_json(protocol_path), read_json(dataset_path)["rows"]
    if (any(protocol[key] != identity[key] for key in ("environment", "strict_rules", "reference_sha256", "timing", "placement_sim_sha256"))
            or protocol["recipes"] != list(placement_phase_agent.RECIPES)
            or protocol["settle_steps"] != placement_phase_agent.SETTLE_STEPS
            or sha256(acquisition / "launch-source.py") != protocol["code_sha256"]
            or read_json(acquisition / "fit.json")["dataset_sha256"] != sha256(dataset_path)
            or [[row["batch"], row["seed"]] for row in rows] != protocol["probe_plan"]):
        raise ValueError("the composed model requires the original timing acquisition and its exact receipts")
    for row in rows:
        placement_phase_agent.audit((checkpoint, *timing, row["batch"], row["seed"]), row)
    return protocol, rows, {"protocol_sha256": sha256(protocol_path), "dataset_sha256": sha256(dataset_path)}


def run(root, reference, timing_run, acquisition, sources, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and verified joint evaluator are required")
    development, confirmation = list(range(block, block + 12)), list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    original, probes, evidence = locked_probes(acquisition, checkpoint, timing, identity)
    inputs = [placement_search.reuse_training(source, checkpoint, timing, identity) for source in sources]
    training = [row for rows, _ in inputs for row in rows]
    used = {row["seed"] for row in training}
    previous = original["schedule"]
    earlier_tests = previous["development"] + previous["confirmation"]["random"]
    if (len(used) != len(training) or used != set(previous["training"])
            or set(development + confirmation).intersection(used | set(earlier_tests))):
        raise ValueError("composition requires the original training scenarios and fresh evaluation seeds")
    splits["training"] = sorted(used)
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    active = screen(probes)
    screened = [entry for entry in active if entry["admitted"]]
    choice = max(screened, key=lambda item: len(item["gains"]), default=None)
    model, seeds = (), []
    if choice:
        column = choice["column"]
        seeds = sorted(row["seed"] for row in training if row["column"] == column)
        model = tuple(placement_phase_agent.PhaseTiming(column, batch, choice["recipe"]) for batch in (0, 1))
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "composition_seeds": seeds, "sampling": evidence,
                 "training_sources": [item for _, item in inputs], "policy_code_sha256": sha256(Path(placement_phase_agent.__file__)),
                 "code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "parity_sha256": sha256(parity), "target_mean_strictly_above": placement_search.TARGET})
    _record_once(root / "fit.json", {"screened": active, "selected": choice, "corrections": placement_search.encode(model)})
    if not model:
        return {"status": "reference_retained_without_repeated_training_gains"}
    print(f"Verifying both batches together in {len(seeds)} whole training episodes in column {column}", flush=True)
    models = {"reference": (), "selected": model}
    policy = placement_phase_agent.PhaseTimingPolicy
    with ProcessPoolExecutor(max_workers=workers) as pool:
        composed = placement_benchmark.evaluate(pool, root / "composition", checkpoint, timing, seeds, models, policy_type=policy)
        joint = select(composed, model)
        _record_once(root / "composition-selection.json", joint)
        if not all(placement_search.passes(stats) and len([entry for entry in stats["changes"] if entry["delta"] > 0]) >= 2
                   for stats in joint.values()):
            return {"status": "reference_retained_after_composition", "training": joint}
        print(f"Composition passed training; evaluating {len(development)} new development pairs", flush=True)
        pairs = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development, models, policy_type=policy)
        dev = select(pairs, model)
        _record_once(root / "development-selection.json", dev)
        if not all(placement_search.passes(stats) for stats in dev.values()):
            return {"status": "reference_retained_after_development", "training": joint, "development": dev}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "corrections": placement_search.encode(model), "seeds": confirmation,
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        print(f"Frozen for {len(confirmation)} independent confirmation pairs", flush=True)
        checked = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                   confirmation[:4], policy_type=policy)
    primary = checked["default"]
    above = primary["mean"] > placement_search.TARGET
    accepted = above and all(placement_search.passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation",
            "training": joint, "development": dev, "confirmation": checked, "target_point_estimate": above,
            "target_lower_bound": primary["mean_ci95"][0] > placement_search.TARGET, "active_policy_above_target": accepted}


def main():
    parser = argparse.ArgumentParser(description="Test composition of repeated placement timing gains across batches.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed-block", type=int, default=126000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.acquisition.resolve(),
                     [item.resolve() for item in args.source], args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
