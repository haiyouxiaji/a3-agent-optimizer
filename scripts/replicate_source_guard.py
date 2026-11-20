from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from a3opt import placement_search
from a3opt.advanced_agent import _record_once
from a3opt.cli import _run_lock
from a3opt.refinement_eval import read_json, sha256

from audit_placement_confirmation import scheduled_seeds
from confirm_expert_grasp import audit, measurements, resume
from confirm_source_guard import calibration, compare, inputs


RETAINED = "reference_retained_after_source_guard_confirmation"


def registered_seeds(plan, source_lock_digest, reserved):
    from a3_dual_arm_sim.sim.randomization import choose_column

    seeds = plan["schedule"]["replication"]
    if (plan["source_lock_sha256"] != source_lock_digest
            or plan["run_if"] != "candidate_promoted_with_absolute_interval_covering_target"
            or plan["target_mean_ci95_lower_bound_exceeds"] != placement_search.TARGET):
        raise ValueError("replication differs from the registered controller or trigger")
    if (not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or plan["episodes"] != len(seeds) or len(set(seeds)) != len(seeds) or set(seeds) & reserved):
        raise ValueError("replication requires the registered number of unused, unique seeds")
    columns = [choose_column("random", seed) for seed in seeds]
    expected = dict.fromkeys(range(1, 5), plan["per_column"])
    if Counter(columns) != expected or set(columns[:4]) != set(expected):
        raise ValueError("replication and its native spot checks must cover four source columns")
    return seeds


def run(source, parent, output, workers):
    checkpoint, timing, constructor, models, lock, development, selection = inputs(source, parent)
    origin = audit(source, checkpoint, timing, models, lock, selection, compare_rows=compare, retained_status=RETAINED)
    lower, upper = origin["confirmation"]["default"]["mean_ci95"]
    if origin["status"] != "candidate_promoted" or not lower <= placement_search.TARGET <= upper:
        raise ValueError("the original confirmation did not trigger the registered replication")
    lock_path = source / "confirmation-lock.json"
    plan_path = source / "replication-plan.json"
    prior_protocols = [source / "protocol.json", parent / "protocol.json", parent / "replication-plan.json"]
    reserved = set().union(*(scheduled_seeds(read_json(path)["schedule"]) for path in prior_protocols if path.is_file()))
    seeds = registered_seeds(read_json(plan_path), sha256(lock_path), reserved)
    prior = read_json(source / "confirmation" / "rollouts" / "results.json")
    replication_lock = {**lock, "seeds": seeds}
    _record_once(output / "protocol.json", {"source_lock_sha256": sha256(lock_path), "plan_sha256": sha256(plan_path),
                 "confirmation_sha256": origin["rollouts_sha256"], "schedule": {"replication": seeds},
                 "replication_driver_sha256": sha256(Path(__file__))})
    _record_once(output / "confirmation-lock.json", replication_lock)
    if not (output / "result.json").is_file():
        expected, parity = calibration(parent, models, development)
        resume(output, checkpoint, timing, constructor, models, replication_lock, expected, selection, workers,
               compare_rows=compare, parity_seeds=parity)
    replication = audit(output, checkpoint, timing, models, replication_lock, selection, compare_rows=compare,
                        retained_status="reference_retained_after_source_guard_replication")
    repeated = read_json(output / "confirmation" / "rollouts" / "results.json")
    pooled = {arm: prior[arm] + repeated[arm] for arm in ("reference", "selected")}
    summary = {"replication": replication, "pooled": measurements(pooled, models, compare_rows=compare)}
    _record_once(output / "replication-audit.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Replay the registered source-guard replication with a frozen controller.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    source, parent, output = args.source.resolve(), args.parent.resolve(), args.out.resolve()
    if output in (source, parent) or args.workers < 1:
        parser.error("replication requires a separate output directory and a positive worker budget")
    with _run_lock(output):
        print(json.dumps(run(source, parent, output, args.workers), indent=2), flush=True)


if __name__ == "__main__":
    main()
