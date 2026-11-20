from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark as benchmark, placement_search, placement_sim
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES

from audit_placement_confirmation import scheduled_seeds
from audit_source_guard import endpoints
from confirm_expert_grasp import audit
from confirm_source_guard import compare, inputs
from replicate_release_hold import NATIVE_FIELDS
from replicate_source_guard import RETAINED


SEED_START = 179600
EPISODES = 192
RULES_NAMES = ("default", "strict-500")


def summarize(rows):
    columns = [row["column"] for row in rows]
    if Counter(columns) != dict.fromkeys(range(1, 5), EPISODES // 4):
        raise ValueError("a validation panel must have four equally represented source columns")
    values = np.array([row["requested_score"] for row in rows])
    return {**endpoints(rows), "episodes": len(rows), "requested_mean_ci95": placement_search._bootstrap(values, columns),
            "by_column": {str(column): endpoints([row for row in rows if row["column"] == column]) for column in range(1, 5)}}


def run(source, parent, output, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, timing, constructor, model, lock, _, selection = inputs(source, parent)
    prior = audit(source, checkpoint, timing, model, lock, selection, compare_rows=compare, retained_status=RETAINED)
    seeds = list(range(SEED_START, SEED_START + EPISODES))
    columns = [choose_column("random", seed) for seed in seeds]
    if Counter(columns) != dict.fromkeys(range(1, 5), EPISODES // 4) or len(set(columns[:4])) != 4:
        raise ValueError("the registered validation and native spot checks require four source columns")
    protocols = [*output.parent.glob("*/protocol.json"), *output.parent.glob("*/replication-plan.json"),
                 *checkpoint.parent.parent.glob("*/*protocol.json"), *checkpoint.parent.parent.glob("*/replication-plan.json")]
    if any(set(seeds) & scheduled_seeds(read_json(path).get("schedule", {})) for path in protocols if path.parent != output):
        raise ValueError("score validation overlaps a previous training or evaluation panel")
    native_seeds = seeds[:4]
    protocol = output / "protocol.json"
    _record_once(protocol, {"environment": simulator_identity(), "rules": RULES, "strict_rules": STRICT_RULES,
                 "checkpoint_sha256": sha256(checkpoint), "source_lock_sha256": sha256(source / "confirmation-lock.json"),
                 "prior_confirmation_sha256": prior["rollouts_sha256"], "corrections": [item.record() for item in model],
                 "driver_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(benchmark.__file__)),
                 "schedule": {"validation": seeds, "native_spot_checks": native_seeds},
                 "target_requested_mean_ci95_lower_bound_exceeds": placement_search.TARGET})
    print(f"Validating the frozen controller on {len(seeds)} new four-column scenes", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = benchmark.evaluate(pool, output / "validation", checkpoint, timing, seeds, {"selected": model}, policy_type=constructor)["selected"]
        chosen = {episode["default"]["seed"]: episode for episode in rows}
        jobs = [(checkpoint, *timing, "random", seed, model, strict) for seed in native_seeds for strict in (False, True)]
        native = _jobs(pool, partial(placement_sim.episode, policy_type=constructor), jobs, output / "native-audit",
                       lambda job: f"{int(job[-1])}/{job[4]}", placement_sim.audit)
        for row in native:
            rule = RULES_NAMES[int(row["strict"])]
            if any(row[key] != chosen[row["seed"]][rule][key] for key in NATIVE_FIELDS):
                raise ValueError(f"native score does not match the validation receipts for seed {row['seed']}")
    replay = {"seeds": native_seeds, "matched_native_replays": len(native), "matched_fields": list(NATIVE_FIELDS)}
    _record_once(output / "native-audit" / "audit.json", replay)
    absolute = {rule: summarize([episode[rule] for episode in rows]) for rule in RULES_NAMES}
    result = {"metrics": absolute, "native_audit": replay, "rollouts_sha256": sha256(output / "validation" / "results.json"),
              "target_lower_bound": absolute["default"]["requested_mean_ci95"][0] > placement_search.TARGET}
    _record_once(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate the absolute advanced score of a frozen controller on 192 reserved scenes.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    source, parent, output = (path.resolve() for path in (args.source, args.parent, args.out))
    if output in (source, parent) or args.workers < 1:
        parser.error("expected separate output and a positive worker budget")
    with _run_lock(output):
        print(json.dumps(run(source, parent, output, args.workers), indent=2), flush=True)


if __name__ == "__main__":
    main()
