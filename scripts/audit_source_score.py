from __future__ import annotations

import argparse
import json
from pathlib import Path

from a3opt import placement_benchmark as benchmark, placement_search, placement_sim
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES

from frozen_source_policy import load_policy
from replicate_release_hold import NATIVE_FIELDS
from validate_source_score import EPISODES, RULES_NAMES, SEED_START, summarize
from validate_source_score import __file__ as validation_driver


def audit(root, source, parent, checkpoint):
    origin_lock = source / "confirmation-lock.json"
    lock = read_json(origin_lock)
    policy = load_policy(origin_lock, parent / "confirmation-lock.json", checkpoint)
    model, timing = (policy.landing, policy.guard), tuple(lock["timing"])
    protocol = read_json(root / "protocol.json")
    seeds = list(range(SEED_START, SEED_START + EPISODES))
    native_seeds = seeds[:4]
    if (protocol["environment"] != simulator_identity() or (protocol["rules"], protocol["strict_rules"]) != (RULES, STRICT_RULES)
            or protocol["checkpoint_sha256"] != sha256(checkpoint) or protocol["source_lock_sha256"] != sha256(origin_lock)
            or protocol["driver_sha256"] != sha256(Path(validation_driver)) or protocol["evaluator_sha256"] != sha256(Path(benchmark.__file__))
            or protocol["corrections"] != [item.record() for item in model]
            or protocol["prior_confirmation_sha256"] != read_json(source / "confirmation-audit.json")["rollouts_sha256"]
            or protocol["schedule"] != {"validation": seeds, "native_spot_checks": native_seeds}
            or protocol["target_requested_mean_ci95_lower_bound_exceeds"] != placement_search.TARGET):
        raise ValueError("score validation does not reproduce the frozen evaluation protocol")
    folder = root / "validation"
    aggregate = folder / "results.json"
    request = read_json(folder / "request.json")
    if request != {"checkpoint_sha256": sha256(checkpoint), "timing": list(timing), "seeds": seeds,
                   "models": {"selected": protocol["corrections"]}, "rules": [RULES, STRICT_RULES]}:
        raise ValueError("requested score validation differs from the protocol")
    rows = read_json(aggregate)["selected"]
    for seed, episode in zip(seeds, rows, strict=True):
        benchmark.check((checkpoint, *timing, seed, model), episode)
        if episode != read_json(folder / "episodes" / "selected" / f"{seed}.json"):
            raise ValueError(f"individual and aggregate score receipts differ for seed {seed}")
    native_root = root / "native-audit"
    for seed, episode in zip(native_seeds, rows[:len(native_seeds)], strict=True):
        for strict, rule in enumerate(RULES_NAMES):
            native = read_json(native_root / str(strict) / f"{seed}.json")
            placement_sim.audit((checkpoint, *timing, "random", seed, model, bool(strict)), native)
            if any(native[key] != episode[rule][key] for key in NATIVE_FIELDS):
                raise ValueError(f"native evidence differs from the {rule} score receipt for seed {seed}")
    replay = {"seeds": native_seeds, "matched_native_replays": 2 * len(native_seeds), "matched_fields": list(NATIVE_FIELDS)}
    if read_json(native_root / "audit.json") != replay:
        raise ValueError("native replay summary differs from the checked receipts")
    metrics = {rule: summarize([episode[rule] for episode in rows]) for rule in RULES_NAMES}
    report = {"metrics": metrics, "native_audit": replay, "rollouts_sha256": sha256(aggregate),
              "target_lower_bound": metrics["default"]["requested_mean_ci95"][0] > placement_search.TARGET}
    if read_json(root / "result.json") != report:
        raise ValueError("absolute score report differs from the audited receipts")
    return report


def main():
    parser = argparse.ArgumentParser(description="Recompute a published advanced score validation from its archived simulator receipts.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("evidence/advanced-20261120/policy.npz"))
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.source, args.parent, args.checkpoint), indent=2), flush=True)


if __name__ == "__main__":
    main()
