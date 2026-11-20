from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from hashlib import sha256 as digest
from pathlib import Path

from a3opt import placement_benchmark, placement_first_hover, placement_hover_agent, placement_search, placement_sim
from a3opt.refinement_eval import read_json, sha256
from audit_placement_confirmation import report_rule, scheduled_seeds
from replay_placement import decode_model, verify_policy_code
from replicate_release_hold import NATIVE_FIELDS


def archived(path, expected):
    raw = gzip.decompress(path.read_bytes())
    if digest(raw).hexdigest() != expected:
        raise ValueError(f"archived receipts differ from their recorded digest: {path.name}")
    return json.loads(raw)


def audit(root, checkpoint):
    original_path, lock_path = root / "protocol.json", root / "confirmation-lock.json"
    original, lock = read_json(original_path), read_json(lock_path)
    protocol = read_json(root / "replication-protocol.json")
    result = read_json(root / "replication-result.json")
    kind = protocol["policy_kind"]
    if kind not in ("release-hold", "release-recovery", "expert-landing"):
        raise ValueError("expected a supported release policy replica")
    if (sha256(lock_path) != protocol["source_lock_sha256"] or sha256(original_path) != lock["protocol_sha256"]
            or sha256(checkpoint) != protocol["checkpoint_sha256"] or lock["checkpoint_sha256"] != protocol["checkpoint_sha256"]
            or lock["corrections"] != protocol["corrections"] or lock["seeds"] != original["schedule"]["confirmation"]
            or sha256(Path(placement_benchmark.__file__)) != protocol["evaluator_sha256"]
            or any(protocol[key] != original[key] for key in ("environment", "rules", "strict_rules"))):
        raise ValueError("replication does not match the original frozen inputs")
    policy_type, model = decode_model(lock["corrections"], kind, incumbent=lock["incumbent"],
                                      late_recovery=lock.get("late_recovery", ()))
    verify_policy_code(original, lock, kind, policy_type)

    prior = archived(root / "confirmation.json.gz", protocol["source_results_sha256"])
    new = archived(root / "replication-rollouts.json.gz", result["replication_sha256"])
    schedule = protocol["schedule"]
    seeds = schedule["replication"]
    if (not seeds or len(seeds) % 4 or len(seeds) != len(set(seeds))
            or set(seeds) & scheduled_seeds(original["schedule"])):
        raise ValueError("replication schedule overlaps model selection or confirmation")
    if kind == "expert-landing":
        plan = read_json(root / "replication-plan.json")
        if (plan["confirmation_lock_sha256"] != sha256(lock_path)
                or plan["schedule"]["replication"] != seeds or plan["episodes"] != len(seeds)
                or plan["by_column"] * 4 != len(seeds) or plan["run_if"] != "candidate_promoted"
                or plan["target_mean_ci95_lower_bound_exceeds"] != placement_search.TARGET):
            raise ValueError("expert landing replication differs from its registered schedule")
    for panel, planned in ((prior, lock["seeds"]), (new, seeds)):
        for arm, corrections in (("reference", ()), ("selected", model)):
            for seed, entry in zip(planned, panel[arm], strict=True):
                placement_benchmark.check((checkpoint, *lock["timing"], seed, corrections), entry)
            if Counter(row["default"]["column"] for row in panel[arm]) != dict.fromkeys(range(1, 5), len(planned) // 4):
                raise ValueError("release evaluation does not have four-column balance")
    pooled = {arm: prior[arm] + new[arm] for arm in ("reference", "selected")}
    for label, panel in (("replication", new), ("pooled", pooled)):
        statistics = placement_hover_agent.pair_stats(panel, model)
        for rule, row in statistics.items():
            if {**row, "by_column": report_rule(panel, rule)} != result[label][rule]:
                raise ValueError(f"archived {label} summary differs from its {rule} receipts")
    promoted = placement_first_hover.supports_confirmation(result["replication"])
    expected = "replication_passed" if promoted else "reference_retained_after_replication"
    target = result["replication"]["default"]["mean_ci95"][0] > placement_search.TARGET
    if result["status"] != expected or result["target_lower_bound"] != target:
        raise ValueError("archived decision differs from the frozen gates")

    parity = read_json(root / "replication-parity.json")
    if (parity["seeds"] != schedule["parity"] or not set(parity["seeds"]).issubset(lock["seeds"])
            or parity["matched_full_rollouts"] != 2 * len(parity["seeds"])):
        raise ValueError("invalid shared-prefix parity report")
    native = json.loads(gzip.decompress((root / "replication-native-replays.json.gz").read_bytes()))
    native_seeds = seeds[:4]
    expected_jobs = {(seed, strict) for seed in native_seeds for strict in (False, True)}
    rows = native["rows"]
    if (native["audit"]["seeds"] != native_seeds or native["audit"]["matched_native_replays"] != 8 or len(rows) != 8
            or {(row["seed"], row["strict"]) for row in rows} != expected_jobs):
        raise ValueError("expected one native replay per source column and scoring rule")
    indexed = {row["default"]["seed"]: row for row in new["selected"]}
    for row in rows:
        seed, strict = row["seed"], row["strict"]
        placement_sim.audit((checkpoint, *lock["timing"], "random", seed, model, strict), row)
        expected_row = indexed[seed]["strict-500" if strict else "default"]
        if any(row[key] != expected_row[key] for key in NATIVE_FIELDS):
            raise ValueError(f"native replay discrepancy for seed {seed}")
    return {"kind": kind, "status": expected, "native_replays_matched": len(rows), "target_lower_bound": target,
            "replication": result["replication"]["default"], "pooled": result["pooled"]["default"]}


def main():
    parser = argparse.ArgumentParser(description="Recompute a published frozen release replication from its archived receipts.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("evidence/advanced-20261120/policy.npz"))
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
