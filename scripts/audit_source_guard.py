from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from a3opt.refinement_eval import read_json, sha256

from confirm_expert_grasp import audit
from confirm_source_guard import compare
from frozen_source_policy import load_policy
from replicate_source_guard import RETAINED


def endpoints(rows):
    occupancy = [sum(cookie >= 0 for cookie in row["terminal"]["slot_occupancy"]) for row in rows]
    return {
        "simulator_raw_mean": mean(row["score"] for row in rows),
        "requested_mean": mean(row["requested_score"] for row in rows),
        "occupied_slots_mean": mean(occupancy),
        "full_slots": sum(count == 10 for count in occupancy),
        "complete": sum(row["success"] for row in rows),
        "wrong_source_cases": [{"seed": row["seed"], "column": row["column"], "counts_by_source": row["by_column"]}
                               for row in rows if row["score"] != row["requested_score"]],
        "safety_stop_cases": [{"seed": row["seed"], "reason": row["safety_reason"]} for row in rows if row["safety_reason"]],
    }


def review(root, parent, checkpoint):
    lock_path = root / "confirmation-lock.json"
    lock = read_json(lock_path)
    policy = load_policy(lock_path, parent / "confirmation-lock.json", checkpoint)
    checkpoint = checkpoint or Path(lock["checkpoint"])
    selection_path = root / "development-selection.json"
    if sha256(selection_path) != lock["selection_sha256"]:
        raise ValueError("development selection differs from the frozen run")
    confirmation_root = root / "confirmation"
    paired = read_json(confirmation_root / "rollouts" / "results.json")
    report = audit(root, checkpoint, tuple(lock["timing"]), (policy.landing, policy.guard), lock, read_json(selection_path),
                   compare_rows=compare, retained_status=RETAINED)
    report["endpoints"] = {rule: {arm: endpoints([entry[rule] for entry in entries]) for arm, entries in paired.items()}
                           for rule in ("default", "strict-500")}
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit archived source-guard confirmation receipts and both score endpoints.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.run, args.parent, args.checkpoint), indent=2), flush=True)


if __name__ == "__main__":
    main()
