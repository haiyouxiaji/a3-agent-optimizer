from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

from a3opt.advanced_agent import TARGET, _record_once
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_outcome_agent import MAJOR_DROP, check_pair


def success_changes(reference: list[dict], candidate: list[dict], field: str) -> dict:
    changes = []
    for baseline, trial in zip(reference, candidate, strict=True):
        if baseline["seed"] != trial["seed"]:
            raise ValueError("success inference requires matched seeds")
        changes.append(int(trial[field]) - int(baseline[field]))
    gains, losses = changes.count(1), changes.count(-1)
    discordant = gains + losses
    exact_p = min(1., 2 * sum(comb(discordant, count) for count in range(min(gains, losses) + 1)) / 2**discordant)
    return {"episodes": len(changes), "gains": gains, "losses": losses, "two_sided_exact_p": exact_p}


def decide(release: Path, focus: Path, strict: Path) -> dict:
    lock_path = release / "confirmation-lock.json"
    lock = read_json(lock_path)
    parent_path = release / "result.json"
    parent = read_json(parent_path)
    parent_receipts = release / "confirmation/results.json"
    focus_protocol_path = focus / "protocol.json"
    focus_protocol = read_json(focus_protocol_path)
    focus_result_path, focus_receipts = focus / "result.json", focus / "paired/results.json"
    focus_result, focus_rows = read_json(focus_result_path), read_json(focus_receipts)
    strict_protocol_path, strict_result_path = strict / "protocol.json", strict / "result.json"
    strict_protocol, strict_result = read_json(strict_protocol_path), read_json(strict_result_path)
    if (focus_protocol["parent_confirmation_lock_sha256"] != sha256(lock_path)
            or focus_protocol["checkpoint_sha256"] != lock["checkpoint_sha256"]
            or focus_result["protocol_sha256"] != sha256(focus_protocol_path)
            or focus_result["results_sha256"] != sha256(focus_receipts)
            or strict_protocol["source_lock_sha256"] != sha256(lock_path)
            or strict_protocol["source_results_sha256"] != sha256(parent_receipts)
            or strict_protocol["checkpoints"]["selected"] != lock["checkpoint_sha256"]
            or focus_protocol["source_column"] != strict_protocol["source_column"]):
        raise ValueError("deployment assessment requires the same locked candidate across all comparisons")
    baseline, trial = (focus_rows[label] for label in ("reference", "selected"))
    for label, group in focus_rows.items():
        expected = focus_protocol["reference_sha256"] if label == "reference" else lock["checkpoint_sha256"]
        if [row["seed"] for row in group] != focus_protocol["seeds"] or any(row["checkpoint_sha256"] != expected for row in group):
            raise ValueError("focused result contains an unexpected scenario or controller")
    for old, new in zip(baseline, trial, strict=True):
        check_pair(old, new)
    strict_baseline, strict_trial = (strict_result["rows"][label] for label in ("reference", "selected"))
    for label, group in strict_result["rows"].items():
        if ([row["seed"] for row in group] != strict_protocol["seeds"]
                or any(row["checkpoint_sha256"] != strict_protocol["checkpoints"][label] for row in group)):
            raise ValueError("strict confirmation contains an unexpected scenario or controller")
    focus_changes = success_changes(baseline, trial, "success")
    strict_changes = success_changes(strict_baseline, strict_trial, "strict_success")
    major_losses = [old["seed"] for old, new in zip(baseline, trial, strict=True)
                    if old["requested_score"] - new["requested_score"] >= MAJOR_DROP]
    total_gain = sum(new["requested_score"] - old["requested_score"] for old, new in zip(baseline, trial, strict=True))
    gates = {"original_confirmation_passed": parent["status"] == "promoted" and parent["scores"]["random"] > TARGET,
             "focused_terminal_count_nonregression": total_gain >= 0, "focused_no_major_loss": not major_losses,
             "released_success_improved_on_triggered_seeds": strict_changes["gains"] > strict_changes["losses"],
             "no_candidate_violations": all(row["score"] == row["requested_score"] and not row["safety_reason"] for row in trial)
             and all(row["strict_score"] == row["strict_requested_score"] and not row["safety_reason"] for row in strict_trial)}
    return {"deployment_recommendation": "trial" if all(gates.values()) else "reference",
            "candidate_sha256": lock["checkpoint_sha256"], "reference_sha256": focus_protocol["reference_sha256"],
            "gates": gates, "major_drop_threshold": MAJOR_DROP, "focused_major_loss_seeds": major_losses,
            "focused_mean_gain": total_gain / len(trial), "focused_success_changes": focus_changes,
            "strict_success_changes_on_triggered_seeds": strict_changes,
            "inputs_sha256": {name: sha256(path) for name, path in {
                "release_result": parent_path, "release_lock": lock_path, "focus_result": focus_result_path,
                "focus_protocol": focus_protocol_path, "strict_result": strict_result_path, "strict_protocol": strict_protocol_path,
                "decision_code": Path(__file__),
            }.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a deployment guard using focused and released-outcome receipts.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--focus", type=Path, required=True)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = decide(args.release, args.focus, args.strict)
    _record_once(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
