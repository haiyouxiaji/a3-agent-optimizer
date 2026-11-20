from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from inspect import getfile
from pathlib import Path
from statistics import mean

from a3opt import placement_benchmark, placement_search, placement_sim
from a3opt.advanced_agent import _record_once
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES
from replay_placement import decode_model


def report_rule(pairs, rule):
    reference = [episode[rule] for episode in pairs["reference"]]
    selected = [episode[rule] for episode in pairs["selected"]]
    details = {}
    for column in range(1, 5):
        columns = [[row for row in rows if row["column"] == column] for rows in (reference, selected)]
        details[str(column)] = {"episodes": len(columns[0]),
                                "means": [mean(row["requested_score"] for row in rows) for rows in columns],
                                "complete": [sum(row["success"] for row in rows) for rows in columns]}
    return details


def audit_confirmation(root, kind):
    protocol_path = root / "protocol.json"
    protocol = read_json(protocol_path)
    lock = read_json(root / "confirmation-lock.json")
    fit_path, selection_path = root / "fit.json", root / "development-selection.json"
    checkpoint = Path(lock["checkpoint"])
    seeds, corrections, timing = lock["seeds"], lock["corrections"], lock["timing"]
    schedule = protocol["schedule"]
    seen = set(schedule["scout"]["random"]) | set(schedule["training"]) | set(schedule["development"])
    if (set(seeds) & seen or len(seeds) != len(set(seeds)) or seeds != schedule["confirmation"]["random"]
            or protocol["environment"] != simulator_identity() or protocol["rules"] != RULES
            or protocol["strict_rules"] != STRICT_RULES or protocol["timing"] != timing
            or protocol["evaluator_sha256"] != sha256(Path(placement_benchmark.__file__))
            or sha256(protocol_path) != lock["protocol_sha256"] or sha256(checkpoint) != lock["checkpoint_sha256"]
            or sha256(fit_path) != lock["fit_sha256"] or sha256(selection_path) != lock["selection_sha256"]
            or corrections != read_json(fit_path)["corrections"]):
        raise ValueError("the confirmation lock, splits, or inputs differ from the saved run")
    policy_type, selected = decode_model(corrections, kind)
    policy_hash = protocol.get("policy_code_sha256", protocol["code_sha256"])
    if sha256(Path(getfile(policy_type))) != policy_hash:
        raise ValueError("the controller differs from the recorded evaluation")
    confirmation = root / "confirmation"
    request = read_json(confirmation / "rollouts" / "request.json")
    if request != {"checkpoint_sha256": lock["checkpoint_sha256"], "timing": timing, "seeds": seeds,
                   "models": {"reference": [], "selected": corrections}, "rules": [RULES, STRICT_RULES]}:
        raise ValueError("rollout request differs from the confirmation lock")
    rows_path = confirmation / "rollouts" / "results.json"
    rows = read_json(rows_path)
    for arm, model in (("reference", ()), ("selected", selected)):
        for seed, episode in zip(seeds, rows[arm], strict=True):
            placement_benchmark.check((checkpoint, *timing, seed, model), episode)
    columns = Counter(episode["default"]["column"] for episode in rows["selected"])
    if set(columns) != {1, 2, 3, 4} or len(set(columns.values())) != 1:
        raise ValueError("confirmation is not four-column balanced")
    stored = read_json(root / "result.json")
    measurements = {}
    for rule in ("default", "strict-500"):
        controls, trials = ([episode[rule] for episode in rows[arm]] for arm in ("reference", "selected"))
        measured = placement_search.compare(controls, trials, selected)
        if measured != stored["confirmation"][rule]:
            raise ValueError(f"the saved {rule} aggregate differs from its paired rollouts")
        measurements[rule] = {**measured, "by_column": report_rule(rows, rule)}
    native = read_json(confirmation / "audit.json")
    if native["seeds"] != seeds[:4] or native["matched_native_replays"] != 8:
        raise ValueError("expected eight native replays, one scene per column in both modes")
    selected_by_seed = dict(zip(seeds, rows["selected"], strict=True))
    native_keys = ("score", "requested_score", "target_cookie_ids", "terminal", "success", "steps",
                   "safety_reason", "action_sha256", "applied_action_sha256", "source_remaining")
    for seed in native["seeds"]:
        for strict, rule in ((False, "default"), (True, "strict-500")):
            native_row = read_json(confirmation / "native-audit" / str(int(strict)) / f"{seed}.json")
            placement_sim.audit((checkpoint, *timing, "random", seed, selected, strict), native_row)
            if any(native_row[key] != selected_by_seed[seed][rule][key] for key in native_keys):
                raise ValueError(f"native replay mismatch for {seed} in {rule}")
    promoted = measurements["default"]["mean"] > placement_search.TARGET and all(
        placement_search.passes(stats) for stats in measurements.values())
    if promoted != (stored["status"] == "candidate_promoted"):
        raise ValueError("saved promotion decision differs from the confirmation evidence")
    return {"run": root.name, "status": stored["status"], "corrections": corrections, "native_replays": 8,
            "evaluation_sha256": sha256(rows_path), "seeds": seeds, "confirmation": measurements}


def scheduled_seeds(schedule):
    if isinstance(schedule, dict):
        return {seed for entry in schedule.values() for seed in scheduled_seeds(entry)}
    if isinstance(schedule, list):
        return {seed for entry in schedule for seed in scheduled_seeds(entry)}
    return {schedule} if type(schedule) is int else set()


def replicate(source, output, kind, seed_start, episodes, workers):
    if seed_start < 0 or seed_start % 4 or episodes < 20 or episodes % 4 or workers < 1:
        raise ValueError("replication requires a positive worker budget and a balanced four-column panel")
    origin = audit_confirmation(source, kind)
    if origin["status"] != "candidate_promoted":
        raise ValueError("replication requires a controller that passed its original confirmation")
    lock_path = source / "confirmation-lock.json"
    lock = read_json(lock_path)
    checkpoint, timing = Path(lock["checkpoint"]), tuple(lock["timing"])
    seeds = list(range(seed_start, seed_start + episodes))
    for path in sorted(output.parent.glob("*/protocol.json")):
        if path.parent != output and set(seeds).intersection(scheduled_seeds(read_json(path).get("schedule", {}))):
            raise ValueError(f"replication overlaps a reserved panel: {path.parent}")
    policy_type, model = decode_model(lock["corrections"], kind)
    _record_once(output / "protocol.json", {"environment": simulator_identity(), "rules": RULES,
                 "strict_rules": STRICT_RULES, "schedule": {"replication": {"random": seeds}},
                 "reference_sha256": sha256(checkpoint), "timing": list(timing), "model": lock["corrections"],
                 "source_lock_sha256": sha256(lock_path), "source_results_sha256": origin["evaluation_sha256"],
                 "policy_code_sha256": sha256(Path(getfile(policy_type))), "audit_code_sha256": sha256(Path(__file__)),
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__)), "target_mean_strictly_above": placement_search.TARGET})
    print(f"Frozen replication of {source.name}: {episodes} new advanced scenes", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        confirmation = placement_search.confirm_combined(pool, output / "confirmation", checkpoint, timing, seeds,
                                                         model, seeds[:4], policy_type=policy_type)
    initial = read_json(source / "confirmation" / "rollouts" / "results.json")
    followup_path = output / "confirmation" / "rollouts" / "results.json"
    followup = read_json(followup_path)
    union = {arm: initial[arm] + followup[arm] for arm in ("reference", "selected")}
    pooled = {}
    for rule in ("default", "strict-500"):
        reference, selected = ([episode[rule] for episode in union[arm]] for arm in ("reference", "selected"))
        confirmation[rule]["by_column"] = report_rule(followup, rule)
        pooled[rule] = {**placement_search.compare(reference, selected, model), "by_column": report_rule(union, rule)}
    supported = confirmation["default"]["mean"] > placement_search.TARGET and all(
        placement_search.passes(stats) for stats in confirmation.values())
    return {"replication_above_target": supported, "replication": confirmation, "pooled": pooled,
            "replication_results_sha256": sha256(followup_path), "source_results_sha256": origin["evaluation_sha256"]}


def main():
    parser = argparse.ArgumentParser(description="Verify a frozen advanced placement confirmation from its episode and native audit receipts.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--policy-kind", required=True, choices=("descent", "servo", "clearance", "landing"))
    parser.add_argument("--replicate-out", type=Path)
    parser.add_argument("--seed-start", type=int, default=138000)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    source = args.run.resolve()
    if args.replicate_out:
        output = args.replicate_out.resolve()
        if source == output:
            raise ValueError("replication requires its own output directory")
        with _run_lock(output):
            result = replicate(source, output, args.policy_kind, args.seed_start, args.episodes, args.workers)
            _record_once(output / "result.json", result)
        print(json.dumps({"replication_above_target": result["replication_above_target"],
                          "replication_mean": result["replication"]["default"]["mean"],
                          "pooled_mean": result["pooled"]["default"]["mean"]}, indent=2))
    else:
        print(json.dumps(audit_confirmation(source, args.policy_kind), indent=2))


if __name__ == "__main__":
    main()
