from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt import placement_benchmark, placement_descent_agent as descent, placement_search, placement_wrist_rgb as wrist
from a3opt.advanced_agent import _record_once
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.refinement_eval import read_json, sha256


def choose_profile(receipts):
    evidence = {}
    for profile in descent.PROFILES:
        gains = {}
        for row in receipts:
            if row["options"] is None:
                continue
            control = row["options"]["reference"]
            alternatives = (end for name, end in row["options"].items() if name.startswith(f"{profile}/"))
            safe = [len(end["counted_ids"]) - len(control["counted_ids"]) for end in alternatives
                    if not (end["safety_reason"] or end["wrong_source"] or (control["success"] and not end["success"]))]
            if safe and max(safe) > 0:
                gains[row["seed"]] = max(safe)
        evidence[profile] = gains
    eligible = [name for name, gained in evidence.items() if len(gained) >= 2]
    profile = max(eligible, key=lambda name: (len(evidence[name]), sum(evidence[name].values())), default=None)
    return profile, {name: {str(seed): delta for seed, delta in gains.items()} for name, gains in evidence.items()}


def select_cases(rows):
    groups = {}
    for column in range(1, 5):
        samples = [row for row in rows if row["column"] == column and [item["batch"] for item in row["samples"]] == [0, 1]]
        failures = sorted((row for row in samples if row["score"] < 10), key=lambda row: (row["score"], row["seed"]))
        controls = sorted(row["seed"] for row in samples if row["score"] == 10)
        if len(failures) >= 2 and len(controls) >= 2:
            groups[column] = {"all_failure_count": len(failures), "failure_cookies": sum(10 - row["score"] for row in failures),
                              "seeds": [row["seed"] for row in failures[:3]] + controls[:3]}
    column = max(groups, key=lambda item: (groups[item]["failure_cookies"], groups[item]["all_failure_count"]), default=None)
    return column, groups


def paired(rows, name, model):
    return {rule: placement_search.compare([row[rule] for row in rows["reference"]], [row[rule] for row in rows[name]], model)
            for rule in ("default", "strict-500")}


def focused_data(path, checkpoint, timing, seeds):
    locations = [path / "training" / "episodes" / f"{seed}.json" for seed in seeds]
    deadline = time.monotonic() + 600
    while not all(item.is_file() for item in locations):
        if time.monotonic() > deadline or any(item.with_suffix(".error.json").is_file() for item in locations):
            raise RuntimeError("the first descent acquisition did not produce all focused training references")
        time.sleep(5.)
    rows = [read_json(item) for item in locations]
    for seed, row in zip(seeds, rows, strict=True):
        descent.audit((checkpoint, *timing, seed), row)
    return rows, [sha256(item) for item in locations]


def run(root, reference, timing_run, parent, acquisition, sources, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("focused descent requires balanced seeds and verified scoring")
    development, confirmation = list(range(block, block + 20)), list(range(block + 100, block + 140))
    parent_protocol_path, wrist_protocol_path = parent / "protocol.json", acquisition / "protocol.json"
    parent_protocol, wrist_protocol = read_json(parent_protocol_path), read_json(wrist_protocol_path)
    prior_seeds = parent_protocol["focused"] + wrist_protocol["schedule"]["training"]
    schedule = {"scout": {"random": []}, "training": prior_seeds, "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, schedule)
    for protocol, module in ((parent_protocol, descent), (wrist_protocol, wrist)):
        previous = protocol["schedule"]
        if (any(protocol[key] != identity[key] for key in ("environment", "reference_sha256", "timing", "strict_rules"))
                or protocol["code_sha256"] != sha256(Path(module.__file__))
                or set(previous["development"] + previous["confirmation"]["random"]).intersection(prior_seeds + development + confirmation)
                or set(previous["training"]).intersection(development + confirmation)):
            raise ValueError("focused descent requires compatible training and unused evaluation scenes")
    sources_data = [placement_search.reuse_training(source, checkpoint, timing, identity) for source in sources]
    if [receipt for _, receipt in sources_data] != parent_protocol["training_sources"]:
        raise ValueError("focused descent requires the parent's original failure-screening receipts")
    original = {row["seed"]: row for group, _ in sources_data for row in group}
    seed_subset = [seed for seed in parent_protocol["focused"] if original[seed]["score"] < 10]
    rows, source_hashes = focused_data(parent, checkpoint, timing, seed_subset)
    if any(row["score"] != original[row["seed"]]["score"] for row in rows):
        raise ValueError("focused descent baseline count changed across training acquisitions")
    dataset_path = acquisition / "training" / "dataset.json"
    acquired = read_json(dataset_path)["rows"]
    if ([row["seed"] for row in acquired] != wrist_protocol["schedule"]["training"]
            or read_json(acquisition / "fit.json")["dataset_sha256"] != sha256(dataset_path)):
        raise ValueError("wrist training receipt list changed")
    for row in acquired:
        wrist.audit((checkpoint, *timing, acquisition / "training" / "frames", row["seed"]), row)
    profile, support = choose_profile(rows)
    column, groups = select_cases(acquired)
    seeds = groups[column]["seeds"] if column else []
    _record_once(root / "protocol.json", {**identity, "schedule": schedule, "sampled_training_seeds": seeds,
                 "descent_training_seeds": seed_subset, "descents": source_hashes,
                 "descent_protocol_sha256": sha256(parent_protocol_path),
                 "wrist_protocol_sha256": sha256(wrist_protocol_path), "wrist_dataset_sha256": sha256(dataset_path),
                 "code_sha256": sha256(Path(__file__)), "policy_code_sha256": sha256(Path(descent.__file__)),
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "target_mean_strictly_above": placement_search.TARGET})
    _record_once(root / "acquisition-plan.json", {"profile": profile, "profile_support": support, "column": column,
                 "candidate_columns": {str(key): value for key, value in groups.items()}, "training_seeds": seeds})
    if not profile or not column:
        return {"status": "reference_retained_without_focused_support"}
    print(f"Comparing {profile} on {len(seeds)} newly selected training scenes in source column {column}", flush=True)
    models = {"reference": (), **{"+".join(map(str, scope)): tuple(descent.Descent(column, batch, profile) for batch in scope)
                                   for scope in descent.SCOPES}}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        samples = placement_benchmark.evaluate(pool, root / "training", checkpoint, timing, seeds, models, policy_type=descent.DescentPolicy)
        stats = {name: paired(samples, name, model) for name, model in models.items() if model}
        eligible = [name for name, measures in stats.items() if all(placement_search.passes(entry)
                    and sum(change["delta"] > 0 for change in entry["changes"]) >= 2 for entry in measures.values())]
        choice = max(eligible, key=lambda name: (stats[name]["default"]["mean"], stats[name]["strict-500"]["complete"], -len(models[name])),
                     default=None)
        _record_once(root / "fit.json", {"selected_scope": choice, "training": stats,
                     "corrections": placement_search.encode(models[choice]) if choice else []})
        if not choice:
            return {"status": "reference_retained_after_focused_training"}
        model = models[choice]
        print(f"Verified repeated gains on training scenes for {choice}; opening development", flush=True)
        dev_rows = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                 {"reference": (), "selected": model}, policy_type=descent.DescentPolicy)
        dev = paired(dev_rows, "selected", model)
        _record_once(root / "development-selection.json", dev)
        if not all(placement_search.passes(stats) for stats in dev.values()):
            return {"status": "reference_retained_after_development", "development": dev}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": placement_search.encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                    confirmation[:4], policy_type=descent.DescentPolicy)
    primary = checked["default"]
    accepted = primary["mean"] > placement_search.TARGET and all(placement_search.passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation", "development": dev,
            "confirmation": checked, "active_policy_above_target": accepted,
            "target_lower_bound": primary["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Focus descent sampling using previous training outcomes and wrist acquisition receipts.")
    for name in ("out", "reference", "timing", "parent", "acquisition", "parity"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-block", type=int, default=132000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.parent.resolve(), args.acquisition.resolve(),
                     [source.resolve() for source in args.source], args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
