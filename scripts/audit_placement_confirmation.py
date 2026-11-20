from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from a3opt import placement_benchmark, placement_compose, placement_descent_agent, placement_first_hover
from a3opt import placement_hover_agent, placement_release_hold, placement_release_recovery, placement_release_router
from a3opt import placement_search
from a3opt import placement_routed_agent, placement_sim, placement_slot_refine, slot_alignment_fit
from a3opt.advanced_agent import _record_once
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES
from replay_placement import RELEASED_REFINEMENTS, decode_model, policy_source, verify_policy_code


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


def released_inputs(root, protocol, lock, fit, kind):
    release_path = Path(lock["checkpoint"]).parent / "controller.json"
    reference = read_json(release_path)
    training = root / "training" / ("rollouts/results.json" if kind == "wall-clearance" else "dataset.json")
    reference_hash = protocol.get("reference_manifest_sha256") or protocol["reference_sha256"]
    if (sha256(release_path) != reference_hash
            or reference["corrections"] != protocol["incumbent"] or reference["corrections"] != lock["incumbent"]
            or reference["checkpoint_sha256"] != lock["checkpoint_sha256"]
            or reference["timing"] != lock["timing"] or reference["sim_revision"] != protocol["environment"]["sim_revision"]
            or reference["policy_code_sha256"] != sha256(Path(placement_descent_agent.__file__))
            or (kind not in ("compose", "clearance-feedback") and fit["dataset_sha256"] != sha256(training))):
        raise ValueError("the fit or its released reference differs from the frozen inputs")
    if kind == "hover":
        if protocol["selection_code_sha256"] != sha256(Path(placement_search.__file__)):
            raise ValueError("the hover selection code differs from the frozen input")
        fitted_model = [placement_hover_agent.Hover(fit["selected"]).record()]
    elif kind == "clearance-feedback":
        estimator = root / "training" / "estimator.json"
        holdout = read_json(root / "sensor-holdout.json")
        parameters = read_json(estimator)
        if (fit["estimator_sha256"] != sha256(estimator) or holdout["estimator_sha256"] != sha256(estimator)
                or parameters["dataset_sha256"] != sha256(training)
                or holdout["dataset_sha256"] != sha256(root / "sensor-holdout" / "dataset.json")
                or fit["rollouts_sha256"] != sha256(root / "training" / "rollouts" / "results.json")):
            raise ValueError("clearance feedback evidence differs from the frozen sensor and rollout data")
        fitted_model = [[batch, fit["recipe"], parameters["estimator"]] for batch in range(2)]
    else:
        if kind == "row" and protocol["fit_code_sha256"] != sha256(Path(slot_alignment_fit.__file__)):
            raise ValueError("the row fit code differs from the frozen input")
        fitted_model = fit["corrections"]
    if kind == "routed":
        dataset = read_json(root / "training" / "dataset.json")
        order = ("late-hover", "early-projection")
        panels = dataset["panels"]
        proposals = dict(zip(order, placement_routed_agent.decode(fitted_model), strict=True))
        regenerated, selection = placement_routed_agent.fit(panels, proposals)
        training_seeds = [seed for name in order for panel in panels[name] for seed in panel["seeds"]]
        if (regenerated != fitted_model or selection != read_json(root / "development-selection.json")
                or dataset["sources"] != protocol["sources"] or training_seeds != protocol["schedule"]["training"]
                or len(training_seeds) != len(set(training_seeds))):
            raise ValueError("the routed fit differs from the frozen training selection")
    if kind == "wall-clearance" and protocol["labels_sha256"] != sha256(root / "training" / "labels.json"):
        raise ValueError("the wall-clearance labels differ from the frozen training data")
    if kind == "release-hold":
        dataset = read_json(training)
        rows = dataset["rows"]
        regenerated, metrics = placement_release_hold.fit(rows, dataset["sampling"]["controls"])
        if (dataset["sampling"] != protocol["sampling"] or [row["seed"] for row in rows] != protocol["schedule"]["training"]
                or [item.record() for item in regenerated] != fitted_model or metrics != fit["metrics"]):
            raise ValueError("the stationary release fit differs from the frozen training selection")
    if kind == "release-recovery":
        panels = read_json(training)["panels"]
        regenerated, metrics = placement_release_recovery.fit(panels)
        groups = {name: [row["seed"] for row in rows] for name, rows in panels.items()}
        if (groups != {name: group["failures"] + group["controls"] for name, group in protocol["sampling"].items()}
                or [item.record() for item in regenerated] != fitted_model
                or any(fit[key] != value for key, value in metrics.items())):
            raise ValueError("the release recovery fit differs from the frozen training selection")
    if kind == "release-router":
        panels = read_json(training)["panels"]
        regenerated, metrics = placement_release_router.fit(panels)
        training_seeds = [row["seed"] for rows in panels.values() for row in rows]
        if (training_seeds != protocol["schedule"]["training"] or len(training_seeds) != len(set(training_seeds))
                or [item.record() for item in regenerated] != fitted_model
                or any(fit[key] != value for key, value in metrics.items())):
            raise ValueError("the release route differs from the frozen training selection")
        fallback = (placement_release_router.ReleaseRoute(regenerated[0].fallback),)
        development = read_json(root / "development" / "results.json")
        development_seeds = protocol["schedule"]["development"]
        for arm, model in (("reference", ()), ("selected", regenerated), ("recovery", fallback)):
            for seed, receipt in zip(development_seeds, development[arm], strict=True):
                placement_benchmark.check((Path(lock["checkpoint"]), *lock["timing"], seed, model), receipt)
        scores = {name: placement_hover_agent.pair_stats({"reference": development[name], "selected": development["selected"]},
                                                       regenerated) for name in ("reference", "recovery")}
        if (scores != read_json(root / "development-selection.json")
                or not all(placement_search.passes(row) for panel in scores.values() for row in panel.values())):
            raise ValueError("release router development gate does not match its paired evaluations")
    if kind == "staged-descent":
        groups = read_json(training)["groups"]
        for column, group in groups.items():
            if group["trials_sha256"] != sha256(root / "training" / column / "results.json"):
                raise ValueError("staged descent trials differ from the frozen training data")
    return fitted_model, reference["evaluator_sha256"]


def supports_promotion(confirmation, kind):
    stricter_gate = ("first-hover", "hover-guard", "hover-guard-refine", "routed", "wall-clearance", "clearance-feedback",
                     "contact-feedback", "late-hover-gate", "staged-descent", "staged-gate", "release-hold", "release-recovery",
                     "release-router")
    if kind in stricter_gate:
        return placement_first_hover.supports_confirmation(confirmation)
    if kind == "compose":
        return placement_compose.stable_gain(confirmation)
    if kind == "row":
        return placement_slot_refine.supports_alignment(confirmation, confirmation=True)
    return confirmation["default"]["mean"] > placement_search.TARGET and all(
        placement_search.passes(stats) and (kind != "hover" or stats["paired_count_ci95"][0] > 0)
        for stats in confirmation.values()
    )


def audit_confirmation(root, kind):
    protocol_path = root / "protocol.json"
    protocol = read_json(protocol_path)
    lock = read_json(root / "confirmation-lock.json")
    fit_path, selection_path = root / "fit.json", root / "development-selection.json"
    checkpoint = Path(lock["checkpoint"])
    seeds, corrections, timing = lock["seeds"], lock["corrections"], lock["timing"]
    schedule = protocol["schedule"]
    seen = scheduled_seeds({key: schedule[key] for key in ("scout", "training", "fit", "sensor_holdout", "development") if key in schedule})
    planned = schedule["confirmation"]
    if isinstance(planned, dict):
        planned = planned["random"]
    fit = read_json(fit_path)
    if kind in RELEASED_REFINEMENTS:
        fitted_model, evaluator_hash = released_inputs(root, protocol, lock, fit, kind)
    else:
        fitted_model, evaluator_hash = fit["corrections"], protocol["evaluator_sha256"]
    if (set(seeds) & seen or len(seeds) != len(set(seeds)) or seeds != planned
            or protocol["environment"] != simulator_identity() or protocol["rules"] != RULES
            or protocol["strict_rules"] != STRICT_RULES or protocol["timing"] != timing
            or evaluator_hash != sha256(Path(placement_benchmark.__file__))
            or sha256(protocol_path) != lock["protocol_sha256"] or sha256(checkpoint) != lock["checkpoint_sha256"]
            or sha256(fit_path) != lock["fit_sha256"] or sha256(selection_path) != lock["selection_sha256"]
            or corrections != fitted_model):
        raise ValueError("the confirmation lock, splits, or inputs differ from the saved run")
    policy_type, selected = decode_model(corrections, kind, incumbent=lock.get("incumbent", ()))
    verify_policy_code(protocol, lock, kind, policy_type)
    confirmation = root / "confirmation"
    request = read_json(confirmation / "rollouts" / "request.json")
    if request != {"checkpoint_sha256": lock["checkpoint_sha256"], "timing": timing, "seeds": seeds,
                   "models": {"reference": [], "selected": corrections}, "rules": [RULES, STRICT_RULES]}:
        raise ValueError("rollout request differs from the confirmation lock")
    rows_path = confirmation / "rollouts" / "results.json"
    rows = read_json(rows_path)
    for arm, model in (("reference", ()), ("selected", selected)):
        for seed, episode in zip(seeds, rows[arm], strict=True):
            receipt = confirmation / "rollouts" / "episodes" / arm / f"{seed}.json"
            if (kind in RELEASED_REFINEMENTS or receipt.is_file()) and episode != read_json(receipt):
                raise ValueError(f"the {arm} aggregate differs from the individual receipt for {seed}")
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
    promoted = supports_promotion(measurements, kind)
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


def replicate(source, output, kind, seed_start, episodes, workers, *, follow_up=False):
    if seed_start < 0 or seed_start % 4 or episodes < 20 or episodes % 4 or workers < 1:
        raise ValueError("replication requires a positive worker budget and a balanced four-column panel")
    origin = audit_confirmation(source, kind)
    if follow_up:
        initial = origin["confirmation"]
        if (kind != "routed" or origin["status"] != "reference_retained_after_confirmation"
                or initial["default"]["mean_ci95"][0] <= placement_search.TARGET
                or not all(placement_search.passes(stats) for stats in initial.values())):
            raise ValueError("follow-up requires a routed candidate with inconclusive paired gains and no observed gate violations")
    elif origin["status"] != "candidate_promoted":
        raise ValueError("replication requires a controller that passed its original confirmation")
    lock_path = source / "confirmation-lock.json"
    lock = read_json(lock_path)
    checkpoint, timing = Path(lock["checkpoint"]), tuple(lock["timing"])
    seeds = list(range(seed_start, seed_start + episodes))
    protocol_path = output / "protocol.json"
    if not protocol_path.exists():
        history = [source / "protocol.json", *sorted(output.parent.glob("*/protocol.json"))]
        if follow_up:
            history.extend(checkpoint.parent / name for name in ("confirmation-protocol.json", "replication-protocol.json"))
        for path in history:
            if path.parent != output and set(seeds).intersection(scheduled_seeds(read_json(path).get("schedule", {}))):
                raise ValueError(f"replication overlaps a reserved panel: {path.parent}")
    incumbent = lock.get("incumbent", [])
    policy_type, model = decode_model(lock["corrections"], kind, incumbent=incumbent)
    audit_hash = (read_json(protocol_path)["audit_code_sha256"] if (output / "result.json").exists()
                  else sha256(Path(__file__)))
    _record_once(protocol_path, {"environment": simulator_identity(), "rules": RULES,
                 "strict_rules": STRICT_RULES, "schedule": {"replication": {"random": seeds}},
                 "reference_sha256": sha256(checkpoint), "timing": list(timing), "model": lock["corrections"],
                 **({"incumbent": incumbent} if kind in RELEASED_REFINEMENTS else {}),
                 "source_lock_sha256": sha256(lock_path), "source_results_sha256": origin["evaluation_sha256"],
                 "policy_code_sha256": sha256(policy_source(policy_type)), "audit_code_sha256": audit_hash,
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__)), "target_mean_strictly_above": placement_search.TARGET,
                 **({"source_status": origin["status"], "decision_basis": "replication_only",
                     "confirmation_requires": placement_routed_agent.REQUIREMENTS,
                     "dependency_sha256": placement_routed_agent.dependencies()} if follow_up else {})})
    print(f"Frozen replication of {source.name}: auditing {episodes} advanced pairs", flush=True)
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
    supported = supports_promotion(confirmation, kind)
    return {"replication_above_target": supported, "replication": confirmation, "pooled": pooled,
            **({"status": "candidate_promoted_after_replication" if supported else "reference_retained_after_replication"}
               if follow_up else {}),
            "replication_results_sha256": sha256(followup_path), "source_results_sha256": origin["evaluation_sha256"]}


def main():
    parser = argparse.ArgumentParser(description="Verify a frozen advanced placement confirmation from its episode and native audit receipts.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--policy-kind", required=True, choices=("descent", "servo", "clearance", "landing", *RELEASED_REFINEMENTS))
    parser.add_argument("--replicate-out", type=Path)
    parser.add_argument("--seed-start", type=int, default=138000)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--follow-up", action="store_true", help="Replicate a routed candidate whose paired interval included zero.")
    args = parser.parse_args()
    if args.follow_up and not args.replicate_out:
        parser.error("--follow-up requires --replicate-out")
    source = args.run.resolve()
    if args.replicate_out:
        output = args.replicate_out.resolve()
        if source == output:
            raise ValueError("replication requires its own output directory")
        with _run_lock(output):
            result = replicate(source, output, args.policy_kind, args.seed_start, args.episodes, args.workers,
                               follow_up=args.follow_up)
            _record_once(output / "result.json", result)
        print(json.dumps({"replication_above_target": result["replication_above_target"],
                          "replication_mean": result["replication"]["default"]["mean"],
                          "pooled_mean": result["pooled"]["default"]["mean"]}, indent=2))
    else:
        print(json.dumps(audit_confirmation(source, args.policy_kind), indent=2))


if __name__ == "__main__":
    main()
