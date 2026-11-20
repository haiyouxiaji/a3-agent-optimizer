from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_sim, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import _bootstrap
from a3opt.placement_agent import reference_setup
from a3opt.placement_sim import Correction
from a3opt.refinement_eval import read_json, sha256


TARGET = 9.0
MAJOR_LOSS = 4
GAP_OFFSETS_M = (.003, .006, .009)


def schedule(block: int, train_only: bool, confirmation_offset: int = 200, training_scenes: int = 16) -> dict:
    if block % 4 or confirmation_offset < 200 or confirmation_offset % 4 or training_scenes not in (8, 12, 16):
        raise ValueError("the random schedule requires a balanced training block")
    return {"scout": {"random": []}, "training": list(range(block, block + training_scenes)),
            "development": [] if train_only else list(range(block + 100, block + 112)),
            "confirmation": {} if train_only else {"random": list(range(block + confirmation_offset, block + confirmation_offset + 20))}}


def branch_stats(branches: list[dict], option: str) -> dict:
    controls = [branch["outcomes"]["reference"] for branch in branches]
    trials = [branch["outcomes"][option] for branch in branches]
    count = [len(trial["counted_ids"]) - len(control["counted_ids"]) for control, trial in zip(controls, trials, strict=True)]
    slots = [trial["filled_slots"] - control["filled_slots"] for control, trial in zip(controls, trials, strict=True)]
    return {"pairs": len(branches), "count_delta": sum(count), "slot_delta": sum(slots),
            "improved": sum(delta > 0 for delta in count), "worse": sum(delta < 0 for delta in count),
            "major_losses": sum(delta <= -MAJOR_LOSS for delta in count),
            "safety_stops": sum(bool(row["safety_reason"]) for row in trials),
            "source_count_changes": sum(old["source_remaining"] != new["source_remaining"] for old, new in zip(controls, trials, strict=True))}


def best_option(options: dict, minimum_gains: int = 1) -> str:
    def rank(name: str) -> tuple:
        row = options[name]
        return row["count_delta"], -row["worse"], row["slot_delta"], name == "reference"

    eligible = [name for name, row in options.items() if name == "reference" or (
        not (row["safety_stops"] or row["major_losses"] or row["source_count_changes"])
        and (minimum_gains == 1 or (row["improved"] >= minimum_gains and row["count_delta"] > 0)))]
    return max(eligible, key=rank)


def fit(rows: list[dict], minimum_gains: int = 1) -> tuple[dict[str, tuple[Correction, ...]], dict]:
    statistics, by_batch, by_column = {}, {}, {}
    for batch in range(2):
        statistics[str(batch)] = {}
        pool = [row["trials"][batch] for row in rows if row["trials"][batch] is not None]
        pooled_stats = {name: branch_stats(pool, name) for name in placement_train.OPTIONS}
        by_batch[batch] = best_option(pooled_stats, minimum_gains)
        statistics[str(batch)]["all"] = pooled_stats
        for column in range(1, 5):
            group = [row["trials"][batch] for row in rows if row["column"] == column and row["trials"][batch] is not None]
            stats = {name: branch_stats(group, name) for name in placement_train.OPTIONS}
            best = best_option(stats, minimum_gains)
            pooled = stats[by_batch[batch]]
            preferred = stats[best]
            improves_count = preferred["count_delta"] > pooled["count_delta"]
            preserves_slots = preferred["count_delta"] == pooled["count_delta"] and pooled["slot_delta"] < 0
            by_column[column, batch] = (best if improves_count or preserves_slots or pooled["major_losses"]
                                         or pooled["safety_stops"] else by_batch[batch])
            statistics[str(batch)][str(column)] = stats
    models = {}
    for name in ("batch", "column-batch"):
        options = {(column, batch): by_batch[batch] if name == "batch" else by_column[column, batch]
                   for column in range(1, 5) for batch in range(2)}
        model = tuple(placement_train.correction(*key, option) for key, option in options.items() if option != "reference")
        if model and model not in models.values():
            models[name] = model
    return models, {"comparisons": statistics, "by_batch": {str(key): val for key, val in by_batch.items()},
                    "by_column_and_batch": {f"{column}/{batch}": val for (column, batch), val in by_column.items()}}


def encode(model: tuple[Correction, ...]) -> list:
    return [item.record() for item in model]


def fit_activation(regular: list[dict], focused: list[dict], model: tuple[Correction, ...],
                   minimum_gains: int) -> tuple[tuple[Correction, ...], dict]:
    def observed_x(row: dict, batch: int) -> float:
        observed = row["descent_observations"][batch]
        return placement_sim.projected_tool_x(observed["eef_xyz_m"], observed["predicted_pose"])

    centers, labels = {}, []
    for item in model:
        relevant = [row for row in regular if row["column"] == item.column and row["descent_observations"][item.batch] is not None]
        if not relevant:
            continue
        center = float(np.median([observed_x(row, item.batch) for row in relevant]))
        centers[item.column, item.batch] = center
        option = next(name for name in placement_train.OPTIONS
                      if placement_train.correction(item.column, item.batch, name).xyz_m == item.xyz_m)
        for row in regular + focused:
            trial = row["trials"][item.batch]
            if row["column"] != item.column or trial is None or row["descent_observations"][item.batch] is None:
                continue
            outcomes = trial["outcomes"]
            labels.append((observed_x(row, item.batch) - center, len(outcomes[option]["counted_ids"])
                           - len(outcomes["reference"]["counted_ids"]), bool(outcomes[option]["safety_reason"])))
    trials = {}
    for relative_mm in (-2, -1, 0, 1, 2, 3, 4, 5):
        enabled = [label for label in labels if label[0] >= relative_mm / 1000]
        trials[relative_mm] = {"activations": len(enabled), "gains": sum(count > 0 for _, count, _ in enabled),
                               "net_count": sum(count for _, count, _ in enabled),
                               "serious_losses": sum(count <= -MAJOR_LOSS or safety for _, count, safety in enabled)}
    eligible = [threshold for threshold, trial in trials.items()
                if trial["gains"] >= minimum_gains and trial["net_count"] > 0 and trial["serious_losses"] == 0]
    if not eligible:
        return (), {"activation_grid": {str(key): value for key, value in trials.items()}}
    selected = max(eligible, key=lambda threshold: (trials[threshold]["net_count"], -trials[threshold]["activations"], threshold))
    gated = tuple(Correction(item.column, item.batch, item.xyz_m, centers[item.column, item.batch] + selected / 1000)
                  for item in model if (item.column, item.batch) in centers)
    return gated, {"activation_grid": {str(key): value for key, value in trials.items()}, "chosen_relative_threshold_mm": selected}


def compare(reference: list[dict], selected: list[dict], model: tuple[Correction, ...], *, prefix_batch: int | None = None) -> dict:
    if not reference or len(reference) != len(selected):
        raise ValueError("paired evaluation requires matching nonempty rows")
    keys = ("seed", "column", "profile", "suite", "strict", "initial_pose", "estimated_pose", "cameras", "light_diffuse", "color")
    pair_deltas, successes, changes, columns = [], [], [], []
    for control, trial in zip(reference, selected, strict=True):
        if any(control[key] != trial[key] for key in keys):
            raise ValueError("paired rows differ in reset, seed, or protocol")
        column = trial["column"]
        active = [item.batch for item in model if item.column == column]
        if active:
            for row in (control, trial):
                if len(row["phase_geometry"]) != 2:
                    raise ValueError("invalid trace for paired episode")
            first = min(active) if prefix_batch is None else prefix_batch
            prefixes = [row["phase_geometry"][first].get("transfer") for row in (control, trial)]
            if prefixes[0] != prefixes[1]:
                raise ValueError("policy altered the shared physical prefix")
        elif any(control[key] != trial[key] for key in ("score", "success", "applied_action_sha256", "action_sha256")):
            raise ValueError("inactive column differs from the reference")
        count = trial["requested_score"] - control["requested_score"]
        pair_deltas.append(count)
        successes.append(int(trial["success"]) - int(control["success"]))
        columns.append(column)
        if count:
            changes.append({"seed": trial["seed"], "column": column, "reference": control["requested_score"],
                            "selected": trial["requested_score"], "delta": count})
    scores = np.asarray([row["requested_score"] for row in selected])
    return {"episodes": len(selected), "reference_mean": float(np.mean([row["requested_score"] for row in reference])),
            "mean": float(scores.mean()), "mean_ci95": _bootstrap(scores, columns),
            "paired_count_delta": float(np.mean(pair_deltas)), "paired_count_ci95": _bootstrap(np.asarray(pair_deltas), columns),
            "reference_complete": sum(row["success"] for row in reference), "complete": sum(row["success"] for row in selected),
            "paired_success_rate_ci95": _bootstrap(np.asarray(successes), columns),
            "major_losses": sum(delta <= -MAJOR_LOSS for delta in pair_deltas),
            "safety_stops": sum(bool(row["safety_reason"]) for row in selected),
            "wrong_source": sum(row["score"] != row["requested_score"] for row in selected),
            "scored_pairs_by_column": {str(column): count for column, count in sorted(Counter(columns).items())}, "changes": changes}


def passes(stats: dict) -> bool:
    return (stats["paired_count_delta"] > 0 and stats["complete"] >= stats["reference_complete"]
            and not (stats["major_losses"] or stats["safety_stops"] or stats["wrong_source"]))


def fit_second_gap(pool, folder: Path, checkpoint: Path, timing: tuple[int, float], rows: list[dict],
                   first: tuple[Correction, ...]) -> tuple[dict[str, tuple[Correction, ...]], dict]:
    if any(item.batch != 0 or item.minimum_tool_x_m is None for item in first):
        raise ValueError("learning a second-batch gap requires a gated first-only correction")
    by_column = {item.column: item for item in first}
    activated = {}
    for row in rows:
        item = by_column.get(row["column"])
        observed = row["descent_observations"][0]
        if item and observed and placement_sim.projected_tool_x(observed["eef_xyz_m"], observed["predicted_pose"]) >= item.minimum_tool_x_m:
            activated[row["seed"]] = row
    seeds = list(activated)
    if len(seeds) < 2:
        return {}, {"active_training_scenes": seeds}
    models = {"first-only": first}
    for offset in GAP_OFFSETS_M:
        models[f"gap-{round(offset * 1000)}mm"] = (*first, *(Correction(item.column, 1, (-offset, 0., 0.), None, True) for item in first))
    outcomes = placement_benchmark.evaluate(pool, folder / "rollouts", checkpoint, timing, seeds, models)
    result = {}
    control = [row["strict-500"] for row in outcomes["first-only"]]
    for name, model in models.items():
        trial = [row["strict-500"] for row in outcomes[name]]
        gains = [row["requested_score"] - activated[row["seed"]]["score"] for row in trial]
        delta = compare(control, trial, model)
        result[name] = {"versus_first_only": delta, "total_gain_from_original": sum(gains),
                        "major_loss_from_original": sum(gain <= -MAJOR_LOSS for gain in gains)}
    _record_once(folder / "paired.json", {"fit_scenes": seeds, "comparisons": result})
    eligible = [name for name in result if name != "first-only" and passes(result[name]["versus_first_only"])
                and result[name]["total_gain_from_original"] > 0 and result[name]["major_loss_from_original"] == 0]
    if not eligible:
        return {}, {"active_training_scenes": seeds, "comparisons": result}
    selected = max(eligible, key=lambda name: (result[name]["total_gain_from_original"], result[name]["versus_first_only"]["complete"]))
    return {selected: models[selected]}, {"active_training_scenes": seeds, "comparisons": result}


def reuse_training(source: Path, checkpoint: Path, timing: tuple[int, float], identity: dict,
                   requested_seeds: list[int] | None = None) -> tuple[list[dict], dict]:
    protocol_path = source / "protocol.json"
    protocol = read_json(protocol_path)
    if (any(protocol[key] != identity[key] for key in ("environment", "timing", "reference_sha256", "strict_rules"))
            or protocol["acquisition_code_sha256"] != sha256(Path(placement_train.__file__))):
        raise ValueError("training reuse requires the same reference, simulator, and acquisition code")
    reserved = protocol["schedule"]["training"]
    seeds = reserved if requested_seeds is None else requested_seeds
    if seeds != reserved[:len(seeds)]:
        raise ValueError("training reuse requires the reserved seeds in acquisition order")
    paths = [source / "training" / "episodes" / f"{seed}.json" for seed in seeds]
    rows = [read_json(path) for path in paths]
    if [row["seed"] for row in rows] != seeds:
        raise ValueError("the reused training data does not cover the locked acquisition schedule")
    for row in rows:
        placement_train.audit((checkpoint, *timing, row["seed"]), row)
    return rows, {"protocol_sha256": sha256(protocol_path), "seeds": seeds,
                  "episode_sha256": [sha256(path) for path in paths]}


def confirm_combined(pool, folder: Path, checkpoint: Path, timing: tuple[int, float], seeds: list[int],
                     model: tuple[Correction, ...], audit_seeds: list[int], *, policy_type=placement_sim.PlacementPolicy,
                     compare_rows=compare) -> dict:
    paired = placement_benchmark.evaluate(pool, folder / "rollouts", checkpoint, timing, seeds,
                                          {"reference": (), "selected": model}, policy_type=policy_type)
    expected = {row["default"]["seed"]: row for row in paired["selected"]}
    jobs = [(checkpoint, *timing, "random", seed, model, strict) for seed in audit_seeds for strict in (False, True)]
    native = _jobs(pool, partial(placement_sim.episode, policy_type=policy_type), jobs, folder / "native-audit",
                   lambda job: f"{int(job[-1])}/{job[4]}", placement_sim.audit)
    audit_keys = ("score", "requested_score", "target_cookie_ids", "terminal", "success", "steps", "safety_reason",
                  "action_sha256", "applied_action_sha256", "source_remaining")
    for row in native:
        rule = "strict-500" if row["strict"] else "default"
        if any(row[key] != expected[row["seed"]][rule][key] for key in audit_keys):
            raise ValueError(f"native score replay does not match the paired rollout for seed {row['seed']}")
    _record_once(folder / "audit.json", {"seeds": audit_seeds, "matched_native_replays": len(native), "matched_fields": list(audit_keys)})
    return {rule: compare_rows([row[rule] for row in paired["reference"]], [row[rule] for row in paired["selected"]], model)
            for rule in ("default", "strict-500")}


def run(root: Path, reference: Path, timing_run: Path, block: int, workers: int, train_only: bool, *,
        reuse: Path | None = None, active: Path | None = None, parity: Path | None = None, training_scenes: int = 16,
        minimum_gains: int = 1, confirmation_offset: int | None = None, gated_placement: bool = False,
        learn_interbatch_spacing: bool = False) -> dict:
    if minimum_gains < 1:
        raise ValueError("minimum positive training observations must be at least one")
    if learn_interbatch_spacing and not gated_placement:
        raise ValueError("second-batch spacing requires the first-batch activation model")
    offset = (300 if reuse else 200) if confirmation_offset is None else confirmation_offset
    splits = schedule(block, train_only, confirmation_offset=offset, training_scenes=training_scenes)
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    training_sources, reused, focused = {}, [], []
    if reuse:
        reused, evidence = reuse_training(reuse, checkpoint, timing, identity, splits["training"])
        training_sources["balanced"] = evidence
    if active:
        focused, evidence = reuse_training(active, checkpoint, timing, identity)
        all_seeds = splits["training"] + splits["development"] + splits["confirmation"].get("random", [])
        if set(evidence["seeds"]).intersection(all_seeds):
            raise ValueError("targeted training overlaps the balanced training or evaluation splits")
        training_sources["active"] = evidence
    if parity and read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("the combined evaluator requires validation against independent native replays")
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "checkpoint": str(checkpoint),
                 "target_mean_strictly_above": TARGET, "major_loss_cookies": MAJOR_LOSS, "train_only": train_only,
                 "minimum_training_gains": minimum_gains, "gated_placement": gated_placement,
                 "spacing_search_m": list(GAP_OFFSETS_M) if learn_interbatch_spacing else [],
                 "training_sources": training_sources,
                 "combined_evaluation": None if parity is None else {"parity_sha256": sha256(parity),
                     "code_sha256": sha256(Path(placement_benchmark.__file__))},
                 "acquisition_code_sha256": sha256(Path(placement_train.__file__)),
                 "search_code_sha256": sha256(Path(__file__)), "confirmation_requires": ["score_gain", "no_major_losses",
                     "no_safety_or_wrong_source", "no_success_regression"], "confirmation_suites": ["default", "strict-500"]})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        train = reused if reuse else placement_train.acquire(pool, root / "training", checkpoint, timing, splits["training"])
        models, stats = fit(train + focused, minimum_gains)
        if gated_placement:
            gated_models, exposure = {}, {}
            for name, model in models.items():
                gated, exposure[name] = fit_activation(train, focused, model, minimum_gains)
                if gated and gated not in gated_models.values():
                    gated_models[name] = gated
            models = gated_models
            stats["activation"] = exposure
        if learn_interbatch_spacing and models:
            first = min(models.values(), key=len)
            models, stats["interbatch_spacing"] = fit_second_gap(pool, root / "training-spacing", checkpoint, timing, train + focused, first)
        dataset_digest = None if reuse else sha256(root / "training/dataset.json")
        _record_once(root / "fit.json", {**stats, "candidates": {name: encode(model) for name, model in models.items()},
                     "dataset_sha256": dataset_digest, "training_sources": training_sources})
        if train_only or not models:
            return {"status": "training_complete" if train_only else "reference_retained_after_training", "fit": stats}
        development = root / "development"
        seeds = splits["development"]
        control_folder = (reuse or root) / "development" / "reference"
        control = placement_sim.evaluate(pool, control_folder, checkpoint, timing, "random", seeds)
        comparisons = {}
        for name, model in models.items():
            trial = placement_sim.evaluate(pool, development / name, checkpoint, timing, "random", seeds, model)
            comparisons[name] = compare(control, trial, model)
            print(f"Development {name}: {comparisons[name]['reference_mean']:.2f} to {comparisons[name]['mean']:.2f}", flush=True)
        _record_once(root / "development-selection.json", comparisons)
        eligible = [name for name in comparisons if passes(comparisons[name])]
        if not eligible:
            return {"status": "reference_retained_after_development", "development": comparisons}
        selected = max(eligible, key=lambda name: (comparisons[name]["mean"], comparisons[name]["complete"], -len(models[name])))
        model = models[selected]
        seeds = splits["confirmation"]["random"]
        audit_seeds = seeds[:4] if parity else []
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "variant": selected, "corrections": encode(model), "suite": "random", "seeds": seeds,
                     "native_audit_seeds": audit_seeds,
                     "protocol_sha256": sha256(root / "protocol.json"), "selection_sha256": sha256(root / "development-selection.json"),
                     "fit_sha256": sha256(root / "fit.json")})
        print(f"Frozen {selected} before evaluating {len(seeds)} new scenes", flush=True)
        if parity:
            confirmation = confirm_combined(pool, root / "confirmation", checkpoint, timing, seeds, model, audit_seeds)
        else:
            confirmation = {}
            for rule, strict in (("default", False), ("strict-500", True)):
                folder = root / "confirmation" / rule
                control = placement_sim.evaluate(pool, folder / "reference", checkpoint, timing, "random", seeds, strict=strict)
                trial = placement_sim.evaluate(pool, folder / "selected", checkpoint, timing, "random", seeds, model, strict=strict)
                confirmation[rule] = compare(control, trial, model)
                _record_once(folder / "paired.json", confirmation[rule])
    confirmed = all(passes(stats) for stats in confirmation.values())
    primary = confirmation["default"]
    return {"status": "candidate_promoted" if confirmed else "reference_retained_after_confirmation", "variant": selected,
            "target_point_estimate": primary["mean"] > TARGET, "target_lower_bound": primary["mean_ci95"][0] > TARGET,
            "active_policy_above_target": confirmed and primary["mean"] > TARGET, "development": comparisons,
            "confirmation": confirmation}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agent's placement acquisition, selection, and independent confirmation.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=110100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--reuse-acquisition", type=Path)
    parser.add_argument("--active-training", type=Path)
    parser.add_argument("--parity-audit", type=Path)
    parser.add_argument("--training-scenes", type=int, default=16)
    parser.add_argument("--minimum-training-gains", type=int, default=1)
    parser.add_argument("--confirmation-offset", type=int)
    parser.add_argument("--gated-placement", action="store_true")
    parser.add_argument("--learn-interbatch-spacing", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.seed_block, args.workers, args.train_only,
                     reuse=args.reuse_acquisition, active=args.active_training, parity=args.parity_audit,
                     training_scenes=args.training_scenes, minimum_gains=args.minimum_training_gains,
                     confirmation_offset=args.confirmation_offset, gated_placement=args.gated_placement,
                     learn_interbatch_spacing=args.learn_interbatch_spacing)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
