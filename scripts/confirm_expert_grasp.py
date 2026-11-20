from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from a3opt import placement_benchmark, placement_expert_grasp as grasp, placement_expert_landing as landing
from a3opt import placement_first_hover, placement_search, placement_sim
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES

from audit_placement_confirmation import audit_expert_landing, report_rule
from replay_placement import decode_model
from replicate_release_hold import NATIVE_FIELDS, check_pair, equivalent, paired_episode


def inputs(root, parent, checkpoint_override=None):
    protocol_path = root / "protocol.json"
    lock_path = root / "confirmation-lock.json"
    fit_path = root / "fit.json"
    selection_path = root / "development-selection.json"
    parent_lock_path = parent / "confirmation-lock.json"
    protocol, lock, fitted, selection, parent_lock = map(
        read_json, (protocol_path, lock_path, fit_path, selection_path, parent_lock_path))
    parent_protocol, parent_fit = (read_json(parent / name) for name in ("protocol.json", "fit.json"))
    checkpoint = checkpoint_override or Path(parent_lock["checkpoint"])
    timing = tuple(parent_lock["timing"])
    audited_anchor, _ = audit_expert_landing(parent, checkpoint, parent_protocol, parent_fit)
    data = root / "training" / "dataset.json"
    parent_data = parent / "training" / "dataset.json"
    dev_path = parent / "development" / "results.json"
    dependencies = {**landing.dependencies(), landing.__name__: sha256(Path(landing.__file__))}
    column, batch = protocol["scope"]
    schedule = protocol["schedule"]
    if (protocol["parent_lock_sha256"] != sha256(parent_lock_path) or lock["parent_lock_sha256"] != sha256(parent_lock_path)
            or protocol["parent_dataset_sha256"] != sha256(parent_data) or protocol["parent_development_sha256"] != sha256(dev_path)
            or any(protocol[key] != parent_protocol[key] for key in ("environment", "rules", "strict_rules", "checkpoint_sha256"))
            or protocol["code_sha256"] != sha256(Path(grasp.__file__)) or protocol["dependencies"] != dependencies
            or any(lock[key] != sha256(path) for key, path in (("checkpoint_sha256", checkpoint), ("protocol_sha256", protocol_path),
                                                             ("fit_sha256", fit_path), ("selection_sha256", selection_path)))
            or fitted["dataset_sha256"] != sha256(data) or column != parent_protocol["scope"][0] or batch != grasp.BATCH
            or audited_anchor != parent_lock["corrections"] or schedule["training"] != parent_protocol["schedule"]["training"]
            or lock["seeds"] != schedule["confirmation"] or len(set(sum(schedule.values(), []))) != sum(map(len, schedule.values()))
            or lock["corrections"] != fitted["corrections"]):
        raise ValueError("grasp refinement differs from its frozen files or assigned data splits")
    rows = grasp.build_dataset(parent, checkpoint, timing, column, read_json(parent_data))
    models, metrics = grasp.fit(rows, column)
    if (read_json(data) != {"rows": rows} or [model.record() for model in models] != lock["corrections"]
            or any(fitted[key] != value for key, value in metrics.items())):
        raise ValueError("grasp model does not reproduce from the frozen expert data")
    parent_policy, anchors = decode_model(parent_lock["corrections"], "expert-landing", incumbent=parent_lock["incumbent"],
                                          late_recovery=parent_lock["late_recovery"])
    constructor = partial(grasp.ExpertGraspPolicy, release_anchor=anchors, **parent_policy.keywords)
    dev = root / "development"
    paired = read_json(dev / "results.json")
    verify_receipts(dev, checkpoint, timing, schedule["development"], models, paired)
    if selection != measurements(paired, models) or not all(placement_search.passes(stats) for stats in selection.values()):
        raise ValueError("grasp selection does not match its development rollouts")
    return checkpoint, timing, constructor, models, lock, paired, selection


def verify_receipts(folder, checkpoint, timing, seeds, models, pairs):
    request = read_json(folder / "request.json")
    if request != {"checkpoint_sha256": sha256(checkpoint), "timing": list(timing), "seeds": seeds,
                   "models": {"reference": [], "selected": [model.record() for model in models]}, "rules": [RULES, STRICT_RULES]}:
        raise ValueError("grasp rollout request differs from the frozen inputs")
    for arm, corrections in (("reference", ()), ("selected", models)):
        for seed, episode in zip(seeds, pairs[arm], strict=True):
            if episode != read_json(folder / "episodes" / arm / f"{seed}.json"):
                raise ValueError("grasp aggregate differs from the individual receipt")
            placement_benchmark.check((checkpoint, *timing, seed, corrections), episode)


def measurements(paired, models):
    return {rule: grasp.compare([row[rule] for row in paired["reference"]], [row[rule] for row in paired["selected"]], models)
            for rule in ("default", "strict-500")}


def audit(root, checkpoint, timing, models, lock, selection):
    folder = root / "confirmation"
    seeds = lock["seeds"]
    rows_path = folder / "rollouts" / "results.json"
    paired = read_json(rows_path)
    verify_receipts(folder / "rollouts", checkpoint, timing, seeds, models, paired)
    confirmed = measurements(paired, models)
    if any(stats["scored_pairs_by_column"] != dict.fromkeys(map(str, range(1, 5)), len(seeds) // 4) for stats in confirmed.values()):
        raise ValueError("grasp confirmation is not four-column balanced")
    replay = read_json(folder / "audit.json")
    if replay["seeds"] != seeds[:4] or replay["matched_native_replays"] != 8:
        raise ValueError("grasp confirmation requires eight native replays")
    for index, seed in enumerate(replay["seeds"]):
        for strict, rule in ((False, "default"), (True, "strict-500")):
            receipt = read_json(folder / "native-audit" / str(int(strict)) / f"{seed}.json")
            placement_sim.audit((checkpoint, *timing, "random", seed, models, strict), receipt)
            if any(receipt[key] != paired["selected"][index][rule][key] for key in NATIVE_FIELDS):
                raise ValueError("grasp native replay diverged from the confirmation")
    status = "candidate_promoted" if placement_first_hover.supports_confirmation(confirmed) else "reference_retained_after_expert_grasp_confirmation"
    result = {"status": status, "confirmation": confirmed, "development": selection, "corrections": lock["corrections"],
              "target_lower_bound": confirmed["default"]["mean_ci95"][0] > placement_search.TARGET}
    _record_once(root / "result.json", result)
    return {**result, "rollouts_sha256": sha256(rows_path), "native_replays": 8,
            "by_column": {rule: report_rule(paired, rule) for rule in confirmed}}


def resume(root, checkpoint, timing, constructor, models, lock, development, selection, workers):
    folder = root / "confirmation"
    seeds = lock["seeds"]
    controls = {row["default"]["seed"]: (row, trial) for row, trial in zip(development["reference"], development["selected"], strict=True)}
    parity = list(dict.fromkeys([*controls.keys()][:3] + [change["seed"] for change in selection["default"]["changes"]]))[:6]
    _record_once(folder / "shared-prefix-protocol.json", {"lock_sha256": sha256(root / "confirmation-lock.json"),
                 "driver_sha256": sha256(Path(__file__)), "paired_episode_sha256": sha256(Path(paired_episode.__code__.co_filename)),
                 "parity_seeds": parity, "confirmation_seeds": seeds})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def evaluate(label, cases):
            jobs = [(checkpoint, timing, constructor, models, seed) for seed in cases]
            return _jobs(pool, paired_episode, jobs, folder / label, lambda job: str(job[-1]), check_pair)

        print(f"Verifying the grasp evaluator against {len(parity)} full development pairs", flush=True)
        for seed, pair in zip(parity, evaluate("shared-prefix-parity", parity), strict=True):
            for arm, expected in zip(("reference", "selected"), controls[seed], strict=True):
                equivalent(pair[arm], expected)
        _record_once(folder / "shared-prefix-audit.json", {"seeds": parity, "matched_full_rollouts": 2 * len(parity)})
        receipts = folder / "rollouts" / "episodes"
        missing = [seed for seed in seeds if any(not (receipts / arm / f"{seed}.json").is_file() for arm in ("reference", "selected"))]
        print(f"Completing {len(missing)} frozen four-column grasp pairs", flush=True)
        for seed, pair in zip(missing, evaluate("shared-prefix-rollouts", missing), strict=True):
            for arm in ("reference", "selected"):
                path = receipts / arm / f"{seed}.json"
                if path.is_file():
                    equivalent(pair[arm], read_json(path))
                else:
                    _record_once(path, pair[arm])
        placement_search.confirm_combined(pool, folder, checkpoint, timing, seeds, models, seeds[:4],
                                          policy_type=constructor, compare_rows=grasp.compare)


def main():
    parser = argparse.ArgumentParser(description="Finish and audit the frozen second-batch grasp evaluation.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.run.resolve()
    with _run_lock(root):
        checkpoint, timing, constructor, models, lock, development, selection = inputs(root, args.parent.resolve(), args.checkpoint)
        if args.resume and not (root / "result.json").is_file():
            resume(root, checkpoint, timing, constructor, models, lock, development, selection, args.workers)
        print(json.dumps(audit(root, checkpoint, timing, models, lock, selection), indent=2), flush=True)


if __name__ == "__main__":
    main()
