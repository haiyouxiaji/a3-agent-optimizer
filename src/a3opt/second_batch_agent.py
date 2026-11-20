from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from a3opt.adaptive_agent import paired_summary
from a3opt.adaptive_sim import phase_model
from a3opt.advanced_agent import TARGET, _npz, _record_once
from a3opt.backends.second_batch import ReleaseRecipe
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.second_batch_fit import fit_release, improvement, nonregression, paired_comparison, prioritize
from a3opt.second_batch_sim import evaluate_schedule


FOCUS_COLUMNS = 2
TRAIN_SIZE = 8
DEV_SIZE = 12


def seed_schedule(block: int) -> dict:
    if block < 0 or block % 4 != 0:
        raise ValueError("seed block must be a nonnegative multiple of four")
    return {"scout": {"random": list(range(block, block + 16))},
            "training": list(range(block + 100, block + 100 + FOCUS_COLUMNS * TRAIN_SIZE)),
            "development": list(range(block + 200, block + 200 + FOCUS_COLUMNS * DEV_SIZE)),
            "confirmation": {"1": list(range(block + 300, block + 320)),
                             "random": list(range(block + 400, block + 420))}}


def frozen_reference(root: Path, schedule: dict) -> tuple[Path, dict]:
    lock_path = root / "confirmation-lock.json"
    lock = read_json(lock_path)
    protocol_path = root / "protocol.json"
    reference = Path(lock["checkpoint"])
    if not reference.exists():
        reference = root / "candidates" / reference.parent.name / reference.name
    if sha256(reference) != lock["checkpoint_sha256"] or sha256(protocol_path) != lock["protocol_sha256"]:
        raise ValueError("frozen reference does not match its confirmation lock")
    protocol = read_json(protocol_path)
    if protocol["profile"] != "advanced" or protocol["rules"] != RULES:
        raise ValueError("reference must use the same 2x5 benchmark contract")
    seeds = set(phase_model(reference)["seeds"].tolist())
    for split in protocol["schedule"].values():
        seeds.update(seed for group in split.values() for seed in group)
    for group in (schedule["scout"]["random"], schedule["training"], schedule["development"],
                  *schedule["confirmation"].values()):
        if seeds.intersection(group):
            raise ValueError("second-batch run requires fresh, disjoint seeds")
        seeds.update(group)
    code = (*protocol["code_sha256"], "second_batch_sim.py", "backends/second_batch.py", "backends/release_residual.py")
    package = Path(__file__).parent
    identity = {"profile": "advanced", "rules": RULES, "environment": simulator_identity(),
                "reference_sha256": sha256(reference), "reference_lock_sha256": sha256(lock_path),
                "code_sha256": {name: sha256(package / name) for name in code}}
    return reference, identity


def _checkpoint(path: Path, reference: Path, column: int, recipe: ReleaseRecipe) -> Path:
    manifest = path.with_suffix(".json")
    if not manifest.exists():
        _npz(path, **phase_model(reference), second_batch_column=column, second_batch_offset_m=recipe.offset_m,
             second_batch_open_hold=recipe.open_hold)
    _record_once(manifest, {"checkpoint_sha256": sha256(path), "reference_sha256": sha256(reference),
                            "column": column, "offset_m": list(recipe.offset_m), "open_hold": recipe.open_hold})
    return path


def _fit_column(pool, reference: Path, root: Path, scout: list[dict], column: int, train: list[int],
                development: list[int]) -> tuple[Path | None, dict]:
    training = {str(column): train}
    baseline = evaluate_schedule(pool, reference, root / "training" / "reference", training)
    proposals, dataset = fit_release([row for row in scout if row["column"] == column] + baseline)
    _record_once(root / "training-dataset.json", {**dataset, "source_column": column,
                 "focused_results_sha256": sha256(root / "training" / "reference" / "results.json")})
    checkpoints, metrics = {}, {}
    for name, recipe in proposals.items():
        checkpoint = _checkpoint(root / "candidates" / name / "policy.npz", reference, column, recipe)
        checkpoints[name] = checkpoint
        rows = evaluate_schedule(pool, checkpoint, root / "training" / name, training)
        metrics[name] = paired_comparison(baseline, rows)
    attempt = {"column": column, "proposals": {name: {**asdict(recipe), "offset_m": list(recipe.offset_m)}
                                                for name, recipe in proposals.items()}, "training": metrics,
               "training_dataset_sha256": sha256(root / "training-dataset.json")}
    eligible = [name for name, comparison in metrics.items() if improvement(comparison)]
    winner = None
    if eligible:
        name = max(eligible, key=lambda name: (metrics[name]["full_success_delta"], metrics[name]["score_delta"]))
        schedule = {str(column): development}
        control = evaluate_schedule(pool, reference, root / "development" / "reference", schedule)
        trial = evaluate_schedule(pool, checkpoints[name], root / "development" / name, schedule)
        comparison = paired_comparison(control, trial)
        attempt.update(development_candidate=name, development=comparison)
        if improvement(comparison):
            winner = checkpoints[name]
    _record_once(root / "selection.json", attempt)
    return winner, attempt


def _confirm(pool, root: Path, reference: Path, winner: Path, schedule: dict, selection: Path) -> dict:
    _record_once(root / "confirmation-lock.json", {"reference_checkpoint_sha256": sha256(reference),
                 "checkpoint_sha256": sha256(winner), "checkpoint": winner.relative_to(root).as_posix(),
                 "seeds": schedule, "protocol_sha256": sha256(root / "protocol.json"),
                 "development_sha256": sha256(selection)})
    print("Agent locked one candidate before confirmation", flush=True)
    control = evaluate_schedule(pool, reference, root / "confirmation" / "reference", schedule)
    trial = evaluate_schedule(pool, winner, root / "confirmation" / "selected", schedule)
    summary = paired_summary(control, trial)
    comparisons = {suite: paired_comparison([row for row in control if row["suite"] == suite],
                                           [row for row in trial if row["suite"] == suite]) for suite in schedule}
    floor = min(group["selected"]["mean_requested_score"] for group in summary.values())
    lower = min(group["selected"]["requested_mean_bootstrap_95"][0] for group in summary.values())
    accepted = (floor > TARGET and all(map(nonregression, comparisons.values()))
                and any(map(improvement, comparisons.values())))
    return {"status": "promoted" if accepted else "reference_retained_after_confirmation",
            "selected_checkpoint": str(winner if accepted else reference), "candidate_checkpoint": str(winner),
            "confirmation": summary, "paired_release_comparison": comparisons, "candidate_score_floor": floor,
            "floor_ci95_lower": lower, "target_point_estimate_met": floor > TARGET, "target_lower_bound_met": lower > TARGET}


def run(root: Path, reference_root: Path, block: int, workers: int, *, scout_only: bool = False) -> dict:
    schedule = seed_schedule(block)
    reference, identity = frozen_reference(reference_root, schedule)
    _record_once(root / "scout-protocol.json", {**identity, "seeds": schedule["scout"]})
    if not scout_only:
        package = Path(__file__).parent
        code = {**identity["code_sha256"], **{name: sha256(package / name) for name in (
            "second_batch_agent.py", "second_batch_fit.py")}}
        _record_once(root / "protocol.json", {**identity, "schedule": schedule, "code_sha256": code,
                     "max_focus_columns": FOCUS_COLUMNS, "training_scenes_per_column": TRAIN_SIZE,
                     "development_scenes_per_column": DEV_SIZE, "target_score_floor": TARGET})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scout = evaluate_schedule(pool, reference, root / "scout", schedule["scout"])
        columns, stages = prioritize(scout)
        diagnosis = {"focus_order": columns, "stages_by_column": stages,
                     "results_sha256": sha256(root / "scout" / "results.json")}
        _record_once(root / "scout-diagnosis.json", diagnosis)
        if scout_only:
            return {"status": "scout_complete", **diagnosis}
        result = {"status": "reference_retained_after_development", "selected_checkpoint": str(reference),
                  "scout": diagnosis, "attempts": []}
        for index, column in enumerate(columns[:FOCUS_COLUMNS]):
            print(f"Agent sampling placement on source column {column}", flush=True)
            folder = root / f"column-{column}"
            train = schedule["training"][index * TRAIN_SIZE:(index + 1) * TRAIN_SIZE]
            development = schedule["development"][index * DEV_SIZE:(index + 1) * DEV_SIZE]
            winner, attempt = _fit_column(pool, reference, folder, scout, column, train, development)
            result["attempts"].append(attempt)
            if winner is not None:
                result.update(_confirm(pool, root, reference, winner, schedule["confirmation"], folder / "selection.json"))
                break
    _record_once(root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect and optimize second-batch placement on a frozen 2x5 policy")
    parser.add_argument("--reference", type=Path, required=True, help="confirmed branch run directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=72000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--scout-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 61:
        parser.error("workers must be between 1 and 61")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.seed_block, args.workers, scout_only=args.scout_only)
    print(f"Agent finished: {result['status']}; candidate confirmation floor: {result.get('candidate_score_floor')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
