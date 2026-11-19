from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple
from pathlib import Path

import numpy as np

from a3opt.adaptive_agent import _evaluate, paired_summary, seed_schedule
from a3opt.adaptive_sim import AdaptiveTrace, check_adaptive_row, phase_model
from a3opt.advanced_agent import MILESTONE, TARGET, _jobs, _npz, _record_once
from a3opt.backends.adaptive_phase import PLACE, Recipe, load_checkpoint
from a3opt.backends.advanced_phase import SAMPLES
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES, run_episode
from a3opt.refinement_eval import read_json, sha256, simulator_identity, write_json
from a3opt.release_agent import _comparison


BRANCH_GAP_RAD = .45
DEMO_FIELDS = ("poses", "columns", "batches", "seeds", "traces", "lengths")


def dominant_branch(model: dict, column: int) -> tuple[dict, dict]:
    group = np.flatnonzero((model["columns"] == column) & (model["batches"] == 0))
    grip_pose = model["traces"][group, sum(SAMPLES[:PLACE + 1]) - 1, 6]
    order = np.argsort(grip_pose, kind="stable")
    gaps = np.diff(grip_pose[order])
    clusters = np.split(order, np.flatnonzero(gaps > BRANCH_GAP_RAD) + 1)
    dominant = max(clusters, key=len)
    rejected = group[np.setdiff1d(order, dominant)] if len(dominant) >= 4 else np.empty(0, dtype=int)
    indices = np.setdiff1d(np.arange(len(model["columns"])), rejected)
    filtered = {name: values[indices] if name in DEMO_FIELDS else values for name, values in model.items()}
    return filtered, {"column": column, "training_rows": len(group), "removed_seeds": model["seeds"][rejected].tolist(),
                      "gap_threshold_rad": BRANCH_GAP_RAD,
                      "clusters": [{"min_rad": float(grip_pose[cluster].min()), "max_rad": float(grip_pose[cluster].max()),
                                    "training_seeds": model["seeds"][group[cluster]].tolist()} for cluster in clusters]}


def _checkpoint(path: Path, model: dict, column: int, *, joint: bool = False) -> Path:
    manifest = path.with_suffix(".json")
    recipes = [list(astuple(Recipe(joint_placement=joint and index == column))) for index in range(1, 5)]
    if not manifest.exists():
        _npz(path, **model, adaptive_recipes=np.asarray(recipes), grip_center=np.zeros(8), grip_scale=np.ones(8),
             grip_coefficients=np.zeros(8), grip_intercept=0.)
    _record_once(manifest, {"checkpoint_sha256": sha256(path), "recipes": recipes})
    return path


def _first_batch(checkpoint: Path, profile: str, suite: str, seed: int) -> dict:
    policy = load_checkpoint(checkpoint)
    trace = AdaptiveTrace(policy, first_batch_only=True)
    row = run_episode(checkpoint, profile, suite, seed, trace=trace)
    row.update(batches=trace.events, retries=policy.retries, first_requested_five=trace.first_requested_five,
               checkpoint_sha256=sha256(checkpoint))
    return row


def _train(pool, root: Path, candidates: dict, suite: str, seeds: list[int]) -> dict:
    hashes = {name: sha256(path) for name, path in candidates.items()}
    _record_once(root / "request.json", {"checkpoints": hashes, "source_column": suite, "seeds": seeds})
    jobs = [(path, "advanced", suite, seed) for path in candidates.values() for seed in seeds]
    rows = _jobs(pool, _first_batch, jobs, root, lambda job: f"{job[0].parent.name}/{job[3]}", check_adaptive_row)
    groups = {name: [row for row in rows if row["checkpoint_sha256"] == digest] for name, digest in hashes.items()}
    results = {name: _comparison(groups["baseline"], group) for name, group in groups.items()}
    _record_once(root / "comparison.json", results)
    return results


def run(root: Path, reference: Path, placement: Path, block: int, workers: int) -> dict:
    schedule = seed_schedule(block)
    focus = placement / "training-focus.json"
    column = read_json(focus)["weakest_column"]
    old = read_json(reference / "confirmation-lock.json")
    old_protocol = reference / "protocol.json"
    placement_protocol = read_json(placement / "protocol.json")
    source = Path(old["selected"]["checkpoint"])
    if (column not in (1, 2, 3, 4) or sha256(source) != old["checkpoint_sha256"]
            or sha256(old_protocol) != old["protocol_sha256"] or sha256(source) != placement_protocol["reference_sha256"]
            or block in (read_json(old_protocol)["budget"]["seed_block"], placement_protocol["schedule"]["scout"][0])):
        raise ValueError("branch fitting requires a verified reference and fresh seeds")
    package = Path(__file__).parent
    focused = {"training": {str(column): schedule["scout"][:8]},
               "development": {str(column): schedule["development"]["random"][:8]}, "confirmation": schedule["confirmation"]}
    code_files = (*placement_protocol["code_sha256"], "branch_agent.py", "release_agent.py")
    protocol = {"profile": "advanced", "rules": RULES, "schedule": focused, "column": column,
                "method": "dominant_branch", "environment": simulator_identity(), "reference_sha256": sha256(source),
                "focus_sha256": sha256(focus), "code_sha256": {name: sha256(package / name) for name in code_files}}
    _record_once(root / "protocol.json", protocol)
    model = phase_model(source)
    filtered, diagnosis = dominant_branch(model, column)
    _record_once(root / "branch-dataset.json", diagnosis)
    print(f"Column {column}: split {diagnosis['training_rows']} demonstrations into {len(diagnosis['clusters'])} clusters", flush=True)
    baseline = _checkpoint(root / "candidates" / "baseline" / "policy.npz", model, column)
    result = {"status": "baseline_retained", "column": column, "branch_training_digest": sha256(root / "branch-dataset.json")}
    if not diagnosis["removed_seeds"]:
        write_json(root / "result.json", result)
        return result
    candidates = {"baseline": baseline,
                  "branch": _checkpoint(root / "candidates" / "branch" / "policy.npz", filtered, column),
                  "branch-joint": _checkpoint(root / "candidates" / "branch-joint" / "policy.npz", filtered, column, joint=True)}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        training = _train(pool, root / "training", candidates, str(column), focused["training"][str(column)])
        eligible = [name for name, row in training.items() if row["paired_delta"] > 0 and not row["new_safety_or_provenance_violations"]]
        result["training"] = training
        if eligible:
            winner = max(eligible, key=lambda name: (training[name]["paired_delta"], training[name]["positive_pairs"]))
            control = _evaluate(pool, baseline, root / "development" / "baseline", focused["development"])
            trial = _evaluate(pool, candidates[winner], root / "development" / winner, focused["development"])
            development = _comparison(control, trial)
            result.update(development=development, development_candidate=winner)
            _record_once(root / "development-selection.json", result)
            if (development["paired_delta"] > 0 and not development["new_safety_or_provenance_violations"]
                    and development["strict_success_delta"] >= 0):
                selected = candidates[winner]
                _record_once(root / "confirmation-lock.json", {"checkpoint_sha256": sha256(selected),
                             "reference_checkpoint_sha256": sha256(baseline), "protocol_sha256": sha256(root / "protocol.json"),
                             "development_sha256": sha256(root / "development-selection.json"), "seeds": focused["confirmation"],
                             "checkpoint": str(selected)})
                control = _evaluate(pool, baseline, root / "confirmation" / "baseline", focused["confirmation"])
                trial = _evaluate(pool, selected, root / "confirmation" / "selected", focused["confirmation"])
                confirmation = paired_summary(control, trial)
                floor = min(group["selected"]["mean_requested_score"] for group in confirmation.values())
                lower = min(group["selected"]["requested_mean_bootstrap_95"][0] for group in confirmation.values())
                result.update(status="confirmed", confirmation=confirmation, score=floor, checkpoint=str(selected),
                              milestone_met=floor >= MILESTONE, target_point_estimate=floor > TARGET,
                              target_lower_bound=lower > TARGET)
    write_json(root / "result.json", result)
    print(f"Branch agent finished: {result['status']}, confirmation floor: {result.get('score')}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit phase trajectories within a single demonstrated wrist posture branch")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=68000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 61:
        parser.error("workers must be between 1 and 61")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    root = args.out.resolve()
    with _run_lock(root):
        run(root, args.reference.resolve(), args.placement.resolve(), args.seed_block, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
