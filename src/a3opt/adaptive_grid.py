from __future__ import annotations

import argparse
import importlib.metadata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from a3opt.adaptive_audit import verify_confirmation
from a3opt.adaptive_sim import adaptive_episode, check_adaptive_row
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import load_checkpoint
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, EpisodeTrace, PROFILES, RULES, _bootstrap
from a3opt.difficulty_transfer import check_row as check_transfer_row
from a3opt.refinement_eval import bootstrap_mean, read_json, sha256, wilson_interval, write_json


SOURCE_COLUMN = "1"
TRANSFER_RULES = {**RULES, "task": "2x10", "first_grasp": "random"}
TASK_CAPACITIES = {"2x5": 10, "2x10": 20}
UNSUPPORTED_INSTRUCTION = "this policy accepts the upstream 2x5 task instructions"
INITIAL_RUNNER_SHA256 = "4f7a5fa481154368c5e85f6ac9cc6111cdeaa7f7db6bca6af37099712873da21"


class UnsupportedTransferInstruction(ValueError):
    pass


class GridTransferTrace(EpisodeTrace):
    def act(self, observation: dict, task: str) -> np.ndarray:
        try:
            return super().act(observation, task)
        except ValueError as error:
            if self.step == 1 and str(error) == UNSUPPORTED_INSTRUCTION:
                raise UnsupportedTransferInstruction(str(error)) from error
            raise

    def add_frame(self, observation, action, info, **context) -> None:
        super().add_frame(observation, action, info, **context)
        self.target_column_counts = list(map(int, info["target_column_counts"]))


def evaluate(checkpoint: Path, task: str, profile: str, seed: int) -> dict:
    if task == "2x5":
        return {**adaptive_episode(checkpoint, profile, SOURCE_COLUMN, seed), "task": task}

    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    bench = make_cookie_benchmark(profile=profile, source_column=SOURCE_COLUMN, **TRANSFER_RULES)
    trace = GridTransferTrace(load_checkpoint(checkpoint))
    env = bench.create_env(render_cameras=False)
    try:
        try:
            result = bench.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        except UnsupportedTransferInstruction:
            metadata = bench.last_attempt_metadata
            return rejected_row(checkpoint, task, metadata["profile"], seed, metadata["first_grasp_count"],
                                metadata["task_instruction"])
        provenance = np.bincount(trace.columns[trace.target], minlength=5)[1:].tolist()
        if result.steps != trace.step or result.score != sum(provenance):
            raise RuntimeError("2x10 benchmark score/steps disagree with the diagnostic")
        metadata = result.randomization
        return {
            "task": task, "profile": metadata["profile"], "seed": result.seed, "column": result.source_column,
            "first_grasp": metadata["first_grasp_count"], "task_prompt": metadata["task_instruction"],
            "score": result.score, "requested_score": provenance[0], "by_column": provenance,
            "target_cookie_ids": np.flatnonzero(trace.target).tolist(),
            "target_column_counts": trace.target_column_counts, "hold_steps": trace.terminal["hold_steps"],
            "peak_count": trace.peak_count, "steps": result.steps, "wall_seconds": result.wall_seconds,
            "source_remaining": result.cookies_in_source, "success": result.success,
            "failure_reason": result.failure_reason, "safety_reason": trace.safety,
            "action_sha256": trace.digest.hexdigest(), "applied_action_sha256": trace.applied_digest.hexdigest(),
            "checkpoint_sha256": sha256(checkpoint),
        }
    finally:
        env.close()


def rejected_row(checkpoint: Path, task: str, profile: str, seed: int, first_grasp: int, prompt: str) -> dict:
    return {"task": task, "profile": profile, "seed": seed, "column": int(SOURCE_COLUMN),
            "first_grasp": first_grasp, "task_prompt": prompt, "checkpoint_sha256": sha256(checkpoint),
            "score": None, "requested_score": None, "success": None, "steps": 0,
            "failure_reason": "unsupported_task_instruction", "detail": UNSUPPORTED_INSTRUCTION}


def check_grid_row(job: tuple, row: dict) -> None:
    checkpoint, task, profile, seed = job
    if row["task"] != task:
        raise ValueError("grid row does not match its task")
    if task == "2x5":
        check_adaptive_row((checkpoint, profile, SOURCE_COLUMN, seed), row)
        return
    if row["score"] is None:
        from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan

        plan = Cookie2x10Plan.from_seed(TRANSFER_RULES["first_grasp"], seed)
        if row != rejected_row(checkpoint, task, profile, seed, plan.first_grasp, plan.prompt(int(SOURCE_COLUMN))):
            raise ValueError("unsupported-instruction receipt does not match its frozen episode")
        return
    check_transfer_row((checkpoint, profile, seed), row)
    if not (row["checkpoint_sha256"] == sha256(checkpoint)
            and row["score"] == len(set(row["target_cookie_ids"])) == sum(row["target_column_counts"])):
        raise ValueError("2x10 score disagrees with the frozen checkpoint or terminal cookie IDs")
    expected_success = (row["target_column_counts"] == [10, 10] and row["source_remaining"] == 60
                        and row["hold_steps"] >= RULES["success_hold_steps"] and not row["safety_reason"])
    if row["success"] != expected_success:
        raise ValueError("2x10 success disagrees with the terminal evidence")


def summarize_cell(rows: list[dict], capacity: int) -> dict:
    completed = sum(row["score"] is not None for row in rows)
    counts = {"episodes": len(rows), "scored_episodes": completed, "capacity": capacity,
              "termination": dict(Counter(row["failure_reason"] or "success" for row in rows))}
    if completed != len(rows):
        return {**counts, "mean_score": None, "mean_requested_score": None, "successes": None}
    scores = np.asarray([row["score"] for row in rows])
    requested_scores = np.asarray([row["requested_score"] for row in rows])
    successes = sum(row["success"] for row in rows)
    requested_successes = sum(row["success"] and row["requested_score"] == capacity for row in rows)
    return {
        **counts, "mean_score": float(scores.mean()),
        "score_ci95": bootstrap_mean(scores), "max_score_observed": int(scores.max()),
        "mean_requested_score": float(requested_scores.mean()), "successes": successes,
        "success_ci95": wilson_interval(successes, len(rows)), "requested_successes": requested_successes,
        "wrong_column_episodes": int(np.count_nonzero(scores != requested_scores)),
        "safety_stops": sum(bool(row["safety_reason"]) for row in rows),
    }


def paired_profiles(rows: list[dict]) -> dict:
    by_profile = {profile: {row["seed"]: row["score"] for row in rows
                            if row["task"] == "2x5" and row["profile"] == profile} for profile in PROFILES}
    comparisons = {}
    for easier, harder in combinations(PROFILES, 2):
        baseline, comparison = by_profile[easier], by_profile[harder]
        if baseline.keys() != comparison.keys():
            raise ValueError("profile comparisons require paired scenes")
        difference = np.asarray([comparison[seed] - baseline[seed] for seed in sorted(baseline)])
        comparisons[f"{harder}-{easier}"] = {"mean_difference": float(difference.mean()),
                                             "ci95": _bootstrap(difference, [1] * len(difference)),
                                             "higher": int(np.count_nonzero(difference > 0)),
                                             "tied": int(np.count_nonzero(difference == 0)),
                                             "lower": int(np.count_nonzero(difference < 0))}
    return comparisons


def episode_key(job: tuple) -> str:
    return "/".join(map(str, job[1:]))


def audit_existing(jobs: list[tuple], receipts: Path, *, initial_runner: bool) -> list[dict]:
    from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan

    rows = []
    for job in jobs:
        checkpoint, task, profile, seed = job
        path = receipts / f"{episode_key(job)}.json"
        error_path = path.with_suffix(".error.json")
        if path.is_file():
            row = read_json(path)
        elif (initial_runner and task == "2x10" and error_path.is_file() and read_json(error_path)
              == {"error": "ValueError", "detail": UNSUPPORTED_INSTRUCTION}):
            plan = Cookie2x10Plan.from_seed(TRANSFER_RULES["first_grasp"], seed)
            row = rejected_row(checkpoint, task, profile, seed, plan.first_grasp, plan.prompt(int(SOURCE_COLUMN)))
        else:
            raise ValueError(f"missing scored episode or matching interface rejection: {path}")
        check_grid_row(job, row)
        rows.append(row)
    return rows


def main() -> None:
    from a3_dual_arm_sim.paths import resource_root

    parser = argparse.ArgumentParser(description="Audit a frozen branch policy on the six course grid cells.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--audit-existing", action="store_true", help="Audit existing simulator receipts without replaying them")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run = args.run.resolve()
    output = args.out.resolve() if args.out else run / "course-grid"
    protocol, checkpoints, confirmation, _ = verify_confirmation(run)
    if protocol["method"] != "dominant_branch" or protocol["rules"] != RULES or protocol["profile"] != "advanced":
        raise ValueError("the grid requires a confirmed adaptive branch policy under the 500-step protocol")
    checkpoint = checkpoints["selected"]
    checkpoint_hash = sha256(checkpoint)
    fixed = [{**row, "task": "2x5"} for row in confirmation["selected"] if row["suite"] == SOURCE_COLUMN]
    seeds = [row["seed"] for row in fixed]
    config = resource_root() / "configs"
    grid_protocol = {
        "confirmation_lock_sha256": sha256(run / "confirmation-lock.json"),
        "checkpoint_sha256": checkpoint_hash, "environment": protocol["environment"],
        "source_column": int(SOURCE_COLUMN), "seeds": seeds, "profiles": list(PROFILES),
        "rules": {"2x5": RULES, "2x10": TRANSFER_RULES}, "reused_confirmation_cell": ["2x5", "advanced"],
        "config_sha256": {name: sha256(config / name) for name in ("cookie_2x10.yaml", "randomization_2x10.yaml")},
        "evaluation_sha256": {name: sha256(Path(__file__).with_name(name))
                              for name in ("adaptive_grid.py", "adaptive_audit.py", "difficulty_transfer.py")},
        "opencv_version": importlib.metadata.version("opencv-python"),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_SAMPLES},
    }
    jobs = [(checkpoint, task, profile, seed) for seed in seeds for task in TASK_CAPACITIES for profile in PROFILES
            if (task, profile) != ("2x5", "advanced")]
    with _run_lock(output):
        protocol_path = output / "protocol.json"
        receipts = output / "episodes"
        if args.audit_existing:
            initial = {**grid_protocol, "evaluation_sha256": {**grid_protocol["evaluation_sha256"],
                                                               "adaptive_grid.py": INITIAL_RUNNER_SHA256}}
            recorded = read_json(protocol_path)
            if recorded not in (grid_protocol, initial):
                raise ValueError("existing course grid protocol differs from the frozen configuration")
            rows = audit_existing(jobs, receipts, initial_runner=recorded == initial)
        else:
            _record_once(protocol_path, grid_protocol)
            print(f"Evaluating {len(jobs)} simulator episodes and auditing {len(fixed)} sealed confirmation rows", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                rows = _jobs(pool, evaluate, jobs, receipts, episode_key, check_grid_row)
        if sha256(checkpoint) != checkpoint_hash:
            raise ValueError("checkpoint changed during the evaluation")
        rows.extend(fixed)
        cells = {}
        for task, capacity in TASK_CAPACITIES.items():
            cells[task] = {}
            for profile in PROFILES:
                cell = [row for row in rows if (row["task"], row["profile"]) == (task, profile)]
                if [row["seed"] for row in cell] != seeds:
                    raise ValueError(f"incomplete grid cell {task}/{profile}")
                cells[task][profile] = summarize_cell(cell, capacity)
        write_json(output / "results.json", {"protocol_sha256": sha256(protocol_path),
                   "audit_sha256": sha256(Path(__file__)), "cells": cells, "rows": rows,
                   "paired_2x5_profile_differences": paired_profiles(rows)})
        scored = sum(row["score"] is not None for row in rows)
        print(f"Saved {len(rows)} audited attempts ({scored} scored) to {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
