from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _jobs, _record_once
from a3opt.advanced_sim import EvaluatedPolicy
from a3opt.backends.advanced_phase import load_advanced_policy
from a3opt.backends.advanced_trajectory import pose_label
from a3opt.cli import _run_lock
from a3opt.refinement_eval import read_json, sha256, simulator_identity, wilson_interval, write_json


PROFILES = ("basic", "medium", "advanced")
SEEDS = {"1": tuple(range(48400, 48420)), "random": tuple(range(48500, 48520))}
RULES = {"task": "2x5", "prompt_style": "task", "max_steps": 500, "success_hold_steps": 5,
         "slot_tolerance_m": [0.015, 0.005], "wall_contact_tolerance_m": 0.030,
         "complete_demonstrations": False, "strict_policy_completion": False}
BOOTSTRAP_SEED = 6102026
BOOTSTRAP_SAMPLES = 20000


class EpisodeTrace(EvaluatedPolicy):
    def reset(self, context=None) -> None:
        super().reset(context)
        self.applied_digest = hashlib.sha256()
        self.peak_count = 0

    def add_frame(self, observation, action, info, **context) -> None:
        super().add_frame(observation, action, info, **context)
        self.applied_digest.update(np.asarray(info["applied_action"], dtype="<f8").tobytes())
        self.peak_count = max(self.peak_count, info["cookies_in_target"])
        self.terminal = {
            "slot_occupancy": [int(cookie) for cookie in info["target_slot_occupancy"]],
            "touches_all_walls": bool(info["target_touches_all_walls"]),
            "hold_steps": int(info["success_hold_count"]),
            "exact_fill": bool(info["exact_2x5_fill"]),
        }


def run_episode(checkpoint: Path, profile: str, suite: str, seed: int, trace=None) -> dict:
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    bench = make_cookie_benchmark(profile=profile, source_column=suite, **RULES)
    env = bench.create_env(render_cameras=False)
    policy = trace if trace is not None else EpisodeTrace(load_advanced_policy(checkpoint))
    try:
        result = bench.run_episode(policy, seed=seed, env=env, diagnostic=policy)
        provenance = np.bincount(policy.columns[policy.target], minlength=5)[1:].tolist()
        metadata = result.randomization
        if result.steps != policy.step or result.score != sum(provenance):
            raise RuntimeError("benchmark score/steps disagree with the diagnostic trace")
        return {
            "profile": metadata["profile"], "suite": suite, "seed": result.seed, "column": result.source_column,
            "score": result.score, "requested_score": provenance[result.source_column - 1],
            "success": result.success, "by_column": provenance, "target_cookie_ids": np.flatnonzero(policy.target).tolist(),
            "steps": result.steps, "wall_seconds": result.wall_seconds, "first_batch_step": policy.first_batch,
            "peak_count": policy.peak_count, "source_remaining": result.cookies_in_source,
            "failure_reason": result.failure_reason, "safety_reason": policy.safety,
            "action_sha256": policy.digest.hexdigest(), "applied_action_sha256": policy.applied_digest.hexdigest(),
            "terminal": policy.terminal, "color": metadata["cookie_color"], "task_prompt": bench._episode_task(result.source_column, seed),
            "initial_pose": pose_label(metadata).tolist(), "estimated_pose": policy.policy.pose.tolist(),
            "cameras": metadata["cameras"], "light_diffuse": metadata["light_diffuse"],
        }
    finally:
        env.close()


def check_row(job: tuple, row: dict) -> None:
    from a3_dual_arm_sim.sim.randomization import choose_column

    _, profile, suite, seed = job
    column = choose_column(suite, seed)
    if (row["profile"], row["suite"], row["seed"], row["column"]) != (profile, suite, seed, column):
        raise ValueError("episode identifiers do not match the frozen schedule")
    if not (row["score"] == len(set(row["target_cookie_ids"])) == sum(row["by_column"])
            and row["requested_score"] == row["by_column"][column - 1]
            and 1 <= row["steps"] <= RULES["max_steps"]):
        raise ValueError("episode scores disagree with terminal provenance or step budget")
    if row["success"] != (row["terminal"]["hold_steps"] >= RULES["success_hold_steps"]
                          and row["score"] == 10 and row["source_remaining"] == 70 and not row["safety_reason"]):
        raise ValueError("episode success disagrees with the terminal evidence")


def _bootstrap(values: np.ndarray, columns: list[int]) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.zeros(BOOTSTRAP_SAMPLES)
    assignments = np.asarray(columns)
    for column in np.unique(assignments):
        group = values[assignments == column]
        estimates += rng.choice(group, size=(BOOTSTRAP_SAMPLES, len(group))).sum(axis=1)
    return np.quantile(estimates / len(values), [0.025, 0.975]).tolist()


def summarize(rows: list[dict]) -> dict:
    cells, contrasts = {}, {}
    for suite, seeds in SEEDS.items():
        suites = {}
        for profile in PROFILES:
            subset = [row for row in rows if (row["profile"], row["suite"]) == (profile, suite)]
            by_seed = {row["seed"]: row for row in subset}
            if len(by_seed) != len(subset) or set(by_seed) != set(seeds):
                raise ValueError(f"incomplete or duplicate episodes for {profile}/{suite}")
            group = [by_seed[seed] for seed in seeds]
            columns = [row["column"] for row in group]
            if Counter(columns) != ({1: 20} if suite == "1" else {column: 5 for column in range(1, 5)}):
                raise ValueError(f"unexpected column balance for {profile}/{suite}")
            source_scores = np.asarray([row["requested_score"] for row in group])
            target_scores = np.asarray([row["score"] for row in group])
            successes = sum(row["success"] and row["requested_score"] == 10 for row in group)
            cells.setdefault(profile, {})[suite] = {
                "n": len(group), "target_mean": float(target_scores.mean()),
                "target_ci95": _bootstrap(target_scores, columns), "requested_mean": float(source_scores.mean()),
                "requested_ci95": _bootstrap(source_scores, columns), "strict_successes": successes,
                "upstream_successes": sum(row["success"] for row in group),
                "success_ci95": wilson_interval(successes, len(group)),
                "requested_tens": int(np.count_nonzero(source_scores == 10)),
                "by_column": {int(column): float(source_scores[np.asarray(columns) == column].mean())
                              for column in sorted(set(columns))},
                "wrong_column_episodes": sum(row["score"] != row["requested_score"] for row in group),
                "termination": dict(Counter(row["failure_reason"] or "success" for row in group)),
            }
            suites[profile] = (source_scores, columns)
        for easier, harder in combinations(PROFILES, 2):
            baseline, columns = suites[easier]
            comparison, other_columns = suites[harder]
            if columns != other_columns:
                raise ValueError("the requested columns differ across paired episodes")
            change = comparison - baseline
            contrasts.setdefault(f"{harder}-{easier}", {})[suite] = {
                "mean_change": float(change.mean()), "ci95": _bootstrap(change, columns),
                "better": int(np.count_nonzero(change > 0)), "tied": int(np.count_nonzero(change == 0)),
                "worse": int(np.count_nonzero(change < 0)),
            }
    return {"cells": cells, "paired_requested_score_differences": contrasts, "episodes": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen 2x5 policy on three difficulty profiles.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    reference, output = args.reference.resolve(), args.out.resolve()
    original_protocol, lock_path = reference / "protocol.json", reference / "confirmation-lock.json"
    lock = read_json(lock_path)
    checkpoint = Path(lock["selected"]["checkpoint"])
    environment = simulator_identity()
    code_root = Path(__file__).parent
    original = read_json(original_protocol)
    if sha256(original_protocol) != lock["protocol_sha256"] or sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("frozen confirmation protocol or checkpoint does not match its lock")
    policy_hashes = {name: sha256(code_root / name) for name in original["code_sha256"]}
    if policy_hashes != original["code_sha256"] or environment != original["environment"]:
        raise ValueError("simulator environment or policy code changed since confirmation")
    protocol = {
        "rules": RULES, "profiles": list(PROFILES), "seeds": {key: list(value) for key, value in SEEDS.items()},
        "checkpoint": str(checkpoint), "checkpoint_sha256": lock["checkpoint_sha256"],
        "confirmation_lock_sha256": sha256(lock_path), "environment": environment,
        "opencv_version": importlib.metadata.version("opencv-python"), "inherited_code_sha256": policy_hashes,
        "evaluation_sha256": sha256(Path(__file__)),
        "bootstrap": {"method": "percentile within requested column", "seed": BOOTSTRAP_SEED,
                      "replicates": BOOTSTRAP_SAMPLES},
    }
    jobs = [(checkpoint, profile, suite, seed) for suite, seeds in SEEDS.items()
            for seed in seeds for profile in PROFILES]
    with _run_lock(output):
        _record_once(output / "protocol.json", protocol)
        print(f"Evaluating {len(jobs)} episodes in {output}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = _jobs(pool, run_episode, jobs, output / "episodes",
                         lambda job: f"{job[1]}/{job[2]}/{job[3]}", check_row)
        if sha256(checkpoint) != lock["checkpoint_sha256"]:
            raise ValueError("checkpoint changed during the evaluation")
        results = summarize(rows)
        write_json(output / "results.json", {"protocol_sha256": sha256(output / "protocol.json"),
                                             "summary": results, "rows": rows})
        print(f"Saved {len(rows)} audited episodes to {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
