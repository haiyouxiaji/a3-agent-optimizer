from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt.advanced_agent import _jobs, _record_once
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES, check_row, run_episode
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import ReleaseTimingPolicy, TimingTrace, compare


def evaluate(checkpoint: Path, column: int, scale: float, seed: int) -> dict:
    trace = TimingTrace(ReleaseTimingPolicy(checkpoint, column, scale), first_batch_only=False)
    row = run_episode(checkpoint, "advanced", "random", seed, trace)
    return {**row, "checkpoint_sha256": sha256(checkpoint), "scale": scale,
            "first_placement_prefix": trace.first_batch_prefix, "second_open_step": trace.events[1].get("open_start")}


def run(out: Path, timing: Path, random: Path, workers: int) -> dict:
    lock_path = timing / "confirmation-lock.json"
    random_protocol = read_json(random / "protocol.json")
    random_path = random / "result.json"
    verified = read_json(random_path)
    lock = read_json(lock_path)
    checkpoint = Path(lock["reference"])
    environment = simulator_identity()
    if (sha256(lock_path) != random_protocol["candidate_lock_sha256"]
            or sha256(checkpoint) != lock["reference_sha256"] or environment != random_protocol["environment"]):
        raise ValueError("course audit requires the verified timing candidate and simulator")
    column, scale = lock["column"], lock["scale"]
    strict = {(row["seed"], row["scale"]): row for group in verified["rows"].values() for row in group
              if row["column"] == column}
    seeds = sorted({seed for seed, _ in strict})
    if len(seeds) != 5 or len(strict) != 10:
        raise ValueError("course audit requires five complete active-column pairs")
    _record_once(out / "protocol.json", {"rules": RULES, "environment": environment, "column": column, "scale": scale,
                 "seeds": seeds, "lock_sha256": sha256(lock_path), "strict_results_sha256": sha256(random_path),
                 "checkpoint_sha256": sha256(checkpoint), "code_sha256": sha256(Path(__file__))})

    def check(job, row):
        path, active_column, variant, seed = job
        check_row((path, "advanced", "random", seed), row)
        expected = strict[seed, variant]
        fields = ("initial_pose", "cameras", "light_diffuse", "color", "task_prompt", "first_placement_prefix")
        if (row["column"] != active_column or row["scale"] != variant
                or row["checkpoint_sha256"] != lock["reference_sha256"]
                or any(row[field] != expected[field] for field in fields)):
            raise ValueError("course audit differs from its locked strict rollout")
        if row["steps"] == expected["steps"] and row["applied_action_sha256"] != expected["applied_action_sha256"]:
            raise ValueError("course and strict applied actions differ on an equal-length rollout")

    jobs = [(checkpoint, column, variant, seed) for seed in seeds for variant in (1., scale)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = _jobs(pool, evaluate, jobs, out / "episodes", lambda job: f"{job[2]}/{job[3]}", check)
    groups = {str(variant): [row for row in rows if row["scale"] == variant] for variant in (1., scale)}
    return {"rows": groups, "default_comparison": compare(groups)[str(scale)]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the timed candidate using the original course scoring contract.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with _run_lock(args.out.resolve()):
        result = run(args.out.resolve(), args.timing.resolve(), args.random.resolve(), args.workers)
        _record_once(args.out / "result.json", result)
    print(result["default_comparison"], flush=True)


if __name__ == "__main__":
    main()
