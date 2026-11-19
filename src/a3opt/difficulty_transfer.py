from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _jobs, _record_once
from a3opt.advanced_sim import EvaluatedPolicy
from a3opt.backends.advanced_phase import load_advanced_policy
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import PROFILES, RULES, _bootstrap
from a3opt.refinement_eval import read_json, sha256, simulator_identity, wilson_interval, write_json


class TransferTrace(EvaluatedPolicy):
    def add_frame(self, observation, action, info, **context) -> None:
        super().add_frame(observation, action, info, **context)
        self.target_column_counts = [int(count) for count in info["target_column_counts"]]
        self.hold_steps = int(info["success_hold_count"])


def evaluate(checkpoint: Path, profile: str, seed: int) -> dict:
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    rules = {**RULES, "task": "2x10", "first_grasp": "random"}
    bench = make_cookie_benchmark(profile=profile, source_column="1", **rules)
    trace = TransferTrace(load_advanced_policy(checkpoint))
    env = bench.create_env(render_cameras=False)
    try:
        result = bench.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        provenance = np.bincount(trace.columns[trace.target], minlength=5)[1:].tolist()
        if result.steps != trace.step or result.score != sum(provenance):
            raise RuntimeError("2x10 benchmark score/steps disagree with the diagnostic")
        return {
            "profile": result.randomization["profile"], "seed": result.seed, "column": result.source_column,
            "task_prompt": result.randomization["task_instruction"], "score": result.score,
            "requested_score": provenance[0], "by_column": provenance, "success": result.success,
            "steps": result.steps, "wall_seconds": result.wall_seconds,
            "first_grasp": result.randomization["first_grasp_count"],
            "target_column_counts": trace.target_column_counts, "hold_steps": trace.hold_steps,
            "source_remaining": result.cookies_in_source, "safety_reason": trace.safety,
            "failure_reason": result.failure_reason, "action_sha256": trace.digest.hexdigest(),
        }
    finally:
        env.close()


def check_row(job: tuple, row: dict) -> None:
    from a3_dual_arm_sim.tasks.cookie_2x10_plan import Cookie2x10Plan

    _, profile, seed = job
    plan = Cookie2x10Plan.from_seed("random", seed)
    if ((row["profile"], row["seed"], row["column"], row["first_grasp"], row["task_prompt"])
            != (profile, seed, 1, plan.first_grasp, plan.prompt(1))):
        raise ValueError("2x10 row does not match its seeded instruction")
    if not (row["score"] == sum(row["by_column"]) and row["requested_score"] == row["by_column"][0]
            and 1 <= row["steps"] <= RULES["max_steps"]):
        raise ValueError("2x10 score does not match terminal provenance or step budget")


def main() -> None:
    import a3_dual_arm_sim

    parser = argparse.ArgumentParser(description="Evaluate transfer of the frozen 2x5 policy on the 2x10 grid cells.")
    parser.add_argument("--paired-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    source = args.paired_run.resolve() / "protocol.json"
    base = read_json(source)
    checkpoint = Path(base["checkpoint"])
    code_root = Path(__file__).parent
    if (sha256(checkpoint) != base["checkpoint_sha256"] or simulator_identity() != base["environment"]
            or sha256(code_root / "difficulty_eval.py") != base["evaluation_sha256"]
            or any(sha256(code_root / name) != digest for name, digest in base["inherited_code_sha256"].items())):
        raise ValueError("the 2x10 transfer requires the same simulator, policy and paired seeds")
    configs = Path(a3_dual_arm_sim.__file__).resolve().parents[2] / "configs"
    rules = {**RULES, "task": "2x10", "source_column": "1", "first_grasp": "random"}
    seeds = base["seeds"]["1"]
    protocol = {
        "paired_protocol_sha256": sha256(source), "checkpoint_sha256": base["checkpoint_sha256"],
        "environment": base["environment"], "rules": rules, "seeds": seeds, "profiles": list(PROFILES),
        "config_sha256": {name: sha256(configs / name) for name in ("cookie_2x10.yaml", "randomization_2x10.yaml")},
        "evaluation_sha256": sha256(Path(__file__)),
    }
    output = args.out.resolve()
    jobs = [(checkpoint, profile, seed) for seed in seeds for profile in PROFILES]
    with _run_lock(output):
        _record_once(output / "protocol.json", protocol)
        print(f"Evaluating {len(jobs)} 2x10 transfer episodes in {output}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = _jobs(pool, evaluate, jobs, output / "episodes", lambda job: f"{job[1]}/{job[2]}", check_row)
        if sha256(checkpoint) != base["checkpoint_sha256"]:
            raise ValueError("checkpoint changed during the 2x10 evaluation")
        cells = {}
        for profile in PROFILES:
            subset = [row for row in rows if row["profile"] == profile]
            correct = np.asarray([row["requested_score"] for row in subset])
            count = np.asarray([row["score"] for row in subset])
            successes = sum(row["success"] and row["requested_score"] == 20 for row in subset)
            cells[profile] = {
                "n": len(subset), "target_mean": float(count.mean()), "requested_mean": float(correct.mean()),
                "target_ci95": _bootstrap(count, [1] * len(subset)), "strict_successes": successes,
                "success_ci95": wilson_interval(successes, len(subset)), "max_score_observed": int(count.max()),
                "termination": dict(Counter(row["failure_reason"] or "success" for row in subset)),
            }
        write_json(output / "results.json", {"protocol_sha256": sha256(output / "protocol.json"),
                                             "summary": cells, "rows": rows})
        print(f"Saved {len(rows)} audited episodes to {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
