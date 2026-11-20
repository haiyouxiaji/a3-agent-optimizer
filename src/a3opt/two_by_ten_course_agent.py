from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _jobs, _record_once
from a3opt.advanced_sim import EvaluatedPolicy
from a3opt.backends.advanced_phase import SAMPLES
from a3opt.backends.advanced_trajectory import POSE_SCALE
from a3opt.backends.trajectory import resample
from a3opt.backends.two_by_ten import TwoByTenPolicy, active_batches
from a3opt.cli import _run_lock
from a3opt.refinement_eval import read_json, sha256
from a3opt.two_by_ten_agent import Budget, admissible, compare, make_protocol
from a3opt.two_by_ten_sim import SHORT_STEPS, TransferTrace, benchmark, check_evaluation


PROFILE = "advanced"
ARMS = ("weighted_average", "nearest_demonstration")
MIN_GAIN = .5


class RetrievedPolicy(TwoByTenPolicy):
    def _plan(self, first_grasp: int) -> np.ndarray:
        eligible = np.flatnonzero(self.model["counts"] == first_grasp)
        if not len(eligible):
            raise ValueError(f"missing demonstrations for first grasp {first_grasp}")
        distances = np.linalg.norm((self.model["poses"][eligible] - self.pose) / POSE_SCALE, axis=1)
        selected = eligible[np.argmin(distances)]
        stages = []
        for batch in active_batches(first_grasp):
            offset = 0
            for size, length in zip(SAMPLES, self.model["lengths"][selected, batch], strict=True):
                stage = resample(self.model["traces"][selected, batch, offset:offset + size], int(length))
                if stages:
                    blend_size = min(len(stage), 3)
                    blend = np.arange(1, blend_size + 1)[:, None] / blend_size
                    stage[:blend_size] = (1 - blend) * stages[-1][-1] + blend * stage[:blend_size]
                stages.append(stage)
                offset += size
        actions = np.concatenate(stages)
        actions[:, [7, 15]] = np.clip(actions[:, [7, 15]], 0, 1)
        if not np.isfinite(actions).all():
            raise ValueError("invalid retrieved joint action plan")
        return actions


def evaluate(checkpoint: Path, seed: int, arm: str) -> dict:
    constructor = dict(zip(ARMS, (TwoByTenPolicy, RetrievedPolicy), strict=True))[arm]
    controller = constructor(checkpoint)
    if controller.recipe != "visual":
        raise ValueError("course comparison requires a shared fitted vision checkpoint")
    engine, trace = benchmark(PROFILE, max_steps=SHORT_STEPS), TransferTrace()
    env = engine.create_env(render_cameras=False)
    try:
        episode = engine.run_episode(EvaluatedPolicy(controller), seed=seed, env=env, diagnostic=trace)
        return {**trace.result(episode, env), "checkpoint_sha256": sha256(checkpoint), "arm": arm,
                "planned_steps": len(controller.actions), "at_500": trace.short or trace.snapshot(env)}
    finally:
        env.close()


def check_receipt(job: tuple, row: dict) -> None:
    checkpoint, seed, arm = job
    check_evaluation((checkpoint, PROFILE, seed), row)
    if (row["arm"] != arm or row["steps"] > SHORT_STEPS
            or row["steps"] < SHORT_STEPS and not (row["success"] or row["safety_reason"])):
        raise ValueError("receipt is inconsistent with the 500-step course protocol")


def evaluate_split(pool, directory: Path, checkpoint: Path, seeds: list[int]) -> dict:
    jobs = [(checkpoint, seed, arm) for seed in seeds for arm in ARMS]
    rows = _jobs(pool, evaluate, jobs, directory / "episodes", lambda job: f"{job[2]}/{job[1]}", check_receipt)
    groups = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    result = compare(*(groups[arm] for arm in ARMS))
    _record_once(directory / "results.json", {"arms": groups, "comparison": result})
    return result


def eligible(result: dict) -> bool:
    return (admissible(result["candidate"]) and result["mean_paired_gain"] >= MIN_GAIN
            and result["success_delta"] >= 0)


def run(out: Path, source: Path, budget: Budget, workers: int) -> dict:
    parent_path = source / "protocol.json"
    parent = read_json(parent_path)
    initial_dir = source / "initial"
    fit_path = initial_dir / "fit.json"
    fit = read_json(fit_path)
    checkpoint, dataset = initial_dir / "visual.npz", initial_dir / "dataset.json"
    if sha256(checkpoint) != fit["checkpoints"]["visual"] or sha256(dataset) != fit["dataset_sha256"]:
        raise ValueError("native checkpoint or demonstration manifest differs from its fit record")
    protocol = make_protocol(budget)
    if any(protocol[key] != parent[key] for key in ("environment", "code", "configs", "opencv_version", "rules")):
        raise ValueError("upstream dynamical or policy implementation differs from the native experiment")
    if abs(budget.seed_block - parent["budget"]["seed_block"]) < 1000:
        raise ValueError("the course experiment requires a fresh seed block")
    protocol.update(rules={**protocol["rules"], "max_steps": SHORT_STEPS}, profiles=[PROFILE],
                    parent_protocol_sha256=sha256(parent_path), parent_fit_sha256=sha256(fit_path),
                    dataset_sha256=sha256(dataset), checkpoint_sha256=sha256(checkpoint),
                    course_agent_sha256=sha256(Path(__file__)), minimum_course_gain=MIN_GAIN, compared_arms=list(ARMS))
    _record_once(out / "protocol.json", protocol)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        development = evaluate_split(pool, out / "development", checkpoint, protocol["development_seeds"])
        if not eligible(development):
            return {"status": "retrieval_rejected_after_development", "development": development}
        _record_once(out / "confirmation-lock.json", {"protocol_sha256": sha256(out / "protocol.json"),
                     "development_sha256": sha256(out / "development/results.json"), "candidate": ARMS[1],
                     "seeds": protocol["confirmation_seeds"]})
        confirmation = evaluate_split(pool, out / "confirmation", checkpoint, protocol["confirmation_seeds"])
    passed = eligible(confirmation) and confirmation["paired_gain_ci95"][0] > 0
    return {"status": "retrieval_confirmed" if passed else "retrieval_rejected_after_confirmation",
            "development": development, "confirmation": confirmation}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare native short-horizon trajectory retrieval with phase averaging.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=88000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with _run_lock(args.out.resolve()):
        result = run(args.out.resolve(), args.source.resolve(), Budget(args.seed_block, feedback_attempts=0), args.workers)
        _record_once(args.out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
