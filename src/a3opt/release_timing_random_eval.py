from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from a3opt import release_timing_agent
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import load_checkpoint
from a3opt.backends.advanced_trajectory import pose_label
from a3opt.cli import _run_lock
from a3opt.refinement_eval import read_json, sha256
from a3opt.second_batch_agent import frozen_reference


class MatchedPolicy(release_timing_agent.ReleaseTimingPolicy):
    def __init__(self, checkpoint: Path, column: int, scale: float):
        self.reference = load_checkpoint(checkpoint)
        super().__init__(checkpoint, column, scale)

    def reset(self, context=None) -> None:
        super().reset(context)
        self.reference.reset(context)
        self.matched_steps = 0

    def act(self, observation, task="") -> np.ndarray:
        action = super().act(observation, task)
        if not np.array_equal(action, self.reference.act(observation, task)):
            raise ValueError("inactive-column controller differs from its reference")
        self.matched_steps += 1
        return action


def episode(checkpoint: Path, focus: int, scale: float, seed: int, shared: bool) -> dict:
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **release_timing_agent.STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        controller = MatchedPolicy if shared else release_timing_agent.ReleaseTimingPolicy
        policy = controller(checkpoint, focus, scale)
        trace = release_timing_agent.TimingTrace(policy, first_batch_only=False)
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        meta = result.randomization
        ids = np.flatnonzero(trace.target).tolist()
        counts = np.bincount(trace.columns[ids], minlength=5)[1:].tolist()
        return {"checkpoint_sha256": sha256(checkpoint), "column": result.source_column, "suite": "random",
                "seed": seed, "profile": meta["profile"], "steps": result.steps, "score": result.score,
                "requested_score": counts[result.source_column - 1], "by_column": counts, "target_cookie_ids": ids,
                "source_remaining": result.cookies_in_source, "success": result.success, "terminal": trace.terminal,
                "safety_reason": trace.safety, "color": meta["cookie_color"], "task_prompt": meta["task_instruction"],
                "initial_pose": pose_label(meta).tolist(), "cameras": meta["cameras"], "light_diffuse": meta["light_diffuse"],
                "first_placement_prefix": trace.first_batch_prefix, "second_open_step": trace.events[1].get("open_start"),
                "applied_action_sha256": trace.applied_digest.hexdigest(), "action_sha256": trace.digest.hexdigest(),
                "scale": scale, "matched_steps": policy.matched_steps if shared else 0}
    finally:
        env.close()


def audit(job: tuple, row: dict) -> None:
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, focus, scale, seed, shared = job
    column = choose_column("random", seed)
    if (row["suite"] != "random" or row["column"] != column or shared != (column != focus)
            or row["matched_steps"] != (row["steps"] if shared else 0)):
        raise ValueError("random verification receipt does not match its assigned column or comparison mode")
    release_timing_agent.audit((checkpoint, column, scale, seed), {**row, "suite": str(column)})


def run(out: Path, timing: Path, reference_run: Path, seed_start: int, workers: int) -> dict:
    from a3_dual_arm_sim.sim.randomization import choose_column

    seeds = list(range(seed_start, seed_start + 20))
    columns = {seed: choose_column("random", seed) for seed in seeds}
    if Counter(columns.values()) != {column: 5 for column in range(1, 5)}:
        raise ValueError("random verification requires five new seeds per column")
    schedule = {"scout": {"random": []}, "training": [], "development": [], "confirmation": {"random": seeds}}
    checkpoint, identity = frozen_reference(reference_run, schedule)
    protocol_path, lock_path = timing / "protocol.json", timing / "confirmation-lock.json"
    protocol, lock = read_json(protocol_path), read_json(lock_path)
    timing_seeds = set(protocol["schedule"]["training"]) | set(protocol["schedule"]["confirmation"]["focused"])
    column, scale = lock["column"], lock["scale"]
    if (set(seeds) & timing_seeds or lock["reference_sha256"] != sha256(checkpoint)
            or lock["protocol_sha256"] != sha256(protocol_path) or protocol["environment"] != identity["environment"]
            or lock["screening_sha256"] != sha256(timing / "screening/results.json")
            or protocol["agent_code_sha256"] != sha256(Path(release_timing_agent.__file__))):
        raise ValueError("random verification requires a locked timing candidate and disjoint seeds")
    _record_once(out / "protocol.json", {**identity, "candidate_lock_sha256": sha256(lock_path),
                 "strict_rules": release_timing_agent.STRICT_RULES, "terminate_on_success": False,
                 "active_column": column, "scale": scale,
                 "expected_columns": {str(seed): assigned for seed, assigned in columns.items()}, "seeds": seeds,
                 "shared_action_check": "all requested actions exactly equal on the shared observation stream",
                 "audit_code_sha256": sha256(Path(__file__))})
    jobs = [(checkpoint, column, trial_scale, seed, assigned != column) for seed, assigned in columns.items()
            for trial_scale in ((1., scale) if assigned == column else (scale,))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = _jobs(pool, episode, jobs, out / "episodes", lambda job: f"{job[2]}/{job[3]}", audit)
    shared, active = {}, {}
    for row in rows:
        if row["matched_steps"]:
            shared[row["seed"]] = row
        else:
            active[row["seed"], row["scale"]] = row
    groups = {str(trial_scale): [{**shared[seed], "scale": trial_scale} if seed in shared else active[seed, trial_scale]
                                  for seed in seeds] for trial_scale in (1., scale)}
    compared = release_timing_agent.compare(groups)[str(scale)]
    focused_groups = {arm: [row for row in group if row["column"] == column] for arm, group in groups.items()}
    advances = release_timing_agent.compare(focused_groups)[str(scale)]
    return {"physical_episodes": len(rows), "shared_episodes": len(shared), "active_column": column,
            "compared": compared, "active_column_comparison": advances, "rows": groups}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a timing candidate on a fresh balanced random-column benchmark.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=90400)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with _run_lock(args.out.resolve()):
        result = run(args.out.resolve(), args.timing.resolve(), args.reference.resolve(), args.seed_start, args.workers)
        _record_once(args.out / "result.json", result)
    print(result["compared"], result["active_column_comparison"], flush=True)


if __name__ == "__main__":
    main()
