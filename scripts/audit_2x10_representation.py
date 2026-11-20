from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _record_once
from a3opt.backends.advanced_phase import SAMPLES
from a3opt.backends.two_by_ten import active_batches
from a3opt.backends.trajectory import resample
from a3opt.refinement_eval import read_json, sha256
from a3opt.two_by_ten_sim import COLLECT_STEPS, TransferTrace, benchmark, check_collection, check_rollout


class Commands(TransferTrace):
    def __init__(self):
        super().__init__()
        self.actions = []

    def add_frame(self, observation, action, info, *, phase, env, **context):
        super().add_frame(observation, action, info, phase=phase, env=env, **context)
        self.actions.append(np.asarray(info["applied_action"], dtype=np.float64).copy())


class FixedActions:
    action_mode = "joint_position"

    def __init__(self, actions: np.ndarray):
        self.actions = actions

    def reset(self, context=None):
        self.cursor = 0

    def act(self, observation, task):
        action = self.actions[self.cursor].copy()
        self.cursor += 1
        return action


def rollout(policy, bench, seed: int, trace: TransferTrace) -> dict:
    env = bench.create_env(render_cameras=False)
    try:
        result = bench.run_episode(policy, seed, env=env, diagnostic=trace)
        return trace.result(result, env)
    finally:
        env.close()


def reconstruct(path: Path, count: int) -> np.ndarray:
    stages = []
    with np.load(path, allow_pickle=False) as archive:
        for batch in active_batches(count):
            start = 0
            for size, length in zip(SAMPLES, archive["lengths"][batch], strict=True):
                stage = resample(archive["traces"][batch, start:start + size], int(length))
                if stages:
                    blend_size = min(3, len(stage))
                    blend = np.arange(1, blend_size + 1)[:, None] / blend_size
                    stage[:blend_size] = (1 - blend) * stages[-1][-1] + blend * stage[:blend_size]
                stages.append(stage)
                start += size
    return np.concatenate(stages).astype(np.float64)


def audit(root: Path, archived: dict) -> dict:
    profile, seed, count = (archived[key] for key in ("profile", "seed", "first_grasp"))
    check_collection((profile, seed, count, root / "shards"), archived)
    trace = Commands()
    expert = rollout("variable_grasp", benchmark(profile, first_grasp=count, max_steps=COLLECT_STEPS,
                                                complete_demonstrations=True), seed, trace)
    for key in ("steps", "scene_sha256", "applied_action_sha256", "target_cookie_ids", "success"):
        if expert[key] != archived[key]:
            raise ValueError(f"expert replay disagrees with original demonstration {seed}: {key}")
    raw = np.stack(trace.actions)
    aligned = reconstruct(root / "shards" / f"{seed}.npz", count)
    if len(aligned) != len(raw):
        raise ValueError("aligned phases do not cover the entire expert rollout")
    outcomes = {name: rollout(FixedActions(actions), benchmark(profile, first_grasp=count, max_steps=len(raw)),
                              seed, TransferTrace())
                for name, actions in (("replay_raw", raw), ("replay_aligned", aligned))}
    for row in outcomes.values():
        check_rollout(row, profile, seed, count, len(raw))
        if row["scene_sha256"] != expert["scene_sha256"]:
            raise ValueError("replay reset differs from expert reset")
    return {"seed": seed, "first_grasp": count, "expert": expert, **outcomes,
            "raw_matches_applied_digest": outcomes["replay_raw"]["applied_action_sha256"] == expert["applied_action_sha256"],
            "alignment_max_joint_deviation_rad": float(np.max(np.abs((aligned - raw)[:, :7])))}


def main():
    parser = argparse.ArgumentParser(description="Replay original and phase-aligned commands on accepted training scenes.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    original = args.run / "initial" / "dataset.json"
    manifest = read_json(original)
    scenarios = [next(row for row in manifest["attempts"] if row["accepted"] and row["profile"] == "advanced"
                      and row["first_grasp"] == count) for count in (0, 5)]
    _record_once(args.out / "request.json", {"dataset_sha256": sha256(original), "seeds": [row["seed"] for row in scenarios]})
    with ProcessPoolExecutor(max_workers=len(scenarios)) as pool:
        futures = [pool.submit(audit, args.run, row) for row in scenarios]
        for future in futures:
            result = future.result()
            _record_once(args.out / f"{result['seed']}.json", result)
            print(json.dumps({"seed": result["seed"], **{name: result[name]["requested_score"]
                             for name in ("expert", "replay_raw", "replay_aligned")}}), flush=True)


if __name__ == "__main__":
    main()
