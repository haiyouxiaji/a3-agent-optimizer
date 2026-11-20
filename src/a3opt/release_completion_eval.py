from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from a3opt.adaptive_sim import AdaptiveTrace
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.advanced_trajectory import pose_label
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES, _bootstrap, check_row
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_feedback_agent import ReleaseFeedbackPolicy
from a3opt.second_batch_agent import frozen_reference


STRICT_HOLD = 20
EXAMPLES = {"reference": (84208, 84219), "initial": (84208, 84219)}


class CompletionTrace(AdaptiveTrace):
    def __init__(self, checkpoint: Path, official_steps: int):
        super().__init__(ReleaseFeedbackPolicy(checkpoint), first_batch_only=False)
        self.official_steps = official_steps

    def reset(self, context=None) -> None:
        super().reset(context)
        self.official = None
        self.first_released_success = None

    def add_frame(self, observation, action, info, *, env, **context) -> None:
        super().add_frame(observation, action, info, env=env, **context)
        if info["success"] and self.first_released_success is None:
            self.first_released_success = self.step
        if self.step != self.official_steps:
            return
        config = env.task_config
        env.task_config = replace(config, require_released=False)
        try:
            counted = tuple(env._cookie_inside_target(index) for index in range(config.cookie_count))
            slots = list(env._target_slot_occupancy(counted))
        finally:
            env.task_config = config
        ids = np.flatnonzero(counted).tolist()
        self.official = {"action_sha256": self.digest.hexdigest(), "applied_action_sha256": self.applied_digest.hexdigest(),
                         "target_cookie_ids": ids, "slot_occupancy": slots,
                         "held_cookie_ids": [index for index in ids if any(env.privileged_left_finger_contacts(index))],
                         "released_count": int(self.target.sum())}


def replay(checkpoint: Path, expected: dict) -> dict:
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    strict_rules = {**RULES, "strict_policy_completion": True, "success_hold_steps": STRICT_HOLD}
    engine = make_cookie_benchmark(profile=expected["profile"], source_column=expected["suite"], **strict_rules)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = CompletionTrace(checkpoint, expected["steps"])
        result = engine.run_episode(trace, seed=expected["seed"], env=env, diagnostic=trace)
        if (pose_label(result.randomization).tolist() != expected["initial_pose"]
                or result.randomization["cameras"] != expected["cameras"]
                or result.randomization["light_diffuse"] != expected["light_diffuse"]
                or result.randomization["cookie_color"] != expected["color"]):
            raise ValueError("strict replay reset differs from the original scene")
        terminal_ids = np.flatnonzero(trace.target).tolist()
        return {"checkpoint_sha256": sha256(checkpoint), "seed": result.seed, "suite": expected["suite"],
                "column": result.source_column, "profile": expected["profile"], "official_steps": expected["steps"],
                "official_score": expected["requested_score"], "official_success": expected["success"],
                "prefix": trace.official, "strict_steps": result.steps, "strict_score": result.score,
                "strict_requested_score": int(np.count_nonzero(trace.target & (trace.columns == result.source_column))),
                "strict_success": result.success, "strict_hold_steps": trace.terminal["hold_steps"],
                "strict_target_cookie_ids": terminal_ids, "strict_source_remaining": result.cookies_in_source,
                "first_strict_success_step": trace.first_released_success, "safety_reason": trace.safety,
                "second_open_step": trace.events[1].get("open_start"), "second_retract_end": trace.events[1].get("retract_end")}
    finally:
        env.close()


def audit(job: tuple, row: dict) -> None:
    checkpoint, expected = job
    check_row((checkpoint, expected["profile"], expected["suite"], expected["seed"]), expected)
    keys = ("seed", "column", "suite", "profile")
    if (row["checkpoint_sha256"] != sha256(checkpoint) or any(row[key] != expected[key] for key in keys)
            or row["official_steps"] != expected["steps"] or row["official_score"] != expected["requested_score"]
            or row["official_success"] != expected["success"]):
        raise ValueError("completion audit does not match the official episode")
    prefix = row["prefix"]
    if (prefix is None or any(prefix[key] != expected[key] for key in ("action_sha256", "applied_action_sha256", "target_cookie_ids"))
            or prefix["slot_occupancy"] != expected["terminal"]["slot_occupancy"]):
        raise ValueError("strict replay did not reproduce the official action or scoring prefix")
    success = (row["strict_score"] == 10 and row["strict_source_remaining"] == 70
               and row["strict_hold_steps"] >= STRICT_HOLD and not row["safety_reason"])
    if (not expected["steps"] <= row["strict_steps"] <= RULES["max_steps"] or row["strict_success"] != success
            or len(set(row["strict_target_cookie_ids"])) != row["strict_score"]
            or row["strict_steps"] < RULES["max_steps"] and not row["safety_reason"]):
        raise ValueError("strict replay violates the released terminal scoring contract")


def summarize(rows: list[dict]) -> dict:
    official = np.array([row["official_score"] for row in rows])
    strict = np.array([row["strict_requested_score"] for row in rows])
    columns = [row["column"] for row in rows]
    return {"episodes": len(rows), "official_mean": float(official.mean()), "official_successes": sum(row["official_success"] for row in rows),
            "strict_mean": float(strict.mean()), "strict_mean_ci95": _bootstrap(strict, columns),
            "strict_successes": sum(row["strict_success"] for row in rows),
            "strict_minus_official_mean": float((strict - official).mean()),
            "strict_minus_official_ci95": _bootstrap(strict - official, columns),
            "official_success_while_in_contact": sum(row["official_success"] and bool(row["prefix"]["held_cookie_ids"]) for row in rows),
            "wrong_source": sum(row["strict_score"] != row["strict_requested_score"] for row in rows),
            "safety_stops": sum(bool(row["safety_reason"]) for row in rows)}


def run(out: Path, branch: Path, release: Path, workers: int, scope: str) -> dict:
    reference, identity = frozen_reference(branch, {"scout": {"random": []}, "training": [], "development": [], "confirmation": {}})
    branch_path = branch / "confirmation/selected/results.json"
    release_path = release / "development/results.json"
    request = read_json(release / "development/request.json")
    release_lock = read_json(release / "confirmation-lock.json")
    selected = release / "candidates" / f"{release_lock['candidate']}.npz"
    checkpoints = {"reference": reference, release_lock["candidate"]: selected}
    if (release_lock["protocol_sha256"] != sha256(release / "protocol.json")
            or release_lock["checkpoint_sha256"] != sha256(selected)
            or any(request["checkpoints"][name] != sha256(path) for name, path in checkpoints.items())):
        raise ValueError("release replay requires the original locked checkpoints")
    if scope == "examples":
        source = read_json(release_path)
        groups = {name: [(checkpoints[name], row) for row in source[name] if row["seed"] in seeds]
                  for name, seeds in EXAMPLES.items()}
        if any(len(groups[name]) != len(seeds) for name, seeds in EXAMPLES.items()):
            raise ValueError("missing source receipts for the example replays")
    else:
        rows = read_json(branch_path)["rows"]
        groups = {suite: [(reference, row) for row in rows if row["suite"] == suite] for suite in ("1", "random")}
        if any(len(group) != 20 for group in groups.values()):
            raise ValueError("missing source receipts for the original confirmation")
    protocol = {**identity, "scope": scope, "audit_rules": {**RULES, "strict_policy_completion": True,
                 "success_hold_steps": STRICT_HOLD, "terminate_on_success": False},
                "source_receipts_sha256": sha256(release_path if scope == "examples" else branch_path),
                "release_lock_sha256": sha256(release / "confirmation-lock.json"),
                "audit_code_sha256": sha256(Path(__file__)), "release_policy_sha256": sha256(Path(__file__).with_name("release_feedback_agent.py")),
                "groups": {label: [row["seed"] for _, row in group] for label, group in groups.items()}}
    _record_once(out / "protocol.json", protocol)
    data = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for label, jobs in groups.items():
            data[label] = _jobs(pool, replay, jobs, out / "episodes" / label, lambda job: str(job[1]["seed"]), audit)
            print(label, summarize(data[label]), flush=True)
    return {"summary": {label: summarize(rows) for label, rows in data.items()}, "rows": data}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the official prefix and audit released placement at the 500-step endpoint.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--scope", choices=("examples", "confirmation"), default="confirmation")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    with _run_lock(args.out.resolve()):
        result = run(args.out.resolve(), args.reference.resolve(), args.release.resolve(), args.workers, args.scope)
        _record_once(args.out / "result.json", result)


if __name__ == "__main__":
    main()
