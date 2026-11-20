from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from a3opt.adaptive_sim import AdaptiveTrace
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import APPROACH, RETRACT, TRANSFER, AdaptivePhasePolicy, load_checkpoint
from a3opt.backends.advanced_trajectory import pose_label
from a3opt.backends.trajectory import resample
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES, check_row
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_completion_eval import STRICT_HOLD
from a3opt.release_outcome_agent import confirmation_stats
from a3opt.second_batch_agent import frozen_reference


SCALES = (1., .85, .75)
STAGES = (APPROACH, TRANSFER)
STRICT_RULES = {**RULES, "strict_policy_completion": True, "success_hold_steps": STRICT_HOLD}


class ReleaseTimingPolicy(AdaptivePhasePolicy):
    def __init__(self, checkpoint: Path, column: int, scale: float):
        if column not in range(1, 5) or scale not in SCALES:
            raise ValueError("invalid source column or transit scale")
        reference = load_checkpoint(checkpoint)
        self.timing_column, self.timing_scale = column, scale
        super().__init__(reference.model, reference.recipes, reference.detector)

    def _phase_actions(self, column, recipe):
        plan = super()._phase_actions(column, recipe)
        if column == self.timing_column and self.timing_scale < 1.:
            for stage in STAGES:
                actions = plan[1][stage]
                length = max(2, round(len(actions) * self.timing_scale))
                if length < len(actions):
                    plan[1][stage] = resample(actions, length)
        return plan


class TimingTrace(AdaptiveTrace):
    def reset(self, context=None) -> None:
        super().reset(context)
        self.first_batch_prefix = None

    def add_frame(self, observation, action, info, **context):
        super().add_frame(observation, action, info, **context)
        if self.policy.batch == 0 and self.policy.stage == RETRACT and self.policy.ended:
            self.first_batch_prefix = {"steps": self.step, "applied": self.applied_digest.hexdigest(), "raw": self.digest.hexdigest()}


def episode(checkpoint: Path, column: int, scale: float, seed: int) -> dict:
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column=str(column), **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = TimingTrace(ReleaseTimingPolicy(checkpoint, column, scale), first_batch_only=False)
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        meta = result.randomization
        ids = np.flatnonzero(trace.target).tolist()
        counts = np.bincount(trace.columns[ids], minlength=5)[1:].tolist()
        return {"checkpoint_sha256": sha256(checkpoint), "column": result.source_column, "suite": str(column),
                "seed": seed, "profile": meta["profile"], "steps": result.steps, "score": result.score,
                "requested_score": counts[column - 1], "by_column": counts, "target_cookie_ids": ids,
                "source_remaining": result.cookies_in_source, "success": result.success, "terminal": trace.terminal,
                "safety_reason": trace.safety, "color": meta["cookie_color"], "task_prompt": meta["task_instruction"],
                "initial_pose": pose_label(meta).tolist(), "cameras": meta["cameras"], "light_diffuse": meta["light_diffuse"],
                "first_placement_prefix": trace.first_batch_prefix, "second_open_step": trace.events[1].get("open_start"),
                "second_retract_end": trace.events[1].get("retract_end"), "applied_action_sha256": trace.applied_digest.hexdigest(),
                "action_sha256": trace.digest.hexdigest(), "scale": scale}
    finally:
        env.close()


def audit(job: tuple, row: dict) -> None:
    checkpoint, column, scale, seed = job
    check_row((checkpoint, "advanced", str(column), seed), {**row, "success": row["terminal"]["hold_steps"] >= RULES["success_hold_steps"]
               and row["score"] == 10 and row["source_remaining"] == 70 and not row["safety_reason"]})
    required = row["terminal"]["hold_steps"] >= STRICT_HOLD and row["score"] == 10 and row["source_remaining"] == 70 and not row["safety_reason"]
    if (row["steps"] < RULES["max_steps"] and not row["safety_reason"] or row["success"] != required
            or row["scale"] != scale or row["checkpoint_sha256"] != sha256(checkpoint)):
        raise ValueError("timing receipt differs from the strict 500-step schedule")


def diagnose(source: Path) -> tuple[int, dict]:
    protocol = read_json(source / "protocol.json")
    result = read_json(source / "result.json")
    evidence = {column: [] for column in range(1, 5)}
    for row in result["rows"]["random"]:
        step = row["second_open_step"]
        if (row["official_success"] and not row["strict_success"] and row["strict_requested_score"] == 10
                and step is not None and step > RULES["max_steps"] - STRICT_HOLD):
            evidence[row["column"]].append(row["seed"])
    focus = max(evidence, key=lambda column: len(evidence[column]))
    if not evidence[focus] or protocol["scope"] != "confirmation":
        raise ValueError("no audited late-release diagnosis")
    return focus, {"late_source_seeds": evidence[focus], "source_sha256": sha256(source / "result.json")}


def compare(rows: dict) -> dict:
    baseline = rows["1.0"]
    result = {}
    for variant, group in rows.items():
        if variant == "1.0":
            continue
        stats = confirmation_stats(baseline, group)
        advances = [old["second_open_step"] - new["second_open_step"] for old, new in zip(baseline, group, strict=True)
                    if old["second_open_step"] is not None and new["second_open_step"] is not None]
        result[variant] = {**stats, "paired_open_step_advance_median": float(np.median(advances)) if advances else None,
                           "paired_open_steps_observed": len(advances)}
    return result


def passes(stats: dict) -> bool:
    return (stats["complete"] > stats["reference_complete"] and stats["mean"] >= stats["reference_mean"]
            and not (stats["major_losses"] or stats["wrong_source"] or stats["safety_stops"]))


def evaluate(pool, out: Path, reference: Path, column: int, seeds: list[int], scales: tuple[float, ...]) -> dict:
    jobs = [(reference, column, scale, seed) for seed in seeds for scale in scales]
    rows = _jobs(pool, episode, jobs, out / "episodes", lambda job: f"{job[2]}/{job[3]}", audit)
    groups = {str(scale): [row for row in rows if row["scale"] == scale] for scale in scales}
    comparisons = compare(groups)
    _record_once(out / "results.json", {"rows": groups, "paired": comparisons})
    return comparisons


def run(out: Path, reference_run: Path, diagnosis: Path, block: int, workers: int) -> dict:
    schedule = {"scout": {"random": []}, "training": list(range(block, block + 8)), "development": [],
                "confirmation": {"focused": list(range(block + 100, block + 120))}}
    reference, identity = frozen_reference(reference_run, schedule)
    parent = read_json(diagnosis / "protocol.json")
    if parent["reference_sha256"] != sha256(reference) or parent["environment"] != identity["environment"]:
        raise ValueError("timing experiment requires the audited frozen reference")
    column, evidence = diagnose(diagnosis)
    _record_once(out / "protocol.json", {**identity, "strict_rules": STRICT_RULES, "terminate_on_success": False,
                 "scales": list(SCALES), "second_batch_stages": ["approach", "transfer"], "schedule": schedule,
                 "column": column, "diagnosis": evidence, "agent_code_sha256": sha256(Path(__file__))})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        train = evaluate(pool, out / "screening", reference, column, schedule["training"], SCALES)
        eligible = [scale for scale in train if passes(train[scale])]
        if not eligible:
            return {"status": "reference_retained_after_screening", "column": column, "screening": train}
        selected = max(eligible, key=lambda scale: (train[scale]["complete"], train[scale]["mean"], float(scale)))
        manifest = {"column": column, "scale": float(selected), "reference": str(reference),
                    "reference_sha256": sha256(reference), "protocol_sha256": sha256(out / "protocol.json"),
                    "screening_sha256": sha256(out / "screening/results.json")}
        _record_once(out / "confirmation-lock.json", manifest)
        confirmation = evaluate(pool, out / "confirmation", reference, column, schedule["confirmation"]["focused"], (1., float(selected)))
    trial = confirmation[selected]
    accepted = passes(trial) and trial["paired_success_rate_ci95"][0] > 0
    return {"status": "focused_timing_confirmed" if accepted else "reference_retained_after_confirmation",
            "column": column, "candidate": manifest, "screening": train, "confirmation": confirmation}


def upstream_policy() -> ReleaseTimingPolicy:
    rule = read_json(Path(os.environ["A3OPT_RELEASE_TIMING_LOCK"]))
    reference = Path(rule["reference"])
    if sha256(reference) != rule["reference_sha256"]:
        raise ValueError("timing policy requires the locked reference")
    return ReleaseTimingPolicy(reference, rule["column"], rule["scale"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize second-batch transit duration for released stability within 500 steps.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=90000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with _run_lock(args.out.resolve()):
        result = run(args.out.resolve(), args.reference.resolve(), args.diagnosis.resolve(), args.seed_block, args.workers)
        _record_once(args.out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
