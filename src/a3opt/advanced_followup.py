from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import (
    MILESTONE, SUITES, TARGET, AdvancedBudget, _actions_dataset, _audit_evaluation, _audit_expert, _calibrate,
    _jobs, _npz, _rank, _record_once, sample_plan, summarize,
)
from a3opt.advanced_sim import EvaluatedPolicy, benchmark, collect_expert
from a3opt.backends.advanced_phase import AdvancedPhasePolicy, align_phases, load_advanced_policy, select_penalties
from a3opt.backends.advanced_trajectory import AdvancedTrajectoryPolicy, pose_label
from a3opt.refinement_eval import read_json, sha256, simulator_identity, write_json


class FollowupBudget(AdvancedBudget):
    def seeds(self, split: str, suite: str = "1") -> tuple[int, ...]:
        if split == "development":
            offset, count = (2100, 12) if suite == "random" else (2000, 8)
            return tuple(range(self.seed_block + offset, self.seed_block + offset + count))
        return super().seeds(split, suite)


def collect(seed: int, column: int, folder: Path) -> dict:
    try:
        return collect_expert(seed, column, folder)
    except RuntimeError as error:
        if not str(error).startswith("same-column batch tilt "):
            raise
        return {"seed": seed, "column": column, "batches": [], "shard": None, "color": None,
                "success": False, "failure_reason": str(error)}


def evaluate(checkpoint: Path, suite: str, seed: int) -> dict:
    import cv2

    cv2.setNumThreads(1)
    bench = benchmark(suite)
    env = bench.create_env(render_cameras=False)
    policy = EvaluatedPolicy(load_advanced_policy(checkpoint))
    try:
        result = bench.run_episode(policy, seed=seed, env=env, diagnostic=policy)
        provenance = np.bincount(policy.columns[policy.target], minlength=5)[1:].tolist()
        if result.score != sum(provenance) or result.steps != policy.step:
            raise RuntimeError("upstream score does not match the recorded steps and cookie IDs")
        return {"seed": seed, "column": result.source_column, "score": result.score, "success": result.success,
                "requested_score": provenance[result.source_column - 1], "by_column": provenance, "steps": result.steps,
                "wall_seconds": result.wall_seconds, "first_batch_step": policy.first_batch,
                "failure_reason": result.failure_reason, "safety_reason": policy.safety,
                "source_remaining": result.cookies_in_source, "action_sha256": policy.digest.hexdigest(),
                "color": result.randomization["cookie_color"], "estimated_pose": policy.policy.pose.tolist(),
                "initial_pose": pose_label(result.randomization).tolist()}
    finally:
        env.close()


def _evaluate(pool, checkpoint: Path, directory: Path, budget: FollowupBudget, split: str) -> dict:
    seeds = {suite: list(budget.seeds(split, suite)) for suite in SUITES}
    _record_once(directory / "request.json", {"checkpoint_sha256": sha256(checkpoint), "split": split, "seeds": seeds})
    inputs = [(checkpoint, suite, seed) for suite in SUITES for seed in seeds[suite]]
    rows = _jobs(pool, evaluate, inputs, directory, lambda job: f"{job[1]}/{job[2]}", _audit_evaluation)
    results = {}
    for suite, requested in seeds.items():
        suite_rows = [row for row in rows if row["seed"] in requested]
        results[suite] = {"metrics": summarize(suite_rows), "rows": suite_rows}
    write_json(directory / "results.json", results)
    return results


def _phase_dataset(dataset: dict, attempts: list[dict], expert: Path) -> dict:
    traces, lengths = [], []
    for row in attempts:
        if not row["batches"]:
            continue
        with np.load(expert / row["shard"], allow_pickle=False) as archive:
            for batch in row["batches"]:
                trace, duration = align_phases(archive[f"actions_{batch}"], archive[f"phases_{batch}"])
                traces.append(trace)
                lengths.append(duration)
    return {**dataset, "traces": np.asarray(traces), "lengths": np.asarray(lengths)}


def run_followup(root: Path, reference: Path, budget: FollowupBudget, workers: int) -> dict:
    budget.validate()
    root, reference = root.resolve(), reference.resolve()
    previous = read_json(reference / "protocol.json")
    previous_lock = read_json(reference / "confirmation-lock.json")
    if budget.seed_block == previous["budget"]["seed_block"] or sha256(reference / "protocol.json") != previous_lock["protocol_sha256"]:
        raise ValueError("follow-up needs a different seed block and an intact reference protocol")
    previous_development = Path(previous_lock["selected"]["development"])
    feedback = read_json(previous_development)
    old_budget = AdvancedBudget(**previous["budget"])
    for suite in SUITES:
        if [row["seed"] for row in feedback[suite]["rows"]] != list(old_budget.seeds("development", suite)):
            raise ValueError("reference feedback must be from the locked development split")
    package = Path(__file__).parent
    protocol = {"search_version": "phase", "budget": vars(budget), "task": "2x5", "profile": "advanced",
                "prompt_style": "task", "max_steps": 500, "milestone_at_least": MILESTONE, "target_above": TARGET,
                "reference_development_sha256": sha256(previous_development), "environment": simulator_identity(),
                "code_sha256": {name: sha256(package / name) for name in (
                    "advanced_followup.py", "advanced_agent.py", "advanced_sim.py", "backends/advanced_trajectory.py",
                    "backends/advanced_phase.py", "backends/trajectory.py", "refinement_eval.py")}}
    _record_once(root / "protocol.json", protocol)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    expert = root / "expert"
    calibration = root / "calibration.npz"
    attempts, incumbent = [], None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        vision, scenes = _calibrate(pool, calibration, budget.seeds("train"))
        print(f"Follow-up Agent calibrated on {len(scenes)} training scenes", flush=True)
        for round_index in range(1, budget.rounds + 1):
            directory = root / f"round-{round_index:02d}"
            count = budget.initial_attempts if round_index == 1 else budget.feedback_attempts
            plan = sample_plan(scenes, attempts, feedback, count)
            _record_once(directory / "plan.json", {"attempts": plan})
            jobs = [(item["seed"], item["column"], expert) for item in plan]
            attempts += _jobs(pool, collect, jobs, expert, lambda job: f"{job[1]}-{job[0]}", _audit_expert)
            data = _actions_dataset(attempts, expert)
            coverage = {column: [int(np.count_nonzero((data["columns"] == column) & (data["batches"] == batch)))
                                 for batch in (0, 1)] for column in range(1, 5)}
            dataset_path = directory / "dataset.json"
            _record_once(dataset_path, {"attempts": len(attempts), "coverage": {str(key): value for key, value in coverage.items()},
                                        "records": attempts})
            print(f"Follow-up round {round_index} retained batches {coverage}", flush=True)
            if any(min(counts) < 4 for counts in coverage.values()):
                continue
            phases = _phase_dataset(data, attempts, expert)
            penalties, validation = select_penalties(phases)
            _record_once(directory / "phase-validation.json", validation)
            variants = ((AdvancedTrajectoryPolicy.name, data, {"mode": "factored", "neighbors": 20, "ridge": .015}),
                        (AdvancedPhasePolicy.name, phases, {"phase_penalties": penalties.tolist()}))
            candidates = []
            for slot, (kind, dataset, parameters) in enumerate(variants, 1):
                candidate = directory / f"candidate-{slot}"
                checkpoint = candidate / "policy.npz"
                _record_once(candidate / "training.json", {"parameters": parameters, "format": kind,
                    "dataset_sha256": sha256(dataset_path), "calibration_sha256": sha256(calibration)})
                if not checkpoint.exists():
                    _npz(checkpoint, format=kind, **vision, **dataset, **parameters)
                development = _evaluate(pool, checkpoint, candidate / "development", budget, "development")
                candidates.append({"checkpoint": str(checkpoint), "parameters": parameters, "rank": list(_rank(development)),
                                   "development": str(candidate / "development/results.json")})
                print(f"Follow-up round {round_index} candidate {slot} development {candidates[-1]['rank']}", flush=True)
            best = max(candidates, key=lambda candidate: candidate["rank"])
            if incumbent is None or best["rank"] > incumbent["rank"]:
                incumbent = best
            feedback = read_json(Path(incumbent["development"]))
            _record_once(directory / "decision.json", {"considered": candidates, "selected": incumbent})
            if incumbent["rank"][0] > TARGET and round_index >= 2:
                break
        if incumbent is None:
            result = {"status": "insufficient_expert_coverage", "collection_attempts": len(attempts)}
        else:
            checkpoint = Path(incumbent["checkpoint"])
            _record_once(root / "confirmation-lock.json", {"selected": incumbent, "checkpoint_sha256": sha256(checkpoint),
                "protocol_sha256": sha256(root / "protocol.json"),
                "seeds": {suite: list(budget.seeds("confirmation", suite)) for suite in SUITES}})
            print("Follow-up Agent froze its development winner", flush=True)
            confirmation = _evaluate(pool, checkpoint, root / "confirmation", budget, "confirmation")
            result = {"status": "confirmed", "checkpoint": str(checkpoint), "collection_attempts": len(attempts),
                      "development": {suite: item["metrics"] for suite, item in feedback.items()},
                      "confirmation": {suite: item["metrics"] for suite, item in confirmation.items()},
                      "milestone_met": _rank(confirmation)[0] >= MILESTONE, "target_met": _rank(confirmation)[0] > TARGET}
    write_json(root / "result.json", result)
    return result
