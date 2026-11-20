from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from a3opt.adaptive_agent import paired_summary
from a3opt.adaptive_sim import phase_model
from a3opt.advanced_agent import TARGET, _jobs, _npz, _record_once
from a3opt.backends.adaptive_phase import AdaptivePhasePolicy, Recipe, load_checkpoint
from a3opt.backends.slot_alignment import MAX_SHIFT_M, SlotAlignmentPolicy, from_adaptive, predict_shift, release_features
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import run_episode
from a3opt.refinement_eval import read_json, sha256
from a3opt.second_batch_agent import frozen_reference
from a3opt.second_batch_fit import improvement, nonregression
from a3opt.slot_alignment_agent import SlotTrace, _compare, check_slot_episode


class GuardedSlotPolicy(SlotAlignmentPolicy):
    def __init__(self, candidate: SlotAlignmentPolicy):
        self.guard = candidate.model["slot_guard_magnitude_m"]
        if (self.guard.shape != (2,) or not 0 <= self.guard[0] < self.guard[1] <= MAX_SHIFT_M):
            raise ValueError("invalid slot correction bounds")
        super().__init__(candidate, candidate.parameters)

    def _phase_actions(self, column: int, recipe: Recipe) -> list[list[np.ndarray]]:
        baseline = AdaptivePhasePolicy._phase_actions(self, column, recipe)
        if column != self.column:
            return baseline
        magnitude = abs(float(predict_shift(self.parameters, release_features(self.pose, baseline[0]))))
        if self.guard[0] <= magnitude <= self.guard[1]:
            return super()._phase_actions(column, recipe)
        return baseline


def upstream_policy() -> GuardedSlotPolicy:
    return GuardedSlotPolicy(from_adaptive(load_checkpoint(Path(os.environ["A3OPT_SLOT_GUARD_CHECKPOINT"]))))


def evaluate(checkpoint: Path, suite: str, seed: int) -> dict:
    trace = SlotTrace(checkpoint, collect=False)
    if "slot_guard_magnitude_m" in trace.policy.model:
        trace.policy = GuardedSlotPolicy(trace.policy)
    row = run_episode(checkpoint, "advanced", suite, seed, trace=trace)
    row.update(checkpoint_sha256=sha256(checkpoint), batches=trace.events, retries=trace.policy.retries,
               first_requested_five=trace.first_requested_five, prefix_sha256=trace.prefix.hexdigest(),
               prefix_steps=trace.prefix_steps, placement_events=trace.placement_events,
               first_placement_prefix={"prefix_steps": trace.first_prefix_steps, "prefix_sha256": trace.first_prefix.hexdigest()},
               first_after_retract=trace.first_after_retract, front_slots=trace.front_slots, collection_only=False)
    return row


def _episodes(pool, checkpoints: dict[str, Path], folder: Path, schedule: dict) -> dict:
    digests = {name: sha256(path) for name, path in checkpoints.items()}
    _record_once(folder / "request.json", {"checkpoints": digests, "seeds": schedule, "profile": "advanced"})
    names = {path: name for name, path in checkpoints.items()}
    jobs = [(path, suite, seed) for path in names for suite, seeds in schedule.items() for seed in seeds]
    rows = _jobs(pool, evaluate, jobs, folder / "episodes", lambda job: f"{names[job[0]]}/{job[1]}/{job[2]}",
                 lambda job, row: check_slot_episode((job[0], "advanced", *job[1:], False), row))
    groups = {name: [row for row in rows if row["checkpoint_sha256"] == digest] for name, digest in digests.items()}
    _record_once(folder / "results.json", groups)
    print(f"{folder.name}: {len(rows)} full episodes", flush=True)
    return groups


def learn_bounds(samples: list[dict]) -> dict | None:
    boundaries = np.linspace(0., MAX_SHIFT_M, 13).tolist()
    candidates = []
    for lower in boundaries:
        for upper in boundaries:
            if upper <= lower:
                continue
            active = [row for row in samples if lower <= row["magnitude_m"] <= upper]
            if any(row["score_delta"] < 0 or row["violation"] for row in active):
                continue
            improvement_count = sum(row["success_delta"] for row in active)
            if improvement_count > 0 and sum(row["success_delta"] > 0 for row in active) >= 2:
                candidates.append({"bounds_m": [lower, upper], "full_success_delta": improvement_count,
                                   "training_seeds_active": [row["seed"] for row in active]})
    return max(candidates, key=lambda choice: (choice["full_success_delta"],
                                               choice["bounds_m"][1] - choice["bounds_m"][0]), default=None)


def _training(source: Path, reference: Path, schedule: dict) -> tuple[Path, list[dict], dict]:
    protocol = read_json(source / "protocol.json")
    reserved = set(schedule["development"] + sum(schedule["confirmation"].values(), []))
    earlier = set(protocol["schedule"]["training"] + protocol["schedule"]["development"]
                  + sum(protocol["schedule"]["confirmation"].values(), []))
    if reserved.intersection(earlier) or protocol["reference_sha256"] != sha256(reference):
        raise ValueError("guard training requires fresh evaluation seeds and the original reference")
    path = source / "candidate" / "policy.npz"
    manifest = read_json(path.with_suffix(".json"))
    results_path = source / "development" / "results.json"
    results = read_json(results_path)
    candidate = from_adaptive(load_checkpoint(path))
    if sha256(path) != manifest["checkpoint_sha256"] or not isinstance(candidate, SlotAlignmentPolicy):
        raise ValueError("guard feedback does not match the frozen calibration")
    _compare(results, candidate.column)
    samples = []
    for base, trial in zip(results["reference"], results["selected"], strict=True):
        if base["checkpoint_sha256"] != sha256(reference) or trial["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
            raise ValueError("guard feedback refers to an unrecognized checkpoint")
        candidate.pose = np.asarray(base["estimated_pose"])
        commands = AdaptivePhasePolicy._phase_actions(candidate, candidate.column, candidate.recipes[candidate.column - 1])
        magnitude = abs(float(predict_shift(candidate.parameters, release_features(candidate.pose, commands[0]))))
        samples.append({"seed": base["seed"], "magnitude_m": magnitude,
                        "score_delta": trial["requested_score"] - base["requested_score"],
                        "success_delta": int(trial["success"]) - int(base["success"]),
                        "violation": bool(trial["safety_reason"] or trial["score"] != trial["requested_score"])})
    if [row["seed"] for row in samples] != protocol["schedule"]["development"]:
        raise ValueError("incomplete feedback sample")
    return path, samples, {"source_column": candidate.column, "feedback_sha256": sha256(results_path),
                           "calibration_sha256": sha256(path)}


def run(root: Path, reference_root: Path, source: Path, block: int, workers: int) -> dict:
    if block < 0 or block % 4:
        raise ValueError("seed block must be a nonnegative multiple of four")
    schedule = {"scout": {"random": []}, "training": [], "development": list(range(block, block + 12)),
                "confirmation": {"1": list(range(block + 100, block + 120)),
                                 "random": list(range(block + 200, block + 220))}}
    reference, identity = frozen_reference(reference_root, schedule)
    calibration, samples, feedback = _training(source, reference, schedule)
    package = Path(__file__).parent
    for name in ("second_batch_agent.py", "second_batch_fit.py", "slot_alignment_agent.py", "slot_alignment_fit.py",
                 "backends/slot_alignment.py", "slot_guard_agent.py"):
        identity["code_sha256"][name] = sha256(package / name)
    _record_once(root / "protocol.json", {**identity, **feedback, "schedule": schedule, "target_score_floor": TARGET})
    choice = learn_bounds(samples)
    _record_once(root / "feedback-dataset.json", {**feedback, "samples": samples, "selected": choice})
    result = {"status": "reference_retained_without_candidate", "selected_checkpoint": str(reference),
              "feedback_selection": choice}
    if choice:
        checkpoint = root / "candidate" / "policy.npz"
        manifest = checkpoint.with_suffix(".json")
        if not manifest.exists():
            _npz(checkpoint, **phase_model(calibration), slot_guard_magnitude_m=choice["bounds_m"])
        _record_once(manifest, {"sha256": sha256(checkpoint), "dataset_sha256": sha256(root / "feedback-dataset.json")})
        column = feedback["source_column"]
        checkpoints = {"reference": reference, "selected": checkpoint}
        print(f"Agent froze a guarded calibration for source column {column} before development", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            development = _episodes(pool, checkpoints, root / "development", {str(column): schedule["development"]})
            comparison = _compare(development, column)
            result.update(status="reference_retained_after_development", development=comparison)
            _record_once(root / "development-selection.json", result)
            if improvement(comparison):
                _record_once(root / "confirmation-lock.json", {"checkpoint_sha256": sha256(checkpoint),
                             "reference_checkpoint_sha256": sha256(reference), "protocol_sha256": sha256(root / "protocol.json"),
                             "development_sha256": sha256(root / "development-selection.json"), "seeds": schedule["confirmation"]})
                print("Agent locked the guarded calibration before independent confirmation", flush=True)
                confirmation = _episodes(pool, checkpoints, root / "confirmation", schedule["confirmation"])
                summary = paired_summary(confirmation["reference"], confirmation["selected"])
                comparisons = {suite: _compare({name: [row for row in group if row["suite"] == suite]
                                                for name, group in confirmation.items()}, column)
                               for suite in schedule["confirmation"]}
                floor = min(group["selected"]["mean_requested_score"] for group in summary.values())
                lower = min(group["selected"]["requested_mean_bootstrap_95"][0] for group in summary.values())
                accepted = (floor > TARGET and all(map(nonregression, comparisons.values()))
                            and any(map(improvement, comparisons.values())))
                result.update(status="promoted" if accepted else "reference_retained_after_confirmation",
                              selected_checkpoint=str(checkpoint if accepted else reference), candidate_score_floor=floor,
                              candidate_floor_ci95_lower=lower, confirmation=summary, paired_comparison=comparisons)
    _record_once(root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Use paired feedback to bound first-batch slot calibration")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=74800)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 61:
        parser.error("workers must be between 1 and 61")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.calibration.resolve(), args.seed_block, args.workers)
    print(f"Agent finished: {result['status']}; candidate confirmation floor: {result.get('candidate_score_floor')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
