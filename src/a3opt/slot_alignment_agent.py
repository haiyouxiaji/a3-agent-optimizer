from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from a3opt.adaptive_agent import paired_summary
from a3opt.adaptive_sim import phase_model
from a3opt.advanced_agent import TARGET, _jobs, _npz, _record_once
from a3opt.backends.adaptive_phase import PLACE, RETRACT
from a3opt.backends.slot_alignment import KEYS, MAX_SHIFT_M, from_adaptive, release_features
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import run_episode
from a3opt.refinement_eval import read_json, sha256
from a3opt.second_batch_agent import frozen_reference
from a3opt.second_batch_fit import improvement, nonregression, paired_comparison
from a3opt.second_batch_sim import SecondBatchTrace, check_episode
from a3opt.slot_alignment_fit import first_row_aligned, first_slot_miss, fit_alignment


class SlotTrace(SecondBatchTrace):
    def __init__(self, checkpoint: Path, collect: bool):
        super().__init__(checkpoint)
        self.policy = from_adaptive(self.policy)
        self.collect = collect

    def bind_env(self, env) -> None:
        super().bind_env(env)
        slots = np.asarray(env.TARGET_SLOTS_LOCAL)
        self.front_slots = np.flatnonzero(~np.isclose(slots[:, 0], self.goal[0])).tolist()
        if len(self.front_slots) != 5:
            raise ValueError("slot calibration requires two target columns of five slots")
        self.front_xy = slots[self.front_slots].tolist()

    def reset(self, context=None) -> None:
        super().reset(context)
        self.first_prefix = hashlib.sha256()
        self.first_prefix_steps = 0
        self.first_after_retract = None

    def add_frame(self, observation, action, info, *, env, **context) -> None:
        super().add_frame(observation, action, info, env=env, **context)
        if self.policy.batch != 0:
            return
        if self.policy.stage < PLACE:
            self.first_prefix.update(np.asarray([action, info["applied_action"]], dtype="<f8").tobytes())
            self.first_prefix_steps = self.step
        if self.policy.stage == RETRACT and self.policy.ended:
            self.first_after_retract = self._snapshot(env, info)
            if self.collect:
                self.placement_events["terminal"] = self.first_after_retract
                self.finished = True


def evaluate(checkpoint: Path, profile: str, suite: str, seed: int, collect: bool) -> dict:
    trace = SlotTrace(checkpoint, collect)
    row = run_episode(checkpoint, profile, suite, seed, trace=trace)
    features = release_features(trace.policy.pose, trace.policy.actions[0]).tolist() if collect else None
    row.update(checkpoint_sha256=sha256(checkpoint), batches=trace.events, retries=trace.policy.retries,
               first_requested_five=trace.first_requested_five, prefix_sha256=trace.prefix.hexdigest(),
               prefix_steps=trace.prefix_steps, placement_events=trace.placement_events, second_batch_goal_m=trace.goal,
               first_placement_prefix={"prefix_steps": trace.first_prefix_steps, "prefix_sha256": trace.first_prefix.hexdigest()},
               first_after_retract=trace.first_after_retract, front_slots=trace.front_slots, front_slot_xy_m=trace.front_xy,
               first_release_features=features, collection_only=collect)
    return row


def check_slot_episode(job: tuple, row: dict) -> None:
    check_episode(job, row)
    prefix = row["first_placement_prefix"]
    retract = row["first_after_retract"]
    if (row["collection_only"] != job[4] or not 0 < prefix["prefix_steps"] <= row["steps"]
            or len(prefix["prefix_sha256"]) != 64 or (retract and retract["step"] <= prefix["prefix_steps"])
            or (job[4] and retract and retract["step"] != row["steps"])):
        raise ValueError("episode does not match the first-batch calibration protocol")


def _episodes(pool, checkpoints: dict[str, Path], folder: Path, schedule: dict, *, collect: bool = False) -> dict:
    digests = {name: sha256(path) for name, path in checkpoints.items()}
    _record_once(folder / "request.json", {"checkpoints": digests, "seeds": schedule, "collection_only": collect})
    jobs = [(path, "advanced", suite, seed, collect) for path in checkpoints.values()
            for suite, seeds in schedule.items() for seed in seeds]
    names = {path: name for name, path in checkpoints.items()}
    rows = _jobs(pool, evaluate, jobs, folder / "episodes", lambda job: f"{names[job[0]]}/{job[2]}/{job[3]}",
                 check_slot_episode)
    groups = {name: [row for row in rows if row["checkpoint_sha256"] == digest] for name, digest in digests.items()}
    _record_once(folder / "results.json", groups)
    print(f"{folder.name}: {len(rows)} episodes", flush=True)
    return groups


def _compare(rows: dict, column: int) -> dict:
    original, altered = rows["reference"], rows["selected"]
    control = [{**row, **row["first_placement_prefix"]} for row in original]
    trial = [{**row, **row["first_placement_prefix"]} for row in altered]
    comparison = paired_comparison(control, trial)
    for base, choice in zip(control, trial, strict=True):
        if base["column"] != column and any(base[key] != choice[key] for key in (
                "score", "success", "steps", "applied_action_sha256", "target_cookie_ids")):
            raise ValueError("calibration changed the outcome outside the selected source column")
    return {**comparison, "first_row_aligned": {name: sum(map(first_row_aligned, group)) for name, group in rows.items()}}


def _focus(diagnosis: Path, reference: Path, schedule: dict) -> tuple[int | None, dict]:
    protocol = diagnosis / "protocol.json"
    previous = read_json(protocol)
    reserved = set(schedule["training"] + schedule["development"] + sum(schedule["confirmation"].values(), []))
    earlier = set(previous["schedule"]["scout"]["random"] + previous["schedule"]["training"]
                  + previous["schedule"]["development"] + sum(previous["schedule"]["confirmation"].values(), []))
    if reserved.intersection(earlier) or previous["reference_sha256"] != sha256(reference):
        raise ValueError("first-batch run requires the same frozen baseline and fresh seeds")
    paths = [diagnosis / "scout" / "results.json", *sorted(diagnosis.glob("column-*/training/reference/results.json"))]
    misses = Counter()
    for path in paths:
        for row in read_json(path)["rows"]:
            if row["checkpoint_sha256"] != previous["reference_sha256"] or row["seed"] not in earlier:
                raise ValueError("unmatched provenance in the diagnostic episode")
            if first_slot_miss(row):
                misses[row["column"]] += 1
    priority = sorted(misses, key=lambda column: (-misses[column], column))
    focus = priority[0] if priority and misses[priority[0]] >= 2 else None
    return focus, {"misses_by_column": {str(key): count for key, count in misses.items()},
                   "diagnostic_results_sha256": {path.relative_to(diagnosis).as_posix(): sha256(path) for path in paths},
                   "diagnostic_protocol_sha256": sha256(protocol)}


def run(root: Path, reference_root: Path, diagnosis: Path, block: int, workers: int) -> dict:
    if block < 0 or block % 4:
        raise ValueError("seed block must be a nonnegative multiple of four")
    schedule = {"scout": {"random": []}, "training": list(range(block, block + 20)),
                "development": list(range(block + 200, block + 212)),
                "confirmation": {"1": list(range(block + 300, block + 320)),
                                 "random": list(range(block + 400, block + 420))}}
    reference, identity = frozen_reference(reference_root, schedule)
    column, focus = _focus(diagnosis, reference, schedule)
    package = Path(__file__).parent
    sources = ("second_batch_agent.py", "second_batch_fit.py", "slot_alignment_agent.py", "slot_alignment_fit.py",
               "backends/slot_alignment.py")
    identity["code_sha256"].update({name: sha256(package / name) for name in sources})
    protocol = {**identity, "schedule": schedule, "focus": focus, "column": column, "maximum_shift_m": MAX_SHIFT_M,
                "target_score_floor": TARGET}
    _record_once(root / "protocol.json", protocol)
    result = {"status": "reference_retained_without_candidate", "selected_checkpoint": str(reference), "column": column}
    if column is None:
        _record_once(root / "result.json", result)
        return result

    print(f"Agent collecting first-batch slot labels on column {column}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        training = _episodes(pool, {"reference": reference}, root / "training", {str(column): schedule["training"]},
                             collect=True)
        model, dataset = fit_alignment(training["reference"], column)
        _record_once(root / "slot-dataset.json", {**dataset, "training_sha256": sha256(root / "training" / "results.json")})
        if model is not None:
            winner = root / "candidate" / "policy.npz"
            manifest = winner.with_suffix(".json")
            if not manifest.exists():
                _npz(winner, **phase_model(reference), **{f"slot_{key}": model[key] for key in KEYS})
            _record_once(manifest, {"checkpoint_sha256": sha256(winner), "dataset_sha256": sha256(root / "slot-dataset.json"),
                                    "parameters": {key: np.asarray(model[key]).tolist() for key in KEYS}})
            checkpoints = {"reference": reference, "selected": winner}
            development = _episodes(pool, checkpoints, root / "development", {str(column): schedule["development"]})
            comparison = _compare(development, column)
            result.update(status="reference_retained_after_development", development=comparison)
            _record_once(root / "development-selection.json", result)
            if improvement(comparison):
                _record_once(root / "confirmation-lock.json", {"checkpoint_sha256": sha256(winner),
                             "reference_checkpoint_sha256": sha256(reference), "protocol_sha256": sha256(root / "protocol.json"),
                             "development_sha256": sha256(root / "development-selection.json"), "seeds": schedule["confirmation"]})
                print("Agent locked slot calibration before independent confirmation", flush=True)
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
                              selected_checkpoint=str(winner if accepted else reference), candidate_score_floor=floor,
                              candidate_floor_ci95_lower=lower, confirmation=summary, paired_comparison=comparisons)
    _record_once(root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Learn first-batch longitudinal slot calibration from settled placements")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=73600)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 61:
        parser.error("workers must be between 1 and 61")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.diagnosis.resolve(), args.seed_block, args.workers)
    print(f"Agent finished: {result['status']}; candidate confirmation floor: {result.get('candidate_score_floor')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
