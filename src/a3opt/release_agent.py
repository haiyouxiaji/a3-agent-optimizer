from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from a3opt.adaptive_agent import paired_summary, seed_schedule
from a3opt.adaptive_sim import AdaptiveTrace, check_adaptive_row, phase_model
from a3opt.advanced_agent import MILESTONE, TARGET, _jobs, _npz, _record_once
from a3opt.backends.adaptive_phase import PLACE
from a3opt.backends.release_residual import MAX_OFFSET, load_checkpoint, nominal_model_path
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES, run_episode
from a3opt.refinement_eval import read_json, sha256, simulator_identity, write_json


class ReleaseTrace(AdaptiveTrace):
    def __init__(self, checkpoint: Path, training: bool):
        super().__init__(load_checkpoint(checkpoint), first_batch_only=training)

    def add_frame(self, observation, action, info, **context) -> None:
        super().add_frame(observation, action, info, **context)
        policy = self.policy
        if not self.first_batch_only or not policy.ended or policy.stage != PLACE:
            return
        from a3_dual_arm_sim.controllers.placement_order import front_camera_column_order

        env = context["env"]
        ids = self.lifted_ids[0]
        self.events[0]["release_labels_available"] = 0
        if len(ids) != 5:
            return
        contacts = np.asarray([env.privileged_left_finger_contacts(int(index)) for index in ids])
        if not contacts.any(axis=0).all():
            return
        positions = np.stack([env.privileged_cookie_target_position(int(index)) for index in ids])
        center = positions.mean(axis=0)
        target_column = front_camera_column_order(env)[0]
        slots = np.asarray(env.TARGET_SLOTS_LOCAL)[target_column::2]
        desired = np.r_[slots.mean(axis=0), env._target_cookie_center_z + .004]
        if (center[2] < .020 or np.max(np.linalg.norm(positions - center, axis=1)) > .030
                or np.linalg.norm((center - desired)[:2]) > .050):
            return
        self.events[0]["release_labels_available"] = 1
        self.events[0]["release_label"] = {"cookie_ids": ids.tolist(), "center_target_frame": center.tolist(),
                                            "training_goal": desired.tolist(), "error": (desired - center).tolist(),
                                            "pad_contacts": contacts.tolist()}


def evaluate(checkpoint: Path, profile: str, suite: str, seed: int, training: bool) -> dict:
    trace = ReleaseTrace(checkpoint, training)
    row = run_episode(checkpoint, profile, suite, seed, trace=trace)
    row.update(batches=trace.events, retries=trace.policy.retries, first_requested_five=trace.first_requested_five,
               checkpoint_sha256=sha256(checkpoint))
    return row


def _schedule_evaluation(checkpoint: Path, folder: Path, schedule: dict, training: bool) -> list[tuple]:
    _record_once(folder / "request.json", {"checkpoint_sha256": sha256(checkpoint), "seeds": schedule,
                                           "mode": "first_batch" if training else "full", "profile": "advanced"})
    return [(checkpoint, "advanced", suite, seed, training) for suite, seeds in schedule.items() for seed in seeds]


def _save_evaluation(folder: Path, rows: list[dict]) -> None:
    _record_once(folder / "results.json", {"rows": rows})
    print(f"{folder.parent.name}/{folder.name}: {len(rows)} episodes", flush=True)


def _evaluate(pool, checkpoint: Path, folder: Path, schedule: dict, *, training: bool = False) -> list[dict]:
    jobs = _schedule_evaluation(checkpoint, folder, schedule, training)
    rows = _jobs(pool, evaluate, jobs, folder, lambda job: f"{job[2]}/{job[3]}", check_adaptive_row)
    _save_evaluation(folder, rows)
    return rows


def _evaluate_candidates(pool, checkpoints: dict[str, Path], folder: Path, schedule: dict,
                         *, training: bool = False) -> dict[str, list[dict]]:
    jobs = [inputs for name, path in checkpoints.items()
            for inputs in _schedule_evaluation(path, folder / name, schedule, training)]
    rows = _jobs(pool, evaluate, jobs, folder, lambda job: f"{job[0].parent.name}/{job[2]}/{job[3]}", check_adaptive_row)
    groups = {name: [row for row in rows if row["checkpoint_sha256"] == sha256(checkpoint)]
              for name, checkpoint in checkpoints.items()}
    for name, group in groups.items():
        _save_evaluation(folder / name, group)
    return groups


def _checkpoint(path: Path, reference: Path, column: int, offset: np.ndarray) -> Path:
    manifest = path.with_suffix(".json")
    if not manifest.exists():
        _npz(path, **phase_model(reference), release_column=column, release_offset=offset)
    _record_once(manifest, {"column": column, "offset_m": offset.tolist(), "reference_sha256": sha256(reference),
                            "checkpoint_sha256": sha256(path)})
    return path


def _fit(rows: list[dict]) -> tuple[np.ndarray, list[dict]]:
    dataset = [{"seed": row["seed"], "applied_action_sha256": row["applied_action_sha256"], **row["batches"][0]["release_label"]}
               for row in rows if not row["safety_reason"] and "release_label" in row["batches"][0]]
    if len(dataset) < 2:
        return np.zeros(3), dataset
    return np.clip(np.median([sample["error"] for sample in dataset], axis=0), -MAX_OFFSET, MAX_OFFSET), dataset


def _comparison(baseline: list[dict], candidate: list[dict]) -> dict:
    pairs = list(zip(baseline, candidate, strict=True))
    if any((before["suite"], before["seed"]) != (after["suite"], after["seed"]) for before, after in pairs):
        raise ValueError("release comparison needs paired scenes")
    deltas = [after["requested_score"] - before["requested_score"] for before, after in pairs]
    violations = sum(bool(after["safety_reason"] and not before["safety_reason"])
                     or after["score"] - after["requested_score"] > before["score"] - before["requested_score"]
                     for before, after in pairs)
    strict_delta = sum(int(after["success"] and after["requested_score"] == 10)
                       - int(before["success"] and before["requested_score"] == 10) for before, after in pairs)
    return {"paired_delta": float(np.mean(deltas)), "baseline": [row["requested_score"] for row in baseline],
            "candidate": [row["requested_score"] for row in candidate], "strict_success_delta": strict_delta,
            "new_safety_or_provenance_violations": violations, "positive_pairs": sum(delta > 0 for delta in deltas)}


def run(root: Path, reference: Path, placement: Path, block: int, workers: int) -> dict:
    schedule = seed_schedule(block)
    first_protocol = read_json(placement / "protocol.json")
    focus_path = placement / "training-focus.json"
    column = read_json(focus_path)["weakest_column"]
    baseline_protocol = reference / "protocol.json"
    lock = read_json(reference / "confirmation-lock.json")
    source = Path(lock["selected"]["checkpoint"])
    if (sha256(source) != first_protocol["reference_sha256"] or sha256(source) != lock["checkpoint_sha256"]
            or sha256(baseline_protocol) != lock["protocol_sha256"]
            or block == first_protocol["schedule"]["scout"][0] or block == read_json(baseline_protocol)["budget"]["seed_block"]
            or column not in (1, 2, 3, 4)):
        raise ValueError("expected an intact frozen baseline and a new seed block")
    calibration = {str(column): schedule["scout"][:8]}
    training = {str(column): schedule["focus"][:6]}
    development = {str(column): schedule["development"]["random"][:8]}
    package = Path(__file__).parent
    code_files = (*first_protocol["code_sha256"], "release_agent.py", "backends/release_residual.py")
    _record_once(root / "protocol.json", {"rules": RULES, "profile": "advanced", "column": column,
                 "reference_sha256": sha256(source), "training_diagnosis_sha256": sha256(focus_path),
                 "environment": simulator_identity(), "kinematics_sha256": sha256(nominal_model_path()),
                 "schedule": {"calibration": calibration, "training": training, "development": development,
                              "confirmation": schedule["confirmation"]},
                 "code_sha256": {name: sha256(package / name) for name in code_files}})
    weights = root / "candidates"
    baseline = _checkpoint(weights / "baseline" / "policy.npz", source, 0, np.zeros(3))
    coherent = _checkpoint(weights / "coherent" / "policy.npz", source, column, np.zeros(3))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        diagnostics = _evaluate(pool, coherent, root / "calibration" / "coherent", calibration, training=True)
        offset, dataset = _fit(diagnostics)
        _record_once(root / "release-dataset.json", {"samples": dataset, "fitted_offset_m": offset.tolist()})
        print(f"Agent learned target-frame offset in mm: {(offset * 1000).round(2).tolist()}", flush=True)
        alternatives = {"baseline": baseline, "coherent": coherent}
        if np.any(offset):
            for name, displacement in (("full-offset", offset), ("reduced-offset", .65 * offset)):
                alternatives[name] = _checkpoint(weights / name / "policy.npz", source, column, displacement)
        episodes = _evaluate_candidates(pool, alternatives, root / "training", training, training=True)
        train_evidence = {name: _comparison(episodes["baseline"], rows) for name, rows in episodes.items()}
        _record_once(root / "training-comparison.json", train_evidence)
        eligible = [name for name, evidence in train_evidence.items()
                    if not evidence["new_safety_or_provenance_violations"] and evidence["paired_delta"] > 0]
        result = {"status": "baseline_retained", "column": column, "geometry_samples": len(dataset),
                  "offset_m": offset.tolist(), "training": train_evidence}
        if eligible:
            winner = max(eligible, key=lambda name: (train_evidence[name]["paired_delta"], train_evidence[name]["positive_pairs"]))
            compared = _evaluate_candidates(pool, {"baseline": baseline, winner: alternatives[winner]}, root / "development", development)
            evidence = _comparison(compared["baseline"], compared[winner])
            result.update(development=evidence, development_candidate=winner)
            _record_once(root / "development-selection.json", result)
            if (evidence["paired_delta"] > 0 and evidence["strict_success_delta"] >= 0
                    and not evidence["new_safety_or_provenance_violations"]):
                selected = alternatives[winner]
                _record_once(root / "confirmation-lock.json", {"checkpoint_sha256": sha256(selected),
                             "reference_checkpoint_sha256": sha256(baseline), "protocol_sha256": sha256(root / "protocol.json"),
                             "development_sha256": sha256(root / "development-selection.json"), "seeds": schedule["confirmation"],
                             "checkpoint": str(selected)})
                compared = _evaluate_candidates(pool, {"baseline": baseline, "selected": selected}, root / "confirmation",
                                               schedule["confirmation"])
                confirmation = paired_summary(compared["baseline"], compared["selected"])
                floor = min(group["selected"]["mean_requested_score"] for group in confirmation.values())
                lower = min(group["selected"]["requested_mean_bootstrap_95"][0] for group in confirmation.values())
                result.update(status="confirmed", confirmation=confirmation, score=floor, checkpoint=str(selected),
                              milestone_met=floor >= MILESTONE, target_point_estimate=floor > TARGET,
                              target_lower_bound=lower > TARGET)
    write_json(root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a bounded placement offset for the hardest source column")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--placement", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed-block", default=64000, type=int)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 61:
        parser.error("workers must be between 1 and 61")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.placement.resolve(), args.seed_block, args.workers)
    print(f"Release agent finished: {result['status']}; confirmation floor: {result.get('score')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
