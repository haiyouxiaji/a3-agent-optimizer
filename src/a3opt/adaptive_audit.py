from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt.adaptive_agent import paired_summary
from a3opt.adaptive_sim import AdaptiveTrace, adaptive_episode, check_adaptive_row, phase_model
from a3opt.advanced_audit import SIZE, VideoPolicy
from a3opt.advanced_agent import _jobs, _record_once, summarize
from a3opt.backends.adaptive_phase import load_checkpoint
from a3opt.backends.release_residual import load_checkpoint as load_release_checkpoint, nominal_model_path
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import run_episode
from a3opt.refinement_eval import read_json, sha256, simulator_identity, write_json


def verify_training(root: Path, protocol: dict, result: dict) -> dict:
    schedule = protocol["schedule"]
    is_release = "kinematics_sha256" in protocol
    is_branch = protocol.get("method") == "dominant_branch"
    if is_release:
        training_groups = [*schedule["calibration"].values(), *schedule["training"].values()]
    elif is_branch:
        training_groups = list(schedule["training"].values())
    else:
        training_groups = [schedule[name] for name in ("scout", "focus", "focus_extra")]
    splits = [*map(set, training_groups),
              *map(set, schedule["development"].values()), *map(set, schedule["confirmation"].values())]
    if len(set.union(*splits)) != sum(map(len, splits)):
        raise ValueError("overlap between scheduled training, development and confirmation scenes")
    if is_branch:
        from a3opt.branch_agent import DEMO_FIELDS, dominant_branch

        manifest = root / "branch-dataset.json"
        locked = read_json(root / "confirmation-lock.json")
        baseline = phase_model(root / "candidates" / "baseline" / "policy.npz")
        filtered, branch = dominant_branch(baseline, protocol["column"])
        selected = phase_model(Path(locked["checkpoint"]))
        if (read_json(manifest) != branch or sha256(manifest) != result["branch_training_digest"]
                or any(not (filtered[key].shape == selected[key].shape and (filtered[key] == selected[key]).all())
                       for key in DEMO_FIELDS)
                or (set(baseline["seeds"].tolist()) & set.union(*splits))):
            raise ValueError("branch checkpoint ancestry differs from the sealed training demonstrations")
        return {"dataset_sha256": sha256(manifest), "samples_verified": branch["training_rows"],
                "removed_seeds": branch["removed_seeds"]}
    if is_release:
        manifest = root / "release-dataset.json"
        samples = read_json(manifest)["samples"]
        allowed = set.union(*map(set, schedule["calibration"].values()))
        if not len(samples) == len({sample["seed"] for sample in samples}) == result["geometry_samples"]:
            raise ValueError("release dataset count does not match its run")
        for sample in samples:
            seed = sample["seed"]
            if seed not in allowed:
                raise ValueError(f"release label is outside the calibration split: {seed}")
            row = read_json(root / "calibration" / "coherent" / str(protocol["column"]) / f"{seed}.json")
            label = row["batches"][0]["release_label"]
            if sample != {"seed": seed, "applied_action_sha256": row["applied_action_sha256"], **label}:
                raise ValueError(f"release label differs from its simulator receipt: {seed}")
        return {"dataset_sha256": sha256(manifest), "samples_verified": len(samples)}
    training = set.union(*map(set, training_groups))
    expected = {}
    for wave in ("scout", "focus"):
        request = read_json(root / wave / "request.json")
        if request["reference_sha256"] != protocol["reference_sha256"]:
            raise ValueError(f"{wave} used a different reference checkpoint")
        for recipe, jobs in request["samples"].items():
            for suite, seed in jobs:
                receipt = f"{wave}/{recipe}/{seed}.json"
                if recipe not in protocol["recipes"] or seed not in training or receipt in expected:
                    raise ValueError(f"unexpected training receipt {receipt}")
                expected[receipt] = (suite, seed, recipe)
    manifest = root / "grip-dataset.json"
    samples = read_json(manifest)["samples"]
    identities = {(sample["row"], sample["seed"]) for sample in samples}
    if len(samples) != len(identities) or len(samples) != result["grip_samples"] or len(expected) != result["training_episodes"]:
        raise ValueError("the reported dataset size disagrees with the training receipts")
    for sample in samples:
        relative = sample["row"]
        if expected.get(relative) != (sample["suite"], sample["seed"], sample["recipe"]):
            raise ValueError(f"unexpected dataset provenance {relative}")
        row = read_json(root / relative)
        event = row["batches"][0]
        if (row["action_sha256"] != sample["action_sha256"] or event["grip_features"] != sample["features"]
                or sample["complete_grasp"] != (event["lifted_requested"] == 5 and event["lifted_other"] == 0)
                or row["column"] != sample["column"] or row["safety_reason"]):
            raise ValueError(f"dataset label or features differ from the simulator receipt {relative}")
    return {"dataset_sha256": sha256(manifest), "samples_verified": len(samples),
            "scheduled_episodes": len(expected), "sampled_seeds": len({sample["seed"] for sample in samples})}


def verify_confirmation(root: Path, *, allow_source_drift: bool = False) -> tuple[dict, dict, dict, list[str]]:
    protocol_path = root / "protocol.json"
    lock_path = root / "confirmation-lock.json"
    selection_path = root / "development-selection.json"
    result = read_json(root / "result.json")
    protocol, lock = read_json(protocol_path), read_json(lock_path)
    package = Path(__file__).parent
    schedule = protocol["schedule"]
    drift = [path for path, digest in protocol["code_sha256"].items() if sha256(package / path) != digest]
    is_release = "kinematics_sha256" in protocol
    is_branch = protocol.get("method") == "dominant_branch"
    if (result["status"] != "confirmed" or sha256(protocol_path) != lock["protocol_sha256"]
            or sha256(selection_path) != lock["development_sha256"] or schedule["confirmation"] != lock["seeds"]
            or protocol["environment"] != simulator_identity()
            or (drift and not allow_source_drift)
            or (is_release and protocol["kinematics_sha256"] != sha256(nominal_model_path()))):
        raise ValueError("confirmation no longer matches its frozen code, selection or protocol")
    selected = Path(lock["checkpoint"]) if is_release or is_branch else root / "selected" / "policy.npz"
    if Path(result["checkpoint"]) != selected:
        raise ValueError("result checkpoint differs from confirmation lock")
    checkpoints = {"baseline": root / "candidates" / "baseline" / "policy.npz", "selected": selected}
    locked_digests = {"baseline": lock["reference_checkpoint_sha256"], "selected": lock["checkpoint_sha256"]}
    confirmation = {}
    expected = [(suite, seed) for suite, seeds in lock["seeds"].items() for seed in seeds]
    for role, checkpoint in checkpoints.items():
        directory = root / "confirmation" / role
        request_path = directory / "request.json"
        request = {"checkpoint_sha256": locked_digests[role], "seeds": lock["seeds"], "profile": protocol["profile"]}
        if is_release:
            request["mode"] = "full"
        if (sha256(checkpoint) != locked_digests[role]
                or read_json(request_path) != request
                or lock_path.stat().st_mtime_ns > request_path.stat().st_mtime_ns):
            raise ValueError(f"confirmation request differs from the frozen recipe for {role}")
        rows = read_json(directory / "results.json")["rows"]
        if [(row["suite"], row["seed"]) for row in rows] != expected:
            raise ValueError(f"missing or out-of-order confirmation scenes for {role}")
        if (Counter(row["column"] for row in rows if row["suite"] == "1") != {1: 20}
                or Counter(row["column"] for row in rows if row["suite"] == "random") != dict.fromkeys(range(1, 5), 5)):
            raise ValueError(f"invalid confirmation balance for {role}")
        for row in rows:
            check_adaptive_row((checkpoint, protocol["profile"], row["suite"], row["seed"]), row)
            if row != read_json(directory / row["suite"] / f"{row['seed']}.json"):
                raise ValueError(f"summary and episode evidence disagree for {role} seed {row['seed']}")
        confirmation[role] = rows
    if not _matches(result["confirmation"], paired_summary(confirmation["baseline"], confirmation["selected"])):
        raise ValueError("reported confirmation metrics differ from the audited rows")
    return protocol, checkpoints, confirmation, drift


def _matches(expected, actual) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _matches(value, actual[key])
                                               for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            _matches(left, right) for left, right in zip(expected, actual, strict=True))
    return expected == actual


class AdaptiveVideoTrace(VideoPolicy, AdaptiveTrace):
    def __init__(self, checkpoint: Path, encoder: Path, output: Path, row: dict, is_release: bool):
        policy = load_release_checkpoint(checkpoint) if is_release else load_checkpoint(checkpoint)
        AdaptiveTrace.__init__(self, policy, first_batch_only=False)
        self.encoder, self.output, self.suite, self.expected = encoder, output, row["suite"], row
        self.writer = None
        self.frames = 0


def record(checkpoint: Path, encoder: Path, output: Path, row: dict, is_release: bool) -> dict:
    import cv2

    cv2.setNumThreads(1)
    temporary = output.with_suffix(".partial.mp4")
    trace = AdaptiveVideoTrace(checkpoint, encoder, temporary, row, is_release)
    try:
        actual = run_episode(checkpoint, row["profile"], row["suite"], row["seed"], trace=trace)
        actual.update(batches=trace.events, retries=trace.policy.retries,
                      first_requested_five=trace.first_requested_five, checkpoint_sha256=sha256(checkpoint))
        fields = sorted(set(row) - {"wall_seconds"})
        mismatches = [field for field in fields if field not in actual or not _matches(row[field], actual[field])]
        if mismatches or trace.frames != actual["steps"] + 1:
            raise ValueError(f"replay differs from {output.name}: {mismatches}")
        hold = trace.finish("full success" if actual["success"] else actual["failure_reason"] or "incomplete")
        capture = cv2.VideoCapture(str(temporary), cv2.CAP_FFMPEG)
        try:
            decoded = 0
            while capture.grab():
                available, frame = capture.retrieve()
                if not available or frame.shape != (SIZE[1], SIZE[0], 3):
                    raise ValueError(f"invalid decoded frame {decoded} in {output.name}")
                decoded += 1
            if decoded != trace.frames + hold or capture.get(cv2.CAP_PROP_FPS) != trace.fps:
                raise ValueError(f"incorrect decoded frame count or frame rate for {output.name}")
        finally:
            capture.release()
        temporary.replace(output)
        return {"file": output.name, "sha256": sha256(output), "matched_fields": fields, "frames": decoded,
                "fps": trace.fps, "episode": row}
    finally:
        trace.close()
        temporary.unlink(missing_ok=True)


def grade_grid(root: Path, protocol: dict, checkpoints: dict, confirmation: dict, workers: int) -> dict:
    if "kinematics_sha256" in protocol:
        raise ValueError("use the upstream release policy plugin to evaluate other profiles")
    column = protocol["column"] if protocol.get("method") == "dominant_branch" else 1
    suite = "random" if protocol.get("method") == "dominant_branch" else "1"
    seeds = [row["seed"] for row in confirmation["baseline"] if (row["suite"], row["column"]) == (suite, column)]
    folder = root / "grade-grid"
    hashes = {role: sha256(path) for role, path in checkpoints.items()}
    _record_once(folder / "request.json", {"suite": suite, "column": column, "seeds": seeds, "checkpoints": hashes,
                                           "rules": protocol["rules"], "confirmation_sha256": sha256(root / "confirmation-lock.json")})
    jobs = [(path, profile, suite, seed) for profile in ("basic", "medium") for path in checkpoints.values() for seed in seeds]
    roles = {path: role for role, path in checkpoints.items()}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        sampled = _jobs(pool, adaptive_episode, jobs, folder, lambda job: f"{job[1]}/{roles[job[0]]}/{job[3]}", check_adaptive_row)
    output = {}
    for profile in ("basic", "medium", "advanced"):
        rows = {role: [row for row in (confirmation[role] if profile == "advanced" else sampled)
                       if row["profile"] == profile and row["checkpoint_sha256"] == digest and row["seed"] in seeds]
                for role, digest in hashes.items()}
        output[profile] = {role: summarize(group) for role, group in rows.items()}
    write_json(folder / "results.json", {"column": column, "suite": suite, "seeds": seeds, "cells": output})
    print(f"Verified paired difficulty results for column {column}, {len(seeds)} seeds per profile", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit confirmed adaptive scores and verified replays")
    parser.add_argument("run", type=Path)
    parser.add_argument("--grid-workers", type=int)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--video-seed", type=int)
    parser.add_argument("--replay-compatible", action="store_true", help="replay archived episode fields with current code")
    args = parser.parse_args()
    if bool(args.ffmpeg) != (args.video_seed is not None):
        parser.error("--ffmpeg and --video-seed must be provided together")
    if args.replay_compatible and not args.ffmpeg:
        parser.error("archived-source verification requires a full replay")
    if args.grid_workers is not None and not 1 <= args.grid_workers <= 61:
        parser.error("grid workers must be between 1 and 61")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    root = args.run.resolve()
    with _run_lock(root):
        protocol, checkpoints, confirmation, drift = verify_confirmation(root, allow_source_drift=args.replay_compatible)
        training = verify_training(root, protocol, read_json(root / "result.json"))
        audit = {"confirmation_lock_sha256": sha256(root / "confirmation-lock.json"), "counts": {},
                 "results_sha256": {}, "training": training, "source_drift": drift}
        for role, rows in confirmation.items():
            audit["counts"][role] = len(rows)
            audit["results_sha256"][role] = sha256(root / "confirmation" / role / "results.json")
        print(f"Verified {sum(audit['counts'].values())} confirmation rows", flush=True)
        if args.ffmpeg:
            output = root / "videos"
            output.mkdir(exist_ok=True)
            encoder = args.ffmpeg.resolve(strict=True)
            jobs = []
            for role, checkpoint in checkpoints.items():
                row = next((item for item in confirmation[role] if item["seed"] == args.video_seed), None)
                if row is None:
                    raise ValueError("video seed must be in the locked confirmation split")
                jobs.append((checkpoint, output / f"{args.video_seed}-{role}.mp4", row))
            with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
                futures = [pool.submit(record, checkpoint, encoder, video, row, "kinematics_sha256" in protocol)
                           for checkpoint, video, row in jobs]
                videos = [future.result() for future in futures]
            write_json(output / f"{args.video_seed}-audit.json", {**audit, "encoder_sha256": sha256(encoder), "videos": videos})
            print(f"Verified both full replays for seed {args.video_seed}", flush=True)
        write_json(root / "confirmation-audit.json", audit)
        if args.grid_workers is not None:
            grade_grid(root, protocol, checkpoints, confirmation, args.grid_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
