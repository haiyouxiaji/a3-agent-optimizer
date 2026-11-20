from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import _record_once
from a3opt.advanced_sim import probe_training_scenes
from a3opt.backends.adaptive_phase import AdaptivePhasePolicy, load_checkpoint
from a3opt.backends.advanced_trajectory import predict_pose
from a3opt.backends.slot_alignment import from_adaptive, predict_shift, release_features
from a3opt.cli import _run_lock
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.slot_alignment_agent import _compare
from a3opt.slot_guard_agent import GuardedSlotPolicy, _episodes


PROBE_COUNT = 64
PAIRED_COUNT = 12


def _probe(checkpoint: Path, seeds: tuple[int, ...]) -> list[dict]:
    policy = GuardedSlotPolicy(from_adaptive(load_checkpoint(checkpoint)))
    rows = []
    for seed, outline, silhouette, _, _ in probe_training_scenes(seeds):
        policy.pose = predict_pose(policy.model, outline, silhouette)
        actions = AdaptivePhasePolicy._phase_actions(policy, policy.column, policy.recipes[policy.column - 1])
        magnitude = abs(float(predict_shift(policy.parameters, release_features(policy.pose, actions[0]))))
        rows.append({"seed": seed, "magnitude_m": magnitude, "active": bool(policy.guard[0] <= magnitude <= policy.guard[1]),
                     "estimated_pose": policy.pose.tolist()})
    return rows


def run(root: Path, guard: Path, block: int, workers: int) -> dict:
    if block < 0:
        raise ValueError("seed block must be nonnegative")
    guard_lock_path = guard / "confirmation-lock.json"
    lock = read_json(guard_lock_path)
    guard_protocol_path = guard / "protocol.json"
    guard_protocol = read_json(guard_protocol_path)
    prior = read_json(guard / "result.json")
    reference = Path(prior["selected_checkpoint"])
    if prior["status"] == "promoted":
        reference = Path(read_json(guard / "development-selection.json")["selected_checkpoint"])
    checkpoint = guard / "candidate" / "policy.npz"
    if (sha256(reference) != lock["reference_checkpoint_sha256"] or sha256(checkpoint) != lock["checkpoint_sha256"]
            or sha256(guard_protocol_path) != lock["protocol_sha256"]):
        raise ValueError("coverage check requires the original locked policy pair")
    schedule = guard_protocol["schedule"]
    previous = set(schedule["development"] + sum(schedule["confirmation"].values(), []))
    seeds = list(range(block, block + PROBE_COUNT))
    if previous.intersection(seeds):
        raise ValueError("coverage check requires new seeds")
    package = Path(__file__).parent
    column = guard_protocol["source_column"]
    _record_once(root / "protocol.json", {"profile": "advanced", "rules": guard_protocol["rules"], "source_column": column,
                 "probe_seeds": seeds, "paired_limit": PAIRED_COUNT, "guards_sha256": sha256(checkpoint),
                 "reference_sha256": sha256(reference), "guard_lock_sha256": sha256(guard_lock_path),
                 "environment": simulator_identity(), "coverage_code_sha256": sha256(package / "slot_guard_coverage.py"),
                 "selection": "first qualifying seeds by number using reset observation only"})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        trace_path = root / "reset-probes.json"
        if trace_path.exists():
            probes = read_json(trace_path)["rows"]
        else:
            jobs = [pool.submit(_probe, checkpoint, tuple(seeds[start:start + 8])) for start in range(0, len(seeds), 8)]
            probes = [row for job in jobs for row in job.result()]
            _record_once(trace_path, {"rows": probes})
        if [row["seed"] for row in probes] != seeds:
            raise ValueError("coverage reset probes do not match the frozen schedule")
        active = [row for row in probes if row["active"]]
        selected = active[:PAIRED_COUNT]
        labels = {row["seed"]: row["estimated_pose"] for row in selected}
        _record_once(root / "paired-lock.json", {"reset_probes_sha256": sha256(trace_path),
                     "protocol_sha256": sha256(root / "protocol.json"), "seeds": list(labels)})
        result = {"probed": len(probes), "qualified": len(active), "paired": len(selected)}
        if selected:
            print(f"Agent selected {len(selected)} active scenarios from {len(probes)} resets", flush=True)
            rows = _episodes(pool, {"reference": reference, "selected": checkpoint}, root / "paired", {str(column): list(labels)})
            for row in rows["reference"]:
                if not np.array_equal(row["estimated_pose"], labels[row["seed"]]):
                    raise ValueError("full episode differs from the reset-only coverage probe")
            result.update(comparison=_compare(rows, column), results_sha256=sha256(root / "paired" / "results.json"),
                          different_action_pairs=sum(base["action_sha256"] != trial["action_sha256"]
                                                     for base, trial in zip(rows["reference"], rows["selected"], strict=True)))
    _record_once(root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a frozen guard on reset-only scenarios where it activates")
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=76000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 61:
        parser.error("workers must be between 1 and 61")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.guard.resolve(), args.seed_block, args.workers)
    print(f"Coverage check complete: {result['paired']} paired scenarios", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
