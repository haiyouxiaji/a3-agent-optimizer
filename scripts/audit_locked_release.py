from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt.advanced_agent import _jobs, _record_once
from a3opt.cli import _run_lock
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_completion_eval import STRICT_HOLD, audit, replay, summarize
from a3opt.second_batch_agent import frozen_reference


def run(out: Path, branch: Path, source: Path, workers: int) -> dict:
    source_protocol_path = source / "protocol.json"
    protocol = read_json(source_protocol_path)
    lock_path = source / "confirmation-lock.json"
    lock = read_json(lock_path)
    reference, identity = frozen_reference(branch, protocol["schedule"])
    selected = source / "candidates" / f"{lock['candidate']}.npz"
    checks = {"reference": sha256(reference), "selected": sha256(selected)}
    results_path = source / "confirmation/results.json"
    original = read_json(results_path)
    if (lock["protocol_sha256"] != sha256(source_protocol_path) or checks["reference"] != protocol["reference_sha256"]
            or checks["selected"] != lock["checkpoint_sha256"] or identity["environment"] != protocol["environment"]):
        raise ValueError("the strict audit requires the original locked pair and simulator")
    sweeps = {label: [row for row in original[label] if row["suite"] == "random"] for label in checks}
    if any([row["seed"] for row in group] != lock["schedule"]["random"] for group in sweeps.values()):
        raise ValueError("candidate and reference must have all locked random-column receipts")
    column = protocol["source_column"]
    arms = {label: [row for row in group if row["column"] == column] for label, group in sweeps.items()}
    seeds = [row["seed"] for row in arms["reference"]]
    if not seeds or seeds != [row["seed"] for row in arms["selected"]]:
        raise ValueError("the random-column audit requires matched active-column scenes")
    _record_once(out / "protocol.json", {**identity, "scope": "all_active_column_scenes_of_locked_random_confirmation",
                 "source_results_sha256": sha256(results_path), "source_lock_sha256": sha256(lock_path),
                 "checkpoints": checks, "source_column": column, "seeds": seeds, "strict_hold_steps": STRICT_HOLD,
                 "audit_code_sha256": sha256(Path(replay.__code__.co_filename))})
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for label, checkpoint in {"reference": reference, "selected": selected}.items():
            jobs = [(checkpoint, row) for row in arms[label]]
            rows[label] = _jobs(pool, replay, jobs, out / "episodes" / label, lambda job: str(job[1]["seed"]), audit)
    return {"summary": {key: summarize(group) for key, group in rows.items()}, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description="Check stable, released outcomes in all triggered random-column confirmation trials.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    with _run_lock(args.out.resolve()):
        result = run(args.out.resolve(), args.reference.resolve(), args.source.resolve(), args.workers)
        _record_once(args.out / "result.json", result)
    print(result["summary"], flush=True)


if __name__ == "__main__":
    main()
