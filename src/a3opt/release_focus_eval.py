from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt.advanced_agent import _record_once
from a3opt.cli import _run_lock
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_feedback_agent import episodes
from a3opt.release_outcome_agent import confirmation_stats
from a3opt.second_batch_agent import frozen_reference, seed_schedule
from a3opt.slot_alignment_agent import _compare


def run(out: Path, source: Path, reference_run: Path, seed_block: int, workers: int) -> dict:
    protocol_path = source / "protocol.json"
    parent = read_json(protocol_path)
    lock_path = source / "confirmation-lock.json"
    lock = read_json(lock_path)
    checkpoint = source / "candidates" / f"{lock['candidate']}.npz"
    if (lock["protocol_sha256"] != sha256(protocol_path) or lock["checkpoint_sha256"] != sha256(checkpoint)
            or lock["development_sha256"] != sha256(source / "development-decision.json")):
        raise ValueError("focused confirmation requires the original locked release candidate")
    schedule = seed_schedule(seed_block)
    seeds = schedule["confirmation"]["1"]
    used = parent["schedule"]
    if set(seeds).intersection(used["training"] + used["development"] + sum(used["confirmation"].values(), [])):
        raise ValueError("focused confirmation requires fresh seeds")
    reference, identity = frozen_reference(reference_run, schedule)
    if any(identity[key] != parent[key] for key in ("reference_sha256", "environment", "code_sha256")):
        raise ValueError("baseline or simulator changed after the release protocol was locked")
    package = Path(__file__).parent
    if sha256(package / "release_feedback_agent.py") != parent["code"]:
        raise ValueError("release policy changed after it was locked")
    column = parent["source_column"]
    protocol = {**identity, "source_column": column, "seeds": seeds, "parent_confirmation_lock_sha256": sha256(lock_path),
                "checkpoint_sha256": sha256(checkpoint), "focus_eval_sha256": sha256(Path(__file__))}
    _record_once(out / "protocol.json", protocol)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = episodes(pool, out / "paired", {"reference": reference, "selected": checkpoint}, {str(column): seeds})
    return {"candidate": lock["candidate"], "column": column, "comparison": _compare(rows, column),
            "statistics": confirmation_stats(rows["reference"], rows["selected"]),
            "major_losses": [{"seed": base["seed"], "reference": base["requested_score"], "candidate": trial["requested_score"]}
                             for base, trial in zip(rows["reference"], rows["selected"], strict=True)
                             if base["requested_score"] - trial["requested_score"] >= 4],
            "protocol_sha256": sha256(out / "protocol.json"), "results_sha256": sha256(out / "paired/results.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently confirm the frozen release candidate on its active source column.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=87000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with _run_lock(args.out.resolve()):
        result = run(args.out.resolve(), args.source.resolve(), args.reference.resolve(), args.seed_block, args.workers)
        _record_once(args.out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
