from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt import placement_sim, placement_train
from a3opt.advanced_agent import _record_once
from a3opt.cli import _run_lock
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES
from a3opt.second_batch_agent import frozen_reference


def reference_setup(reference: Path, timing_run: Path, splits: dict) -> tuple[Path, tuple[int, float], dict]:
    checkpoint, identity = frozen_reference(reference, splits)
    protocol_path, lock_path = timing_run / "protocol.json", timing_run / "confirmation-lock.json"
    protocol, lock = read_json(protocol_path), read_json(lock_path)
    if (protocol["environment"] != identity["environment"] or protocol["strict_rules"] != STRICT_RULES
            or lock["protocol_sha256"] != sha256(protocol_path) or lock["reference_sha256"] != sha256(checkpoint)):
        raise ValueError("placement Agent requires the verified timing-locked reference")
    previous_seeds = set(protocol["schedule"]["training"]) | set(protocol["schedule"]["confirmation"]["focused"])
    for seeds in (splits["scout"]["random"], splits["training"], splits["development"], *splits["confirmation"].values()):
        if previous_seeds.intersection(seeds):
            raise ValueError("placement Agent requires seeds disjoint from timing search and confirmation")
    timing = (lock["column"], lock["scale"])
    return checkpoint, timing, {**identity, "timing": list(timing), "timing_lock_sha256": sha256(lock_path),
                               "strict_rules": STRICT_RULES, "placement_sim_sha256": sha256(Path(placement_sim.__file__))}


def scout(root: Path, reference: Path, timing_run: Path, archive: Path, seed_start: int, workers: int) -> None:
    archive_path = archive / "result.json"
    archive_rows = read_json(archive_path)["rows"]
    seeds = list(range(seed_start, seed_start + 16))
    if seed_start % 4:
        raise ValueError("random scout requires an aligned four-column block")
    splits = {"scout": {"random": seeds}, "training": [], "development": [], "confirmation": {}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    previous = archive_rows[str(timing[1])]
    archived = {row["seed"]: row for row in previous if row["requested_score"] < 10}
    if not archived or set(seeds) & {row["seed"] for row in previous}:
        raise ValueError("archived scenes must be isolated from collection")
    _record_once(root / "scout-lock.json", {**identity, "archive_results_sha256": sha256(archive_path),
                 "archive_diagnosis_seeds": list(archived), "scout_seeds": seeds})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        replayed = placement_sim.evaluate(pool, root / "archive-diagnosis", checkpoint, timing, "random", list(archived))
        keys = ("source_remaining", "target_cookie_ids", "action_sha256", "applied_action_sha256", "terminal", "steps")
        for row in replayed:
            if any(row[key] != archived[row["seed"]][key] for key in keys):
                raise ValueError("archived failure replay differs from its locked receipt")
        placement_sim.evaluate(pool, root / "scout", checkpoint, timing, "random", seeds)


def prioritize_failures(rows: list[dict], budget: int = 8) -> list[int]:
    def lost_after_transfer(row: dict) -> int:
        losses = 0
        for batch, event in enumerate(row["batches"]):
            if event.get("retained_through_transfer") == 5:
                terminal = row["physical_terminal"]["batches"][batch]
                losses += 5 - sum(cookie["scored"] for cookie in terminal)
        return losses

    groups = {}
    for row in sorted(rows, key=lambda row: (-lost_after_transfer(row), row["seed"])):
        if row["requested_score"] < 10 and lost_after_transfer(row) > 0 and not row["safety_reason"]:
            groups.setdefault(row["column"], []).append(row["seed"])
    return [seeds[index] for index in range(budget) for seeds in groups.values() if index < len(seeds)][:budget]


def focus_training(root: Path, reference: Path, timing_run: Path, scout_run: Path, workers: int) -> None:
    source = scout_run / "scout" / "results.json"
    rows = read_json(source)["rows"]
    scout_lock = read_json(scout_run / "scout-lock.json")
    seeds = prioritize_failures(rows)
    splits = {"scout": {"random": []}, "training": seeds, "development": [], "confirmation": {}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    for row in rows:
        placement_sim.audit((checkpoint, *timing, "random", row["seed"], (), True), row)
    if ([row["seed"] for row in rows] != scout_lock["scout_seeds"] or scout_lock["reference_sha256"] != sha256(checkpoint)
            or scout_lock["placement_sim_sha256"] != identity["placement_sim_sha256"]):
        raise ValueError("focus sampling requires the locked scout receipts")
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "source_sha256": sha256(source),
                 "source_role": "training-scout", "acquisition_code_sha256": sha256(Path(placement_train.__file__)),
                 "agent_code_sha256": sha256(Path(__file__))})
    if not seeds:
        _record_once(root / "result.json", {"status": "no_observed_placement_shortfalls", "seeds": seeds})
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        placement_train.acquire(pool, root / "training", checkpoint, timing, seeds)
    _record_once(root / "result.json", {"status": "active_collection_complete", "seeds": seeds,
                 "dataset_sha256": sha256(root / "training/dataset.json")})


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect stage-labeled 2x5 placement episodes with frozen timing.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--archive", type=Path)
    sources.add_argument("--focus-scout", type=Path)
    parser.add_argument("--seed-start", type=int, default=110000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with _run_lock(args.out.resolve()):
        if args.focus_scout:
            focus_training(args.out.resolve(), args.reference.resolve(), args.timing.resolve(), args.focus_scout.resolve(), args.workers)
        else:
            scout(args.out.resolve(), args.reference.resolve(), args.timing.resolve(), args.archive.resolve(), args.seed_start, args.workers)


if __name__ == "__main__":
    main()
