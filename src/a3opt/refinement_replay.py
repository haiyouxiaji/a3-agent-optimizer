from pathlib import Path

from a3opt.refinement_eval import PolicySpec, evaluate, read_json, sha256, write_json

REPLAY_FIELDS = ("score", "success", "steps", "action_sha256", "first_batch_step", "target_peak", "layout_features")


def replay_candidate(lock_path: Path, workers: int, initial: dict) -> list[dict]:
    lock = read_json(lock_path)
    entry = lock["policies"]["candidate"]
    candidate = PolicySpec(entry["kind"], entry["checkpoint"], entry["horizon"])
    if candidate.identity() != entry:
        raise ValueError("candidate changed before replay")
    directory = lock_path.parent / "confirmation-eager-replay"
    replay = evaluate({"candidate": candidate}, tuple(lock["seeds"]), directory, workers, eager=True)["candidate"]
    rows = [
        {"seed": after["seed"], "matches": {field: before[field] == after[field] for field in REPLAY_FIELDS}}
        for before, after in zip(initial["rows"], replay["rows"], strict=True)
    ]
    write_json(directory / "parity.json", {"lock_sha256": sha256(lock_path), "rows": rows})
    mismatches = [row["seed"] for row in rows if not all(row["matches"].values())]
    if mismatches:
        raise RuntimeError(f"eager replay differs on seeds {mismatches}")
    return rows
