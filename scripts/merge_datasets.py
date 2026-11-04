from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("front", "left_wrist", "right_wrist", "state", "action", "seeds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge multi-camera expert datasets")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parts = {key: [] for key in KEYS}
    reports: list[dict] = []
    for name in args.inputs:
        path = Path(name)
        with np.load(path) as data:
            for key in KEYS:
                parts[key].append(np.asarray(data[key]))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.is_file():
            reports.extend(json.loads(meta_path.read_text(encoding="utf-8")).get("episode_reports", []))
    merged = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    seeds = sorted({int(seed) for seed in merged["seeds"].tolist()})
    meta = {
        "format": "a3opt-expert-multicam-v1",
        "dataset_path": str(output.resolve()),
        "inputs": [str(Path(name).resolve()) for name in args.inputs],
        "frames": int(merged["front"].shape[0]),
        "episodes": len(seeds),
        "seeds": seeds,
        "image_size": int(merged["front"].shape[1]),
        "cameras": ["front", "left_wrist", "right_wrist"],
        "episode_reports": reports,
    }
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"merged {len(args.inputs)} datasets -> {output} frames={meta['frames']} episodes={meta['episodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
