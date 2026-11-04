from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SIM_SRC = ROOT.parent / "a3_dual_arm_sim" / "src"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))

FRONT = "observation.images.front"
LEFT = "observation.images.left_wrist"
RIGHT = "observation.images.right_wrist"
STATE = "observation.state"


def downsample(image: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(image)
    height, width = int(array.shape[0]), int(array.shape[1])
    if height == size and width == size:
        return np.ascontiguousarray(array)
    factor_y = height // size
    factor_x = width // size
    if factor_y * size == height and factor_x * size == width:
        blocks = array.reshape(size, factor_y, size, factor_x, array.shape[2])
        return blocks.mean(axis=(1, 3)).astype(np.uint8)
    rows = np.minimum((np.arange(size) * height) // size, height - 1)
    cols = np.minimum((np.arange(size) * width) // size, width - 1)
    return np.ascontiguousarray(array[rows][:, cols])


class MultiCameraRecorder:
    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)
        self.front: list[np.ndarray] = []
        self.left: list[np.ndarray] = []
        self.right: list[np.ndarray] = []
        self.state: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.seeds: list[int] = []
        self.episodes: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._frames = 0

    def start_episode(self, context: Any, controller_type: str = "") -> None:
        self._current = {
            "seed": int(context.seed),
            "task": str(context.task),
            "action_mode": str(context.action_mode),
            "controller": str(controller_type),
        }
        self._frames = 0

    def set_episode_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        return

    def add_frame(self, observation: Mapping[str, Any], action: np.ndarray) -> None:
        size = self.image_size
        self.front.append(downsample(observation[FRONT], size))
        self.left.append(downsample(observation[LEFT], size))
        self.right.append(downsample(observation[RIGHT], size))
        self.state.append(np.asarray(observation[STATE], dtype=np.float32).copy())
        self.action.append(np.asarray(action, dtype=np.float32).copy())
        self.seeds.append(int(self._current["seed"]))
        self._frames += 1

    def finish_episode(self, success: bool | None = None) -> None:
        if self._current is None:
            return
        episode = dict(self._current)
        episode["success"] = None if success is None else bool(success)
        episode["frames"] = self._frames
        self.episodes.append(episode)
        self._current = None

    def discard_episode(self) -> None:
        if self._current is None:
            return
        if self._frames:
            for store in (self.front, self.left, self.right, self.state, self.action, self.seeds):
                del store[-self._frames:]
        self._current = None

    def close(self) -> None:
        if self._current is not None:
            self.discard_episode()

    def export(self) -> dict[str, np.ndarray]:
        return {
            "front": np.stack(self.front, axis=0).astype(np.uint8),
            "left_wrist": np.stack(self.left, axis=0).astype(np.uint8),
            "right_wrist": np.stack(self.right, axis=0).astype(np.uint8),
            "state": np.stack(self.state, axis=0).astype(np.float32),
            "action": np.stack(self.action, axis=0).astype(np.float32),
            "seeds": np.asarray(self.seeds, dtype=np.int32),
        }


def parse_seeds(text: str) -> list[int]:
    seeds = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record multi-camera A3 expert demonstrations")
    parser.add_argument("--dataset", default=str(ROOT / "artifacts" / "datasets" / "expert_2x5_multicam_20ep.npz"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--task", default="2x5")
    parser.add_argument("--profile", default="basic")
    parser.add_argument("--source-column", default="1")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    args = build_parser().parse_args()
    dataset_path = Path(args.dataset)
    if dataset_path.is_file() and not args.force:
        print(f"dataset exists: {dataset_path}; pass --force to overwrite", flush=True)
        return 1
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(args.seed_start) + index for index in range(int(args.episodes))]
    recorder = MultiCameraRecorder(int(args.image_size))
    benchmark = make_cookie_benchmark(
        task=args.task,
        profile=args.profile,
        source_column=str(args.source_column),
        max_steps=int(args.max_steps),
    )
    env = benchmark.create_env(render_cameras=True)
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index, seed in enumerate(seeds):
            result = benchmark.run_episode(
                policy="same_column",
                seed=seed,
                episode_idx=index + 1,
                recorder=recorder,
                env=env,
            )
            report = {
                "seed": seed,
                "score": float(result.score),
                "success": bool(result.success),
                "steps": int(result.steps),
                "wall_seconds": float(result.wall_seconds),
                "failure_reason": result.failure_reason,
            }
            reports.append(report)
            print(
                f"[{index + 1:02d}/{len(seeds):02d}] seed={seed} score={report['score']:.0f} "
                f"success={report['success']} steps={report['steps']} wall={report['wall_seconds']:.1f}s",
                flush=True,
            )
    finally:
        env.close()
        recorder.close()
    arrays = recorder.export()
    np.savez_compressed(dataset_path, **arrays)
    meta = {
        "format": "a3opt-expert-multicam-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_path": str(dataset_path.resolve()),
        "task": args.task,
        "profile": args.profile,
        "source_column": str(args.source_column),
        "image_size": int(args.image_size),
        "frames": int(arrays["front"].shape[0]),
        "episodes": len({int(seed) for seed in arrays["seeds"].tolist()}),
        "seeds": seeds,
        "cameras": ["front", "left_wrist", "right_wrist"],
        "episode_reports": reports,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    dataset_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {dataset_path} frames={meta['frames']} episodes={meta['episodes']} wall={meta['wall_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
