from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from html import escape
from pathlib import Path

import numpy as np

from a3opt.advanced_agent import AdvancedBudget, SUITES, _audit_evaluation, _rank, summarize
from a3opt.advanced_sim import EvaluatedPolicy, benchmark
from a3opt.backends.advanced_phase import load_advanced_policy
from a3opt.backends.advanced_trajectory import pose_label
from a3opt.refinement_eval import read_json, render_frames, sha256, simulator_identity, write_json


CAMERAS = ("front", "left_wrist", "right_wrist")
PANELS = ((16, 57, 580, 423), (610, 57, 332, 205), (610, 275, 332, 205))
SIZE = (960, 540)


def load_confirmation(root: Path) -> tuple[Path, dict, dict]:
    protocol_path = root / "protocol.json"
    protocol = read_json(protocol_path)
    lock_path = root / "confirmation-lock.json"
    lock = read_json(lock_path)
    result = read_json(root / "result.json")
    checkpoint = Path(lock["selected"]["checkpoint"])
    package = Path(__file__).parent
    if (sha256(protocol_path) != lock["protocol_sha256"] or sha256(checkpoint) != lock["checkpoint_sha256"]
            or simulator_identity() != protocol["environment"] or result["checkpoint"] != str(checkpoint)
            or any(sha256(package / name) != digest for name, digest in protocol["code_sha256"].items())):
        raise ValueError("confirmation protocol, simulator, or checkpoint differs from the frozen run")
    if protocol.get("search_version") == "phase":
        from a3opt.advanced_followup import FollowupBudget

        budget = FollowupBudget(**protocol["budget"])
    else:
        budget = AdvancedBudget(**protocol["budget"])
    budget.validate()
    expected = {suite: list(budget.seeds("confirmation", suite)) for suite in SUITES}
    evaluation = root / "confirmation"
    request_path = evaluation / "request.json"
    request = read_json(request_path)
    if (lock["seeds"] != expected or request != {"checkpoint_sha256": lock["checkpoint_sha256"],
            "split": "confirmation", "seeds": expected}
            or lock_path.stat().st_mtime_ns > request_path.stat().st_mtime_ns):
        raise ValueError("confirmation used a different seed list or checkpoint")
    train = set(budget.seeds("train"))
    development = {seed for suite in SUITES for seed in budget.seeds("development", suite)}
    confirmation = {seed for seeds in expected.values() for seed in seeds}
    if train & development or train & confirmation or development & confirmation or len(confirmation) != 40:
        raise ValueError("train, development, and confirmation splits overlap")
    training = read_json(checkpoint.parent / "training.json")
    manifest_path = checkpoint.parent.parent / "dataset.json"
    manifest = read_json(manifest_path)
    attempts = [row["seed"] for row in manifest["records"]]
    if (training["dataset_sha256"] != sha256(manifest_path)
            or training["calibration_sha256"] != sha256(root / "calibration.npz")
            or len(set(attempts)) != len(attempts) or set(attempts) - train):
        raise ValueError("frozen training receipts do not match the training split")
    with np.load(checkpoint, allow_pickle=False) as archive:
        entries = list(zip(archive["seeds"].tolist(), archive["columns"].tolist(), archive["batches"].tolist(), strict=True))
    retained = [(row["seed"], row["column"], batch) for row in manifest["records"] for batch in row["batches"]]
    with np.load(root / "calibration.npz", allow_pickle=False) as archive:
        if archive["seeds"].tolist() != list(budget.seeds("train")) or entries != retained:
            raise ValueError("checkpoint or calibration is inconsistent with the locked training records")
    development_path = Path(lock["selected"]["development"])
    development_results = read_json(development_path)
    for suite in SUITES:
        if [row["seed"] for row in development_results[suite]["rows"]] != list(budget.seeds("development", suite)):
            raise ValueError("candidate selection used a different development split")
    if list(_rank(development_results)) != lock["selected"]["rank"]:
        raise ValueError("frozen candidate rank differs from its development results")
    results = read_json(evaluation / "results.json")
    if set(results) != set(SUITES) or result["status"] != "confirmed":
        raise ValueError("confirmation is incomplete")
    for suite in SUITES:
        rows = results[suite]["rows"]
        if [row["seed"] for row in rows] != expected[suite]:
            raise ValueError(f"confirmation row order or coverage differs for {suite}")
        for row in rows:
            _audit_evaluation((checkpoint, suite, row["seed"]), row)
            if row != read_json(evaluation / suite / f"{row['seed']}.json"):
                raise ValueError(f"confirmation summary disagrees with raw episode {row['seed']}")
        metrics = json.loads(json.dumps(summarize(rows)))
        if result["confirmation"][suite] != metrics or results[suite]["metrics"] != metrics:
            raise ValueError(f"confirmation aggregate disagrees with its raw episodes for {suite}")
    if Counter(row["column"] for row in results["random"]["rows"]) != {column: 5 for column in range(1, 5)}:
        raise ValueError("random-column confirmation does not have five episodes per column")
    floor = _rank(results)[0]
    if (result["milestone_met"] != (floor >= protocol["milestone_at_least"])
            or result["target_met"] != (floor > protocol["target_above"])):
        raise ValueError("milestone flags disagree with confirmation scores")
    return checkpoint, lock, results


class VideoPolicy(EvaluatedPolicy):
    def __init__(self, checkpoint: Path, encoder: Path, output: Path, suite: str, row: dict):
        super().__init__(load_advanced_policy(checkpoint))
        self.encoder, self.output, self.suite, self.expected = encoder, output, suite, row
        self.writer = None
        self.frames = 0

    def bind_env(self, env) -> None:
        super().bind_env(env)
        self.env = env

    def reset(self, context=None) -> None:
        super().reset(context)
        self.fps = self.env.config.control_hz
        self.writer = subprocess.Popen([
            str(self.encoder), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo",
            "-pixel_format", "bgr24", "-video_size", f"{SIZE[0]}x{SIZE[1]}", "-framerate", str(self.fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "21", "-threads", "1",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.output),
        ], stdin=subprocess.PIPE)
        self.write_frame(0, 0, 80)

    def label(self, text: str, origin: tuple[int, int], scale: float = .59) -> None:
        import cv2

        cv2.putText(self.canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 242, 246), 1, cv2.LINE_AA)

    def write_frame(self, score: int, requested: int, source: int) -> None:
        import cv2

        self.canvas = np.full((SIZE[1], SIZE[0], 3), (31, 23, 19), dtype=np.uint8)
        images = render_frames(self.env, CAMERAS)
        for camera, (left, top, width, height) in zip(CAMERAS, PANELS, strict=True):
            rgb = images[f"observation.images.{camera}"]
            if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.any(rgb):
                raise ValueError(f"invalid simulator RGB frame from {camera} at step {self.step}")
            scale = min(width / rgb.shape[1], height / rgb.shape[0])
            resized_width, resized_height = round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)
            x_offset = left + (width - resized_width) // 2
            y_offset = top + (height - resized_height) // 2
            self.canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = cv2.resize(
                rgb[:, :, ::-1], (resized_width, resized_height), interpolation=cv2.INTER_LINEAR,
            )
            self.label(camera.replace("_", " ").upper(), (left + 8, top + 18), .46)
        seed, column = self.expected["seed"], self.expected["column"]
        profile = self.expected.get("profile", "advanced")
        self.label(f"MuJoCo | 2x5 {profile} | seed {seed} | suite {self.suite} | requested col {column}", (18, 28))
        self.label(f"STEP {self.step:03d}/500    {self.step / self.fps:05.2f}s    TARGET {score}/10    "
                   f"REQUESTED {requested}/10    SOURCE {source}/70", (18, 502))
        self.writer.stdin.write(self.canvas.tobytes())
        self.frames += 1

    def add_frame(self, observation, action, info, **context) -> None:
        super().add_frame(observation, action, info, **context)
        requested = int(np.count_nonzero(self.columns[self.target] == self.expected["column"]))
        self.write_frame(info["cookies_in_target"], requested, info["cookies_in_source"])

    def finish(self, outcome: str) -> int:
        self.label(f"END | {outcome}", (18, 527), .52)
        hold_frames = 2 * self.fps
        ending = self.canvas.tobytes()
        for _ in range(hold_frames):
            self.writer.stdin.write(ending)
        self.writer.stdin.close()
        if self.writer.wait():
            raise RuntimeError(f"could not encode {self.output}")
        return hold_frames

    def close(self) -> None:
        if self.writer is not None and self.writer.poll() is None:
            self.writer.kill()
            self.writer.wait()


def record(checkpoint: Path, checkpoint_digest: str, encoder: Path, output: Path, suite: str, expected: dict) -> dict:
    import cv2

    cv2.setNumThreads(1)
    seed = expected["seed"]
    video = output / suite / f"{seed}.mp4"
    receipt = video.with_suffix(".json")
    if receipt.is_file() and video.is_file():
        cached = read_json(receipt)
        if (cached.get("checkpoint_sha256") == checkpoint_digest and cached["expected"] == expected
                and cached["sha256"] == sha256(video)):
            return cached
    video.parent.mkdir(parents=True, exist_ok=True)
    partial = video.with_name(f".{video.stem}.{os.getpid()}.mp4")
    bench = benchmark(suite)
    env = bench.create_env(render_cameras=False)
    policy = VideoPolicy(checkpoint, encoder, partial, suite, expected)
    try:
        result = bench.run_episode(policy, seed=seed, env=env, diagnostic=policy)
        provenance = np.bincount(policy.columns[policy.target], minlength=5)[1:].tolist()
        actual = {"seed": seed, "column": result.source_column, "score": result.score, "success": result.success,
                  "requested_score": provenance[result.source_column - 1], "by_column": provenance,
                  "steps": result.steps, "first_batch_step": policy.first_batch, "failure_reason": result.failure_reason,
                  "safety_reason": policy.safety, "source_remaining": result.cookies_in_source,
                  "action_sha256": policy.digest.hexdigest(), "color": result.randomization["cookie_color"],
                  "estimated_pose": policy.policy.pose.tolist(), "initial_pose": pose_label(result.randomization).tolist()}
        mismatches = [name for name, value in actual.items() if value != expected[name]]
        if mismatches or policy.frames != result.steps + 1:
            raise RuntimeError(f"confirmation replay differs for {suite}/{seed}: {mismatches}, frames={policy.frames}")
        hold = policy.finish("full success" if result.success else result.failure_reason or "incomplete")
        capture = cv2.VideoCapture(str(partial), cv2.CAP_FFMPEG)
        try:
            decoded = 0
            while capture.grab():
                available, frame = capture.retrieve()
                if not available or frame.shape != (SIZE[1], SIZE[0], 3):
                    raise RuntimeError(f"cannot decode video frame {decoded} for {suite}/{seed}")
                decoded += 1
            if decoded != policy.frames + hold or capture.get(cv2.CAP_PROP_FPS) != policy.fps:
                raise RuntimeError(f"video frame count or rate differs for {suite}/{seed}")
        finally:
            capture.release()
        partial.replace(video)
        row = {"suite": suite, "checkpoint_sha256": checkpoint_digest, "expected": expected,
               "matched_fields": sorted(actual), "frames": decoded,
               "control_frames": policy.frames, "hold_frames": hold, "fps": policy.fps, "sha256": sha256(video)}
        write_json(receipt, row)
        return row
    finally:
        policy.close()
        env.close()
        partial.unlink(missing_ok=True)


def write_index(path: Path, videos: list[dict], results: dict) -> None:
    sections = []
    for suite, title in (("1", "固定第 1 列"), ("random", "随机指定列")):
        items = []
        for video in (item for item in videos if item["suite"] == suite):
            row = video["expected"]
            text = (f"种子 {row['seed']} · 第 {row['column']} 列 · 正确列 {row['requested_score']}/10 · "
                    f"{row['steps']} 步 · {'完整成功' if row['success'] else '未完整成功'}")
            items.append(f'<li><a href="{suite}/{row["seed"]}.mp4" target="rollout">{escape(text)}</a></li>')
        mean = results[suite]["metrics"]["mean_requested_score"]
        sections.append(f"<section><h2>{title} · {mean:.2f}/10</h2><ul>{''.join(items)}</ul></section>")
    first_seed = videos[0]["expected"]["seed"]
    first_video = f'{videos[0]["suite"]}/{first_seed}.mp4'
    path.write_text("""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>A3 最高档确认录像</title><style>
body{font:16px system-ui,sans-serif;color:#142537;background:#f1f5f8;margin:32px auto;padding:0 20px;max-width:1200px}
iframe{width:100%;height:auto;aspect-ratio:16/9;border:0;background:#171c23}main{display:flex;gap:36px;flex-wrap:wrap}
ul{padding-left:20px}li{margin:8px 0}a{color:#175b96}h2{font-size:20px}
</style><h1>A3 2×5 · advanced 确认录像</h1>
<p>点击种子切换录像。每段含重置、每个控制步和终局停留。三路画面来自同一局仿真。</p>
""" + f'<iframe title="MuJoCo 确认局录像" name="rollout" src="{first_video}" allowfullscreen></iframe><main>'
                    + "".join(sections) + "</main></html>\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay locked advanced confirmation and verify full simulator videos")
    parser.add_argument("run", type=Path)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.run.resolve()
    checkpoint, lock, results = load_confirmation(root)
    print("Verified checkpoint, protocol, splits, raw episode receipts, and score aggregation", flush=True)
    output = root / "videos"
    encoder = args.ffmpeg.resolve(strict=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(record, checkpoint, lock["checkpoint_sha256"], encoder, output, suite, row)
                for suite in reversed(SUITES) for row in results[suite]["rows"]]
        videos = []
        for job in jobs:
            video = job.result()
            print(f"Replay {video['suite']}/{video['expected']['seed']}: {video['frames']} verified frames", flush=True)
            videos.append(video)
    audit = {"confirmation_lock_sha256": sha256(root / "confirmation-lock.json"),
             "checkpoint_sha256": lock["checkpoint_sha256"], "encoder_sha256": sha256(encoder),
             "splits_disjoint": True, "rows": videos}
    write_json(output / "replay-audit.json", audit)
    write_index(output / "index.html", videos, results)


if __name__ == "__main__":
    main()
