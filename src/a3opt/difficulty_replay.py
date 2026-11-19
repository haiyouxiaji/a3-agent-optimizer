from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from a3opt.advanced_audit import SIZE, VideoPolicy
from a3opt.difficulty_eval import PROFILES, EpisodeTrace, check_row, run_episode
from a3opt.refinement_eval import read_json, sha256, simulator_identity, write_json


class DifficultyVideoPolicy(VideoPolicy, EpisodeTrace):
    pass


def record(checkpoint: Path, encoder: Path, output: Path, expected: dict) -> dict:
    import cv2

    profile, suite, seed = (expected[key] for key in ("profile", "suite", "seed"))
    name = f"{seed}-{profile}.mp4"
    partial = output / f"{seed}-{profile}.partial.mp4"
    video = output / name
    trace = DifficultyVideoPolicy(checkpoint, encoder, partial, suite, expected)
    try:
        actual = run_episode(checkpoint, profile, suite, seed, trace)
        check_row((checkpoint, profile, suite, seed), actual)
        matched_keys = sorted(set(expected) - {"wall_seconds"})
        mismatches = [key for key in matched_keys if actual[key] != expected[key]]
        if mismatches or trace.frames != actual["steps"] + 1:
            raise ValueError(f"replay differs for {name}: {mismatches}")
        outcome = "full success" if actual["success"] else actual["failure_reason"] or "incomplete"
        hold = trace.finish(outcome)
        capture = cv2.VideoCapture(str(partial), cv2.CAP_FFMPEG)
        try:
            decoded = 0
            while capture.grab():
                available, frame = capture.retrieve()
                if not available or frame.shape != (SIZE[1], SIZE[0], 3):
                    raise ValueError(f"invalid decoded frame {decoded} in {name}")
                decoded += 1
            if decoded != trace.frames + hold or capture.get(cv2.CAP_PROP_FPS) != trace.fps:
                raise ValueError(f"incorrect frame count or frame rate for {name}")
        finally:
            capture.release()
        partial.replace(video)
        return {"file": name, "sha256": sha256(video), "matched_fields": matched_keys, "frames": decoded,
                "control_frames": trace.frames, "hold_frames": hold, "fps": trace.fps, "episode": expected}
    finally:
        trace.close()
        partial.unlink(missing_ok=True)


def write_index(output: Path, videos: list[dict], seeds: list[int]) -> None:
    sections = []
    for seed in seeds:
        group = [video for video in videos if video["episode"]["seed"] == seed]
        column = group[0]["episode"]["column"]
        cards = []
        for video in group:
            row, name = video["episode"], video["file"]
            passed = row["success"] and row["requested_score"] == 10
            label = "完整成功" if passed else "未完整成功"
            cards.append(f'<figure><figcaption><strong>{row["profile"]}</strong> · {row["requested_score"]}/10 · '
                         f'{row["steps"]} 步 · {label}</figcaption><video controls preload="metadata" '
                         f'src="{name}"></video><a href="{name}" download>下载 MP4</a></figure>')
        sections.append(f'<section><h2>种子 {seed} · 第 {column} 列</h2><div>{"".join(cards)}</div></section>')
    (output / "index.html").write_text(
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>A3 三档配对仿真录像</title><style>
body { font: 16px system-ui, sans-serif; color: #172738; background: #f2f5f8; max-width: 1460px; margin: 28px auto; padding: 0 24px; }
h1 { font-size: 28px; } h2 { font-size: 20px; margin: 28px 0 12px; }
section > div { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap: 14px; }
figure { margin: 0; background: white; padding: 12px; border: 1px solid #d4dce6; border-radius: 8px; }
video { width: 100%; aspect-ratio: 16 / 9; background: #131d29; display: block; margin: 10px 0; }
a { color: #175b96; }
</style><h1>A3 三档配对仿真录像</h1>
<p>每段包含重置、全部控制步与 2 秒终局停留。画面为正面与双腕相机，分数为指定列终局入盒数。</p>
""" + "\n".join(sections) + "\n</html>\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render full three-camera replays for selected paired seeds.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--seeds", required=True, type=int, nargs="+")
    args = parser.parse_args()
    root = args.run.resolve()
    protocol_path, results_path = root / "protocol.json", root / "results.json"
    protocol, results = read_json(protocol_path), read_json(results_path)
    checkpoint = Path(protocol["checkpoint"])
    evaluation = Path(__file__).with_name("difficulty_eval.py")
    if (results["protocol_sha256"] != sha256(protocol_path) or sha256(checkpoint) != protocol["checkpoint_sha256"]
            or simulator_identity() != protocol["environment"] or sha256(evaluation) != protocol["evaluation_sha256"]
            or any(sha256(evaluation.parent / name) != digest
                   for name, digest in protocol["inherited_code_sha256"].items())):
        raise ValueError("replay inputs differ from the frozen protocol")
    indexed = {(row["seed"], row["profile"]): row for row in results["rows"]}
    requested = [indexed[seed, profile] for seed in dict.fromkeys(args.seeds) for profile in PROFILES]
    output = root / "videos"
    output.mkdir(exist_ok=True)
    encoder = args.ffmpeg.resolve(strict=True)
    with ProcessPoolExecutor(max_workers=min(6, len(requested))) as pool:
        jobs = [pool.submit(record, checkpoint, encoder, output, row) for row in requested]
        videos = [job.result() for job in jobs]
    write_json(output / "replay-audit.json", {"results_sha256": sha256(results_path), "videos": videos})
    write_index(output, videos, list(dict.fromkeys(args.seeds)))
    print(f"Verified {len(videos)} full videos and saved the index to {output}", flush=True)


if __name__ == "__main__":
    main()
