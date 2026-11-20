from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from a3opt.advanced_audit import SIZE, VideoPolicy
from a3opt.backends.advanced_trajectory import pose_label
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import Correction, PlacementPolicy, PlacementTrace, audit
from a3opt import placement_clearance_agent, placement_descent_agent, placement_retarget_agent, placement_servo_agent, placement_visual_geometry
from a3opt.refinement_eval import read_json, sha256, simulator_identity, write_json
from a3opt.release_timing_agent import STRICT_RULES


class PlacementVideoTrace(VideoPolicy, PlacementTrace):
    def __init__(self, policy, encoder, output, row):
        PlacementTrace.__init__(self, policy)
        self.encoder, self.output, self.suite, self.expected = encoder, output, row["suite"], row
        self.writer = None
        self.frames = 0


def decode_model(records, policy_kind):
    if policy_kind == "geometry":
        model = tuple(placement_visual_geometry.GeometryPlacement(column, recipe, tuple(center), tuple(scale), tuple(weights), intercept)
                      for column, recipe, center, scale, weights, intercept in records)
        return placement_visual_geometry.GeometryPolicy, model
    if policy_kind == "descent":
        return placement_descent_agent.DescentPolicy, tuple(placement_descent_agent.Descent(*item) for item in records)
    if policy_kind == "servo":
        return placement_servo_agent.ServoPolicy, tuple(placement_servo_agent.Servo(*item) for item in records)
    if policy_kind == "clearance":
        return placement_clearance_agent.ClearancePolicy, tuple(placement_clearance_agent.Clearance(*item) for item in records)
    if policy_kind == "landing":
        return placement_retarget_agent.LandingPolicy, tuple(placement_retarget_agent.Landing(*item) for item in records)
    model = tuple(Correction(column, batch, tuple(xyz), *limits) for column, batch, xyz, *limits in records)
    return PlacementPolicy, model


def record(checkpoint, timing, encoder, output, name, expected, policy_kind):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    seed, suite, strict = (expected[key] for key in ("seed", "suite", "strict"))
    policy_type, corrections = decode_model(expected["corrections"], policy_kind)
    audit((checkpoint, *timing, suite, seed, corrections, strict), expected)
    engine = make_cookie_benchmark(profile=expected["profile"], source_column=suite, **(STRICT_RULES if strict else RULES))
    env = engine.create_env(render_cameras=False)
    if strict:
        env.task_config = replace(env.task_config, terminate_on_success=False)
    path = output / f"{seed}-{name}.mp4"
    partial = path.with_suffix(".partial.mp4")
    trace = PlacementVideoTrace(policy_type(checkpoint, *timing, corrections), encoder, partial, expected)
    try:
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        random = result.randomization
        ids = np.flatnonzero(trace.target)
        actual = {"steps": result.steps, "score": result.score, "requested_score": int(np.count_nonzero(trace.columns[ids] == result.source_column)),
                  "success": result.success, "target_cookie_ids": ids.tolist(), "source_remaining": result.cookies_in_source,
                  "safety_reason": trace.safety, "terminal": trace.terminal, "initial_pose": pose_label(random).tolist(),
                  "color": random["cookie_color"], "cameras": random["cameras"], "light_diffuse": random["light_diffuse"],
                  "action_sha256": trace.digest.hexdigest(), "applied_action_sha256": trace.applied_digest.hexdigest()}
        differences = [key for key in actual if actual[key] != expected[key]]
        if differences or trace.frames != result.steps + 1:
            raise ValueError(f"replay differs for {name}/{seed}: {differences}")
        hold = trace.finish("full success" if result.success else "incomplete")
        capture = cv2.VideoCapture(str(partial), cv2.CAP_FFMPEG)
        try:
            frames = 0
            while capture.grab():
                valid, frame = capture.retrieve()
                if not valid or frame.shape != (SIZE[1], SIZE[0], 3):
                    raise ValueError(f"invalid replay frame {frames}")
                frames += 1
            if frames != trace.frames + hold or capture.get(cv2.CAP_PROP_FPS) != trace.fps:
                raise ValueError("video does not cover all control steps")
        finally:
            capture.release()
        partial.replace(path)
        evidence = {"arm": name, "seed": seed, "video": path.name, "sha256": sha256(path), "frames": frames,
                    "control_frames": trace.frames, "fps": trace.fps, "matched_fields": sorted(actual), "episode": actual}
        write_json(path.with_suffix(".json"), evidence)
        return evidence
    finally:
        trace.close()
        env.close()
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Reproduce both controllers on the same locked advanced scene and record every step.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--rule", choices=("default", "strict-500"), default="strict-500")
    parser.add_argument("--policy-kind", choices=("offset", "geometry", "descent", "servo", "clearance", "landing"), default="offset")
    parser.add_argument("--stage", choices=("development", "confirmation"), default="confirmation")
    args = parser.parse_args()
    root = args.run.resolve()
    protocol, lock = root / "protocol.json", root / "confirmation-lock.json"
    configuration, frozen = read_json(protocol), read_json(lock)
    checkpoint, timing = Path(frozen["checkpoint"]), tuple(frozen["timing"])
    stages = root / "confirmation" / "rollouts" if args.stage == "confirmation" else root / "development"
    rows_path = stages / "results.json"
    request = read_json(stages / "request.json")
    seeds = frozen["seeds"] if args.stage == "confirmation" else configuration["schedule"]["development"]
    if (simulator_identity() != configuration["environment"] or sha256(protocol) != frozen["protocol_sha256"]
            or sha256(checkpoint) != frozen["checkpoint_sha256"] or args.seed not in seeds or request["seeds"] != seeds
            or request["models"] != {"reference": [], "selected": frozen["corrections"]}):
        raise ValueError("replay does not match the frozen evaluation")
    policy_module = {"geometry": placement_visual_geometry, "descent": placement_descent_agent,
                     "servo": placement_servo_agent, "clearance": placement_clearance_agent,
                     "landing": placement_retarget_agent}.get(args.policy_kind)
    if policy_module is not None:
        policy_hash = configuration.get("policy_code_sha256", configuration["code_sha256"])
        if sha256(Path(policy_module.__file__)) != policy_hash:
            raise ValueError("policy code differs from the frozen evaluation")
    rows = read_json(rows_path)
    pair = {name: next(row[args.rule] for row in rows[name] if row[args.rule]["seed"] == args.seed)
            for name in ("reference", "selected")}
    if pair["reference"]["corrections"] or pair["selected"]["corrections"] != frozen["corrections"]:
        raise ValueError("replay candidates differ from the pre-evaluation lock")
    output = root / "videos" / args.rule
    if args.stage == "development":
        output /= "development"
    output.mkdir(parents=True, exist_ok=True)
    encoder = args.ffmpeg.resolve(strict=True)
    with _run_lock(output):
        with ProcessPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(record, checkpoint, timing, encoder, output, name, row, args.policy_kind) for name, row in pair.items()]
            evidence = [job.result() for job in jobs]
        write_json(output / f"{args.seed}-audit.json", {"stage": args.stage, "results_sha256": sha256(rows_path), "videos": evidence})
    print(f"Verified {len(evidence)} full simulator videos for seed {args.seed}", flush=True)


if __name__ == "__main__":
    main()
