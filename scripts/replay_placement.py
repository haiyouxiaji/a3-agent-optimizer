from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from inspect import getfile
from pathlib import Path

import numpy as np

from a3opt.advanced_audit import SIZE, VideoPolicy
from a3opt.backends.advanced_trajectory import pose_label
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import Correction, PlacementPolicy, PlacementTrace, audit
from a3opt import placement_clearance_agent, placement_clearance_feedback, placement_compose, placement_contact_feedback
from a3opt import placement_descent_agent, placement_first_hover
from a3opt import placement_hover_agent, placement_hover_guard, placement_hover_guard_refine, placement_retarget_agent
from a3opt import placement_late_hover_gate, placement_release_hold, placement_release_recovery, placement_release_router
from a3opt import placement_staged_descent, placement_staged_gate
from a3opt import placement_routed_agent, placement_servo_agent, placement_slot_refine, placement_visual_geometry
from a3opt import placement_wall_clearance_agent
from a3opt.refinement_eval import read_json, sha256, simulator_identity, write_json
from a3opt.release_timing_agent import STRICT_RULES


class PlacementVideoTrace(VideoPolicy, PlacementTrace):
    def __init__(self, policy, encoder, output, row):
        PlacementTrace.__init__(self, policy)
        self.encoder, self.output, self.suite, self.expected = encoder, output, row["suite"], row
        self.writer = None
        self.frames = 0


RELEASED_REFINEMENTS = ("hover", "row", "compose", "first-hover", "hover-guard", "hover-guard-refine", "routed",
                        "wall-clearance", "clearance-feedback", "contact-feedback", "late-hover-gate", "staged-descent",
                        "staged-gate", "release-hold", "release-recovery", "release-router")


def decode_model(records, policy_kind, *, incumbent=()):
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
    if policy_kind in RELEASED_REFINEMENTS:
        base = tuple(placement_descent_agent.Descent(*item) for item in incumbent)
        if policy_kind == "clearance-feedback":
            model = tuple(placement_clearance_feedback.Feedback(batch, recipe, placement_clearance_feedback.WallModel(
                mode, tuple(center), tuple(scale), tuple(weights), intercept))
                for batch, recipe, (mode, center, scale, weights, intercept) in records)
            return partial(placement_clearance_feedback.FeedbackPolicy, incumbent=base), model
        if policy_kind == "contact-feedback":
            model = tuple(placement_contact_feedback.ContactGate(*item) for item in records)
            return partial(placement_contact_feedback.ContactPolicy, incumbent=base), model
        if policy_kind == "late-hover-gate":
            model = tuple(placement_late_hover_gate.LateHoverGate(*item) for item in records)
            return partial(placement_late_hover_gate.LateHoverPolicy, incumbent=base), model
        if policy_kind == "staged-descent":
            model = tuple(placement_staged_descent.StagedEntry(*item) for item in records)
            return partial(placement_staged_descent.StagedEntryPolicy, incumbent=base), model
        if policy_kind == "staged-gate":
            model = tuple(placement_staged_gate.StagedGate(placement_staged_descent.StagedEntry(*entry),
                placement_clearance_feedback.WallModel(mode, tuple(center), tuple(scale), tuple(weights), intercept), cutoff)
                for entry, (mode, center, scale, weights, intercept), cutoff in records)
            return partial(placement_staged_gate.StagedGatePolicy, incumbent=base), model
        if policy_kind == "release-hold":
            model = tuple(placement_release_hold.ReleaseHold(recipe) for _, _, recipe in records)
            if len(model) > 1 or [item.record() for item in model] != list(records):
                raise ValueError("unexpected stationary release scope")
            return partial(placement_release_hold.ReleaseHoldPolicy, incumbent=base), model
        if policy_kind == "release-recovery":
            model = tuple(placement_release_recovery.Recovery(*item[2:]) for item in records)
            if len(model) > 1 or [item.record() for item in model] != list(records):
                raise ValueError("unexpected release recovery scope")
            return partial(placement_release_recovery.RecoveryPolicy, incumbent=base), model
        if policy_kind == "release-router":
            model = tuple(placement_release_router.decode(item) for item in records)
            if len(model) > 1 or [item.record() for item in model] != list(records):
                raise ValueError("invalid release router model")
            return partial(placement_release_router.ReleaseRouterPolicy, incumbent=base), model
        if policy_kind == "routed":
            model = placement_routed_agent.decode(records) if records else ()
            if any(item.column == adaptation.column for item in base for adaptation in model):
                raise ValueError("routed correction overlaps the incumbent controller")
            return partial(placement_routed_agent.RoutedPlacementPolicy, incumbent=base), model
        if policy_kind == "wall-clearance":
            model = tuple(placement_wall_clearance_agent.WallClearance(recipe, tuple(cutoffs)) for _, recipe, cutoffs in records)
            if len(model) > 1 or [item.record() for item in model] != list(records):
                raise ValueError("invalid frozen wall-clearance controller")
            return partial(placement_wall_clearance_agent.WallClearancePolicy, incumbent=base), model
        if policy_kind == "first-hover":
            model = tuple(placement_first_hover.FirstHover(column, recipe) for column, _, recipe in records)
            if [item.record() for item in model] != list(records) or len({item.column for item in model}) != len(model):
                raise ValueError("invalid frozen first-batch hover controller")
            return partial(placement_first_hover.FirstHoverPolicy, incumbent=base), model
        if policy_kind == "hover-guard":
            model = tuple(placement_hover_guard.HoverGate(column, recipe, feature, cutoff) for column, _, recipe, feature, cutoff in records)
            if len(model) > 1 or [item.record() for item in model] != list(records):
                raise ValueError("invalid frozen guarded hover controller")
            return partial(placement_hover_guard.GatedHoverPolicy, incumbent=base), model
        if policy_kind == "hover-guard-refine":
            model = tuple(placement_hover_guard_refine.PersistentProjection(column, recipe, cutoff)
                          for column, _, recipe, cutoff in records)
            if len(model) > 1 or [item.record() for item in model] != list(records):
                raise ValueError("invalid frozen projection gate")
            return partial(placement_hover_guard_refine.PersistentProjectionPolicy, incumbent=base), model
        if policy_kind == "hover":
            variant = placement_hover_agent.HoverPolicy
            model = tuple(placement_hover_agent.Hover(recipe) for _, _, recipe in records)
        elif policy_kind == "row":
            variant = placement_slot_refine.RowPolicy
            model = tuple(placement_slot_refine.RowAlignment(tuple(center), tuple(scale), tuple(weights), bias)
                          for _, _, center, scale, weights, bias in records)
        else:
            variant = placement_compose.ComposedPolicy
            model = placement_compose.decode(records) if records else ()
        if (policy_kind != "compose" and len(model) > 1) or [item.record() for item in model] != list(records) or any(
            item.column == adaptation.column for item in base for adaptation in model
        ):
            raise ValueError("invalid frozen placement controller")
        return partial(variant, incumbent=base), model
    model = tuple(Correction(column, batch, tuple(xyz), *limits) for column, batch, xyz, *limits in records)
    return PlacementPolicy, model


def policy_source(policy_type):
    return Path(getfile(policy_type.func if isinstance(policy_type, partial) else policy_type))


def verify_policy_code(configuration, frozen, kind, constructor):
    policy_hash = configuration.get("policy_code_sha256") or configuration["code_sha256"]
    if sha256(policy_source(constructor)) != policy_hash:
        raise ValueError("policy code differs from the frozen evaluation")
    if kind in RELEASED_REFINEMENTS and configuration["incumbent"] != frozen["incumbent"]:
        raise ValueError("reference controller differs from the frozen evaluation")
    if kind == "compose" and (
        sha256(Path(placement_slot_refine.__file__)) != configuration["row_code_sha256"]
        or sha256(Path(placement_hover_agent.__file__)) != configuration["hover_code_sha256"]
    ):
        raise ValueError("composed policy dependencies differ from the frozen evaluation")
    if kind in ("hover-guard", "hover-guard-refine") and sha256(Path(placement_first_hover.__file__)) != configuration["first_hover_code_sha256"]:
        raise ValueError("first-batch hover dependency differs from the frozen evaluation")
    if kind in ("hover-guard-refine", "wall-clearance") and sha256(Path(placement_hover_guard.__file__)) != configuration["guard_code_sha256"]:
        raise ValueError("projection guard dependency differs from the frozen evaluation")
    if kind == "routed" and placement_routed_agent.dependencies() != configuration["dependency_sha256"]:
        raise ValueError("routed controller dependencies differ from the frozen evaluation")
    if kind == "release-recovery" and sha256(Path(placement_contact_feedback.__file__)) != configuration["contact_code_sha256"]:
        raise ValueError("contact signal code differs from the frozen evaluation")
    if kind == "release-router" and placement_release_router.dependencies() != configuration["dependencies"]:
        raise ValueError("release router dependencies differ from the frozen evaluation")


def upstream_policy():
    lock_path = Path(os.environ["A3OPT_PLACEMENT_LOCK"]).resolve(strict=True)
    kind = os.environ["A3OPT_PLACEMENT_KIND"]
    if kind not in RELEASED_REFINEMENTS:
        raise ValueError("expected a supported frozen placement refinement")
    protocol_path = lock_path.parent / "protocol.json"
    lock, protocol = read_json(lock_path), read_json(protocol_path)
    checkpoint = Path(lock["checkpoint"])
    constructor, model = decode_model(lock["corrections"], kind, incumbent=lock["incumbent"])
    if (sha256(protocol_path) != lock["protocol_sha256"] or sha256(checkpoint) != lock["checkpoint_sha256"]
            or protocol["environment"] != simulator_identity()):
        raise ValueError("the requested controller differs from the pre-evaluation lock")
    verify_policy_code(protocol, lock, kind, constructor)
    return constructor(checkpoint, *lock["timing"], model)


def record(checkpoint, timing, encoder, output, name, expected, policy_kind, incumbent=()):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    seed, suite, strict = (expected[key] for key in ("seed", "suite", "strict"))
    policy_type, corrections = decode_model(expected["corrections"], policy_kind, incumbent=incumbent)
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


def replay_pair(root, configuration, frozen, stage, kind, seed, rule, reference_run=None):
    seeds = frozen["seeds"] if stage == "confirmation" else configuration["schedule"]["development"]
    if kind in ("first-hover", "contact-feedback", "staged-descent") and stage == "development":
        from a3_dual_arm_sim.sim.randomization import choose_column

        active = {entry[0] for entry in frozen["corrections"]}
        seeds = [item for item in seeds if choose_column("random", item) in active]
    if seed not in seeds:
        raise ValueError("replay seed was not in the frozen evaluation")
    if kind == "compose" and stage == "development":
        fit_path = root / "fit.json"
        if reference_run is None or sha256(fit_path) != frozen["fit_sha256"]:
            raise ValueError("composed development replay requires the frozen fit and --reference-run")
        reference = reference_run / "development"
        selected = root / "interaction"
        fit = read_json(fit_path)
        if sha256(reference / "results.json") != fit["row"]["rollout_sha256"]:
            raise ValueError("the reference development receipts differ from the frozen composition input")
        folders = {"reference": reference, "selected": selected}
    else:
        folder = root / "confirmation" / "rollouts" if stage == "confirmation" else root / "development"
        folders = {"reference": folder, "selected": folder}
    corrections = {"reference": [], "selected": frozen["corrections"]}
    paths = {arm: folder / "results.json" for arm, folder in folders.items()}
    pair = {}
    for arm, folder in folders.items():
        request = read_json(folder / "request.json")
        if (request["seeds"] != seeds or request["models"].get(arm) != corrections[arm]
                or request["checkpoint_sha256"] != frozen["checkpoint_sha256"] or request["timing"] != frozen["timing"]):
            raise ValueError(f"replay request for {arm} differs from the frozen evaluation")
        entry = read_json(paths[arm])[arm][seeds.index(seed)]
        if entry[rule]["seed"] != seed or entry[rule]["corrections"] != corrections[arm]:
            raise ValueError(f"replay receipt for {arm} differs from the frozen evaluation")
        pair[arm] = entry[rule]
    digest = sha256(paths["reference"]) if paths["reference"] == paths["selected"] else {arm: sha256(path) for arm, path in paths.items()}
    return pair, digest


def main():
    parser = argparse.ArgumentParser(description="Reproduce both controllers on the same locked advanced scene and record every step.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--rule", choices=("default", "strict-500"), default="strict-500")
    parser.add_argument("--policy-kind", choices=("offset", "geometry", "descent", "servo", "clearance", "landing", *RELEASED_REFINEMENTS),
                        default="offset")
    parser.add_argument("--stage", choices=("development", "confirmation"), default="confirmation")
    parser.add_argument("--reference-run", type=Path, help="row-alignment run that supplied the composed development reference")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    root = args.run.resolve()
    protocol, lock = root / "protocol.json", root / "confirmation-lock.json"
    configuration, frozen = read_json(protocol), read_json(lock)
    checkpoint, timing = Path(frozen["checkpoint"]), tuple(frozen["timing"])
    if (simulator_identity() != configuration["environment"] or sha256(protocol) != frozen["protocol_sha256"]
            or sha256(checkpoint) != frozen["checkpoint_sha256"]):
        raise ValueError("replay does not match the frozen evaluation")
    incumbent = frozen.get("incumbent", ())
    if args.policy_kind != "offset":
        policy_type, _ = decode_model(frozen["corrections"], args.policy_kind, incumbent=incumbent)
        verify_policy_code(configuration, frozen, args.policy_kind, policy_type)
    pair, results_digest = replay_pair(root, configuration, frozen, args.stage, args.policy_kind, args.seed, args.rule, args.reference_run)
    output = root / "videos" / args.rule
    if args.stage == "development":
        output /= "development"
    output.mkdir(parents=True, exist_ok=True)
    encoder = args.ffmpeg.resolve(strict=True)
    with _run_lock(output):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(record, checkpoint, timing, encoder, output, name, row, args.policy_kind, incumbent)
                    for name, row in pair.items()]
            evidence = [job.result() for job in jobs]
        write_json(output / f"{args.seed}-audit.json", {"stage": args.stage, "results_sha256": results_digest, "videos": evidence})
    print(f"Verified {len(evidence)} full simulator videos for seed {args.seed}", flush=True)


if __name__ == "__main__":
    main()
