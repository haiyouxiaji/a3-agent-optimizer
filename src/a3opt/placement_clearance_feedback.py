from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_descent_agent as descent, placement_first_hover as first
from a3opt import placement_hover_agent as hover, placement_routed_agent as routed, placement_search
from a3opt import placement_wall_clearance_agent as wall, placement_wrist_rgb as wrist
from a3opt.advanced_agent import _jobs, _npz, _record_once
from a3opt.backends.adaptive_phase import PLACE, TRANSFER, task_column
from a3opt.backends.release_residual import nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementPolicy, PlacementTrace, projected_tool_x
from a3opt.refinement_eval import read_json, render_frames, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


NEAR_WALL_M = .002
FEATURES = {"plan": 2, "pose": 8, "force": 17, "wrist": 67}
PENALTIES = {"plan": 2., "pose": 6., "force": 16., "wrist": 32.}
PANEL_SIZES = {"fit": 16, "sensor_holdout": 24, "development": 20, "confirmation": 64}


def readings(policy, observation, batch):
    measured = np.asarray(observation["observation.eef_pose"], dtype=float)
    commanded = descent.tool_xyz(nominal_kinematics(), policy.actions[batch][TRANSFER][-1, :7])
    direction = wall.OUTWARD[batch]
    projected = direction * projected_tool_x(commanded, policy.pose)
    actual = direction * projected_tool_x(measured[:3], policy.pose)
    quaternion = measured[3:7] * (1. if measured[3] >= 0. else -1.)
    sensors = np.asarray(observation["observation.force"], dtype=float)
    return np.r_[batch, projected, actual - projected, measured[1:3] - commanded[1:3], quaternion[1:],
                 sensors[:6], sensors[[12, 13, 16]]]


@dataclass(frozen=True)
class WallModel:
    mode: str
    center: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    intercept: float

    def __post_init__(self):
        size = FEATURES.get(self.mode, 0)
        if (not size or any(len(values) != size or not np.isfinite(values).all()
                           for values in (self.center, self.scale, self.weights))
                or min(self.scale) <= 0. or not np.isfinite(self.intercept)):
            raise ValueError("invalid wall clearance estimator")

    def record(self):
        return [self.mode, list(self.center), list(self.scale), list(self.weights), self.intercept]

    def predict(self, values):
        normalized = np.clip((np.asarray(values[:FEATURES[self.mode]]) - self.center) / self.scale, -3., 3.)
        return float(normalized @ self.weights + self.intercept)


@dataclass(frozen=True)
class Feedback:
    batch: int
    recipe: str
    estimator: WallModel
    column = wall.COLUMN

    def __post_init__(self):
        if self.batch not in range(2) or self.recipe not in wall.RECIPES or self.estimator.mode == "plan":
            raise ValueError("invalid clearance feedback correction")

    def record(self):
        return [self.batch, self.recipe, self.estimator.record()]


class FeedbackPolicy(descent.DescentPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        self.feedback = {item.batch: item for item in corrections}
        if len(self.feedback) != len(corrections):
            raise ValueError("expected one clearance correction per batch")
        super().__init__(checkpoint, focus, scale, incumbent)

    def pending(self):
        if self.actions is not None and self.cursor is not None and self.cursor[1:] == (PLACE, 0) and not self.retry_remaining:
            return self.feedback.get(self.cursor[0])
        return None

    @property
    def camera_names(self):
        item = self.pending()
        if item is not None and item.estimator.mode == "wrist":
            return ("left_wrist",)
        return super().camera_names

    def act(self, observation, task=""):
        item = self.pending()
        if item is not None and task_column(task) == item.column:
            values = readings(self, observation, item.batch)
            if item.estimator.mode == "wrist":
                frame = observation.get(wrist.WRIST)
                if frame is None or not np.any(frame):
                    return super().act(observation, task)
                values = np.r_[values, wrist.wrist_features(frame)]
            if np.isfinite(values).all() and item.estimator.predict(values) <= NEAR_WALL_M:
                prior = self.profiles.get((item.column, item.batch))
                if prior is not None:
                    descent.adjust(self.actions[item.batch], observation["observation.eef_pose"], prior)
                inset, rise = wall.RECIPES[item.recipe]
                self.shift_placement(item.batch, np.array([-wall.OUTWARD[item.batch] * inset, 0., rise]))
                return PlacementPolicy.act(self, observation, task)
        return super().act(observation, task)


def clearance(snapshot, batch, scoring_tolerance):
    group = snapshot["batches"][batch]
    if len(group) != 5 or any(cookie["abs_vertical_axis"] < .98 or cookie["bottom_above_floor_m"] < .035 for cookie in group):
        return None
    direction = wall.OUTWARD[batch]
    return min(cookie["footprint_margin_m"][0] - scoring_tolerance
               + abs(cookie["xyz_m"][0]) - direction * cookie["xyz_m"][0] for cookie in group)


class TransferTrace(PlacementTrace):
    def reset(self, context=None):
        super().reset(context)
        self.samples, self.frames = [], {}

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        policy = self.policy
        if not policy.ended or policy.stage != TRANSFER or self.safety:
            return
        batch = policy.batch
        snapshot = self.stages[batch]["transfer"]
        frame = render_frames(env, ("left_wrist",))[wrist.WRIST]
        public = readings(policy, env._observation(), batch)
        self.frames[str(batch)] = frame
        self.samples.append({"batch": batch, "step": self.step, "prefix_sha256": snapshot["prefix_sha256"],
                             "features": np.r_[public, wrist.wrist_features(frame)].tolist(),
                             "margin_m": clearance(snapshot, batch, env.CONTACT_CONTAINMENT_TOLERANCE_M)})
        self.finished = batch == 1


def collect(checkpoint, timing, incumbent, shards, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = TransferTrace(descent.DescentPolicy(checkpoint, *timing, incumbent))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        shard = shards / f"{seed}.npz"
        _npz(shard, **trace.frames)
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "timing": list(timing), "steps": result.steps,
                "safety_reason": trace.safety, "shard_sha256": sha256(shard), "samples": trace.samples}
    finally:
        env.close()


def audit(job, row):
    checkpoint, timing, _, shards, seed = job
    shard = shards / f"{seed}.npz"
    batches = [sample["batch"] for sample in row["samples"]]
    if (row["seed"] != seed or row["column"] != wall.COLUMN or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != list(timing)
            or row["shard_sha256"] != sha256(shard) or batches != list(range(len(batches))) or len(batches) > 2
            or (batches == [0, 1] and row["steps"] != row["samples"][-1]["step"])
            or (batches != [0, 1] and not row["safety_reason"] and row["steps"] != RULES["max_steps"])):
        raise ValueError("transfer sample differs from the frozen acquisition")
    with np.load(shard, allow_pickle=False) as frames:
        if set(frames.files) != set(map(str, batches)):
            raise ValueError("wrist frame inventory differs from the receipt")
        for sample in row["samples"]:
            features = sample["features"]
            margin = sample["margin_m"]
            if (len(features) != FEATURES["wrist"] or not np.isfinite(features).all()
                    or margin is not None and not np.isfinite(margin)
                    or len(sample["prefix_sha256"]) != 2 or any(len(digest) != 64 for digest in sample["prefix_sha256"])
                    or not np.array_equal(features[FEATURES["force"]:], wrist.wrist_features(frames[str(sample["batch"])]))):
                raise ValueError("invalid transfer observation or label")


def fit(scenes, mode):
    samples = [sample for scene in scenes for sample in scene["samples"] if sample["margin_m"] is not None]
    inputs = np.asarray([sample["features"][:FEATURES[mode]] for sample in samples])
    labels = np.asarray([sample["margin_m"] for sample in samples])
    center, scale = inputs.mean(axis=0), np.maximum(inputs.std(axis=0), .00001)
    features = np.clip((inputs - center) / scale, -3., 3.)
    design = np.column_stack((features, np.ones(len(samples))))
    penalty = np.diag([*([PENALTIES[mode]] * FEATURES[mode]), 0.])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ labels)
    return WallModel(mode, tuple(center), tuple(scale), tuple(coefficients[:-1]), float(coefficients[-1]))


def predictions(scenes, model):
    return [{"seed": scene["seed"], "batch": sample["batch"], "margin_m": sample["margin_m"],
             "estimated_m": model.predict(sample["features"])} for scene in scenes for sample in scene["samples"]]


def metrics(rows):
    stats = dict(near_wall=0, clear=0, missed=0, false_triggers=0, unsupported=0, unsupported_triggers=0)
    errors = []
    for row in rows:
        margin, estimate = row["margin_m"], row["estimated_m"]
        trigger = estimate <= NEAR_WALL_M
        if margin is None:
            stats["unsupported"] += 1
            stats["unsupported_triggers"] += trigger
            continue
        danger = margin <= NEAR_WALL_M
        stats["near_wall"] += danger
        stats["clear"] += not danger
        stats["missed"] += danger and not trigger
        stats["false_triggers"] += trigger and not danger
        errors.append(abs(margin - estimate))
    return {**stats, "mae_mm": 1000. * float(np.mean(errors)) if errors else None}


def errors(stats):
    return stats["missed"], stats["false_triggers"] + stats["unsupported_triggers"]


def improves_sensor(baseline, selected):
    before, after = errors(baseline), errors(selected)
    return (min(selected["near_wall"], selected["clear"]) >= 4 and before != after
            and all(trial <= control for control, trial in zip(before, after, strict=True)))


def active_plan(rows):
    failures, controls = set(), set()
    for batch in range(2):
        missed, healthy = [], []
        for row in rows:
            if row["column"] != wall.COLUMN or row["safety_reason"] or row["score"] != row["requested_score"]:
                continue
            stages = row["phase_geometry"][batch]
            margin = clearance(stages.get("transfer", {"batches": [[], []]}), batch, 0.)
            if margin is None:
                continue
            placed = sum(cookie["scored"] for cookie in row["physical_terminal"]["batches"][batch])
            if placed < 5:
                missed.append((margin, row["seed"]))
            elif row["success"]:
                healthy.append((margin, row["seed"]))
        failures.update(seed for _, seed in sorted(missed)[:4])
        controls.update(seed for _, seed in sorted(healthy)[:3])
    return {"failures": sorted(failures), "controls": sorted(controls - failures)}


def schedule(root, block, historic_seeds, active):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4:
        raise ValueError("expected an aligned seed block")

    def focused(name, offset):
        return [seed for seed in range(block + offset, block + offset + 4 * PANEL_SIZES[name])
                if choose_column("random", seed) == wall.COLUMN]

    panels = {"fit": focused("fit", 0), "sensor_holdout": focused("sensor_holdout", 100),
              "development": focused("development", 240),
              "confirmation": list(range(block + 400, block + 400 + PANEL_SIZES["confirmation"]))}
    reserved = set().union(*panels.values())
    if (any(len(panels[name]) != size for name, size in PANEL_SIZES.items()) or sum(PANEL_SIZES.values()) != len(reserved)
            or reserved & historic_seeds or any(path.parent != root and reserved & hover.scheduled_seeds(
                read_json(path).get("schedule", {})) for path in root.parent.glob("*/protocol.json"))):
        raise ValueError("clearance evaluation must use mutually disjoint, untouched scenes")
    panels["fit"] += active["failures"] + active["controls"]
    return panels


def run(root, release, source, block, workers):
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint, timing = release / frozen["checkpoint"], tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*item) for item in frozen["corrections"])
    common = {"environment": simulator_identity(), "incumbent": frozen["corrections"], "timing": list(timing),
              "rules": RULES, "strict_rules": STRICT_RULES}
    if (common["environment"]["sim_revision"] != frozen["sim_revision"] or sha256(checkpoint) != frozen["checkpoint_sha256"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("clearance feedback requires the frozen published controller")
    history, archive = first.historical_rows(release, (), frozen, common, checkpoint, timing, incumbent)
    if source is not None:
        observed, fingerprints = wall.read_training_source(source, common, checkpoint, timing)
        history += observed
        archive["routed_replication"] = fingerprints
    previous = {row["seed"]: row for row in history}
    if len(previous) != len(history):
        raise ValueError("duplicated clearance training source")
    active = active_plan(history)
    panels = schedule(root, block, set(previous), active)
    protocol = root / "protocol.json"
    _record_once(protocol, {**common, "reference_manifest_sha256": sha256(manifest), "source": archive, "schedule": panels,
                 "active_sampling": active, "code_sha256": sha256(Path(__file__)), "near_wall_m": NEAR_WALL_M,
                 "penalties": PENALTIES, "recipes": {name: list(values) for name, values in wall.RECIPES.items()},
                 "sensor_holdout_requires": ["at_least_four_near_wall_and_four_clear_labels", "fewer_misses_or_spurious_triggers",
                                             "no_increase_in_either_error"], "confirmation_requires": routed.REQUIREMENTS})
    policy_type = partial(FeedbackPolicy, incumbent=incumbent)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        shards = root / "training" / "frames"
        training_jobs = [(checkpoint, timing, incumbent, shards, seed) for seed in panels["fit"]]
        print(f"Agent acquiring {len(training_jobs)} training transfer scenes", flush=True)
        training = _jobs(pool, collect, training_jobs, root / "training" / "episodes", lambda job: str(job[-1]), audit)
        for row in training:
            if row["seed"] in previous:
                expected = previous[row["seed"]]["phase_geometry"]
                if any(sample["prefix_sha256"] != expected[sample["batch"]]["transfer"]["prefix_sha256"] for sample in row["samples"]):
                    raise ValueError("active transfer sample did not replay the archived prefix")
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"fit": training})
        if sum(sample["margin_m"] is not None for scene in training for sample in scene["samples"]) < 12:
            return {"status": "reference_retained_without_sensor_labels"}
        cross_validation = {}
        for mode in FEATURES:
            estimates = []
            for scene in training:
                grouped = [other for other in training if other["seed"] != scene["seed"]]
                estimates.extend(predictions([scene], fit(grouped, mode)))
            cross_validation[mode] = metrics(estimates)
        mode = min((mode for mode in FEATURES if mode != "plan"),
                   key=lambda mode: (sum(errors(cross_validation[mode])), *errors(cross_validation[mode]),
                                     cross_validation[mode]["mae_mm"], FEATURES[mode]))
        baseline, estimator = fit(training, "plan"), fit(training, mode)
        candidate = root / "training" / "estimator.json"
        _record_once(candidate, {"cross_validation": cross_validation, "baseline": baseline.record(),
                     "estimator": estimator.record(), "fit_seeds": panels["fit"], "dataset_sha256": sha256(dataset)})
        heldout_jobs = [(checkpoint, timing, incumbent, shards, seed) for seed in panels["sensor_holdout"]]
        print(f"Agent froze {mode} before acquiring {len(heldout_jobs)} sensor holdout scenes", flush=True)
        heldout = _jobs(pool, collect, heldout_jobs, root / "sensor-holdout" / "episodes", lambda job: str(job[-1]), audit)
        holdout_dataset = root / "sensor-holdout" / "dataset.json"
        _record_once(holdout_dataset, {"sensor_holdout": heldout})
        independent = {model.mode: predictions(heldout, model) for model in (baseline, estimator)}
        sensor = {name: metrics(rows) for name, rows in independent.items()}
        _record_once(root / "sensor-holdout.json", {"metrics": sensor, "predictions": independent,
                     "estimator_sha256": sha256(candidate), "dataset_sha256": sha256(holdout_dataset)})
        print(f"Agent tested frozen {mode} estimates on independent scenes: {sensor}", flush=True)
        if not improves_sensor(sensor["plan"], sensor[mode]):
            return {"status": "reference_retained_after_sensor_holdout", "sensor_holdout": sensor, "mode": mode}
        probes = active["failures"] + active["controls"]
        models = {name: tuple(Feedback(batch, name, estimator) for batch in range(2)) for name in wall.RECIPES}
        paired = placement_benchmark.evaluate(pool, root / "training" / "rollouts", checkpoint, timing, probes,
                                              {"reference": (), **models}, policy_type=policy_type)
        for pair in paired["reference"]:
            row = pair["strict-500"]
            if any(row[key] != previous[row["seed"]][key] for key in hover.MATCH_FIELDS):
                raise ValueError("the training reference did not replay its archived episode")
        scores = {name: hover.pair_stats({"reference": paired["reference"], "selected": paired[name]}, model)
                  for name, model in models.items()}
        eligible = [name for name, stats in scores.items() if all(placement_search.passes(row) and
                    sum(change["delta"] > 0 for change in row["changes"]) >= 2 for row in stats.values())]
        selected = max(eligible, key=lambda name: (scores[name]["default"]["mean"], -wall.RECIPES[name][1]), default=None)
        _record_once(root / "fit.json", {"recipe": selected, "metrics": scores, "estimator_sha256": sha256(candidate),
                     "rollouts_sha256": sha256(root / "training" / "rollouts" / "results.json")})
        if selected is None:
            return {"status": "reference_retained_after_training", "sensor_holdout": sensor, "training": scores, "mode": mode}
        model = models[selected]
        print(f"Agent selected {selected} for {len(panels['development'])} unseen development pairs", flush=True)
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, panels["development"],
                                              {"reference": (), "selected": model}, policy_type=policy_type)
        development = hover.pair_stats(paired, model)
        _record_once(root / "development-selection.json", development)
        result = {"sensor_holdout": sensor, "mode": mode, "recipe": selected, "development": development}
        if not all(placement_search.passes(stats) for stats in development.values()):
            return {**result, "status": "reference_retained_after_development"}
        confirmation = panels["confirmation"]
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "seeds": confirmation,
                     "corrections": [item.record() for item in model], "protocol_sha256": sha256(protocol),
                     "fit_sha256": sha256(root / "fit.json"), "selection_sha256": sha256(root / "development-selection.json")})
        print(f"Agent froze {selected} before {len(confirmation)} four-column confirmation pairs", flush=True)
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                   model, confirmation[:4], policy_type=policy_type)
    balance = {str(column): len(confirmation) // 4 for column in range(1, 5)}
    if any(stats["scored_pairs_by_column"] != balance for stats in final.values()):
        raise ValueError("confirmation requires four-column balance")
    return {**result, "status": "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_confirmation",
            "confirmation": final, "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_CLEARANCE_FEEDBACK_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("clearance feedback checkpoint differs from the confirmation lock")
    incumbent = tuple(descent.Descent(*item) for item in lock["incumbent"])
    corrections = tuple(Feedback(batch, recipe, WallModel(mode, tuple(center), tuple(scale), tuple(weights), intercept))
                        for batch, recipe, (mode, center, scale, weights, intercept) in lock["corrections"])
    return FeedbackPolicy(checkpoint, *lock["timing"], corrections, incumbent=incumbent)


def main():
    parser = argparse.ArgumentParser(description="Acquire and validate observable tray-wall clearance feedback.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=150000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.source.resolve() if args.source else None, args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
