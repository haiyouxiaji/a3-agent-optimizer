from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_search
from a3opt.advanced_agent import _jobs, _npz, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, TRANSFER, task_column
from a3opt.backends.release_residual import nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.placement_descent_agent import tool_xyz
from a3opt.placement_sim import PlacementPolicy, PlacementTrace
from a3opt.refinement_eval import read_json, render_frames, sha256
from a3opt.release_timing_agent import STRICT_RULES


WRIST = "observation.images.left_wrist"
SPLIT_SIZES = (32, 12, 12)
PENALTIES = (1., 6., 24.)
MAX_SHIFT = .006


def basis(yaw):
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array([[cosine, -sine], [sine, cosine]])


def wrist_features(image):
    import cv2

    pixels = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)[20:102, 26:102].astype(np.float32)
    peak = pixels.max(axis=-1)
    chroma = np.ptp(pixels, axis=-1) / np.maximum(40., peak)
    objects = np.clip((chroma - .15) * 3., 0., 1.) * (peak > 45.)
    tray = (chroma < .23) & (peak > 65.)
    return np.concatenate([cv2.resize(mask.astype(np.float32), (5, 5), interpolation=cv2.INTER_AREA).ravel()
                           for mask in (objects, tray)])


def sensors(policy, observation, batch):
    observed = np.asarray(observation["observation.eef_pose"][:3])
    plan = policy.actions[batch][OPEN][0, :7]
    commanded = tool_xyz(nominal_kinematics(), plan)
    local = (observed[:2] - policy.pose[2:4]) @ basis(policy.pose[4])
    observation_features = np.r_[local / .04, (commanded - observed) / .06, policy.pose[4] / .08]
    return observation_features, commanded[:2], observed[:2]


class WristTrace(PlacementTrace):
    def reset(self, context=None):
        super().reset(context)
        self.samples, self.images = [], {}

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.policy.stage != TRANSFER or not self.policy.ended or self.safety:
            return
        batch = self.policy.batch
        stage = self.stages[batch]["transfer"]
        group = stage["batches"][batch]
        if len(group) != 5 or min(cookie["bottom_above_floor_m"] for cookie in group) < .035:
            return
        features, _, tool = sensors(self.policy, env._observation(), batch)
        points = np.array([cookie["xyz_m"][:2] for cookie in group])
        center = (points.min(axis=0) + points.max(axis=0)) / 2.
        target = env.data.xpos[env._target_bin_body, :2]
        rotation = env.data.xmat[env._target_bin_body].reshape(3, 3)[:2, :2]
        image = render_frames(env, ("left_wrist",))[WRIST]
        self.images[str(batch)] = image
        self.samples.append({"batch": batch, "prefix_sha256": stage["prefix_sha256"], "sensors": features.tolist(),
                             "wrist": wrist_features(image).tolist(), "offset": (center - (tool - target) @ rotation).tolist()})


def collect(checkpoint, focus, scale, shards, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = WristTrace(PlacementPolicy(checkpoint, focus, scale))
        run = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        shard = shards / f"{seed}.npz"
        _npz(shard, **trace.images, seed=seed, column=run.source_column)
        return {"seed": seed, "column": run.source_column, "profile": run.randomization["profile"], "score": run.score,
                "steps": run.steps, "checkpoint_sha256": sha256(checkpoint), "timing": [focus, scale],
                "shard_sha256": sha256(shard), "samples": trace.samples, "action_sha256": trace.digest.hexdigest()}
    finally:
        env.close()


def audit(job, row):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, focus, scale, shards, seed = job
    shard = shards / f"{seed}.npz"
    if (row["seed"] != seed or row["column"] != choose_column("random", seed) or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != [focus, scale]
            or row["shard_sha256"] != sha256(shard) or len(row["action_sha256"]) != 64):
        raise ValueError("wrist acquisition receipt differs from the reserved scene")
    with np.load(shard, allow_pickle=False) as archive:
        batches = [sample["batch"] for sample in row["samples"]]
        if (batches != sorted(set(batches)) or set(archive.files) != {"seed", "column", *map(str, batches)}
                or (int(archive["seed"]), int(archive["column"])) != (seed, row["column"])):
            raise ValueError("wrist dataset record has missing or inconsistent frames")
        for sample in row["samples"]:
            if (len(sample["prefix_sha256"]) != 2 or any(len(value) != 64 for value in sample["prefix_sha256"])
                    or not np.isfinite(sample["offset"] + sample["sensors"]).all()
                    or not np.array_equal(sample["wrist"], wrist_features(archive[str(sample["batch"])]))):
                raise ValueError("wrist features or trajectory provenance changed")


def table(rows):
    samples = [(row["column"], sample) for row in rows for sample in row["samples"]]
    return (np.array([(column - 1) * 2 + sample["batch"] for column, sample in samples], dtype=int),
            np.array([sample["sensors"] for _, sample in samples]), np.array([sample["wrist"] for _, sample in samples]),
            np.array([sample["offset"] for _, sample in samples]))


def fit(rows):
    fit_end = SPLIT_SIZES[0]
    choice_end = sum(SPLIT_SIZES[:2])
    group, sensors_fit, wrist_fit, labels = table(rows[:fit_end])
    choice_group, sensors_choice, wrist_choice, expected_choice = table(rows[fit_end:choice_end])
    audit_group, sensors_audit, wrist_audit, expected_audit = table(rows[choice_end:])
    if (np.bincount(group, minlength=8).min() < 3 or np.bincount(choice_group, minlength=8).min() < 1
            or np.bincount(audit_group, minlength=8).min() < 1):
        return None, {"status": "insufficient_held_batches_by_column"}
    mean = np.stack([labels[group == index].mean(axis=0) for index in range(8)])
    residual = labels - mean[group]
    choices = []
    for name, train, choice, holdout in (("eef", sensors_fit, sensors_choice, sensors_audit),
            ("wrist", np.c_[sensors_fit, wrist_fit], np.c_[sensors_choice, wrist_choice], np.c_[sensors_audit, wrist_audit])):
        center, scale = train.mean(axis=0), np.maximum(.05, train.std(axis=0))
        normalized = np.clip((train - center) / scale, -4., 4.)
        for ridge in PENALTIES:
            coeff = np.linalg.solve(normalized.T @ normalized + ridge * np.eye(len(center)), normalized.T @ residual)
            estimate = mean[choice_group] + np.clip((choice - center) / scale, -4., 4.) @ coeff
            errors = np.abs(estimate - expected_choice)
            parameters = {"center": center, "scale": scale, "coeff": coeff, "mean": mean}
            predicted = mean[audit_group] + np.clip((holdout - center) / scale, -4., 4.) @ coeff
            choices.append((name, ridge, float(errors.mean()), parameters, predicted))
    eef, wrist = (min((item for item in choices if item[0] == name), key=lambda item: item[2]) for name in ("eef", "wrist"))
    base_errors = np.abs(eef[4] - expected_audit)
    wrist_errors = np.abs(wrist[4] - expected_audit)
    base_p80, wrist_p80 = np.quantile(base_errors, .8, axis=0), np.quantile(wrist_errors, .8, axis=0)
    active = (wrist_errors.mean(axis=0) < .9 * base_errors.mean(axis=0)) & (wrist_p80 <= base_p80) & (wrist_p80 <= .004)
    report = {"choices": [{"features": name, "ridge": ridge, "mae_m": loss} for name, ridge, loss, _, _ in choices],
              "counts": [len(labels), len(expected_choice), len(expected_audit)],
              "audit_eef_mae_mm": (1000. * base_errors.mean(axis=0)).tolist(), "audit_wrist_mae_mm": (1000. * wrist_errors.mean(axis=0)).tolist(),
              "audit_eef_p80_mm": (1000. * base_p80).tolist(), "audit_wrist_p80_mm": (1000. * wrist_p80).tolist(),
              "active": active.tolist()}
    if not active.any():
        return None, report
    return {**wrist[3], "active": active, "deadband": np.maximum(.0015, wrist_p80)}, report


@dataclass(frozen=True)
class WristCorrection:
    column: int
    batch: int
    checkpoint: str
    digest: str

    def record(self):
        return [self.column, self.batch, self.checkpoint, self.digest]


class WristPlacementPolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

        self.adapters = {(item.column, item.batch) for item in corrections}
        self.model_wrist = None
        if corrections:
            (artifact,) = {(item.checkpoint, item.digest) for item in corrections}
            path, digest = artifact
            if sha256(Path(path)) != digest:
                raise ValueError("wrist model differs from the frozen fit")
            with np.load(path, allow_pickle=False) as archive:
                self.model_wrist = dict(archive)
        slots = np.asarray(make_cookie_benchmark(task="2x5").config.cookie_transfer.target_slots_local_m)
        self.goals = np.stack([slots[slots[:, 0] == position].mean(axis=0) for position in sorted(set(slots[:, 0]), reverse=True)])
        super().__init__(checkpoint, focus, scale)

    @property
    def camera_names(self):
        if self.adapters and self.cursor is not None and self.cursor[1:] == (PLACE, 0):
            return ("left_wrist",)
        return super().camera_names

    def act(self, observation, task=""):
        if self.adapters and self.cursor is not None and self.cursor[1:] == (PLACE, 0):
            column, batch = task_column(task), self.cursor[0]
            if (column, batch) in self.adapters:
                model = self.model_wrist
                readings, tool_xy, _ = sensors(self, observation, batch)
                features = np.r_[readings, wrist_features(observation[WRIST])]
                offset = model["mean"][(column - 1) * 2 + batch] + np.clip((features - model["center"]) / model["scale"], -4., 4.) @ model["coeff"]
                predicted = (tool_xy - self.pose[2:4]) @ basis(self.pose[4]) + offset
                error = self.goals[batch] - predicted
                correction = np.sign(error) * np.clip(np.abs(error) - model["deadband"], 0., MAX_SHIFT) * model["active"]
                if correction.any():
                    self.shift_placement(batch, np.r_[correction, 0.])
        return super().act(observation, task)


def run(root, reference, timing_run, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("wrist acquisition requires balanced seeds and verified scoring")
    training = list(range(block, block + sum(SPLIT_SIZES)))
    development, confirmation = list(range(block + 100, block + 120)), list(range(block + 200, block + 240))
    schedule = {"scout": {"random": []}, "training": training, "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, schedule)
    _record_once(root / "protocol.json", {**identity, "schedule": schedule, "split_sizes": list(SPLIT_SIZES),
                 "sensing": [WRIST, "observation.eef_pose"], "code_sha256": sha256(Path(__file__)),
                 "parity_sha256": sha256(parity), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "kinematics_adapter_sha256": sha256(Path(tool_xyz.__code__.co_filename)),
                 "target_mean_strictly_above": placement_search.TARGET})
    print(f"Reserved {len(training)} advanced rollouts with wrist observations at both transfer endpoints", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(checkpoint, *timing, root / "training" / "frames", seed) for seed in training]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]), audit)
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"rows": rows})
        arrays, evidence = fit(rows)
        artifact = root / "training" / "wrist.npz"
        model = ()
        if arrays:
            if not artifact.is_file():
                _npz(artifact, **arrays)
            else:
                with np.load(artifact, allow_pickle=False) as archive:
                    if set(archive.files) != set(arrays) or any(not np.array_equal(archive[key], value) for key, value in arrays.items()):
                        raise ValueError("wrist checkpoint differs from the fitted model")
            digest = sha256(artifact)
            model = tuple(WristCorrection(column, batch, str(artifact), digest) for column in range(1, 5) for batch in range(2))
        _record_once(root / "fit.json", {**evidence, "dataset_sha256": sha256(dataset), "corrections": placement_search.encode(model)})
        print(f"Completed wrist geometry audit; admitted axes {evidence.get('active')}", flush=True)
        if not model:
            return {"status": "reference_retained_after_wrist_audit"}
        rows = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                            {"reference": (), "selected": model}, policy_type=WristPlacementPolicy)
        dev = {rule: placement_search.compare([row[rule] for row in rows["reference"]],
                                               [row[rule] for row in rows["selected"]], model)
               for rule in ("default", "strict-500")}
        _record_once(root / "development-selection.json", dev)
        if not all(placement_search.passes(stats) for stats in dev.values()):
            return {"status": "reference_retained_after_development", "development": dev}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": placement_search.encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                    confirmation[:4], policy_type=WristPlacementPolicy)
    primary = checked["default"]
    accepted = primary["mean"] > placement_search.TARGET and all(placement_search.passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation", "development": dev,
            "confirmation": checked, "active_policy_above_target": accepted,
            "target_lower_bound": primary["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Acquire wrist geometry samples and verify observable alignment feedback.")
    for name in ("out", "reference", "timing", "parity"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-block", type=int, default=131000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
