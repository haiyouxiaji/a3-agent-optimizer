from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from a3opt import advanced_sim, placement_benchmark, placement_search
from a3opt.advanced_agent import _npz, _record_once
from a3opt.backends import advanced_trajectory
from a3opt.backends.adaptive_phase import load_checkpoint, task_column
from a3opt.backends.trajectory import FRONT
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import _bootstrap
from a3opt.placement_agent import reference_setup
from a3opt.placement_sim import PlacementPolicy
from a3opt.refinement_eval import read_json, render_frames, sha256


VIEWS = ("front", "left_wrist")
WRIST = "observation.images.left_wrist"
POSE_LIMIT = np.array([.003, .003, .003, .003, .020])
GROUPS = {"source_xy": slice(0, 2), "target_xy_yaw": slice(2, 5)}
PENALTIES = (.3, 1., 3., 10.)
BLENDS = (0., .5, 1.)
FIT_ROWS, CHOICE_ROWS, AUDIT_ROWS = 240, 60, 60


def wrist_features(image):
    import cv2

    pixels = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)[:80].astype(np.float32)
    peak = pixels.max(axis=-1)
    neutral = (peak > 54.) & (np.ptp(pixels, axis=-1) < .30 * peak)
    return cv2.resize(neutral.astype(np.float32), (24, 16), interpolation=cv2.INTER_AREA).ravel()


class CalibrationProbe(advanced_sim.Probe):
    def bind_env(self, env):
        images = render_frames(env, VIEWS)
        self.row = (*advanced_trajectory.visual_features(images[FRONT]), wrist_features(images[WRIST]),
                    advanced_trajectory.pose_label(env.randomization_metadata), env.randomization_metadata["cookie_color"])


def collect(seeds):
    import cv2

    cv2.setNumThreads(1)
    engine, probe = advanced_sim.benchmark("random"), CalibrationProbe()
    env = engine.create_env(render_cameras=False)
    try:
        rows = []
        for seed in seeds:
            result = engine.run_episode(probe, seed=seed, env=env)
            if result.randomization["profile"] != "advanced" or result.steps != 1:
                raise ValueError("calibration requires one initial observation per advanced reset")
            rows.append((seed, *probe.row))
        return rows
    finally:
        env.close()


def design(outline, silhouette, wrist, center, scale, multiview):
    contours = np.clip((outline - center) / scale, -6., 6.)
    parts = (contours, 3. * silhouette, 2. * wrist) if multiview else (contours, 3. * silhouette)
    return np.concatenate(parts, axis=-1)


def bounded_pose(reference, predicted, blend):
    return reference + np.clip(predicted - reference, -POSE_LIMIT, POSE_LIMIT) * blend


def errors(predicted, labels, group):
    absolute = np.abs(predicted[:, group] - labels[:, group]) / advanced_trajectory.POSE_SCALE[group]
    return np.array([absolute.mean(), np.quantile(absolute.mean(axis=1), .9)])


def fit(data, reference, fit_rows, choice_rows):
    outline, silhouette, wrist, labels = (data[key] for key in ("outline", "silhouette", "wrist", "labels"))
    fit_slice = slice(None, fit_rows)
    choice = slice(fit_rows, fit_rows + choice_rows)
    audit = slice(choice.stop, None)
    center, scale = outline[fit_slice].mean(axis=0), np.maximum(1., outline[fit_slice].std(axis=0))
    mean_pose = labels[fit_slice].mean(axis=0)
    labels_centered = labels[fit_slice] - mean_pose
    incumbent = np.stack([advanced_trajectory.predict_pose(reference, *views)
                          for views in zip(outline, silhouette, strict=True)])
    choices, models = [], []
    for multiview in (False, True):
        features = design(outline, silhouette, wrist, center, scale, multiview)
        mean = features[fit_slice].mean(axis=0)
        features -= mean
        left, singular, right = np.linalg.svd(features[fit_slice], full_matrices=False)
        for ridge in PENALTIES:
            weights = (right.T * (singular / (singular**2 + ridge))) @ (left.T @ labels_centered)
            predictions = features @ weights + mean_pose
            blend, score = np.zeros(5), 0.
            for group in GROUPS.values():
                ranks = [(errors(bounded_pose(incumbent[choice], predictions[choice], value), labels[choice], group)[0], value)
                         for value in BLENDS]
                loss, blend[group] = min(ranks)
                score += loss
            choices.append({"views": list(VIEWS if multiview else VIEWS[:1]), "ridge": ridge, "choice_loss": score,
                            "blend": blend.tolist()})
            models.append((multiview, mean, weights, predictions, blend))
    winner = min(range(len(choices)), key=lambda index: choices[index]["choice_loss"])
    multiview, mean, weights, predictions, blend = models[winner]
    combined = bounded_pose(incumbent[audit], predictions[audit], blend)
    validation = {}
    for name, group in GROUPS.items():
        old = errors(incumbent[audit], labels[audit], group)
        new = errors(combined, labels[audit], group)
        keep = bool(new[0] < .95 * old[0] and new[1] <= old[1])
        if not keep:
            blend[group] = 0.
        validation[name] = {"reference": old.tolist(), "candidate": new.tolist(), "keep": keep}
    return {"multiview": multiview, "center": center, "scale": scale, "mean": mean, "weights": weights,
            "mean_pose": mean_pose, "blend": blend}, {"choices": choices, "winner": choices[winner], "audit": validation,
                                                    "deployment_blend": blend.tolist()}


@dataclass(frozen=True)
class Calibration:
    checkpoint: str
    digest: str

    def record(self):
        return [self.checkpoint, self.digest]


class CalibrationPolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.calibration = None
        if corrections:
            (item,) = corrections
            path = Path(item.checkpoint)
            if sha256(path) != item.digest:
                raise ValueError("pose calibrator differs from the frozen checkpoint")
            with np.load(path, allow_pickle=False) as archive:
                self.calibration = dict(archive)
        super().__init__(checkpoint, focus, scale)

    @property
    def camera_names(self):
        if self.actions is None and self.calibration is not None and self.calibration["multiview"].item():
            return VIEWS
        return super().camera_names

    def act(self, observation, task=""):
        if self.actions is None and self.calibration is not None:
            model = self.calibration
            outline, silhouette = advanced_trajectory.visual_features(observation[FRONT])
            wrist = wrist_features(observation[WRIST]) if model["multiview"].item() else None
            features = design(outline, silhouette, wrist, model["center"], model["scale"], wrist is not None)
            pose = advanced_trajectory.predict_pose(self.model, outline, silhouette)
            self.pose = bounded_pose(pose, (features - model["mean"]) @ model["weights"] + model["mean_pose"], model["blend"])
            column = task_column(task)
            self.recipe = self.recipes[column - 1]
            self.actions = self._phase_actions(column, self.recipe)
            self.cursor = (0, 0, 0)
        return super().act(observation, task)


def compare(reference, selected, model):
    if not model or not reference or len(reference) != len(selected):
        raise ValueError("calibration needs complete paired rollouts")
    keys = ("seed", "column", "profile", "suite", "strict", "task_prompt", "initial_pose", "cameras", "light_diffuse", "color")
    for old, new in zip(reference, selected, strict=True):
        drift = np.abs(np.asarray(new["estimated_pose"]) - old["estimated_pose"])
        if any(old[key] != new[key] for key in keys) or not np.all(drift <= POSE_LIMIT + 1e-12):
            raise ValueError("paired reset or calibrated pose is outside the locked contract")
        if not drift.any() and any(old[key] != new[key] for key in ("action_sha256", "applied_action_sha256")):
            raise ValueError("an unchanged pose must reproduce the reference commands")
    columns = np.array([row["column"] for row in selected])
    old = np.array([row["requested_score"] for row in reference])
    new = np.array([row["requested_score"] for row in selected])
    delta = new - old
    complete = np.array([row["success"] for row in selected], dtype=int)
    reference_complete = np.array([row["success"] for row in reference], dtype=int)
    return {"episodes": len(new), "reference_mean": float(old.mean()), "mean": float(new.mean()),
            "mean_ci95": _bootstrap(new, columns), "paired_count_delta": float(delta.mean()),
            "paired_count_ci95": _bootstrap(delta, columns), "reference_complete": int(reference_complete.sum()),
            "complete": int(complete.sum()), "paired_success_rate_ci95": _bootstrap(complete - reference_complete, columns),
            "major_losses": int(np.sum(delta <= -placement_search.MAJOR_LOSS)),
            "wrong_source": sum(row["score"] != row["requested_score"] for row in selected),
            "safety_stops": sum(bool(row["safety_reason"]) for row in selected),
            "by_column": {str(column): {"episodes": int(np.sum(columns == column)),
                                         "reference_mean": float(old[columns == column].mean()),
                                         "mean": float(new[columns == column].mean())} for column in sorted(set(columns))}}


def run(root, reference, timing_run, previous, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("the calibration run requires balanced seeds and verified scoring")
    training = list(range(block, block + FIT_ROWS + CHOICE_ROWS + AUDIT_ROWS))
    development = list(range(block + 400, block + 416))
    confirmation = list(range(block + 500, block + 540))
    splits = {"training": training, "development": development, "scout": {"random": []}, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    previous_schedule = read_json(previous)["schedule"]
    prior_seeds = {*previous_schedule["training"], *previous_schedule["development"], *previous_schedule["confirmation"]["random"]}
    if prior_seeds.intersection(training + development + confirmation):
        raise ValueError("calibration seeds overlap the prior placement protocol")
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "camera_names": list(VIEWS),
                 "calibration_split_sizes": [FIT_ROWS, CHOICE_ROWS, AUDIT_ROWS], "target_mean_strictly_above": placement_search.TARGET,
                 "prior_protocol_sha256": sha256(previous), "parity_sha256": sha256(parity),
                 "calibration_code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__))})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        dataset = root / "training" / "calibration.npz"
        if not dataset.is_file():
            chunks = [tuple(training[start:start + 12]) for start in range(0, len(training), 12)]
            rows = [row for group in pool.map(collect, chunks) for row in group]
            _npz(dataset, **dict(zip(("seeds", "outline", "silhouette", "wrist", "labels", "colors"), zip(*rows), strict=True)))
        with np.load(dataset, allow_pickle=False) as data:
            if data["seeds"].tolist() != training:
                raise ValueError("calibration sample IDs differ from the locked split")
            arrays, report = fit(data, load_checkpoint(checkpoint).model, FIT_ROWS, CHOICE_ROWS)
        print(f"Audited {len(training)} reset observations; retained blend {report['deployment_blend']}", flush=True)
        calibrated = root / "training" / "calibrator.npz"
        if calibrated.is_file():
            with np.load(calibrated, allow_pickle=False) as archive:
                if set(archive.files) != set(arrays) or any(not np.array_equal(archive[key], value) for key, value in arrays.items()):
                    raise ValueError("resumed calibrator differs from the fitted parameters")
        else:
            _npz(calibrated, **arrays)
        model = (Calibration(str(calibrated), sha256(calibrated)),) if arrays["blend"].any() else ()
        _record_once(root / "fit.json", {**report, "dataset_sha256": sha256(dataset), "corrections": placement_search.encode(model)})
        if not model:
            return {"status": "reference_retained_after_calibration_audit"}
        rows = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                            {"reference": (), "selected": model}, policy_type=CalibrationPolicy)
        dev = {rule: compare([row[rule] for row in rows["reference"]], [row[rule] for row in rows["selected"]], model)
               for rule in ("default", "strict-500")}
        _record_once(root / "development-selection.json", dev)
        if not all(placement_search.passes(stats) for stats in dev.values()):
            return {"status": "reference_retained_after_development", "development": dev}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": placement_search.encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                    confirmation[:4], policy_type=CalibrationPolicy, compare_rows=compare)
    primary = checked["default"]
    accepted = primary["mean"] > placement_search.TARGET and all(placement_search.passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation", "development": dev,
            "confirmation": checked, "active_policy_above_target": accepted,
            "target_lower_bound": primary["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Acquire RGB training labels and independently verify a bounded pose recalibration.")
    for name in ("out", "reference", "timing", "previous", "parity"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed-block", type=int, default=129000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.previous.resolve(), args.parity.resolve(),
                     args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
