from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_pooled, placement_search, placement_tail, placement_visual_feedback
from a3opt.advanced_agent import _record_once
from a3opt.backends.adaptive_phase import PLACE, task_column
from a3opt.backends.advanced_trajectory import visual_features
from a3opt.backends.trajectory import FRONT
from a3opt.cli import _run_lock
from a3opt.placement_agent import reference_setup
from a3opt.placement_sim import PlacementPolicy
from a3opt.refinement_eval import read_json, sha256


WIDTH = 48
CELLS = 4
RIDGE = .75
FEATURES = 2 * CELLS**2
CAPTURE = placement_visual_feedback.CAPTURE


def rectifier(image):
    import cv2

    corners = visual_features(image)[0][:8].reshape(4, 2).astype(np.float32)
    if cv2.contourArea(corners) < 100:
        return None
    center = corners.mean(axis=0)
    interior = (center + .86 * (corners - center)).astype(np.float32)
    edge = WIDTH - 1
    destination = np.array([[0, 0], [edge, 0], [edge, edge], [0, edge]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(interior, destination)
    return matrix if np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-8 else None


def tray_chroma(image, matrix):
    import cv2

    pixels = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    rectified = cv2.warpPerspective(pixels, matrix, (WIDTH, WIDTH)).astype(float)
    return np.ptp(rectified, axis=-1) / np.maximum(40., rectified.max(axis=-1))


def describe(initial, observed):
    import cv2

    delta = np.maximum(0., observed - initial)
    cells = cv2.resize(delta, (CELLS, CELLS), interpolation=cv2.INTER_AREA).ravel()
    return np.r_[cells, cells / max(.04, float(cells.mean()))]


@dataclass(frozen=True)
class GeometryPlacement:
    column: int
    recipe: str
    center: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    intercept: float

    def __post_init__(self):
        if (self.column not in range(1, 5) or self.recipe not in placement_tail.OFFSETS or self.recipe == "reference"
                or any(len(values) != FEATURES or not np.isfinite(values).all() for values in (self.center, self.scale, self.weights))
                or min(self.scale) <= 0. or not np.isfinite(self.intercept)):
            raise ValueError("invalid visual geometry checkpoint")

    @property
    def batch(self):
        return 1

    def record(self):
        return [self.column, self.recipe, list(self.center), list(self.scale), list(self.weights), self.intercept]


def prediction(features, parameters):
    center, scale, weights, intercept = parameters
    return float(np.clip((features - center) / scale, -3., 3.) @ weights + intercept)


class GeometryPolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.models = {item.column: item for item in corrections}
        if len(self.models) != len(corrections):
            raise ValueError("duplicate visual geometry policy")
        super().__init__(checkpoint, focus, scale)

    def reset(self, context=None):
        super().reset(context)
        self.matrix = self.initial_chroma = self.risk = None

    @property
    def camera_names(self):
        return ("front",) if self.actions is None or (self.matrix is not None and self.cursor == CAPTURE) else ()

    def act(self, observation, task=""):
        item = self.models.get(task_column(task))
        if item is not None:
            if self.actions is None:
                self.matrix = rectifier(observation[FRONT])
                if self.matrix is not None:
                    self.initial_chroma = tray_chroma(observation[FRONT], self.matrix)
            elif self.matrix is not None and self.cursor == CAPTURE:
                current = tray_chroma(observation[FRONT], self.matrix)
                self.risk = prediction(describe(self.initial_chroma, current), (item.center, item.scale, item.weights, item.intercept))
            elif self.cursor == (1, PLACE, 0) and self.risk is not None and self.risk > .5:
                self.shift_placement(1, np.asarray(placement_tail.OFFSETS[item.recipe]))
        return super().act(observation, task)


def regressor(scenes):
    rows = [scene for scene in scenes if scene["label"] is not None]
    labels = np.asarray([row["label"] for row in rows], dtype=float)
    counts = np.bincount(labels.astype(int), minlength=2)
    if counts.min() < 2:
        return None
    features = np.stack([row["features"] for row in rows])
    center, scale = features.mean(axis=0), np.maximum(features.std(axis=0), .02)
    design = np.column_stack((np.clip((features - center) / scale, -3., 3.), np.ones(len(rows))))
    sample_weights = (len(rows) / (2. * counts))[labels.astype(int)]
    weighted = design.T * sample_weights
    penalty = np.diag([*([RIDGE] * FEATURES), 0.])
    params = np.linalg.solve(weighted @ design + penalty, weighted @ labels)
    return center, scale, params[:-1], float(params[-1])


def learn(scenes, outcomes):
    model = regressor(scenes)
    scores = []
    for scene in scenes:
        seed = scene["seed"]
        heldout = regressor([row for row in scenes if row["seed"] != seed])
        scores.append({"seed": seed, "column": scene["column"], "label": scene["label"],
                       "full_risk": prediction(scene["features"], model) if model else None,
                       "out_of_fold_risk": prediction(scene["features"], heldout) if heldout else None})

    options = {}
    for recipe in placement_tail.OFFSETS:
        if recipe == "reference":
            continue
        selected = []
        for row in scores:
            recipes = outcomes[row["seed"]].get("fork")
            full, heldout = row["full_risk"], row["out_of_fold_risk"]
            oof_trigger = heldout is not None and heldout > .5
            full_trigger = full is not None and full > .5
            if recipes is None or not (oof_trigger or full_trigger):
                continue
            original, alternative = (recipes["outcomes"][name]["physics"] for name in ("reference", recipe))
            delta = len(alternative["counted_ids"]) - len(original["counted_ids"])
            violation = bool(recipes["outcomes"][recipe]["wrong_source"] or alternative["safety_reason"]
                             or alternative["source_remaining"] != original["source_remaining"]
                             or alternative["filled_slots"] < original["filled_slots"])
            selected.append({"seed": row["seed"], "column": row["column"], "delta": delta, "violation": violation,
                             "full_trigger": full_trigger, "oof_trigger": oof_trigger})
        options[recipe] = selected
    eligible = [name for name, rows in options.items() if not any(row["delta"] < 0 or row["violation"] for row in rows)
                and len({row["column"] for row in rows if row["delta"] > 0 and row["oof_trigger"] and row["full_trigger"]}) >= 2]
    winner = max(eligible, key=lambda name: (sum(row["delta"] for row in options[name] if row["full_trigger"]),
                                              -sum(map(abs, placement_tail.OFFSETS[name]))), default=None)
    if winner is None:
        return (), {"chosen_recipe": None, "scenes": scores, "options": options}
    center, scale, weights, intercept = model
    corrections = tuple(GeometryPlacement(column, winner, tuple(center.tolist()), tuple(scale.tolist()), tuple(weights.tolist()), intercept)
                        for column in range(1, 5))
    return corrections, {"chosen_recipe": winner, "scenes": scores, "options": options}


def dataset(acquisition, receipt, checkpoint, timing, expected):
    dataset_path = acquisition / "training" / "dataset.json"
    shards = acquisition / "training" / "frames"
    rows = read_json(dataset_path)["rows"]
    if (read_json(acquisition / "fit.json")["dataset_sha256"] != sha256(dataset_path)
            or [row["seed"] for row in rows] != list(expected) or receipt["schedule"]["training"] != list(expected)):
        raise ValueError("visual geometry requires the complete recorded training acquisition")
    scenes = []
    for row in rows:
        seed = row["seed"]
        placement_visual_feedback.audit((checkpoint, *timing, shards, seed), row, expected)
        if row["image_sha256"] is None:
            continue
        with np.load(shards / f"{seed}.npz", allow_pickle=False) as frames:
            matrix = rectifier(frames["initial"])
            if matrix is None:
                continue
            features = describe(tray_chroma(frames["initial"], matrix), tray_chroma(frames["observed"], matrix))
        count = row["post_first_count"]
        label = 1 if count <= 1 else 0 if count >= 4 else None
        scenes.append({"seed": seed, "column": row["column"], "features": features, "label": label})
    return scenes, sha256(dataset_path)


def run(root, reference, timing_run, source, acquisition, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and verified joint evaluator are required")
    development, confirmation = list(range(block, block + 12)), list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    pairs, provenance = placement_pooled.data(source, identity, checkpoint, timing, development + confirmation)
    expected = {row["reference"]["seed"]: row["reference"] for row in pairs}
    prior_path = acquisition / "protocol.json"
    prior = read_json(prior_path)
    previous = prior["schedule"]
    if (any(prior[key] != identity[key] for key in ("environment", "reference_sha256", "timing", "strict_rules"))
            or prior["training"] != provenance or prior["code_sha256"] != sha256(Path(placement_visual_feedback.__file__))
            or set(previous["development"] + previous["confirmation"]["random"]).intersection(development + confirmation)):
        raise ValueError("visual geometry requires compatible acquisition and unused evaluation seeds")
    splits["training"] = list(expected)
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    scenes, dataset_hash = dataset(acquisition, prior, checkpoint, timing, expected)
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "collection_sha256": sha256(prior_path),
                 "dataset_sha256": dataset_hash, "training": provenance, "feature_cells": CELLS, "ridge": RIDGE,
                 "code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "parity_sha256": sha256(parity), "target_mean_strictly_above": placement_search.TARGET})
    model, report = learn(scenes, expected)
    _record_once(root / "fit.json", {**report, "corrections": placement_search.encode(model)})
    print(f"Compared geometric feedback on {len(scenes)} image pairs; selected {report['chosen_recipe']}", flush=True)
    if not model:
        return {"status": "reference_retained_after_training", "training_pairs": len(scenes)}
    models = {"reference": (), "selected": model}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        observed = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development, models,
                                                policy_type=GeometryPolicy)
        dev = placement_visual_feedback.paired_stats(observed, model)
        _record_once(root / "development-selection.json", dev)
        if not all(placement_search.passes(stats) for stats in dev.values()):
            return {"status": "reference_retained_after_development", "development": dev}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": placement_search.encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                   confirmation[:4], policy_type=GeometryPolicy)
    score = checked["default"]
    accepted = score["mean"] > placement_search.TARGET and all(placement_search.passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation", "development": dev,
            "confirmation": checked, "active_policy_above_target": accepted,
            "target_point_estimate": score["mean"] > placement_search.TARGET,
            "target_lower_bound": score["mean_ci95"][0] > placement_search.TARGET}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_PLACEMENT_GEOMETRY_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("visual geometry requires the frozen policy checkpoint")
    models = tuple(GeometryPlacement(column, recipe, tuple(center), tuple(scale), tuple(weights), intercept)
                   for column, recipe, center, scale, weights, intercept in lock["corrections"])
    return GeometryPolicy(checkpoint, *lock["timing"], models)


def main():
    parser = argparse.ArgumentParser(description="Fit a spatial placement feedback gate from audited RGB image pairs.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--seed-block", type=int, default=128000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.source.resolve(), args.acquisition.resolve(),
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
