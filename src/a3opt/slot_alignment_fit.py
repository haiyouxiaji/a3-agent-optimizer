from __future__ import annotations

import numpy as np

from a3opt.backends.slot_alignment import FEATURE_COUNT, predict_shift
from a3opt.difficulty_eval import RULES


SLOT_TOLERANCE = np.asarray(RULES["slot_tolerance_m"])
FEATURE_GROUPS = ((), (5, 6), tuple(range(5)), tuple(range(FEATURE_COUNT)))


def first_row_aligned(row: dict) -> bool:
    snapshot = row.get("first_after_retract")
    if snapshot is None:
        return False
    return {cookie["slot"] for cookie in snapshot["batches"][0]} == set(row["front_slots"])


def first_slot_miss(row: dict) -> bool:
    snapshots = row["placement_events"]
    pre_place = snapshots.get("before_descent")
    if row["requested_score"] != 10 or row["success"] or pre_place is None:
        return False
    first = pre_place["batches"][0]
    terminal = snapshots["terminal"]["batches"]
    if len(first) != 5 or any(not cookie["scored"] for cookie in first):
        return False
    return (any(cookie["slot"] is None for cookie in first)
            and any(cookie["slot"] is None for cookie in terminal[0])
            and len(terminal[1]) == 5 and all(cookie["slot"] is not None for cookie in terminal[1]))


def _fit(features: np.ndarray, labels: np.ndarray, mask: tuple[int, ...], ridge: float, column: int) -> dict:
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-5)
    weights = np.zeros(FEATURE_COUNT)
    bias = float(np.median(labels))
    if mask:
        design = np.clip(((features - center) / scale)[:, mask], -3., 3.)
        design = np.column_stack((np.ones(len(design)), design))
        penalty = np.diag([0., *([ridge] * len(mask))])
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ labels)
        bias = float(coefficients[0])
        weights[list(mask)] = coefficients[1:]
    return {"center": center, "scale": scale, "weights": weights, "bias": bias, "column": column}


def _metrics(errors: np.ndarray) -> dict:
    worst = np.max(np.abs(errors), axis=1)
    return {"rows_within_y_tolerance": int(np.count_nonzero(worst <= SLOT_TOLERANCE[1])),
            "mean_worst_y_error_mm": float(1000 * worst.mean())}


def _rank(metrics: dict) -> tuple:
    return metrics["rows_within_y_tolerance"], -metrics["mean_worst_y_error_mm"]


def fit_alignment(rows: list[dict], column: int) -> tuple[dict | None, dict]:
    samples, rejected = [], []
    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(seeds) or any(row["column"] != column for row in rows):
        raise ValueError("fit requires one independently sampled episode per seed in the selected column")
    for row in rows:
        snapshot = row["first_after_retract"]
        cookies = snapshot["batches"][0] if snapshot is not None else []
        if (len(cookies) != 5 or not all(cookie["scored"] for cookie in cookies) or row["score"] != row["requested_score"]
                or row["safety_reason"]):
            rejected.append({"seed": row["seed"], "reason": "first batch did not settle with five correct cookies"})
            continue
        positions = np.asarray([cookie["target_xyz_m"] for cookie in cookies])[:, :2]
        slots = np.asarray(row["front_slot_xy_m"])
        errors = np.sort(positions[:, 1]) - np.sort(slots[:, 1])
        if np.any(np.abs(positions[:, 0] - slots[:, 0]) > SLOT_TOLERANCE[0]) or np.ptp(errors) > .008:
            rejected.append({"seed": row["seed"], "reason": "out-of-column or nonrigid group"})
            continue
        samples.append({"seed": row["seed"], "cookie_ids": [cookie["id"] for cookie in cookies],
                        "features": row["first_release_features"], "residuals_y_m": errors.tolist(),
                        "label_shift_m": float(-np.median(errors)), "label_step": snapshot["step"],
                        "action_sha256": row["action_sha256"]})
    dataset = {"samples": samples, "rejected": rejected}
    if len(samples) < 8:
        return None, dataset
    features = np.asarray([sample["features"] for sample in samples])
    labels = np.asarray([sample["label_shift_m"] for sample in samples])
    errors = np.asarray([sample["residuals_y_m"] for sample in samples])
    baseline = _metrics(errors)
    comparisons = []
    indices = np.arange(len(samples))
    for mask in FEATURE_GROUPS:
        for ridge in ((0.,) if not mask else (1., 4., 16.)):
            predicted = []
            for held in indices:
                train = indices != held
                model = _fit(features[train], labels[train], mask, ridge, column)
                predicted.append(float(predict_shift(model, features[held])))
            metrics = _metrics(errors + np.asarray(predicted)[:, None])
            comparisons.append({"feature_indices": list(mask), "ridge": ridge, **metrics})
    winner = max(comparisons, key=_rank)
    dataset.update(validation={"method": "leave_one_seed_out", "baseline": baseline, "alternatives": comparisons,
                               "selected": winner if _rank(winner) > _rank(baseline) else None})
    if dataset["validation"]["selected"] is None:
        return None, dataset
    return _fit(features, labels, tuple(winner["feature_indices"]), winner["ridge"], column), dataset
