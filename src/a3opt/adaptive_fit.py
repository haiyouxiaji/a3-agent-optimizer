from __future__ import annotations

from collections import defaultdict

import numpy as np

from a3opt.backends.adaptive_phase import FEATURES


def _logistic(features: np.ndarray, labels: np.ndarray, penalty: float) -> dict:
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), .001)
    design = np.column_stack((np.ones(len(features)), (features - center) / scale))
    weights = np.zeros(design.shape[1])
    regularizer = np.diag([1e-5, *([penalty] * FEATURES)])
    for _ in range(30):
        probability = 1 / (1 + np.exp(-np.clip(design @ weights, -35., 35.)))
        gradient = design.T @ (probability - labels) / len(labels) + regularizer @ weights
        hessian = (design.T * (probability * (1 - probability))) @ design / len(labels) + regularizer
        delta = np.linalg.solve(hessian, gradient)
        weights -= delta
        if np.linalg.norm(delta) < 1e-7:
            break
    return {"center": center, "scale": scale, "coefficients": weights[1:], "intercept": weights[0]}


def fit_grip(samples: list[dict]) -> tuple[dict, dict]:
    if not samples:
        raise ValueError("no training rollouts reached the grip observation")
    features = np.asarray([sample["features"] for sample in samples], dtype=float)
    labels = np.asarray([sample["complete_grasp"] for sample in samples], dtype=int)
    seeds = np.asarray([sample["seed"] for sample in samples])
    if features.shape != (len(labels), FEATURES) or not np.isfinite(features).all():
        raise ValueError("invalid public-observation features")
    seed_folds = {seed: index % 4 for index, seed in enumerate(np.unique(seeds))}
    folds = np.asarray([seed_folds[seed] for seed in seeds])
    threshold, penalty = 0., .2
    validation = {"samples": len(labels), "scenes": len(seed_folds), "complete_grasps": int(labels.sum()),
                  "heldout_unit": "scene_seed", "choices": [], "selected_gate": None}
    if all(len(np.unique(labels[folds != index])) == 2 and np.any(folds == index) for index in range(4)):
        predictions = {}
        for candidate in (.02, .2, 2.):
            probability = np.zeros(len(labels))
            for index in range(4):
                heldout = folds == index
                model = _logistic(features[~heldout], labels[~heldout], candidate)
                design = (features[heldout] - model["center"]) / model["scale"]
                logits = model["intercept"] + design @ model["coefficients"]
                probability[heldout] = 1 / (1 + np.exp(-np.clip(logits, -35., 35.)))
            loss = float(np.mean((probability - labels) ** 2))
            predictions[candidate] = (loss, probability)
            validation["choices"].append({"penalty": candidate, "brier": loss})
        penalty = min(predictions, key=lambda candidate: predictions[candidate][0])
        _, probability = predictions[penalty]
        gates = []
        for cutoff in (.15, .25, .35, .45, .55, .65):
            alerts = probability < cutoff
            hits = int(np.count_nonzero(alerts & (labels == 0)))
            false_alarms = int(np.count_nonzero(alerts & (labels == 1)))
            if hits >= 3 and hits / (hits + false_alarms) >= .65 and false_alarms / labels.sum() <= .25:
                gates.append((hits, -false_alarms, -cutoff))
        if gates:
            hits, negative_alarms, negative_cutoff = max(gates)
            threshold = -negative_cutoff
            validation["selected_gate"] = {"cutoff": threshold, "failure_alerts": hits, "false_alarms": -negative_alarms}
    validation.update(penalty=penalty, retry_probability=threshold)
    return _logistic(features, labels, penalty), validation


def _safe(rows: list[dict]) -> bool:
    return all(not row["safety_reason"] and row["score"] == row["requested_score"] for row in rows)


def rank_training(rows: dict[str, list[dict]], column: int, *, prioritize_lift: bool = False) -> list[str]:
    groups = {name: [row for row in subset if row["column"] == column] for name, subset in rows.items()}
    base_seeds = {row["seed"] for row in groups["baseline"]}
    if not base_seeds:
        raise ValueError(f"missing training scenes for column {column}")
    rankings = {}
    for name, group in groups.items():
        if not group:
            continue
        if len(group) != len(base_seeds) or {row["seed"] for row in group} != base_seeds:
            raise ValueError(f"incomplete paired training for column {column} and {name}")
        first = [row["batches"][0] for row in group]
        correct = sum(row["requested_score"] for row in group)
        lifted = sum(min(event.get("lifted_requested", 0), 5) for event in first)
        five = sum(row["requested_score"] >= 5 for row in group)
        rankings[name] = (_safe(group), *((lifted, correct) if prioritize_lift else (correct, lifted)),
                          five, name == "baseline", -sum(row["steps"] for row in group))
    return sorted(rankings, key=rankings.get, reverse=True)


def select_columns(evidence: dict[str, list[dict]]) -> list[str]:
    by_column = defaultdict(dict)
    for name, rows in evidence.items():
        for column in range(1, 5):
            subset = {(row["suite"], row["seed"]): row for row in rows if row["column"] == column}
            if not subset or len(subset) != sum(row["column"] == column for row in rows):
                raise ValueError(f"missing or duplicate development scenes for column {column}")
            by_column[column][name] = subset
    selections = []
    for column in range(1, 5):
        group = by_column[column]
        base = group["baseline"]
        winner, winning_rank = "baseline", (0, 0)
        for name, paired in group.items():
            if paired.keys() != base.keys():
                raise ValueError("candidate and baseline development schedules differ")
            if any((row["safety_reason"] and not base[key]["safety_reason"])
                   or row["score"] - row["requested_score"] > base[key]["score"] - base[key]["requested_score"]
                   for key, row in paired.items()):
                continue
            deltas = defaultdict(int)
            for key, row in paired.items():
                deltas[key[0]] += row["requested_score"] - base[key]["requested_score"]
            strict_gain = sum(int(row["success"] and row["requested_score"] == 10)
                              - int(base[key]["success"] and base[key]["requested_score"] == 10)
                              for key, row in paired.items())
            rank = sum(deltas.values()), strict_gain
            if min(deltas.values()) >= 0 and strict_gain >= 0 and rank > winning_rank:
                winner, winning_rank = name, rank
        selections.append(winner)
    return selections
