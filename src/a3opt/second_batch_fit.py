from __future__ import annotations

from collections import Counter

import numpy as np

from a3opt.backends.second_batch import MAX_OFFSET_M, ReleaseRecipe
from a3opt.difficulty_eval import _bootstrap


def _scored(cookies: list[dict]) -> int:
    return sum(cookie["scored"] for cookie in cookies)


def _placement_ready(row: dict) -> bool:
    before = row["placement_events"].get("before_descent")
    return (before is not None and _scored(before["batches"][0]) == 5
            and row["batches"][1]["retained_through_transfer"] == 5
            and not row["safety_reason"] and row["score"] == row["requested_score"])


def failure_stage(row: dict) -> str:
    if row["safety_reason"] or row["score"] != row["requested_score"]:
        return "safety_or_source"
    if row["success"]:
        return "complete"
    events = row["placement_events"]
    if not _placement_ready(row):
        return "before_second_descent"
    if "before_open" not in events:
        return "before_second_open"
    first, second = events["terminal"]["batches"]
    if _scored(first) < 5:
        return "first_batch_disturbed"
    if _scored(second) < 5:
        return "second_batch_shortfall"
    return "slot_or_stability"


def prioritize(rows: list[dict]) -> tuple[list[int], dict]:
    stages = {column: Counter(failure_stage(row) for row in rows if row["column"] == column)
              for column in range(1, 5)}
    actionable = ("before_second_open", "first_batch_disturbed", "second_batch_shortfall", "slot_or_stability")
    counts = {column: sum(counts[name] for name in actionable) for column, counts in stages.items()}
    order = sorted((column for column, count in counts.items() if count), key=lambda column: (-counts[column], column))
    return order, {str(column): dict(counts) for column, counts in stages.items()}


def fit_release(rows: list[dict]) -> tuple[dict[str, ReleaseRecipe], dict]:
    samples = []
    rejected = []
    for row in rows:
        key = {"suite": row["suite"], "seed": row["seed"], "stage": failure_stage(row)}
        event = row["placement_events"].get("before_open")
        cookies = event["batches"][1] if event is not None else []
        if not _placement_ready(row) or len(cookies) != 5:
            rejected.append({**key, "reason": "no five-Cookie release label"})
            continue
        positions = np.array([cookie["target_xyz_m"] for cookie in cookies])
        pad_contacts = np.array([cookie["pad_contacts"] for cookie in cookies])
        center = positions.mean(axis=0)
        error = np.asarray(row["second_batch_goal_m"]) - center
        if not pad_contacts.any(axis=0).all() or np.any(np.abs(error) > .05):
            rejected.append({**key, "reason": "release group lacks two-sided pad contact or is far from the slot column"})
            continue
        samples.append({**key, "label_step": event["step"], "cookie_ids": [cookie["id"] for cookie in cookies],
                        "center_m": center.tolist(), "error_m": error.tolist(), "action_sha256": row["action_sha256"]})
    failures = [sample for sample in samples if sample["stage"] != "complete"]
    fit = failures if len(failures) >= 2 else samples
    fitted = np.zeros(3)
    if len(fit) >= 2:
        fitted = np.clip(np.median([sample["error_m"] for sample in fit], axis=0), -MAX_OFFSET_M, MAX_OFFSET_M)
    proposals = {
        "open-hold": ReleaseRecipe(open_hold=6),
        "clearance": ReleaseRecipe((0., 0., .006)),
    }
    if np.linalg.norm(fitted) >= .001:
        proposals["alignment"] = ReleaseRecipe(tuple(fitted))
        proposals["alignment-hold"] = ReleaseRecipe(tuple(fitted), open_hold=6)
    return proposals, {"samples": samples, "rejected": rejected, "fitted_labels": len(fit),
                       "fitted_offset_m": fitted.tolist()}


def paired_comparison(baseline: list[dict], candidate: list[dict]) -> dict:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("paired comparison requires a complete nonempty baseline and candidate")
    columns, counts, successes = [], [], []
    candidate_violations = 0
    for base, trial in zip(baseline, candidate, strict=True):
        matched = ("suite", "seed", "profile", "column", "prefix_steps", "prefix_sha256",
                   "initial_pose", "estimated_pose", "cameras", "light_diffuse", "color")
        if any(base[key] != trial[key] for key in matched):
            raise ValueError("seed, randomization, or pre-placement actions differ within an episode pair")
        columns.append(base["column"])
        counts.append(trial["requested_score"] - base["requested_score"])
        successes.append(int(trial["success"] and trial["requested_score"] == 10)
                         - int(base["success"] and base["requested_score"] == 10))
        candidate_violations += bool(trial["safety_reason"] or trial["score"] != trial["requested_score"])
    return {"n": len(baseline), "score_delta": float(np.mean(counts)), "full_success_delta": sum(successes),
            "score_delta_ci95": _bootstrap(np.asarray(counts), columns),
            "full_success_rate_delta_ci95": _bootstrap(np.asarray(successes), columns),
            "better_score": sum(count > 0 for count in counts), "worse_score": sum(count < 0 for count in counts),
            "candidate_violations": candidate_violations,
            "baseline_stages": dict(Counter(map(failure_stage, baseline))),
            "candidate_stages": dict(Counter(map(failure_stage, candidate)))}


def nonregression(comparison: dict) -> bool:
    return (comparison["candidate_violations"] == 0 and comparison["score_delta"] >= 0
            and comparison["full_success_delta"] >= 0)


def improvement(comparison: dict) -> bool:
    return nonregression(comparison) and (comparison["score_delta"] > 0 or comparison["full_success_delta"] > 0)
