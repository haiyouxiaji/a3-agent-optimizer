from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from a3opt.adaptive_sim import phase_model
from a3opt.advanced_agent import TARGET, _jobs, _npz, _record_once
from a3opt.backends.adaptive_phase import PLACE, task_column
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import _bootstrap, check_row, run_episode
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_feedback_agent import PARAMETERS, ReleaseFeedbackPolicy, evaluate, observed_release_features, predict_shift
from a3opt.second_batch_agent import frozen_reference, seed_schedule
from a3opt.slot_alignment_agent import SlotTrace, check_slot_episode


ARMS = ("reference", "initial", "feedback")
SIGNALS = {7: "projected_tracking_error_m", 8: "log_wrist_force"}
MIN_LEAF = 6
MAJOR_DROP = 4


def outcomes(rows: list[dict]) -> np.ndarray:
    return np.asarray([(row["requested_score"], row["success"] and row["requested_score"] == 10,
                        bool(row["safety_reason"] or row["score"] != row["requested_score"])) for row in rows], dtype=int)


def objective(values: np.ndarray, reference: np.ndarray) -> tuple[int, int, int]:
    count, success, _ = values.sum(axis=0)
    major_losses = np.count_nonzero(reference[:, 0] - values[:, 0] >= MAJOR_DROP)
    return int(success), int(count), -int(major_losses)


def admissible(values: np.ndarray) -> bool:
    return not values[:, 2].any()


def best_constant(labels: np.ndarray) -> int:
    reference = labels[:, 0]
    candidates = [index for index in range(len(ARMS)) if admissible(labels[:, index])
                  and labels[:, index, 0].sum() >= reference[:, 0].sum()]
    if not candidates:
        raise ValueError("there is no safe fixed controller with acceptable terminal count")
    return max(candidates, key=lambda index: objective(labels[:, index], reference))


def select_arm(rule: dict, features: np.ndarray) -> int:
    signal = rule["signal"]
    if signal is None or not np.isfinite(features[signal]):
        return rule["fallback"]
    return rule["left"] if features[signal] <= rule["threshold"] else rule["right"]


def fit_rule(features: np.ndarray, labels: np.ndarray) -> dict:
    fixed = best_constant(labels)
    reference = labels[:, 0]
    fixed_rank = objective(labels[:, fixed], reference)
    best_rank = fixed_rank
    best = {"signal": None, "fallback": fixed}
    for signal in SIGNALS:
        unique = np.unique(features[:, signal])
        for threshold in (unique[:-1] + unique[1:]) / 2:
            left = features[:, signal] <= threshold
            if min(left.sum(), (~left).sum()) < MIN_LEAF:
                continue
            for low in range(len(ARMS)):
                for high in range(len(ARMS)):
                    if low == high:
                        continue
                    values = labels[np.arange(len(labels)), np.where(left, low, high)]
                    rank = objective(values, reference)
                    if admissible(values) and rank[1] >= fixed_rank[1] and rank[2] >= fixed_rank[2] and rank > best_rank:
                        best_rank = rank
                        best = {"signal": signal, "threshold": float(threshold), "left": low, "right": high, "fallback": fixed}
    return best


def validate_rule(features: np.ndarray, labels: np.ndarray) -> dict:
    routed, fixed = [], []
    for held in range(len(labels)):
        train = np.arange(len(labels)) != held
        rule = fit_rule(features[train], labels[train])
        routed.append(select_arm(rule, features[held]))
        fixed.append(rule["fallback"])
    reference = labels[:, 0]
    fixed_arms = labels[np.arange(len(labels)), fixed]
    routed_arms = labels[np.arange(len(labels)), routed]
    fixed_rank, route_rank = objective(fixed_arms, reference), objective(routed_arms, reference)
    eligible = (admissible(routed_arms) and route_rank[0] > fixed_rank[0]
                and route_rank[1] >= fixed_rank[1] and route_rank[2] >= fixed_rank[2])
    return {"eligible": bool(eligible), "fixed_objective": list(fixed_rank), "routed_objective": list(route_rank),
            "fixed_arms": [ARMS[index] for index in fixed], "routed_arms": [ARMS[index] for index in routed]}


class OutcomeReleasePolicy(ReleaseFeedbackPolicy):
    def __init__(self, checkpoint: Path):
        super().__init__(checkpoint)
        self.column = int(self.model["outcome_column"])
        signal, threshold, left, right, fallback = self.model["outcome_rule"]
        self.rule = {"signal": int(signal), "threshold": float(threshold), "left": int(left), "right": int(right),
                     "fallback": int(fallback)}
        self.calibrations = {index: {key: self.model[f"outcome_{arm}_{key}"] for key in PARAMETERS}
                             for index, arm in enumerate(ARMS) if index}

    def reset(self, context=None):
        super().reset(context)
        self.active_arm = "reference"

    def act(self, observation, task=""):
        if self.cursor == (0, PLACE, 0) and self.release_features is None and task_column(task) == self.column:
            self.release_features = observed_release_features(self, observation)
            choice = select_arm(self.rule, self.release_features)
            if not np.isfinite(self.release_features).all():
                choice = 0
            self.active_arm = ARMS[choice]
            if choice:
                self.release_shift_m = float(predict_shift(self.calibrations[choice], self.release_features))
                if self.release_shift_m:
                    self.apply_shift()
        return super().act(observation, task)


def upstream_policy() -> OutcomeReleasePolicy:
    return OutcomeReleasePolicy(Path(os.environ["A3OPT_OUTCOME_RELEASE_CHECKPOINT"]))


def evaluate_route(checkpoint: Path, profile: str, suite: str, seed: int) -> dict:
    trace = SlotTrace(checkpoint, False)
    trace.policy = OutcomeReleasePolicy(checkpoint)
    result = run_episode(checkpoint, profile, suite, seed, trace=trace)
    return {**result, "active_arm": trace.policy.active_arm, "checkpoint_sha256": sha256(checkpoint),
            "first_placement_prefix": {"prefix_steps": trace.first_prefix_steps, "prefix_sha256": trace.first_prefix.hexdigest()}}


def check_pair(reference: dict, candidate: dict) -> None:
    keys = ("seed", "profile", "suite", "column", "task_prompt", "initial_pose", "color", "cameras", "light_diffuse",
            "first_placement_prefix")
    if any(reference[key] != candidate[key] for key in keys):
        raise ValueError("paired policies differ before the first descent")


def training_pairs(source: Path, reference: Path, column: int) -> tuple[dict, list[dict]]:
    directory = source / "development"
    request = read_json(directory / "request.json")
    paths = {"reference": reference, **{arm: source / "candidates" / f"{arm}.npz" for arm in ARMS[1:]}}
    if request["collection_only"] or request["checkpoints"] != {arm: sha256(path) for arm, path in paths.items()}:
        raise ValueError("training labels require all three locked release controllers")
    suite = str(column)
    groups = read_json(directory / "results.json")
    if (set(groups) != set(ARMS) or set(request["seeds"]) != {suite}
            or len(set(request["seeds"][suite])) <= 2 * MIN_LEAF):
        raise ValueError("expected a completed three-arm focused development split")
    for arm in ARMS:
        if [row["seed"] for row in groups[arm]] != request["seeds"][suite]:
            raise ValueError("missing or unscheduled development labels")
    pairs = []
    for group in zip(*(groups[arm] for arm in ARMS), strict=True):
        baseline = group[0]
        features = np.asarray(baseline["feedback_features"], dtype=float)
        if features.shape != (9,) or not np.isfinite(features).all():
            raise ValueError("paired development data requires finite pre-descent observations")
        for arm, row in zip(ARMS, group, strict=True):
            check_slot_episode((paths[arm], "advanced", suite, baseline["seed"], False), row)
            check_pair(baseline, row)
            receipt = directory / "episodes" / arm / suite / f"{row['seed']}.json"
            if read_json(receipt) != row or row["checkpoint_sha256"] != request["checkpoints"][arm]:
                raise ValueError("paired development label does not match its receipt")
            if row["feedback_features"] != baseline["feedback_features"]:
                raise ValueError("paired release sensors must be measured before the intervention")
        pairs.append({"seed": baseline["seed"], "features": features.tolist(), "outcomes": outcomes(list(group)).tolist()})
    return paths, pairs


def confirmation_stats(reference: list[dict], candidate: list[dict]) -> dict:
    for baseline, trial in zip(reference, candidate, strict=True):
        check_pair(baseline, trial)
    old, new = outcomes(reference), outcomes(candidate)
    count, success = (new - old)[:, :2].T
    columns = [row["column"] for row in reference]
    return {"episodes": len(candidate), "reference_mean": float(old[:, 0].mean()), "mean": float(new[:, 0].mean()),
            "reference_complete": int(old[:, 1].sum()), "complete": int(new[:, 1].sum()),
            "mean_ci95": _bootstrap(new[:, 0], columns), "paired_count_ci95": _bootstrap(count, columns),
            "paired_success_rate_ci95": _bootstrap(success, columns), "major_losses": int((count <= -MAJOR_DROP).sum()),
            "wrong_source": sum(row["score"] != row["requested_score"] for row in candidate),
            "safety_stops": sum(bool(row["safety_reason"]) for row in candidate)}


def confirm(out: Path, selected: Path, fixed: Path, schedule: dict, column: int, workers: int) -> dict:
    policies = {"fixed": (evaluate, fixed), "outcome": (evaluate_route, selected)}
    cells = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for suite, seeds in schedule.items():
            rows = {}
            for name, (runner, path) in policies.items():
                digest = sha256(path)
                args = (False,) if name == "fixed" else ()

                def audit(job, row):
                    check_row(job[:4], row)
                    if row["checkpoint_sha256"] != digest or (name == "outcome" and row["column"] != column
                                                               and row["active_arm"] != "reference"):
                        raise ValueError("confirmation receipt has an unexpected controller")

                jobs = [(path, "advanced", suite, seed, *args) for seed in seeds]
                rows[name] = _jobs(pool, runner, jobs, out / "confirmation" / name / suite, lambda job: str(job[3]), audit)
            cells[suite] = confirmation_stats(rows["fixed"], rows["outcome"])
    focused = cells[str(column)]
    passed = (cells["random"]["mean"] > TARGET and focused["paired_success_rate_ci95"][0] > 0
              and all(cell["mean"] >= cell["reference_mean"] and cell["complete"] >= cell["reference_complete"]
                      and not (cell["major_losses"] or cell["wrong_source"] or cell["safety_stops"]) for cell in cells.values()))
    return {"status": "confirmation_passed" if passed else "fixed_retained_after_confirmation", "confirmation": cells}


def run(out: Path, reference_run: Path, source: Path, seed_block: int, workers: int) -> dict:
    original = read_json(source / "protocol.json")
    column = original["source_column"]
    if column not in range(1, 5):
        raise ValueError("release experiment has no focused source column")
    schedule = seed_schedule(seed_block)
    schedule["confirmation"][str(column)] = schedule["confirmation"].pop("1")
    previous = original["schedule"]
    used = set(previous["training"] + previous["development"] + sum(previous["confirmation"].values(), []))
    if used.intersection(sum(schedule["confirmation"].values(), [])):
        raise ValueError("confirmation seeds overlap the parent release experiment")
    reference, identity = frozen_reference(reference_run, schedule)
    if (sha256(reference) != original["reference_sha256"] or identity["environment"] != original["environment"]
            or identity["code_sha256"] != original["code_sha256"]):
        raise ValueError("release experiment used another frozen baseline or simulator environment")
    paths, pairs = training_pairs(source, reference, column)
    if [pair["seed"] for pair in pairs] != previous["development"]:
        raise ValueError("training pairs must come from the parent's development split")
    package = Path(__file__).parent
    code = {name: sha256(package / name) for name in ("release_outcome_agent.py", "release_feedback_agent.py", "slot_alignment_agent.py")}
    if code["release_feedback_agent.py"] != original["code"]:
        raise ValueError("release controller changed since the paired development collection")
    protocol = {**identity, "release_protocol_sha256": sha256(source / "protocol.json"), "code": code, "column": column,
                "training_receipts_sha256": sha256(source / "development/results.json"),
                "confirmation_seeds": schedule["confirmation"], "fixed_checkpoints": {arm: sha256(path) for arm, path in paths.items()},
                "signals": [[index, name] for index, name in SIGNALS.items()], "min_leaf": MIN_LEAF,
                "major_drop": MAJOR_DROP, "score_target": TARGET}
    _record_once(out / "protocol.json", protocol)
    _record_once(out / "dataset.json", {"arms": list(ARMS), "pairs": pairs})
    features = np.asarray([pair["features"] for pair in pairs])
    labels = np.asarray([pair["outcomes"] for pair in pairs], dtype=int)
    validation = validate_rule(features, labels)
    rule = fit_rule(features, labels)
    fixed = ARMS[rule["fallback"]]
    _record_once(out / "fit.json", {"dataset_sha256": sha256(out / "dataset.json"), "rule": rule, "validation": validation})
    if not validation["eligible"] or rule["signal"] is None:
        return {"status": "no_routing_gain_in_cross_validation", "fixed_comparator": fixed, "validation": validation}
    checkpoint = out / "outcome.npz"
    arrays = {f"outcome_{arm}_{key}": value for arm in ARMS[1:]
              for key, value in ReleaseFeedbackPolicy(paths[arm]).calibration.items()}
    if not checkpoint.exists():
        _npz(checkpoint, **phase_model(reference), **arrays, outcome_column=column,
             outcome_rule=np.asarray([rule[key] for key in ("signal", "threshold", "left", "right", "fallback")]))
    _record_once(out / "confirmation-lock.json", {"checkpoint_sha256": sha256(checkpoint), "fixed_sha256": sha256(paths[fixed]),
                 "fit_sha256": sha256(out / "fit.json"), "protocol_sha256": sha256(out / "protocol.json")})
    result = confirm(out, checkpoint, paths[fixed], schedule["confirmation"], column, workers)
    return {**result, "fixed_comparator": fixed, "validation": validation}


def main():
    parser = argparse.ArgumentParser(description="Learn first-release routing from paired closed-loop rollout outcomes.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=86000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with _run_lock(args.out.resolve()):
        result = run(args.out.resolve(), args.reference.resolve(), args.source.resolve(), args.seed_block, args.workers)
        _record_once(args.out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
