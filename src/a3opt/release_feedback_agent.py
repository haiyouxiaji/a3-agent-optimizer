from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from a3opt.adaptive_sim import phase_model
from a3opt.advanced_agent import TARGET, _jobs, _npz, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER, AdaptivePhasePolicy, load_checkpoint, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.backends.slot_alignment import MAX_SHIFT_M, release_features
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import run_episode
from a3opt.refinement_eval import sha256
from a3opt.second_batch_agent import frozen_reference, seed_schedule
from a3opt.second_batch_fit import improvement, nonregression
from a3opt.slot_alignment_agent import SlotTrace, _compare, _focus, check_slot_episode
from a3opt.slot_alignment_fit import _metrics, _rank, fit_alignment


PARAMETERS = ("center", "scale", "weights", "bias", "column")
FEATURES = {"initial": (5, 6), "feedback": (5, 6, 7, 8)}
RIDGES = (1., 4., 16.)


def observed_release_features(policy: AdaptivePhasePolicy, observation: dict) -> np.ndarray:
    phases = policy.actions[0]
    solver = nominal_kinematics()
    low, high = solver.ranges.T
    solver.data.qpos[solver.positions] = np.clip(phases[TRANSFER][-1, :7], low, high)
    solver.mujoco.mj_kinematics(solver.model, solver.data)
    direction = np.array([-np.sin(policy.pose[4]), np.cos(policy.pose[4])])
    error = observation["observation.eef_pose"][:2] - solver.data.site_xpos[solver.site, :2]
    force = np.linalg.norm(observation["observation.force"][:3])
    return np.r_[release_features(policy.pose, phases), error @ direction, np.log1p(force)]


def predict_shift(model: dict, features: np.ndarray) -> np.ndarray:
    normalized = np.clip((features - model["center"]) / model["scale"], -3., 3.)
    return np.clip(model["bias"] + normalized @ model["weights"], -MAX_SHIFT_M, MAX_SHIFT_M)


class ReleaseFeedbackPolicy(AdaptivePhasePolicy):
    def __init__(self, checkpoint: Path):
        reference = load_checkpoint(checkpoint)
        self.calibration = ({key: reference.model[f"release_feedback_{key}"] for key in PARAMETERS}
                            if "release_feedback_column" in reference.model else None)
        super().__init__(reference.model, reference.recipes, reference.detector)

    def reset(self, context=None):
        super().reset(context)
        self.release_features = None
        self.release_shift_m = 0.

    def act(self, observation, task=""):
        if self.actions is not None and self.cursor == (0, PLACE, 0) and self.release_features is None:
            self.release_features = observed_release_features(self, observation)
            if self.calibration is not None and task_column(task) == int(self.calibration["column"]):
                self.release_shift_m = float(predict_shift(self.calibration, self.release_features))
                if self.release_shift_m:
                    self.apply_shift()
        return super().act(observation, task)

    def apply_shift(self) -> None:
        direction = self.release_shift_m * np.array([-np.sin(self.pose[4]), np.cos(self.pose[4]), 0.])
        solver = nominal_kinematics()
        for phase in (PLACE, OPEN, RETRACT):
            actions = self.actions[0][phase]
            progress = _ramp(np.arange(1, len(actions) + 1) / len(actions))
            if phase == OPEN:
                progress[:] = 1.
            elif phase == RETRACT:
                progress = 1. - progress
            for action, weight in zip(actions, progress, strict=True):
                if weight:
                    action[:7] = solver.shift(action[:7], direction * weight)


def evaluate(checkpoint: Path, profile: str, suite: str, seed: int, collect: bool) -> dict:
    trace = SlotTrace(checkpoint, collect)
    trace.policy = ReleaseFeedbackPolicy(checkpoint)
    row = run_episode(checkpoint, profile, suite, seed, trace=trace)
    features = trace.policy.release_features
    return {**row, "checkpoint_sha256": sha256(checkpoint), "batches": trace.events, "retries": trace.policy.retries,
            "first_requested_five": trace.first_requested_five, "prefix_sha256": trace.prefix.hexdigest(),
            "prefix_steps": trace.prefix_steps, "placement_events": trace.placement_events,
            "first_placement_prefix": {"prefix_steps": trace.first_prefix_steps, "prefix_sha256": trace.first_prefix.hexdigest()},
            "first_after_retract": trace.first_after_retract, "front_slots": trace.front_slots,
            "front_slot_xy_m": trace.front_xy, "collection_only": collect,
            "first_release_features": features[:7].tolist() if features is not None else None,
            "feedback_features": features.tolist() if features is not None else None,
            "release_shift_m": trace.policy.release_shift_m}


def episodes(pool, folder: Path, policies: dict, suites: dict, *, collect: bool = False) -> dict:
    models = {label: sha256(path) for label, path in policies.items()}
    _record_once(folder / "request.json", {"checkpoints": models, "seeds": suites, "collection_only": collect})
    jobs = [(path, "advanced", suite, seed, collect) for path in policies.values()
            for suite, seeds in suites.items() for seed in seeds]
    labels = {digest: label for label, digest in models.items()}
    cache_labels = {path: label for label, path in policies.items()}
    rows = _jobs(pool, evaluate, jobs, folder / "episodes", lambda job: f"{cache_labels[job[0]]}/{job[2]}/{job[3]}",
                 check_slot_episode)
    groups = {label: [] for label in policies}
    for row in rows:
        groups[labels[row["checkpoint_sha256"]]].append(row)
    _record_once(folder / "results.json", groups)
    return groups


def regress(features: np.ndarray, targets: np.ndarray, indices: tuple[int, ...], ridge: float, column: int) -> dict:
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-5)
    selected = np.clip(((features - center) / scale)[:, indices], -3., 3.)
    design = np.column_stack((np.ones(len(features)), selected))
    penalty = np.diag([0., *([ridge] * len(indices))])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    weights = np.zeros(features.shape[1])
    weights[list(indices)] = coefficients[1:]
    return {"center": center, "scale": scale, "weights": weights, "bias": coefficients[0], "column": column}


def fit(rows: list[dict], column: int) -> tuple[dict, dict]:
    _, labels = fit_alignment(rows, column)
    samples = labels["samples"]
    diagnostics = {"samples": len(samples), "rejected": labels["rejected"]}
    if len(samples) < 8:
        return {}, diagnostics
    by_seed = {row["seed"]: row["feedback_features"] for row in rows}
    features = np.asarray([by_seed[sample["seed"]] for sample in samples])
    targets = np.asarray([sample["label_shift_m"] for sample in samples])
    errors = np.asarray([sample["residuals_y_m"] for sample in samples])
    indices = np.arange(len(samples))
    diagnostics["baseline"] = _metrics(errors)
    models = {}
    for name, columns in FEATURES.items():
        trials = []
        for ridge in RIDGES:
            shifts = []
            for held in indices:
                train = indices != held
                fitted = regress(features[train], targets[train], columns, ridge, column)
                shifts.append(predict_shift(fitted, features[held]))
            trials.append({"ridge": ridge, **_metrics(errors + np.asarray(shifts)[:, None])})
        diagnostics[name] = max(trials, key=_rank)
        model = regress(features, targets, columns, diagnostics[name]["ridge"], column)
        if _rank(diagnostics[name]) > _rank(diagnostics["baseline"]):
            models[name] = model
    return models, diagnostics


def run(out: Path, reference_run: Path, diagnosis: Path, block: int, workers: int) -> dict:
    schedule = seed_schedule(block)
    reference, identity = frozen_reference(reference_run, schedule)
    column, evidence = _focus(diagnosis, reference, schedule)
    _record_once(out / "protocol.json", {**identity, "source_column": column, "diagnosis": evidence,
                 "schedule": schedule, "features": {name: list(indices) for name, indices in FEATURES.items()},
                 "code": sha256(Path(__file__))})
    if column is None:
        return {"status": "no_focused_column", "selected_checkpoint": str(reference)}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        training = episodes(pool, out / "training", {"reference": reference}, {str(column): schedule["training"]}, collect=True)
        models, diagnostics = fit(training["reference"], column)
        _record_once(out / "fit.json", {**diagnostics, "training_sha256": sha256(out / "training" / "results.json")})
        if not models:
            return {"status": "no_validation_gain", "selected_checkpoint": str(reference), "fit": diagnostics}
        checkpoints = {"reference": reference}
        for name, model in models.items():
            path = out / "candidates" / f"{name}.npz"
            if not path.exists():
                _npz(path, **phase_model(reference), **{f"release_feedback_{key}": val for key, val in model.items()})
            checkpoints[name] = path
        dev = episodes(pool, out / "development", checkpoints, {str(column): schedule["development"]})
        comparisons = {name: _compare({"reference": dev["reference"], "selected": dev[name]}, column) for name in models}
        choices = [name for name, stats in comparisons.items() if improvement(stats)]
        candidate = max(choices, key=lambda name: (comparisons[name]["full_success_delta"], comparisons[name]["score_delta"])) if choices else None
        _record_once(out / "development-decision.json", {"candidate": candidate, "comparisons": comparisons})
        if candidate is None:
            return {"status": "reference_retained_after_development", "selected_checkpoint": str(reference), "development": comparisons}
        path = checkpoints[candidate]
        _record_once(out / "confirmation-lock.json", {"candidate": candidate, "checkpoint_sha256": sha256(path),
                     "protocol_sha256": sha256(out / "protocol.json"), "schedule": schedule["confirmation"],
                     "development_sha256": sha256(out / "development-decision.json")})
        evaluated = episodes(pool, out / "confirmation", {"reference": reference, "selected": path}, schedule["confirmation"])
        cells = {suite: _compare({label: [row for row in rows if row["suite"] == suite]
                                  for label, rows in evaluated.items()}, column) for suite in schedule["confirmation"]}
        scores = {suite: np.mean([row["requested_score"] for row in evaluated["selected"] if row["suite"] == suite])
                  for suite in cells}
        promoted = min(scores.values()) > TARGET and all(map(nonregression, cells.values())) and any(map(improvement, cells.values()))
        return {"status": "promoted" if promoted else "reference_retained_after_confirmation",
                "selected_checkpoint": str(path if promoted else reference), "scores": scores, "confirmation": cells}


def main():
    parser = argparse.ArgumentParser(description="Learn bounded first-placement corrections from rollout feedback.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=84000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with _run_lock(args.out.resolve()):
        result = run(args.out.resolve(), args.reference.resolve(), args.diagnosis.resolve(), args.seed_block, args.workers)
        _record_once(args.out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
