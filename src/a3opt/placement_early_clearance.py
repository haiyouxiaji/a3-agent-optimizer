from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_contact_feedback as contact, placement_descent_agent as descent
from a3opt import placement_first_hover as first, placement_first_recovery as history, placement_hover_agent as hover
from a3opt import placement_late_hover_gate as late, placement_release, placement_release_recovery as recovery
from a3opt import placement_routed_agent as routed, placement_search, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementTrace
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


BATCH = 0
WINDOW = .35
RECIPES = {"raise8": (.008, 0.), "raise14": (.014, 0.), "raise20": (.020, 0.),
           "inset4-raise14": (.014, .004)}
SIGNALS = ("vertical_tracking_error_m", "wall_axis_tracking_error_m", "lateral_tracking_error_m", "force_change_n")


@dataclass(frozen=True)
class Clearance:
    column: int
    recipe: str
    signal: int | None = None
    threshold: float | None = None
    above: bool = True
    batch = BATCH

    def __post_init__(self):
        if (self.column not in range(1, 5) or self.recipe not in RECIPES
                or (self.signal is None) != (self.threshold is None) or not isinstance(self.above, bool)
                or (self.signal is not None and (self.signal not in range(len(SIGNALS)) or not np.isfinite(self.threshold)))):
            raise ValueError("invalid early clearance gate")

    def record(self):
        return [self.column, self.batch, self.recipe, self.signal, self.threshold, self.above]

    def activates(self, values):
        if not np.isfinite(values).all():
            return False
        if self.signal is None:
            return True
        value = values[self.signal]
        return value >= self.threshold if self.above else value < self.threshold


def replan(phases, frame, measured_xyz, wall_axis, recipe, solver):
    lowering, opening, retreat = (phases[stage] for stage in (PLACE, OPEN, RETRACT))
    remaining = len(lowering) - frame
    if frame < 1 or remaining < 6 or not np.isfinite(np.r_[measured_xyz, wall_axis]).all():
        return False
    rise, inset = RECIPES[recipe]
    nominal = lowering[-1, :7].copy()
    target = descent.tool_xyz(solver, nominal)
    target[:2] -= inset * wall_axis
    target[2] += rise
    if not 0. <= measured_xyz[2] - target[2] <= .070:
        return False
    hold = recovery.joint_target(solver, nominal, target)
    if hold is None:
        return False
    previous = lowering[frame - 1, :7].copy()
    progress = _ramp(np.arange(1, remaining + 1) / remaining)[:, None]
    lowering[frame:, :7] = previous + progress * (hold - previous)
    opening[:, :7] = hold
    if len(retreat) > 1:
        fade = 1. - _ramp(np.linspace(0., 1., len(retreat)))[:, None]
        retreat[:, :7] = np.clip(retreat[:, :7] + fade * (hold - retreat[0, :7]), *solver.ranges.T)
    return True


class EarlyClearancePolicy(recovery.RecoveryPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent, late_recovery, probe_column=None):
        if len(corrections) > 1:
            raise ValueError("expected one early clearance gate")
        self.clearance = corrections[0] if corrections else None
        self.probe_column = probe_column
        super().__init__(checkpoint, focus, scale, late_recovery, incumbent=incumbent)

    def reset(self, context=None):
        super().reset(context)
        self.start_force = None

    def wall_axis(self):
        yaw = self.pose[4]
        return np.array([np.cos(yaw), np.sin(yaw)])

    def early_signals(self, observation, task):
        column = self.clearance.column if self.clearance else self.probe_column
        if (column is None or self.actions is None or self.cursor is None or self.cursor[:2] != (BATCH, PLACE)
                or self.retry_remaining or task_column(task) != column):
            return None
        frame = self.cursor[2]
        lowering = self.actions[BATCH][PLACE]
        if frame == 0:
            self.start_force = contact.force_observation(observation)[:3]
        if frame != max(1, int(WINDOW * len(lowering))) or self.start_force is None:
            return None
        command = descent.tool_xyz(nominal_kinematics(), lowering[frame - 1, :7])
        error = np.asarray(observation["observation.eef_pose"][:3]) - command
        force_change = contact.force_observation(observation)[:3] - self.start_force
        axis = self.wall_axis()
        return np.array([error[2], error[:2] @ axis, error[:2] @ np.array([-axis[1], axis[0]]),
                         np.linalg.norm(force_change)])

    def act(self, observation, task=""):
        values = self.early_signals(observation, task)
        if values is not None and self.clearance and self.clearance.activates(values):
            replan(self.actions[BATCH], self.cursor[2], np.asarray(observation["observation.eef_pose"][:3]),
                   self.wall_axis(), self.clearance.recipe, nominal_kinematics())
        return super().act(observation, task)


class ClearanceTrace(history.FirstTrace):
    def trial(self, observation, task, recipe):
        branch = copy(self.env)
        branch.data = copy(self.env.data)
        policy = copy(self.policy)
        policy.actions = [[stage.copy() for stage in phases] for phases in policy.actions]
        if recipe != "reference":
            policy.clearance = Clearance(self.column, recipe)
        digest = hashlib.sha256()
        while policy.step < RULES["max_steps"]:
            action = policy.act(observation, task)
            observation, _, terminated, truncated, info = branch.step(action)
            digest.update(placement_train.command_bytes(action, info))
            if terminated or truncated:
                break
        return placement_release.terminal(branch, info, self.columns, self.column, digest)

    def act(self, observation, task):
        signals = self.policy.early_signals(observation, task)
        if signals is not None:
            self.trials = {"features": signals.tolist(), "step": self.step,
                           "outcomes": {name: self.trial(observation, task, name) for name in ("reference", *RECIPES)}}
            self.suffix_digest = hashlib.sha256()
        return PlacementTrace.act(self, observation, task)


def collect(checkpoint, timing, policy_type, column, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = ClearanceTrace(policy_type(checkpoint, *timing, probe_column=column))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "steps": result.steps, "column": result.source_column,
                "profile": result.randomization["profile"], "checkpoint_sha256": sha256(checkpoint),
                "prefix": trace.stages[BATCH]["transfer"], "trials": trace.trials, "continuation": trace.continuation,
                "target_cookie_ids": np.flatnonzero(trace.target).tolist(), "score": result.score}
    finally:
        env.close()


def audit_sample(job, row, *, baseline):
    checkpoint, _, _, column, seed = job
    data = row["trials"]
    if (row["seed"] != seed or row["column"] != column or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint)
            or row["prefix"] != baseline[seed]["phase_geometry"][BATCH]["transfer"]
            or data is None or len(data["features"]) != len(SIGNALS) or not np.isfinite(data["features"]).all()
            or set(data["outcomes"]) != {"reference", *RECIPES}
            or row["continuation"] != data["outcomes"]["reference"]
            or row["continuation"]["counted_ids"] != row["target_cookie_ids"]
            or row["score"] != len(row["target_cookie_ids"])):
        raise ValueError("early clearance branch did not reproduce its control")


def best_gate(panels, column):
    rows = [row for group in panels.values() for row in group]
    splits = [(None, None, True)]
    for signal in range(len(SIGNALS)):
        values = sorted({row["trials"]["features"][signal] for row in rows})
        for low, high in zip(values, values[1:]):
            splits.extend((signal, (low + high) / 2., above) for above in (True, False))
    candidates = []
    for recipe in RECIPES:
        for signal, threshold, above in splits:
            gate = Clearance(column, recipe, signal, threshold, above)
            metrics = recovery.effect(rows, gate)
            if recovery.usable(metrics) and sum(recovery.effect(group, gate)["gain"] > 0 for group in panels.values()) >= 2:
                rank = (metrics["gain"], metrics["complete_gain"], signal is None, -metrics["activations"])
                candidates.append((rank, gate))
    return max(candidates, key=lambda item: item[0], default=(None, None))[1]


def fit(panels, column):
    held_out, predictions = {}, []
    for name, rows in panels.items():
        gate = best_gate({key: panel for key, panel in panels.items() if key != name}, column)
        held_out[name] = {"gate": gate.record() if gate else None, "effect": recovery.effect(rows, gate)}
        predictions.extend(contact.gate_outcome(row, gate) for row in rows)
    gate = best_gate(panels, column)
    admitted = gate is not None and contact.eligible(predictions) and sum(
        group["effect"]["gain"] > 0 for group in held_out.values()) >= 2
    rows = [row for group in panels.values() for row in group]
    return ((gate,) if admitted else ()), {"held_out": held_out, "admitted": admitted, "fold_predictions": predictions,
        "unconditional": {recipe: recovery.effect(rows, Clearance(column, recipe)) for recipe in RECIPES}}


def dependencies():
    modules = (history, recovery, contact, descent, placement_benchmark, placement_search, placement_train)
    return {module.__name__: sha256(Path(module.__file__)) for module in modules}


def run(root, release, archive, recovery_archive, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected an aligned seed block and a positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint, timing = release / frozen["checkpoint"], tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*item) for item in frozen["corrections"])
    common = {"environment": simulator_identity(), "incumbent": frozen["corrections"], "timing": list(timing),
              "rules": RULES, "strict_rules": STRICT_RULES}
    if (common["environment"]["sim_revision"] != frozen["sim_revision"] or sha256(checkpoint) != frozen["checkpoint_sha256"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("early clearance requires the frozen published checkpoint")
    late_recovery = history.frozen_recovery(recovery_archive, checkpoint, timing, incumbent, common)
    published, release_hashes = first.historical_rows(release, (), frozen, common, checkpoint, timing, incumbent)
    _, archived, archived_hashes = late.routed_panels(archive, checkpoint, timing, common)
    panels = {name: [pair["strict-500"] for pair in panel["reference"]] for name, panel in archived.items()}
    for name in ("confirmation", "replication"):
        schedule = read_json(release / f"{name}-protocol.json")["schedule"][name]
        seeds = set(schedule["random"] if isinstance(schedule, dict) else schedule)
        panels[f"release-{name}"] = [row for row in published if row["seed"] in seeds]
    baseline = {row["seed"]: row for rows in panels.values() for row in rows}
    if len(baseline) != sum(map(len, panels.values())):
        raise ValueError("duplicate historical scene")
    column, plan = history.sample_plan(panels)
    if not plan:
        return {"status": "reference_retained_without_early_clearance_cases"}

    training = [seed for group in plan.values() for seed in group["failures"] + group["controls"]]
    development = [seed for seed in range(block, block + 96) if choose_column("random", seed) == column]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    prior = [recovery_archive / "protocol.json", recovery_archive / "replication-protocol.json",
             *root.parent.glob("*/protocol.json")]
    if reserved & set(baseline) or any(path.is_file() and path.parent != root and reserved & hover.scheduled_seeds(
            read_json(path).get("schedule", {})) for path in prior):
        raise ValueError("early clearance requires fresh evaluation scenes")
    protocol = root / "protocol.json"
    fallback_records = [item.record() for item in late_recovery]
    _record_once(protocol, {**common, "reference_sha256": sha256(manifest), "code_sha256": sha256(Path(__file__)),
                 "checkpoint_sha256": sha256(checkpoint), "late_recovery": fallback_records, "scope": [column, BATCH],
                 "sources": {"release": release_hashes, "routed": archived_hashes,
                             "recovery_lock": sha256(recovery_archive / "confirmation-lock.json")},
                 "dependency_sha256": dependencies(), "descent_fraction": WINDOW, "recipes_m": RECIPES, "signals": SIGNALS,
                 "sampling": plan, "schedule": {"training": training, "development": development, "confirmation": confirmation},
                 "confirmation_requires": routed.REQUIREMENTS})
    policy_type = partial(EarlyClearancePolicy, incumbent=incumbent, late_recovery=late_recovery)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent selected column {column} and is collecting {len(training)} paired early clearance scenes", flush=True)
        jobs = [(checkpoint, timing, policy_type, column, seed) for seed in training]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]),
                     partial(audit_sample, baseline=baseline))
        indexed = {row["seed"]: row for row in rows}
        training_panels = {name: [indexed[seed] for seed in group["failures"] + group["controls"]] for name, group in plan.items()}
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"panels": training_panels})
        model, metrics = fit(training_panels, column)
        records = [item.record() for item in model]
        fit_path = root / "fit.json"
        _record_once(fit_path, {**metrics, "corrections": records, "dataset_sha256": sha256(dataset)})
        if not model:
            return {"status": "reference_retained_after_early_clearance_training", "fit": metrics}
        print(f"Agent froze one clearance gate for {len(development)} new development pairs", flush=True)
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                              {"reference": (), "selected": model}, policy_type=policy_type)
        scores = hover.pair_stats(paired, model)
        selection = root / "development-selection.json"
        _record_once(selection, scores)
        if not all(placement_search.passes(row) for row in scores.values()):
            return {"status": "reference_retained_after_early_clearance_development", "development": scores, "corrections": records}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "late_recovery": fallback_records,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection),
                     "seeds": confirmation, "corrections": records})
        print(f"Agent started {len(confirmation)} independent four-column confirmation pairs", flush=True)
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                   confirmation[:4], policy_type=policy_type)
    if any(stats["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for stats in final.values()):
        raise ValueError("confirmation requires four-column balance")
    return {"status": "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_early_clearance_confirmation",
            "confirmation": final, "corrections": records, "development": scores,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Learn an earlier first-batch clearance correction from paired full suffixes.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--archive", type=Path, default=Path("evidence/routed-20261120"))
    parser.add_argument("--recovery-archive", type=Path, default=Path("evidence/release-recovery-20261121"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=164000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.archive.resolve(), args.recovery_archive.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
