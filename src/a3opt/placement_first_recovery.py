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
from a3opt import placement_first_hover as first, placement_hover_agent as hover, placement_late_hover_gate as late
from a3opt import placement_release, placement_release_recovery as recovery, placement_routed_agent as routed
from a3opt import placement_search, placement_servo_agent as servo, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementTrace
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


BATCH = 0


@dataclass(frozen=True)
class FirstRecovery:
    column: int
    gate: recovery.Recovery
    batch = BATCH

    def __post_init__(self):
        if self.column not in range(1, 5):
            raise ValueError("invalid first-batch recovery column")

    def record(self):
        return [self.column, self.batch, *self.gate.record()[2:]]


def dependencies():
    return {module.__name__: sha256(Path(module.__file__))
            for module in (recovery, contact, servo, placement_benchmark, placement_search)}


def raise_for_release(policy, observation, recipe):
    phases = policy.actions[BATCH]
    descent_actions, opening, retreat = (phases[stage] for stage in (PLACE, OPEN, RETRACT))
    frame = policy.cursor[2]
    steps = len(descent_actions) - frame
    current = np.asarray(observation["observation.eef_pose"][:3], dtype=float)
    measured_joints = np.asarray(observation["observation.state"][:7], dtype=float)
    yaw = policy.pose[4]
    if steps < 6 or not np.isfinite(np.r_[current, measured_joints, yaw]).all():
        return False
    solver = nominal_kinematics()
    rise, outward = recovery.RECIPES[recipe]
    planned_z = descent.tool_xyz(solver, descent_actions[-1, :7])[2]
    transfer_z = descent.tool_xyz(solver, phases[TRANSFER][-1, :7])[2]
    raised = current.copy()
    raised[2] = min(transfer_z - .004, max(current[2] + .006, planned_z + rise))
    if not .004 <= raised[2] - current[2] <= .030:
        return False
    lifted_joints = recovery.joint_target(solver, measured_joints, raised)
    if lifted_joints is None:
        return False
    raised[:2] += outward * np.array([np.cos(yaw), np.sin(yaw)])
    release_joints = recovery.joint_target(solver, lifted_joints, raised) if outward else lifted_joints
    if release_joints is None:
        return False

    lift_steps = (steps + 1) // 2 if outward else steps
    last_joints = descent_actions[frame - 1, :7].copy()
    lift_progress = _ramp(np.arange(1, lift_steps + 1) / lift_steps)[:, None]
    descent_actions[frame:frame + lift_steps, :7] = last_joints + lift_progress * (lifted_joints - last_joints)
    if outward:
        move_steps = steps - lift_steps
        move_progress = _ramp(np.arange(1, move_steps + 1) / move_steps)[:, None]
        descent_actions[frame + lift_steps:, :7] = lifted_joints + move_progress * (release_joints - lifted_joints)
    opening[:, :7] = release_joints
    fade = 1. - _ramp(np.linspace(0., 1., len(retreat)))[:, None]
    retreat[:, :7] = np.clip(retreat[:, :7] + fade * (release_joints - retreat[0, :7]), *solver.ranges.T)
    return True


class FirstRecoveryPolicy(recovery.RecoveryPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent, late_recovery, probe_column=None):
        if len(corrections) > 1:
            raise ValueError("expected one first-batch recovery")
        self.first_recovery = corrections[0] if corrections else None
        self.probe_column = probe_column
        super().__init__(checkpoint, focus, scale, late_recovery, incumbent=incumbent)

    def reset(self, context=None):
        super().reset(context)
        self.first_force = None

    def first_signals(self, observation, task):
        scope = self.first_recovery.column if self.first_recovery else self.probe_column
        if (scope is None or self.actions is None or self.cursor is None or self.cursor[:2] != (BATCH, PLACE)
                or self.retry_remaining or task_column(task) != scope):
            return None
        frame = self.cursor[2]
        trajectory = self.actions[BATCH][PLACE]
        if frame == 0:
            self.first_force = contact.force_observation(observation)
        if frame != max(1, int(recovery.WINDOW * len(trajectory))) or self.first_force is None:
            return None
        force = contact.force_observation(observation) - self.first_force
        command = descent.tool_xyz(nominal_kinematics(), trajectory[frame - 1, :7])
        error = np.asarray(observation["observation.eef_pose"][:3]) - command
        return np.array([np.linalg.norm(force[:3]), np.linalg.norm(force[3:6]), error[2], np.linalg.norm(error[:2])])

    def act(self, observation, task=""):
        signals = self.first_signals(observation, task)
        item = self.first_recovery
        if signals is not None and item and item.gate.activates(signals):
            raise_for_release(self, observation, item.gate.recipe)
        return super().act(observation, task)


class FirstTrace(PlacementTrace):
    def bind_env(self, env):
        super().bind_env(env)
        self.env = env

    def reset(self, context=None):
        super().reset(context)
        self.trials = self.suffix_digest = self.continuation = None

    def trial(self, observation, task, recipe):
        branch = copy(self.env)
        branch.data = copy(self.env.data)
        policy = copy(self.policy)
        policy.actions = [[stage.copy() for stage in phases] for phases in policy.actions]
        if recipe:
            policy.first_recovery = FirstRecovery(self.column, recovery.Recovery(recipe))
        digest = hashlib.sha256()
        while policy.step < RULES["max_steps"]:
            action = policy.act(observation, task)
            observation, _, terminated, truncated, info = branch.step(action)
            digest.update(placement_train.command_bytes(action, info))
            if terminated or truncated:
                break
        return placement_release.terminal(branch, info, self.columns, self.column, digest)

    def act(self, observation, task):
        signals = self.policy.first_signals(observation, task)
        if signals is not None:
            recipes = {"reference": None, **{name: name for name in recovery.RECIPES}}
            self.trials = {"features": signals.tolist(), "step": self.step,
                           "outcomes": {name: self.trial(observation, task, recipe) for name, recipe in recipes.items()}}
            self.suffix_digest = hashlib.sha256()
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.suffix_digest is not None:
            self.suffix_digest.update(placement_train.command_bytes(action, info))
            if self.safety or self.step == RULES["max_steps"]:
                self.continuation = placement_release.terminal(env, info, self.columns, self.column, self.suffix_digest)


def collect(checkpoint, timing, policy_type, column, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = FirstTrace(policy_type(checkpoint, *timing, probe_column=column))
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
            or data is None or len(data["features"]) != len(recovery.SIGNALS) or not np.isfinite(data["features"]).all()
            or set(data["outcomes"]) != {"reference", *recovery.RECIPES}
            or row["continuation"] != data["outcomes"]["reference"]
            or row["continuation"]["counted_ids"] != row["target_cookie_ids"]
            or row["score"] != len(row["target_cookie_ids"])):
        raise ValueError("first-batch training differs from the calibrated reference")


def sample_plan(panels):
    options = []
    for column in range(1, 5):
        groups, missing = {}, 0
        for name, rows in panels.items():
            cases = [row for row in rows if row["column"] == column and not row["safety_reason"]
                     and row["score"] == row["requested_score"] and "transfer" in row["phase_geometry"][BATCH]]
            losses = sorted((-servo.lost_during_descent({**row, "terminal": row["physical_terminal"]}, BATCH), row["seed"])
                            for row in cases)
            controls = sorted(row["seed"] for row in cases if row["success"])
            failures = [(loss, seed) for loss, seed in losses if loss < 0]
            if failures and controls:
                missing -= sum(loss for loss, _ in failures)
                groups[name] = {"failures": [seed for _, seed in failures[:2]], "controls": controls[:2]}
        if len(groups) >= 3:
            options.append((missing, -column, groups))
    if not options:
        return None, {}
    _, column, plan = max(options, key=lambda option: option[:2])
    return -column, plan


def frozen_recovery(folder, checkpoint, timing, incumbent, common):
    lock = read_json(folder / "confirmation-lock.json")
    protocol_path = folder / "protocol.json"
    protocol = read_json(protocol_path)
    result = read_json(folder / "result.json")
    audit = read_json(folder / "confirmation-audit.json")
    if (any(protocol[key] != value for key, value in common.items()) or lock["timing"] != list(timing)
            or lock["incumbent"] != [item.record() for item in incumbent]
            or lock["checkpoint_sha256"] != sha256(checkpoint) or lock["protocol_sha256"] != sha256(protocol_path)
            or protocol["code_sha256"] != sha256(Path(recovery.__file__))
            or protocol["contact_code_sha256"] != sha256(Path(contact.__file__))
            or not first.supports_confirmation(result["confirmation"]) or audit["native_replays"] != 8
            or result["status"] != "candidate_promoted" or audit["status"] != result["status"]
            or not (audit["corrections"] == result["corrections"] == lock["corrections"])):
        raise ValueError("first-batch search requires an audited contact recovery checkpoint")
    gate, = [recovery.Recovery(*item[2:]) for item in lock["corrections"]]
    if lock["corrections"] != [gate.record()]:
        raise ValueError("unexpected late recovery scope")
    return (gate,)


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
        raise ValueError("first-batch search requires the frozen published checkpoint")
    late_recovery = frozen_recovery(recovery_archive, checkpoint, timing, incumbent, common)
    published, release_hashes = first.historical_rows(release, (), frozen, common, checkpoint, timing, incumbent)
    _, archived, archived_hashes = late.routed_panels(archive, checkpoint, timing, common)
    panels = {name: [pair["strict-500"] for pair in panel["reference"]] for name, panel in archived.items()}
    for name in ("confirmation", "replication"):
        schedule = read_json(release / f"{name}-protocol.json")["schedule"][name]
        seeds = set(schedule["random"] if isinstance(schedule, dict) else schedule)
        panels[f"release-{name}"] = [row for row in published if row["seed"] in seeds]
    baseline = {row["seed"]: row for rows in panels.values() for row in rows}
    if len(baseline) != sum(map(len, panels.values())):
        raise ValueError("duplicate first-batch training source scene")
    column, plan = sample_plan(panels)
    if not plan:
        return {"status": "reference_retained_without_first_batch_cases"}

    training = [seed for group in plan.values() for seed in group["failures"] + group["controls"]]
    development = [seed for seed in range(block, block + 64) if choose_column("random", seed) == column]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    prior = [recovery_archive / "protocol.json", *root.parent.glob("*/protocol.json")]
    if reserved & set(baseline) or any(path.parent != root and reserved & hover.scheduled_seeds(
            read_json(path).get("schedule", {})) for path in prior):
        raise ValueError("first-batch search requires fresh evaluation scenes")
    protocol = root / "protocol.json"
    fallback_records = [item.record() for item in late_recovery]
    _record_once(protocol, {**common, "reference_sha256": sha256(manifest), "code_sha256": sha256(Path(__file__)),
                 "late_recovery": fallback_records, "scope": [column, BATCH],
                 "sources": {"release": release_hashes, "routed": archived_hashes,
                             "recovery_lock": sha256(recovery_archive / "confirmation-lock.json")},
                 "dependency_sha256": dependencies(),
                 "sampling": plan, "schedule": {"training": training, "development": development, "confirmation": confirmation},
                 "confirmation_requires": routed.REQUIREMENTS})
    policy_type = partial(FirstRecoveryPolicy, incumbent=incumbent, late_recovery=late_recovery)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent selected column {column} and is collecting {len(training)} matched first-batch recoveries", flush=True)
        jobs = [(checkpoint, timing, policy_type, column, seed) for seed in training]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]),
                     partial(audit_sample, baseline=baseline))
        indexed = {row["seed"]: row for row in rows}
        training_panels = {name: [indexed[seed] for seed in group["failures"] + group["controls"]] for name, group in plan.items()}
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"panels": training_panels})
        studied, metrics = recovery.fit(training_panels)
        model = tuple(FirstRecovery(column, item) for item in studied)
        records = [item.record() for item in model]
        fit = root / "fit.json"
        _record_once(fit, {**metrics, "corrections": records, "dataset_sha256": sha256(dataset)})
        if not model:
            return {"status": "reference_retained_after_first_batch_training", "fit": metrics}
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                              {"reference": (), "selected": model}, policy_type=policy_type)
        scores = hover.pair_stats(paired, model)
        selection = root / "development-selection.json"
        _record_once(selection, scores)
        if not all(placement_search.passes(row) for row in scores.values()):
            return {"status": "reference_retained_after_first_batch_development", "development": scores, "corrections": records}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "late_recovery": fallback_records,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit), "selection_sha256": sha256(selection),
                     "seeds": confirmation, "corrections": records})
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model,
                                                   confirmation[:4], policy_type=policy_type)
    if any(stats["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for stats in final.values()):
        raise ValueError("confirmation requires four-column balance")
    return {"status": "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_first_batch_confirmation",
            "confirmation": final, "corrections": records, "development": scores,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Learn first-batch contact recovery while retaining the second-batch controller.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--archive", type=Path, default=Path("evidence/routed-20261120"))
    parser.add_argument("--recovery-archive", type=Path, default=Path("evidence/release-recovery-20261121"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=163000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.archive.resolve(), args.recovery_archive.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
