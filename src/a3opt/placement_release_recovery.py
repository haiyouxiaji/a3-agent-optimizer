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
from a3opt import placement_release, placement_routed_agent as routed, placement_search, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementTrace
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


COLUMN, BATCH = hover.COLUMN, hover.BATCH
WINDOW = .75
EXTRA_STEPS = 8
RECIPES = {"raise-in-place": (.016, 0.), "raise-then-outward": (.024, .004)}
SIGNALS = ("force_change", "torque_change", "vertical_tracking_error", "horizontal_tracking_error")
MAX_JOINT_DELTA = .18


@dataclass(frozen=True)
class Recovery:
    recipe: str
    signal: int | None = None
    threshold: float | None = None
    column = COLUMN
    batch = BATCH

    def __post_init__(self):
        if (self.recipe not in RECIPES or (self.signal is None) != (self.threshold is None)
                or (self.signal is not None and (self.signal not in range(len(SIGNALS)) or not np.isfinite(self.threshold)))):
            raise ValueError("invalid release recovery")

    def activates(self, values):
        return np.isfinite(values).all() and (self.signal is None or values[self.signal] >= self.threshold)

    def record(self):
        return [self.column, self.batch, self.recipe, self.signal, self.threshold]


def joint_target(solver, start, target):
    solution = start.copy()
    for _ in range(4):
        position = descent.tool_xyz(solver, solution)
        if np.linalg.norm(target - position) <= .0025:
            return solution
        solution = solver.shift(solution, target - position)
        if not np.isfinite(solution).all() or np.max(np.abs(solution - start)) > MAX_JOINT_DELTA:
            return None
    return solution if np.linalg.norm(descent.tool_xyz(solver, solution) - target) <= .0025 else None


def apply_recovery(policy, observation, recipe):
    frame = policy.cursor[2]
    phases = policy.actions[BATCH]
    descent_actions, opening, retreat = (phases[stage] for stage in (PLACE, OPEN, RETRACT))
    lift_steps = len(descent_actions) - frame + EXTRA_STEPS
    if policy.step + lift_steps + len(opening) + STRICT_RULES["success_hold_steps"] > RULES["max_steps"]:
        return False
    current = np.asarray(observation["observation.eef_pose"][:3], dtype=float)
    joints = np.asarray(observation["observation.state"][:7], dtype=float)
    if not np.isfinite(current).all() or not np.isfinite(joints).all() or not np.isfinite(policy.pose[4]):
        return False
    solver = nominal_kinematics()
    rise, outward = RECIPES[recipe]
    nominal = descent.tool_xyz(solver, descent_actions[-1, :7])
    high = descent.tool_xyz(solver, phases[TRANSFER][-1, :7])[2]
    raised = current.copy()
    raised[2] = min(high - .004, max(current[2] + .006, nominal[2] + rise))
    if not .004 <= raised[2] - current[2] <= .030:
        return False
    lift_joints = joint_target(solver, joints, raised)
    if lift_joints is None:
        return False
    release_xy = raised.copy()
    release_xy[:2] -= outward * np.array([np.cos(policy.pose[4]), np.sin(policy.pose[4])])
    release_joints = joint_target(solver, lift_joints, release_xy) if outward else lift_joints
    if release_joints is None:
        return False
    lift_count = (lift_steps + 1) // 2 if outward else lift_steps
    weights = _ramp(np.arange(1, lift_count + 1) / lift_count)[:, None]
    start = descent_actions[frame - 1, :7]
    path = start + weights * (lift_joints - start)
    if outward:
        move_count = lift_steps - lift_count
        weights = _ramp(np.arange(1, move_count + 1) / move_count)[:, None]
        path = np.concatenate([path, lift_joints + weights * (release_joints - lift_joints)])
    times = np.linspace(frame - 1, len(descent_actions) - 1, lift_steps + 1)[1:]
    unexecuted = np.stack([np.interp(times, np.arange(len(descent_actions)), channel) for channel in descent_actions.T], axis=1)
    unexecuted[:, :7] = path
    phases[PLACE] = np.concatenate([descent_actions[:frame], unexecuted])
    opening[:, :7] = release_joints
    weights = 1. - _ramp(np.linspace(0., 1., len(retreat)))[:, None]
    retreat[:, :7] = np.clip(retreat[:, :7] + weights * (release_joints - retreat[0, :7]), *solver.ranges.T)
    return True


class RecoveryPolicy(descent.DescentPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent, probe=False):
        if len(corrections) > 1:
            raise ValueError("expected one release recovery")
        self.recovery = corrections[0] if corrections else None
        self.probe = probe
        super().__init__(checkpoint, focus, scale, incumbent)

    def reset(self, context=None):
        super().reset(context)
        self.transfer_force = None
        self.recovery_checked = False

    def signal(self, observation, task):
        if (self.recovery_checked or (self.recovery is None and not self.probe) or self.actions is None or self.cursor is None
                or self.cursor[:2] != (BATCH, PLACE) or self.retry_remaining or task_column(task) != COLUMN):
            return None
        frame = self.cursor[2]
        trajectory = self.actions[BATCH][PLACE]
        if frame == 0:
            self.transfer_force = contact.force_observation(observation)
        if frame != max(1, int(WINDOW * len(trajectory))) or self.transfer_force is None:
            return None
        force = contact.force_observation(observation) - self.transfer_force
        nominal = descent.tool_xyz(nominal_kinematics(), trajectory[frame - 1, :7])
        error = np.asarray(observation["observation.eef_pose"][:3]) - nominal
        return np.array([np.linalg.norm(force[:3]), np.linalg.norm(force[3:6]), error[2], np.linalg.norm(error[:2])])

    def act(self, observation, task=""):
        values = self.signal(observation, task)
        if values is not None:
            self.recovery_checked = True
            if self.recovery is not None and self.recovery.activates(values):
                apply_recovery(self, observation, self.recovery.recipe)
        return super().act(observation, task)


def suffix(env, policy, observation, task, columns, recipe):
    branch = copy(env)
    branch.data = copy(env.data)
    controller = copy(policy)
    if recipe:
        controller.actions = policy.actions.copy()
        controller.actions[BATCH] = [phase.copy() for phase in policy.actions[BATCH]]
        controller.recovery = Recovery(recipe)
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated:
            break
    return placement_release.terminal(branch, info, columns, COLUMN, digest)


class RecoveryTrace(PlacementTrace):
    def bind_env(self, env):
        super().bind_env(env)
        self.env = env

    def reset(self, context=None):
        super().reset(context)
        self.trials = self.suffix_digest = self.continuation = None

    def act(self, observation, task):
        values = self.policy.signal(observation, task)
        if values is not None:
            recipes = {"reference": None, **{name: name for name in RECIPES}}
            self.trials = {"features": values.tolist(), "step": self.step, "prefix_sha256": self.digest.hexdigest(),
                           "outcomes": {name: suffix(self.env, self.policy, observation, task, self.columns, recipe)
                                        for name, recipe in recipes.items()}}
            self.suffix_digest = hashlib.sha256()
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.suffix_digest is not None:
            self.suffix_digest.update(placement_train.command_bytes(action, info))
            if self.safety or self.step == RULES["max_steps"]:
                self.continuation = placement_release.terminal(env, info, self.columns, COLUMN, self.suffix_digest)


def collect(checkpoint, timing, incumbent, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = RecoveryTrace(RecoveryPolicy(checkpoint, *timing, incumbent=incumbent, probe=True))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "steps": result.steps, "score": result.score, "success": result.success,
                "target_cookie_ids": np.flatnonzero(trace.target).tolist(), "safety_reason": trace.safety,
                "action_sha256": trace.digest.hexdigest(), "applied_action_sha256": trace.applied_digest.hexdigest(),
                "trials": trace.trials, "continuation": trace.continuation}
    finally:
        env.close()


def audit_training(job, row, *, history):
    checkpoint, _, _, seed = job
    prior, trials = history[seed], row["trials"]
    if (row["seed"] != seed or row["column"] != COLUMN or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or any(row[key] != prior[key] for key in hover.MATCH_FIELDS)
            or trials is None or len(trials["features"]) != len(SIGNALS) or not np.isfinite(trials["features"]).all()
            or set(trials["outcomes"]) != {"reference", *RECIPES} or trials["outcomes"]["reference"] != row["continuation"]
            or row["continuation"]["counted_ids"] != row["target_cookie_ids"]):
        raise ValueError("release recovery fork does not match its archived control")


def effect(rows, gate):
    outcomes = [contact.gate_outcome(row, gate) for row in rows]
    return {"gain": sum(row["gain"] for row in outcomes), "improved": sum(row["gain"] > 0 for row in outcomes),
            "complete_gain": sum(row["complete_gain"] for row in outcomes), "harmed": sum(row["gain"] < 0 for row in outcomes),
            "unsafe": sum(row["unsafe"] for row in outcomes), "activations": sum(row["triggered"] for row in outcomes)}


def usable(summary):
    return (summary["improved"] >= 2 and summary["complete_gain"] >= 0 and not (summary["harmed"] or summary["unsafe"]))


def select_gate(panels):
    rows = [row for panel in panels.values() for row in panel]
    thresholds = [(None, None)]
    for signal in range(len(SIGNALS)):
        values = sorted({row["trials"]["features"][signal] for row in rows})
        thresholds.extend((signal, (low + high) / 2.) for low, high in zip(values, values[1:]))
    candidates = []
    for recipe in RECIPES:
        for signal, threshold in thresholds:
            gate = Recovery(recipe, signal, threshold)
            result = effect(rows, gate)
            if usable(result) and sum(effect(panel, gate)["gain"] > 0 for panel in panels.values()) >= 2:
                rank = (result["gain"], result["complete_gain"], signal is None, -result["activations"])
                candidates.append((rank, gate))
    return max(candidates, key=lambda item: item[0], default=(None, None))[1]


def fit(panels):
    held_out, predictions = {}, []
    for name, rows in panels.items():
        gate = select_gate({key: panel for key, panel in panels.items() if key != name})
        held_out[name] = {"gate": gate.record() if gate else None, "effect": effect(rows, gate)}
        predictions.extend(contact.gate_outcome(row, gate) for row in rows)
    gate = select_gate(panels)
    admitted = (gate is not None and contact.eligible(predictions)
                and sum(item["effect"]["gain"] > 0 for item in held_out.values()) >= 2)
    return ((gate,) if admitted else ()), {"held_out": held_out, "admitted": admitted, "fold_predictions": predictions}


def run(root, release, archive, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected aligned seeds and a positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint, timing = release / frozen["checkpoint"], tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*item) for item in frozen["corrections"])
    common = {"environment": simulator_identity(), "incumbent": frozen["corrections"], "timing": list(timing),
              "rules": RULES, "strict_rules": STRICT_RULES}
    if (common["environment"]["sim_revision"] != frozen["sim_revision"] or sha256(checkpoint) != frozen["checkpoint_sha256"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("release recovery requires the frozen published controller")
    published, release_sources = first.historical_rows(release, (), frozen, common, checkpoint, timing, incumbent)
    _, panels, archive_sources = late.routed_panels(archive, checkpoint, timing, common)
    source_panels = {name: [pair["strict-500"] for pair in panel["reference"]] for name, panel in panels.items()}
    for phase in ("confirmation", "replication"):
        phase_protocol = read_json(release / f"{phase}-protocol.json")
        seeds = phase_protocol["schedule"][phase]
        seeds = set(seeds["random"] if isinstance(seeds, dict) else seeds)
        rows = [row for row in published if row["seed"] in seeds]
        if {row["seed"] for row in rows} != seeds:
            raise ValueError("published panel differs from its reserved seed block")
        source_panels[f"release-{phase}"] = rows
    history = {row["seed"]: row for rows in source_panels.values() for row in rows}
    if len(history) != sum(map(len, source_panels.values())):
        raise ValueError("duplicate historical source scene")
    plan = {}
    for name, rows in source_panels.items():
        cases = hover.plan_cases([{"strict-500": row} for row in rows])
        plan[name] = {"failures": cases["failures"][:4], "controls": cases["controls"][:3]}
    training = [seed for group in plan.values() for seed in group["failures"] + group["controls"]]
    development = [seed for seed in range(block, block + 64) if choose_column("random", seed) == COLUMN]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    if reserved & set(history) or any(path.parent != root and reserved & hover.scheduled_seeds(read_json(path).get("schedule", {}))
                                     for path in root.parent.glob("*/protocol.json")):
        raise ValueError("development and confirmation require untouched scenes")
    protocol = root / "protocol.json"
    _record_once(protocol, {**common, "reference_sha256": sha256(manifest), "code_sha256": sha256(Path(__file__)),
                 "contact_code_sha256": sha256(Path(contact.__file__)),
                 "sources": {"release": release_sources, "routed": archive_sources}, "sampling": plan,
                 "schedule": {"training": training, "development": development, "confirmation": confirmation},
                 "window": WINDOW, "signals": list(SIGNALS), "extra_steps": EXTRA_STEPS,
                 "recipes": {name: list(values) for name, values in RECIPES.items()}, "confirmation_requires": routed.REQUIREMENTS})
    policy_type = partial(RecoveryPolicy, incumbent=incumbent)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent collecting {len(training)} contact-matched release recovery scenes", flush=True)
        jobs = [(checkpoint, timing, incumbent, seed) for seed in training]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]),
                     partial(audit_training, history=history))
        indexed = {row["seed"]: row for row in rows}
        samples = {name: [indexed[seed] for seed in group["failures"] + group["controls"]] for name, group in plan.items()}
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"panels": samples})
        model, metrics = fit(samples)
        records = [item.record() for item in model]
        fit_path = root / "fit.json"
        _record_once(fit_path, {**metrics, "corrections": records, "dataset_sha256": sha256(dataset)})
        if not model:
            return {"status": "reference_retained_after_release_recovery_training", **metrics}
        print(f"Agent froze a release recovery for {len(development)} unseen pairs", flush=True)
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                              {"reference": (), "selected": model}, policy_type=policy_type)
        scores = hover.pair_stats(paired, model)
        selection = root / "development-selection.json"
        _record_once(selection, scores)
        if not all(placement_search.passes(stats) for stats in scores.values()):
            return {"status": "reference_retained_after_release_recovery_development", "development": scores, "corrections": records}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "seeds": confirmation, "corrections": records,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)})
        print(f"Agent started {len(confirmation)} independent four-column pairs", flush=True)
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                   model, confirmation[:4], policy_type=policy_type)
    if any(stats["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for stats in final.values()):
        raise ValueError("confirmation requires four-column balance")
    return {"status": "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_release_recovery_confirmation",
            "corrections": records, "development": scores, "confirmation": final,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Fit a contact-triggered lift before second-batch release.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--archive", type=Path, default=Path("evidence/routed-20261120"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=159000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.archive.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
