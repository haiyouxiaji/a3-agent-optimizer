from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_descent_agent as descent, placement_first_hover as first
from a3opt import placement_hover_agent as hover, placement_release, placement_routed_agent as routed
from a3opt import placement_search, placement_servo_agent as servo, placement_train, placement_wall_clearance_agent as wall
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementTrace
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


WINDOW = .55
RECIPES = {"raise8": (0., .008), "raise16": (0., .016), "inset4-raise12": (.004, .012)}
SIGNALS = ("wrist_force_change", "wrist_torque_change", "gripper_load_change", "vertical_tracking_error")


@dataclass(frozen=True)
class ContactGate:
    column: int
    batch: int
    recipe: str
    signal: int
    above: bool
    threshold: float

    def __post_init__(self):
        if (self.column not in range(1, 5) or self.batch not in range(2) or self.recipe not in RECIPES
                or self.signal not in range(len(SIGNALS)) or not np.isfinite(self.threshold)):
            raise ValueError("invalid contact guard")

    def activates(self, values):
        value = values[self.signal]
        return np.isfinite(value) and (value >= self.threshold if self.above else value <= self.threshold)

    def record(self):
        return [self.column, self.batch, self.recipe, self.signal, self.above, self.threshold]


def force_observation(observation):
    forces = np.asarray(observation["observation.force"], dtype=float)
    return forces[[0, 1, 2, 3, 4, 5, 16]]


class ContactPolicy(descent.DescentPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent, probe=None):
        self.contact_gates = {(gate.column, gate.batch): gate for gate in corrections}
        self.probe = probe
        if len(self.contact_gates) != len(corrections):
            raise ValueError("duplicate contact gate")
        super().__init__(checkpoint, focus, scale, incumbent)

    def reset(self, context=None):
        super().reset(context)
        self.initial_force = None

    def pending(self, task):
        if self.actions is None or self.cursor is None or self.cursor[1] != PLACE or self.retry_remaining:
            return None
        key = task_column(task), self.cursor[0]
        return key if key == self.probe or key in self.contact_gates else None

    def at_window(self):
        batch, _, frame = self.cursor
        return frame == max(1, int(len(self.actions[batch][PLACE]) * WINDOW))

    def contact_signals(self, observation):
        batch, _, frame = self.cursor
        forces = force_observation(observation) - self.initial_force
        target = self.actions[batch][PLACE][frame - 1, :7]
        commanded_z = descent.tool_xyz(nominal_kinematics(), target)[2]
        return np.r_[np.linalg.norm(forces[:3]), np.linalg.norm(forces[3:6]), abs(forces[6]),
                     float(observation["observation.eef_pose"][2]) - commanded_z]

    def act(self, observation, task=""):
        key = self.pending(task)
        if key is not None:
            if self.cursor[2] == 0:
                self.initial_force = force_observation(observation)
            elif self.at_window() and key in self.contact_gates:
                gate = self.contact_gates[key]
                if self.initial_force is not None and gate.activates(self.contact_signals(observation)):
                    batch, _, frame = self.cursor
                    apply_correction(self, batch, frame, gate.recipe)
        return super().act(observation, task)


def apply_correction(controller, batch, frame, recipe):
    inset, rise = RECIPES[recipe]
    yaw = controller.pose[4]
    lateral = -wall.OUTWARD[batch] * inset
    offset = np.array([lateral * np.cos(yaw), lateral * np.sin(yaw), rise])
    solver = nominal_kinematics()
    phases = controller.actions[batch]
    remaining = phases[PLACE][frame:]
    release, retract = phases[OPEN], phases[RETRACT]
    for actions, weights in ((remaining, _ramp(np.arange(1, len(remaining) + 1) / len(remaining))),
                             (release, np.ones(len(release))),
                             (retract, 1. - _ramp(np.arange(1, len(retract) + 1) / len(retract)))):
        for action, weight in zip(actions, weights, strict=True):
            if weight:
                action[:7] = solver.shift(action[:7], weight * offset)


def fork(env, policy, task, observation, columns, column, recipe):
    branch = copy(env)
    branch.data = copy(env.data)
    controller = copy(policy)
    batch, _, frame = controller.cursor
    if recipe != "reference":
        controller.actions = policy.actions.copy()
        controller.actions[batch] = [phase.copy() for phase in policy.actions[batch]]
        apply_correction(controller, batch, frame, recipe)
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated or (controller.batch == batch and controller.stage == RETRACT and controller.ended):
            break
    return placement_release.terminal(branch, info, columns, column, digest)


class ContactTrace(PlacementTrace):
    def bind_env(self, env):
        super().bind_env(env)
        self.env = env

    def reset(self, context=None):
        super().reset(context)
        self.trials = self.suffix = self.continuation = None

    def act(self, observation, task):
        policy = self.policy
        if policy.pending(task) is not None and policy.at_window() and policy.initial_force is not None:
            values = policy.contact_signals(observation)
            self.trials = {"step": self.step, "features": values.tolist(), "prefix_sha256": self.digest.hexdigest(),
                           "outcomes": {name: fork(self.env, policy, task, observation, self.columns, self.column, name)
                                        for name in ("reference", *RECIPES)}}
            self.suffix = hashlib.sha256()
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.suffix is not None:
            self.suffix.update(placement_train.command_bytes(action, info))
            policy = self.policy
            if (self.step == RULES["max_steps"] or self.safety
                    or (policy.batch == policy.probe[1] and policy.stage == RETRACT and policy.ended)):
                self.continuation = placement_release.terminal(env, info, self.columns, self.column, self.suffix)
                self.suffix = None


def collect(checkpoint, timing, incumbent, column, batch, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = ContactTrace(ContactPolicy(checkpoint, *timing, incumbent=incumbent, probe=(column, batch)))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "column": result.source_column, "batch": batch, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "timing": list(timing), "action_sha256": trace.digest.hexdigest(),
                "applied_action_sha256": trace.applied_digest.hexdigest(), "score": result.score,
                "target_cookie_ids": np.flatnonzero(trace.target).tolist(), "safety_reason": trace.safety,
                "trials": trace.trials, "continuation": trace.continuation}
    finally:
        env.close()


def audit(job, row):
    checkpoint, timing, _, column, batch, seed = job
    trials = row["trials"]
    if (row["seed"] != seed or row["column"] != column or row["batch"] != batch or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != list(timing)
            or trials is None or len(trials["features"]) != len(SIGNALS) or not np.isfinite(trials["features"]).all()
            or set(trials["outcomes"]) != {"reference", *RECIPES} or row["continuation"] != trials["outcomes"]["reference"]):
        raise ValueError("contact control fork does not reproduce the reference episode")


def sample_plan(history):
    choices = []
    for column in range(1, 5):
        members = [{**row, "terminal": row["physical_terminal"]} for row in history
                   if row["column"] == column and not row["safety_reason"] and row["score"] == row["requested_score"]]
        controls = sorted(row["seed"] for row in members if row["success"])
        batches = []
        for batch in range(2):
            losses = [(servo.lost_during_descent(row, batch), row["seed"]) for row in members]
            failures = [seed for lost, seed in sorted(losses, key=lambda item: (-item[0], item[1])) if lost]
            if len(failures) >= 4 and len(controls) >= 3:
                batches.append((sum(lost for lost, _ in losses), batch, failures[:4]))
        if batches:
            total, batch, failures = max(batches, key=lambda item: (item[0], -item[1]))
            choices.append({"column": column, "batch": batch, "lost_during_descent": total,
                            "failures": failures, "controls": controls[:3]})
    return sorted(choices, key=lambda item: (-item["lost_during_descent"], item["column"]))[:2]


def gate_outcome(row, gate):
    sample = row["trials"]
    options = sample["outcomes"]
    triggered = gate is not None and gate.activates(sample["features"])
    original, selected = options["reference"], options[gate.recipe] if triggered else options["reference"]
    return {"seed": row["seed"], "triggered": bool(triggered), "gain": len(selected["counted_ids"]) - len(original["counted_ids"]),
            "complete_gain": int(selected["success"]) - int(original["success"]),
            "unsafe": bool(selected["safety_reason"] or selected["wrong_source"])}


def eligible(outcomes):
    return (sum(outcome["gain"] > 0 for outcome in outcomes) >= 2 and sum(outcome["complete_gain"] for outcome in outcomes) >= 0
            and not any(outcome["unsafe"] or outcome["gain"] < 0 for outcome in outcomes))


def best_gate(rows, column, batch):
    options = []
    for signal in range(len(SIGNALS)):
        values = sorted({row["trials"]["features"][signal] for row in rows})
        for low, high in zip(values, values[1:]):
            for recipe in RECIPES:
                for above in (True, False):
                    gate = ContactGate(column, batch, recipe, signal, above, (low + high) / 2.)
                    outcomes = [gate_outcome(row, gate) for row in rows]
                    if eligible(outcomes):
                        rank = (sum(row["gain"] for row in outcomes), sum(row["complete_gain"] for row in outcomes),
                                -sum(row["triggered"] for row in outcomes), -RECIPES[recipe][1])
                        options.append((rank, gate))
    return max(options, key=lambda option: option[0])[1] if options else None


def fit(plan, rows):
    decisions, models = [], []
    for item in plan:
        column, batch = item["column"], item["batch"]
        cases = [row for row in rows if row["column"] == column]
        gate = best_gate(cases, column, batch)
        leave_one_out = [gate_outcome(row, best_gate([other for other in cases if other["seed"] != row["seed"]], column, batch))
                         for row in cases]
        admitted = gate is not None and eligible(leave_one_out)
        decisions.append({"column": column, "batch": batch, "gate": gate.record() if gate else None,
                          "leave_one_out": leave_one_out, "admitted": admitted})
        if admitted:
            models.append(gate)
    return tuple(models), decisions


def run(root, release, source, block, workers):
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
        raise ValueError("contact feedback requires the frozen published controller")
    history, archive = first.historical_rows(release, (), frozen, common, checkpoint, timing, incumbent)
    if source is not None:
        observed, fingerprints = wall.read_training_source(source, common, checkpoint, timing)
        history += observed
        archive["routed_replication"] = fingerprints
    previous = {row["seed"]: row for row in history}
    if len(previous) != len(history):
        raise ValueError("duplicated contact training source")
    plan = sample_plan(history)
    jobs = [(checkpoint, timing, incumbent, group["column"], group["batch"], seed)
            for group in plan for seed in group["failures"] + group["controls"]]
    columns = {item["column"] for item in plan}
    development = [seed for seed in range(block, block + 48) if choose_column("random", seed) in columns]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    if reserved & set(previous) or any(path.parent != root and reserved & hover.scheduled_seeds(read_json(path).get("schedule", {}))
                                       for path in root.parent.glob("*/protocol.json")):
        raise ValueError("contact development and confirmation require unseen scenes")
    protocol = root / "protocol.json"
    _record_once(protocol, {**common, "reference_manifest_sha256": sha256(manifest), "sources": archive, "sampling": plan,
                 "schedule": {"training": [job[-1] for job in jobs], "development": development, "confirmation": confirmation},
                 "code_sha256": sha256(Path(__file__)), "window": WINDOW, "signals": list(SIGNALS),
                 "training_endpoint": "targeted_retract_or_episode_end",
                 "recipes": {name: list(values) for name, values in RECIPES.items()}, "confirmation_requires": routed.REQUIREMENTS})
    if not jobs:
        return {"status": "reference_retained_without_contact_cases"}
    policy_type = partial(ContactPolicy, incumbent=incumbent)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent selected {len(jobs)} scenes for matched descent interventions", flush=True)
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]), audit)
        match = ("score", "target_cookie_ids", "safety_reason", "action_sha256", "applied_action_sha256")
        if any(row[key] != previous[row["seed"]][key] for row in rows for key in match):
            raise ValueError("contact control episode did not match its original receipt")
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"plan": plan, "rows": rows})
        model, decisions = fit(plan, rows)
        fit_path = root / "fit.json"
        records = [gate.record() for gate in model]
        _record_once(fit_path, {"decisions": decisions, "corrections": records, "dataset_sha256": sha256(dataset)})
        if not model:
            return {"status": "reference_retained_after_contact_training", "decisions": decisions}
        development = [seed for seed in development if choose_column("random", seed) in {gate.column for gate in model}]
        print(f"Agent selected {len(model)} descent gates for {len(development)} new development pairs", flush=True)
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                              {"reference": (), "selected": model}, policy_type=policy_type)
        results = hover.pair_stats(paired, model)
        selection = root / "development-selection.json"
        _record_once(selection, results)
        if not all(placement_search.passes(row) for row in results.values()):
            return {"status": "reference_retained_after_contact_development", "development": results, "corrections": records}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "seeds": confirmation, "corrections": records,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)})
        print(f"Agent froze the contact policy before {len(confirmation)} four-column confirmation pairs", flush=True)
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                   model, confirmation[:4], policy_type=policy_type)
    if any(row["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for row in final.values()):
        raise ValueError("confirmation requires four-column balance")
    return {"status": "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_contact_confirmation",
            "development": results, "confirmation": final, "corrections": records,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Train a bounded contact-triggered correction during descent.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=152000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.source.resolve() if args.source else None, args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
