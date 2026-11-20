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
from a3opt import placement_hover_agent as hover, placement_late_hover_gate as late, placement_release
from a3opt import placement_routed_agent as routed, placement_search, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementTrace
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


COLUMN, BATCH = hover.COLUMN, hover.BATCH
RECIPES = {"planned-18": (.018, 0., 0.), "anchored-24": (.024, 1., 0.), "outward-24": (.024, 1., .004)}
MAX_ANCHOR_M = .012
MAX_JOINT_DELTA = .18
POSITION_TOLERANCE_M = .003


@dataclass(frozen=True)
class ReleaseHold:
    recipe: str
    column = COLUMN
    batch = BATCH

    def __post_init__(self):
        if self.recipe not in RECIPES:
            raise ValueError("unknown stationary release recipe")

    def record(self):
        return [self.column, self.batch, self.recipe]


def replan(phases, observed_xyz, yaw, recipe, solver):
    height, blend, outward = RECIPES[recipe]
    if not np.isfinite(observed_xyz).all() or not np.isfinite(yaw):
        return False
    start = phases[TRANSFER][-1, :7]
    descent_actions, opening, retreat = (phases[stage] for stage in (PLACE, OPEN, RETRACT))
    nominal = descent_actions[-1, :7].copy()
    target = descent.tool_xyz(solver, nominal)
    target[:2] += blend * np.clip(observed_xyz[:2] - target[:2], -MAX_ANCHOR_M, MAX_ANCHOR_M)
    target[:2] -= outward * np.array([np.cos(yaw), np.sin(yaw)])
    target[2] += height
    if not 0. <= observed_xyz[2] - target[2] <= .080:
        return False
    hold = nominal
    for _ in range(4):
        hold = solver.shift(hold, target - descent.tool_xyz(solver, hold))
        if not np.isfinite(hold).all() or np.max(np.abs(hold - nominal)) > MAX_JOINT_DELTA:
            return False
        if np.linalg.norm(descent.tool_xyz(solver, hold) - target) <= POSITION_TOLERANCE_M:
            break
    else:
        return False
    weights = _ramp(np.arange(1, len(descent_actions) + 1) / len(descent_actions))[:, None]
    descent_actions[:, :7] = start + weights * (hold - start)
    opening[:, :7] = hold
    if len(retreat) > 1:
        weights = 1. - _ramp(np.linspace(0., 1., len(retreat)))[:, None]
        retreat[:, :7] = np.clip(retreat[:, :7] + weights * (hold - retreat[0, :7]), *solver.ranges.T)
    return True


class ReleaseHoldPolicy(descent.DescentPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        if len(corrections) > 1:
            raise ValueError("expected a single stationary release recipe")
        self.release_hold = corrections[0] if corrections else None
        super().__init__(checkpoint, focus, scale, incumbent)

    def act(self, observation, task=""):
        if (self.release_hold is not None and self.actions is not None
                and self.cursor == (BATCH, PLACE, 0) and not self.retry_remaining
                and task_column(task) == COLUMN):
            observed = np.asarray(observation["observation.eef_pose"][:3], dtype=float)
            replan(self.actions[BATCH], observed, self.pose[4], self.release_hold.recipe, nominal_kinematics())
        return super().act(observation, task)


def fork(env, policy, task, columns, recipe):
    branch = copy(env)
    branch.data = copy(env.data)
    controller = copy(policy)
    if recipe is not None:
        controller.actions = policy.actions.copy()
        controller.actions[BATCH] = [stage.copy() for stage in policy.actions[BATCH]]
        controller.release_hold = ReleaseHold(recipe)
    observation = branch._observation()
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated:
            break
    return placement_release.terminal(branch, info, columns, COLUMN, digest)


class ReleaseHoldTrace(PlacementTrace):
    def reset(self, context=None):
        super().reset(context)
        self.task = ""
        self.prefix = self.trials = self.suffix = self.continuation = None

    def act(self, observation, task):
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.suffix is not None:
            self.suffix.update(placement_train.command_bytes(action, info))
            if self.safety or self.step == RULES["max_steps"]:
                self.continuation = placement_release.terminal(env, info, self.columns, COLUMN, self.suffix)
        if (self.column == COLUMN and self.policy.cursor == (BATCH, PLACE, 0)
                and self.policy.stage == TRANSFER and self.policy.ended and not self.safety
                and self.step < RULES["max_steps"]):
            self.prefix = self.stages[BATCH]["transfer"]["prefix_sha256"]
            self.trials = {"reference": fork(env, self.policy, self.task, self.columns, None),
                           **{name: fork(env, self.policy, self.task, self.columns, name) for name in RECIPES}}
            self.suffix = hashlib.sha256()


def collect(checkpoint, timing, policy_type, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = ReleaseHoldTrace(policy_type(checkpoint, *timing))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "steps": result.steps, "score": result.score,
                "success": result.success, "target_cookie_ids": np.flatnonzero(trace.target).tolist(),
                "safety_reason": trace.safety, "action_sha256": trace.digest.hexdigest(),
                "applied_action_sha256": trace.applied_digest.hexdigest(), "prefix_sha256": trace.prefix,
                "trials": trace.trials, "continuation": trace.continuation}
    finally:
        env.close()


def audit_training(job, row, *, history):
    checkpoint, _, _, seed = job
    prior = history[seed]
    if (row["seed"] != seed or row["column"] != COLUMN or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint)
            or row["prefix_sha256"] != prior["phase_geometry"][BATCH]["transfer"]["prefix_sha256"]
            or any(row[key] != prior[key] for key in hover.MATCH_FIELDS)):
        raise ValueError("stationary release training did not replay the archived controller")
    if (row["trials"] is None or set(row["trials"]) != {"reference", *RECIPES}
            or row["trials"]["reference"] != row["continuation"]
            or row["continuation"]["counted_ids"] != row["target_cookie_ids"]):
        raise ValueError("unchanged fork does not match the reference continuation")


def fit(rows, controls):
    metrics = {}
    for recipe in RECIPES:
        changes = {}
        strict_delta = unsafe = 0
        for row in rows:
            reference, trial = (row["trials"][name] for name in ("reference", recipe))
            changes[row["seed"]] = len(trial["counted_ids"]) - len(reference["counted_ids"])
            strict_delta += int(trial["success"]) - int(reference["success"])
            unsafe += bool(trial["safety_reason"] or trial["wrong_source"])
        gain = sum(changes.values())
        improved = sum(delta > 0 for delta in changes.values())
        harmed = sum(delta < 0 for delta in changes.values())
        admitted = (gain > 0 and improved >= 2 and strict_delta >= 0 and not unsafe
                    and all(delta >= -1 for delta in changes.values())
                    and all(changes[seed] >= 0 for seed in controls))
        metrics[recipe] = {"gain": gain, "strict_delta": strict_delta, "improved": improved, "harmed": harmed,
                           "unsafe": unsafe, "changes": {str(seed): delta for seed, delta in changes.items()},
                           "admitted": admitted}
    selected = max((name for name in RECIPES if metrics[name]["admitted"]), default=None,
                   key=lambda name: (metrics[name]["gain"], metrics[name]["strict_delta"], -RECIPES[name][0]))
    return (ReleaseHold(selected),) if selected else (), metrics


def run(root, release, archive, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected aligned seeds and a positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint, timing = release / frozen["checkpoint"], tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*record) for record in frozen["corrections"])
    common = {"environment": simulator_identity(), "incumbent": frozen["corrections"], "timing": list(timing),
              "rules": RULES, "strict_rules": STRICT_RULES}
    if (common["environment"]["sim_revision"] != frozen["sim_revision"] or sha256(checkpoint) != frozen["checkpoint_sha256"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("stationary release requires the frozen published controller")
    published, release_sources = first.historical_rows(release, (), frozen, common, checkpoint, timing, incumbent)
    _, panels, archive_sources = late.routed_panels(archive, checkpoint, timing, common)
    history = {row["seed"]: row for row in published}
    for panel in panels.values():
        for pair in panel["reference"]:
            row = pair["strict-500"]
            if row["seed"] in history:
                raise ValueError("duplicate historical reference scene")
            history[row["seed"]] = row
    plan = hover.plan_cases([{"strict-500": row} for row in history.values()])
    training = plan["failures"] + plan["controls"]
    development = [seed for seed in range(block, block + 64) if choose_column("random", seed) == COLUMN]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    if reserved & set(history) or any(path.parent != root and reserved & hover.scheduled_seeds(read_json(path).get("schedule", {}))
                                     for path in root.parent.glob("*/protocol.json")):
        raise ValueError("development and confirmation require untouched scenes")
    protocol = root / "protocol.json"
    _record_once(protocol, {**common, "reference_sha256": sha256(manifest), "code_sha256": sha256(Path(__file__)),
                 "sources": {"release": release_sources, "routed": archive_sources}, "sampling": plan,
                 "schedule": {"training": training, "development": development, "confirmation": confirmation},
                 "recipes": {name: list(values) for name, values in RECIPES.items()},
                 "confirmation_requires": routed.REQUIREMENTS})
    if len(plan["failures"]) < 2 or len(plan["controls"]) < 2:
        return {"status": "reference_retained_without_release_hold_cases"}
    policy_type = partial(ReleaseHoldPolicy, incumbent=incumbent)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent collecting {len(training)} matched second-batch release scenes", flush=True)
        jobs = [(checkpoint, timing, policy_type, seed) for seed in training]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]),
                     partial(audit_training, history=history))
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"sampling": plan, "rows": rows})
        model, metrics = fit(rows, plan["controls"])
        records = [item.record() for item in model]
        fit_path = root / "fit.json"
        _record_once(fit_path, {"corrections": records, "metrics": metrics, "dataset_sha256": sha256(dataset)})
        if not model:
            return {"status": "reference_retained_after_release_hold_training", "metrics": metrics}
        print(f"Agent froze {model[0].recipe} for {len(development)} unseen development pairs", flush=True)
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                              {"reference": (), "selected": model}, policy_type=policy_type)
        scores = hover.pair_stats(paired, model)
        selection = root / "development-selection.json"
        _record_once(selection, scores)
        if not all(placement_search.passes(stats) for stats in scores.values()):
            return {"status": "reference_retained_after_release_hold_development", "development": scores,
                    "corrections": records}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "seeds": confirmation, "corrections": records,
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)})
        print(f"Agent started {len(confirmation)} independent four-column pairs", flush=True)
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                   model, confirmation[:4], policy_type=policy_type)
    if any(stats["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for stats in final.values()):
        raise ValueError("confirmation requires four-column balance")
    return {"status": "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_release_hold_confirmation",
            "corrections": records, "development": scores, "confirmation": final,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Collect and select fixed-pose second-batch release trajectories.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--archive", type=Path, default=Path("evidence/routed-20261120"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=158000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.archive.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
