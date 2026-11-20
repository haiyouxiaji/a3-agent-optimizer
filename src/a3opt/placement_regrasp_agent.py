from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_grasp, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import CLOSE, LIFT, OPEN, grip_features, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_agent import reference_setup
from a3opt.placement_release import terminal
from a3opt.placement_search import TARGET, compare, confirm_combined, encode, passes
from a3opt.placement_sim import PlacementPolicy, PlacementTrace
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES


RECIPES = ("reference", "lower-reclose", "withdraw-insert", "nudge-left", "nudge-right")
GAPS = (.06, .10, .14, .18)
RETRY_STEPS = 12
FRESH_EPISODES = 20


@dataclass(frozen=True)
class Regrasp:
    column: int
    gap: float
    recipe: str

    def __post_init__(self):
        if self.column not in range(1, 5) or self.gap not in GAPS or self.recipe not in RECIPES[1:]:
            raise ValueError("invalid gripper recovery model")

    def record(self):
        return [self.column, self.gap, self.recipe]

    @property
    def batch(self):
        return 1


def insert_retry(policy, observation, recipe):
    if recipe == "reference":
        return
    phases = policy.actions[1]
    reserve = placement_grasp.EXTRA_STEPS if recipe == "lower-reclose" else RETRY_STEPS
    if policy.step + reserve + sum(map(len, phases[LIFT:OPEN + 1])) > RULES["max_steps"]:
        return
    if recipe == "lower-reclose":
        placement_grasp.recover(policy, recipe)
        return
    offset = {"withdraw-insert": 0., "nudge-left": -.005, "nudge-right": .005}[recipe]
    solver = nominal_kinematics()
    closed = phases[CLOSE][-1]
    measured = np.asarray(observation["observation.state"], dtype=float)
    opening = max(closed[7], min(.70, measured[7] + .045))
    raised = solver.shift(closed[:7], np.array([0., 0., .005]))
    reinserted = solver.shift(closed[:7], np.array([0., offset, -.004]))
    anchors = np.repeat(closed[None], 3, axis=0)
    anchors[0, :8] = measured[:8]
    anchors[1, :7], anchors[1, 7] = raised, opening
    anchors[2, :7] = reinserted
    steps = RETRY_STEPS // 3
    weights = _ramp(np.arange(1, steps + 1) / steps)[:, None]
    prefix = np.concatenate([anchors[0] + weights * (anchors[1] - anchors[0]),
                             anchors[1] + weights * (anchors[2] - anchors[1]),
                             np.repeat(anchors[2:], steps, axis=0)])
    lift = phases[LIFT]
    blend = min(8, len(lift))
    fade = 1. - _ramp(np.linspace(0., 1., blend))
    lift[:blend, :7] += fade[:, None] * (reinserted - lift[0, :7])
    phases[LIFT] = np.concatenate((prefix, lift))


class RegraspPolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.regrasp = {entry.column: entry for entry in corrections}
        if len(self.regrasp) != len(corrections):
            raise ValueError("duplicate gripper recovery model")
        super().__init__(checkpoint, focus, scale)

    def act(self, observation, task=""):
        if self.actions is not None and self.cursor == (1, LIFT, 0):
            config = self.regrasp.get(task_column(task))
            if config is not None:
                features = grip_features(observation, self.actions[1][CLOSE][-1])
                if features[1] >= config.gap:
                    insert_retry(self, observation, config.recipe)
        return super().act(observation, task)


def finish_branch(env, policy, task, recipe, columns, column):
    branch = placement_grasp.cloned(env)
    controller = copy(policy)
    controller.actions = [policy.actions[0], [stage.copy() for stage in policy.actions[1]]]
    observation = branch._observation()
    insert_retry(controller, observation, recipe)
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated:
            break
    return terminal(branch, info, columns, column, digest)


class RecoveryTrace(PlacementTrace):
    def __init__(self, policy, probe_control):
        self.probe_control = probe_control
        super().__init__(policy)

    def reset(self, context=None):
        super().reset(context)
        self.task = ""
        self.gap = self.prefix = self.trials = self.original = self.digest_suffix = None

    def act(self, observation, task):
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.digest_suffix is not None:
            self.digest_suffix.update(placement_train.command_bytes(action, info))
            if self.safety or self.step == RULES["max_steps"]:
                self.original = terminal(env, info, self.columns, self.column, self.digest_suffix)
        if self.policy.cursor == (1, LIFT, 0) and not self.safety and self.step < RULES["max_steps"]:
            observed = env._observation()
            self.gap = float(grip_features(observed, self.policy.actions[1][CLOSE][-1])[1])
            if self.probe_control or self.gap >= min(GAPS):
                self.prefix = {"step": self.step, "physics": placement_train.physics_digest(env),
                               "raw": self.digest.hexdigest(), "applied": self.applied_digest.hexdigest()}
                self.trials = {recipe: finish_branch(env, self.policy, self.task, recipe, self.columns, self.column) for recipe in RECIPES}
                self.digest_suffix = hashlib.sha256()


def collect(checkpoint, focus, scale, seed, probe_control):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = RecoveryTrace(PlacementPolicy(checkpoint, focus, scale), probe_control)
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "timing": [focus, scale], "probe_control": probe_control,
                "gap": trace.gap, "lifted": trace.events[1].get("lifted_requested"), "prefix": trace.prefix,
                "score": result.score, "terminal_ids": np.flatnonzero(trace.target).tolist(),
                "safety": trace.safety, "trials": trace.trials, "original": trace.original}
    finally:
        env.close()


def audit(job, row):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, focus, scale, seed, probe_control = job
    if (row["seed"] != seed or row["column"] != choose_column("random", seed) or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != [focus, scale]
            or row["probe_control"] != probe_control or row["score"] != len(row["terminal_ids"])):
        raise ValueError("regrasp receipt differs from its acquisition request")
    gap, trials = row["gap"], row["trials"]
    if gap is not None and not np.isfinite(gap):
        raise ValueError("invalid gripper gap observation")
    expected = gap is not None and (probe_control or gap >= min(GAPS))
    if expected != bool(trials) or bool(trials) != bool(row["prefix"]):
        raise ValueError("regrasp receipt lacks its requested paired branches")
    if trials and (set(trials) != set(RECIPES) or row["original"] != trials["reference"]
                   or trials["reference"]["counted_ids"] != row["terminal_ids"]
                   or any(not row["prefix"]["step"] < item["step"] <= RULES["max_steps"] for item in trials.values())):
        raise ValueError("regrasp branches failed replay parity or the step budget")


def fit(rows, column):
    blocked = [row for row in rows if row["gap"] is not None and row["gap"] >= min(GAPS) and row["lifted"] is not None
               and row["lifted"] < 5 and row["column"] == column]
    normal = [row["gap"] for row in rows if row["lifted"] == 5 and row["gap"] is not None and row["column"] == column]
    report = []
    for gap in GAPS:
        if not normal or max(normal) >= gap:
            continue
        active = [row for row in rows if row["column"] == column and row["gap"] is not None and row["gap"] >= gap]
        for recipe in RECIPES[1:]:
            totals = [len(row["trials"][recipe]["counted_ids"]) - row["score"] for row in active]
            violations = sum(bool(row["trials"][recipe]["safety_reason"] or row["trials"][recipe]["wrong_source"]
                                  or (row["original"]["success"] and not row["trials"][recipe]["success"])) for row in active)
            report.append({"gap": gap, "recipe": recipe, "active": [row["seed"] for row in active], "net_gain": sum(totals),
                           "gains": sum(delta > 0 for delta in totals), "min_delta": min(totals, default=0), "violations": violations})
    eligible = [item for item in report if item["gains"] >= 2 and item["net_gain"] >= 3 and item["min_delta"] >= 0 and not item["violations"]]
    best = max(eligible, default=None, key=lambda item: (item["net_gain"], item["gains"], -len(item["active"]), item["gap"]))
    model = () if best is None else (Regrasp(column, best["gap"], best["recipe"]),)
    return model, {"column": column, "screened": report, "blocked_seeds": [row["seed"] for row in blocked],
                   "normal_grasps": len(normal), "normal_maximum_gap": max(normal, default=None), "corrections": encode(model)}


def acquisition_plan(prior_rows, block):
    from a3_dual_arm_sim.sim.randomization import choose_column

    failures = [row for row in prior_rows if not row["safety_reason"] and row["features"] and row["features"][1] >= min(GAPS)
                and row["lifted"] is not None and row["lifted"] < 5]
    if not failures:
        return None, [], []
    counts = {column: sum(row["column"] == column for row in failures) for column in range(1, 5)}
    focus = max(counts, key=counts.get)
    fresh = [seed for seed in range(block, block + 4 * FRESH_EPISODES) if choose_column("random", seed) == focus]
    archived = [row["seed"] for row in failures if row["column"] == focus]
    return focus, archived + fresh, fresh[:2]


def run(root, reference, timing_run, prior, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and verified joint evaluator are required")
    development, confirmation = list(range(block + 100, block + 112)), list(range(block + 200, block + 220))
    splits = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    prior_path, prior_data = prior / "protocol.json", prior / "training" / "dataset.json"
    manifest, dataset = read_json(prior_path), read_json(prior_data)["rows"]
    if (any(manifest[key] != identity[key] for key in ("environment", "timing", "reference_sha256", "placement_sim_sha256", "strict_rules"))
            or manifest["acquisition_code_sha256"] != sha256(Path(placement_grasp.__file__))
            or read_json(prior / "result.json")["dataset_sha256"] != sha256(prior_data)
            or manifest["schedule"]["training"] != [row["seed"] for row in dataset]
            or set(manifest["schedule"]["training"]).intersection(range(block, block + 220))):
        raise ValueError("regrasp screening requires audited, disjoint training evidence")
    for row in dataset:
        placement_grasp.audit((checkpoint, *timing, row["seed"], row["seed"] in manifest["control_seeds"]), row)
    column, seeds, controls = acquisition_plan(dataset, block)
    splits["training"] = sorted(set(seeds) | set(manifest["schedule"]["training"]))
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "probe_seeds": seeds, "control_seeds": controls,
                 "focus_column": column, "recipes": list(RECIPES), "gaps": list(GAPS), "retry_steps": RETRY_STEPS,
                 "prior_data_sha256": sha256(prior_data), "prior_protocol_sha256": sha256(prior_path),
                 "parity_sha256": sha256(parity), "code_sha256": sha256(Path(__file__)),
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__)), "target_mean_strictly_above": TARGET})
    if column is None:
        return {"status": "reference_retained_without_observed_gripper_obstruction"}
    print(f"Reserved {len(seeds)} training scenes in column {column}; {len(controls)} forced controls", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(checkpoint, *timing, seed, seed in controls) for seed in seeds]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-2]), audit)
        samples = root / "training" / "dataset.json"
        _record_once(samples, {"rows": rows})
        model, evidence = fit(rows, column)
        _record_once(root / "fit.json", {**evidence, "dataset_sha256": sha256(samples)})
        print(f"Evaluated {len(rows)} training scenes; selected {len(model)} gripper recovery policies", flush=True)
        if not model:
            return {"status": "reference_retained_after_training", "blocked_scenes": len(evidence["blocked_seeds"])}
        pairs = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                             {"reference": (), "selected": model}, policy_type=RegraspPolicy)
        stats = {rule: compare_regrasp([row[rule] for row in pairs["reference"]], [row[rule] for row in pairs["selected"]], model)
                 for rule in ("default", "strict-500")}
        _record_once(root / "development-selection.json", stats)
        if not all(passes(item) for item in stats.values()):
            return {"status": "reference_retained_after_development", "development": stats}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model, confirmation[:4],
                                   policy_type=RegraspPolicy, compare_rows=compare_regrasp)
    primary = checked["default"]
    accepted = primary["mean"] > TARGET and all(passes(item) for item in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation", "development": stats,
            "confirmation": checked, "active_policy_above_target": accepted, "target_point_estimate": primary["mean"] > TARGET,
            "target_lower_bound": primary["mean_ci95"][0] > TARGET}


def compare_regrasp(reference, selected, model):
    columns = {entry.column for entry in model}
    for control, trial in zip(reference, selected, strict=True):
        if control["column"] in columns and control["phase_geometry"][0] != trial["phase_geometry"][0]:
            raise ValueError("second-batch recovery changed the first-batch trajectory")
    return compare(reference, selected, model, prefix_batch=0)


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_REGRASP_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("gripper recovery requires the frozen checkpoint")
    return RegraspPolicy(checkpoint, *lock["timing"], tuple(Regrasp(*item) for item in lock["corrections"]))


def main():
    parser = argparse.ArgumentParser(description="Learn a bounded second-batch recovery for obstructed grippers.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-block", type=int, default=125000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.prior.resolve(), args.parity.resolve(),
                     args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
