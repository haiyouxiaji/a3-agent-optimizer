from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_descent_agent, placement_release, placement_search, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends import release_residual
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, TRANSFER, task_column
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_agent import reference_setup
from a3opt.placement_servo_agent import lost_during_descent, paired
from a3opt.placement_sim import PlacementPolicy, PlacementTrace
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES


RECIPES = {"clearance-4mm": (.004, False), "frame-4mm": (.004, True), "frame-7mm": (.007, True)}
MAX_RAISE_M = .008
MAX_ROTATION_RAD = .080


@dataclass(frozen=True)
class Clearance:
    column: int
    batch: int
    recipe: str
    floor_tool_z: float

    def __post_init__(self):
        if (self.column not in range(1, 5) or self.batch not in (0, 1) or self.recipe not in RECIPES
                or not .65 <= self.floor_tool_z <= .90):
            raise ValueError("invalid calibrated clearance")

    def record(self):
        return [self.column, self.batch, self.recipe, self.floor_tool_z]


def orientation_error(current, target):
    error = .5 * np.cross(current.T, target.T).sum(axis=0)
    size = np.linalg.norm(error)
    return error * min(1., MAX_ROTATION_RAD / max(size, MAX_ROTATION_RAD))


def shift_command(solver, joints, release_height, anchor, weight):
    start = np.clip(joints, *solver.ranges.T)
    solver.data.qpos[solver.positions] = start
    mj = solver.mujoco
    mj.mj_kinematics(solver.model, solver.data)
    rise = np.clip(release_height - solver.data.site_xpos[solver.site, 2], 0., MAX_RAISE_M)
    rotation = np.zeros(3) if anchor is None else orientation_error(solver.data.site_xmat[solver.site].reshape(3, 3), anchor)
    if not rise and not np.any(rotation):
        return joints
    mj.mj_comPos(solver.model, solver.data)
    mj.mj_jacSite(solver.model, solver.data, solver.jac_position, solver.jac_rotation, solver.site)
    jacobian = np.vstack((solver.jac_position[:, solver.dofs], .10 * solver.jac_rotation[:, solver.dofs]))
    twist = np.r_[0., 0., rise, .10 * weight * rotation]
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .004**2 * np.eye(6), twist)
    magnitude = np.max(np.abs(delta))
    limit = release_residual.MAX_JOINT_CORRECTION
    return np.clip(start + delta * min(1., limit / max(magnitude, limit)), *solver.ranges.T)


def adjust(phases, pose, recipe, floor_tool_z):
    clearance, keep_frame = RECIPES[recipe]
    solver = release_residual.nominal_kinematics()
    anchor = None
    quaternion = np.asarray(pose[3:7], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if keep_frame and np.isfinite(norm) and norm > .9:
        anchor = np.empty((3, 3))
        solver.mujoco.mju_quat2Mat(anchor.ravel(), quaternion / norm)
    original_open = phases[OPEN][-1, :7].copy()
    for phase in (PLACE, OPEN):
        actions = phases[phase]
        for frame, action in enumerate(actions, start=1):
            weight = release_residual._ramp(4. * frame / len(actions)) if phase == PLACE else 1.
            action[:7] = shift_command(solver, action[:7], floor_tool_z + clearance, anchor, weight)
    retreat = phases[RETRACT]
    weights = 1. - release_residual._ramp(np.arange(1, len(retreat) + 1) / len(retreat))
    change = phases[OPEN][-1, :7] - original_open
    if np.any(change):
        retreat[:-1, :7] = np.clip(retreat[:-1, :7] + weights[:-1, None] * change, *solver.ranges.T)


class ClearancePolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, heights=None):
        self.profiles = {(item.column, item.batch): item.recipe for item in corrections}
        self.heights = heights if heights is not None else {(item.column, item.batch): item.floor_tool_z for item in corrections}
        if len(self.profiles) != len(corrections):
            raise ValueError("duplicate release clearance correction")
        super().__init__(checkpoint, focus, scale)

    def act(self, observation, task=""):
        if self.actions is not None and self.cursor is not None and self.cursor[1:] == (PLACE, 0):
            batch = self.cursor[0]
            key = task_column(task), batch
            if key in self.profiles:
                adjust(self.actions[batch], observation["observation.eef_pose"], self.profiles[key], self.heights[key])
        return super().act(observation, task)


def height_samples(rows, batch):
    samples = []
    for row in rows:
        event = row["phase_geometry"][batch].get("transfer")
        measurement = row["descent_observations"][batch]
        if event is None or measurement is None:
            continue
        held = event["batches"][batch]
        if len(held) != 5 or min(item["abs_vertical_axis"] for item in held) < .985:
            continue
        bottom = min(item["bottom_above_floor_m"] for item in held)
        if bottom >= .035:
            samples.append([row["seed"], measurement["eef_xyz_m"][2] - bottom])
    return samples


def sampling_plan(rows):
    plan = {}
    for column in range(1, 5):
        local = [row for row in rows if row["column"] == column and not row["safety_reason"]]
        for batch in range(2):
            measurements = height_samples(local, batch)
            errors = sorted((-lost_during_descent(row, batch), row["seed"]) for row in local)
            failures = [seed for loss, seed in errors if loss < 0]
            controls = sorted(row["seed"] for row in local if row["score"] == 10)
            if len(measurements) >= 3 and failures and len(controls) >= 2:
                plan[column, batch] = {"height_samples": measurements, "floor_tool_z": float(np.median([z for _, z in measurements])),
                                       "loss_count": -sum(loss for loss, _ in errors), "seeds": failures[:3] + controls[:2]}
    ranked = sorted(plan, key=lambda key: (-plan[key]["loss_count"], key))[:3]
    return {key: plan[key] for key in ranked}


class ClearanceTrace(PlacementTrace):
    def __init__(self, policy, batch):
        self.focus_batch = batch
        super().__init__(policy)

    def reset(self, context=None):
        super().reset(context)
        self.task = ""
        self.options = self.digest_suffix = self.continuation = self.prefix = None

    def act(self, observation, task):
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.digest_suffix is not None:
            self.digest_suffix.update(placement_train.command_bytes(action, info))
            if self.safety or self.step == RULES["max_steps"]:
                self.continuation = placement_release.terminal(env, info, self.columns, self.column, self.digest_suffix)
        if (self.policy.batch == self.focus_batch and self.policy.stage == TRANSFER and self.policy.ended
                and not self.safety and self.step < RULES["max_steps"]):
            self.prefix = self.stages[self.focus_batch]["transfer"]["prefix_sha256"]
            self.options = {"reference": placement_descent_agent.fork(env, self.policy, self.task, (), None, self.columns, self.column)}
            for recipe in RECIPES:
                self.options[recipe] = placement_descent_agent.fork(env, self.policy, self.task, (self.focus_batch,), recipe,
                                                                  self.columns, self.column)
            self.digest_suffix = hashlib.sha256()


def collect(checkpoint, focus, scale, seed, column, batch, floor):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = ClearanceTrace(ClearancePolicy(checkpoint, focus, scale, heights={(column, batch): floor}), batch)
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": result.seed, "column": result.source_column, "batch": batch, "profile": result.randomization["profile"],
                "score": result.score, "terminal_ids": np.flatnonzero(trace.target).tolist(), "checkpoint_sha256": sha256(checkpoint),
                "timing": [focus, scale], "floor_tool_z": floor, "prefix": trace.prefix, "options": trace.options,
                "continuation": trace.continuation, "wall_seconds": result.wall_seconds}
    finally:
        env.close()


def audit(job, row):
    checkpoint, focus, scale, seed, column, batch, floor = job
    if (row["seed"] != seed or row["column"] != column or row["batch"] != batch or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != [focus, scale]
            or row["floor_tool_z"] != floor or len(row["terminal_ids"]) != row["score"]):
        raise ValueError("calibrated clearance training receipt differs from its request")
    if row["options"] is not None and (not row["prefix"] or set(row["options"]) != {"reference", *RECIPES}
            or row["options"]["reference"] != row["continuation"] or row["continuation"]["counted_ids"] != row["terminal_ids"]):
        raise ValueError("calibrated clearance requires an exact replay of the baseline suffix")


def fit(plan, rows):
    model, evidence = [], {}
    for (column, batch), entry in plan.items():
        observed = [row["options"] for row in rows if row["column"] == column and row["batch"] == batch and row["options"] is not None]
        report, eligible = {}, {}
        for recipe in RECIPES:
            samples = [options[recipe] for options in observed]
            references = [options["reference"] for options in observed]
            changes = [len(new["counted_ids"]) - len(old["counted_ids"]) for old, new in zip(references, samples, strict=True)]
            controls = sum(len(old["counted_ids"]) == 10 for old in references)
            complete_delta = sum(new["success"] for new in samples) - sum(old["success"] for old in references)
            unsafe = sum(bool(row["wrong_source"] or row["safety_reason"]) for row in samples)
            report[recipe] = {"paired_changes": changes, "complete_delta": complete_delta, "controls": controls, "unsafe": unsafe}
            if (controls >= 2 and sum(change > 0 for change in changes) >= 2 and sum(changes) > 0 and complete_delta >= 0
                    and not unsafe and min(changes) > -placement_search.MAJOR_LOSS):
                eligible[recipe] = (sum(changes), complete_delta)
        selected = max(eligible, key=eligible.get, default=None)
        if selected:
            model.append(Clearance(column, batch, selected, entry["floor_tool_z"]))
        evidence[f"{column}/{batch}"] = {"selection": selected, "comparisons": report}
    return tuple(model), evidence


def run(root, reference, timing_run, sources, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("calibrated clearance requires a balanced schedule and native scoring audit")
    development = list(range(block, block + 20))
    confirmation = list(range(block + 100, block + 140))
    schedule = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, schedule)
    data = [placement_search.reuse_training(source, checkpoint, timing, identity) for source in sources]
    rows = [row for group, _ in data for row in group]
    archive = {row["seed"] for row in rows}
    if len(archive) != len(rows) or archive.intersection(development + confirmation):
        raise ValueError("calibrated clearance requires disjoint training and evaluation scenes")
    plan = sampling_plan(rows)
    schedule["training"] = sorted({seed for entry in plan.values() for seed in entry["seeds"]})
    reference_setup(reference, timing_run, schedule)
    _record_once(root / "protocol.json", {**identity, "schedule": schedule, "recipes": {name: list(values) for name, values in RECIPES.items()},
                 "max_raise_m": MAX_RAISE_M, "max_rotation_rad": MAX_ROTATION_RAD,
                 "training_sources": [evidence for _, evidence in data], "training_plan": {f"{c}/{b}": entry for (c, b), entry in plan.items()},
                 "target_mean_strictly_above": placement_search.TARGET, "parity_sha256": sha256(parity),
                 "code_sha256": sha256(Path(__file__)), "descent_sha256": sha256(Path(placement_descent_agent.__file__)),
                 "evaluator_sha256": sha256(Path(placement_benchmark.__file__))})
    if not plan:
        return {"status": "reference_retained_without_training_support"}
    jobs = [(checkpoint, *timing, seed, column, batch, entry["floor_tool_z"])
            for (column, batch), entry in plan.items() for seed in entry["seeds"]]
    print(f"Reserved {len(jobs)} calibrated clearance probes across {len(plan)} placement phases", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        data = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: f"{job[4]}-{job[5]}/{job[3]}", audit)
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"rows": data})
        selected, evidence = fit(plan, data)
        fit_path = root / "fit.json"
        _record_once(fit_path, {"evidence": evidence, "corrections": placement_search.encode(selected), "dataset_sha256": sha256(dataset)})
        print(f"Audited {len(data)} clearance probes; selected {placement_search.encode(selected)}", flush=True)
        if not selected:
            return {"status": "reference_retained_after_training"}
        rollouts = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                                {"reference": (), "selected": selected}, policy_type=ClearancePolicy)
        measured = paired(rollouts, "selected", selected)
        selection = root / "development-selection.json"
        _record_once(selection, measured)
        if not all(placement_search.passes(stats) for stats in measured.values()):
            return {"status": "reference_retained_after_development", "development": measured}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": placement_search.encode(selected),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(fit_path),
                     "selection_sha256": sha256(selection)})
        checked = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, selected,
                                                    confirmation[:4], policy_type=ClearancePolicy)
    primary = checked["default"]
    accepted = primary["mean"] > placement_search.TARGET and all(placement_search.passes(stats) for stats in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation", "development": measured,
            "confirmation": checked, "active_policy_above_target": accepted,
            "target_lower_bound": primary["mean_ci95"][0] > placement_search.TARGET}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_CLEARANCE_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("calibrated clearance requires the frozen reference checkpoint")
    return ClearancePolicy(checkpoint, *lock["timing"], tuple(Clearance(*item) for item in lock["corrections"]))


def main():
    parser = argparse.ArgumentParser(description="Fit permitted landing-height and tool-orientation corrections from training rollouts.")
    for name in ("out", "reference", "timing", "parity"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-block", type=int, default=136000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), [path.resolve() for path in args.source],
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
