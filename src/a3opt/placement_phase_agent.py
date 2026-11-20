from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_release, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, TRANSFER, task_column
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_agent import reference_setup
from a3opt.placement_search import TARGET, compare, confirm_combined, encode, passes, reuse_training
from a3opt.placement_sim import PlacementPolicy, PlacementTrace
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES


RECIPES = ("reference", "soft-landing", "early-unload", "settle-before-open")
SETTLE_STEPS = 4
MAX_FAILURES_PER_BATCH = 4
MAX_CONTROLS_PER_BATCH = 2


@dataclass(frozen=True)
class PhaseTiming:
    column: int
    batch: int
    recipe: str

    def __post_init__(self) -> None:
        if self.column not in range(1, 5) or self.batch not in (0, 1) or self.recipe not in RECIPES[1:]:
            raise ValueError("invalid placement phase timing")

    def record(self) -> list:
        return [self.column, self.batch, self.recipe]


def adjust(policy: PlacementPolicy, batch: int, recipe: str) -> None:
    phases = policy.actions[batch]
    descent, opening = phases[PLACE], phases[OPEN]
    if recipe == "soft-landing":
        phase = np.linspace(0., 1., len(descent))
        warped = phase + .65 * phase * (1. - phase)
        phases[PLACE] = np.column_stack([np.interp(warped, phase, command) for command in descent.T])
    elif recipe == "early-unload":
        overlap = min(5, len(descent) - 1, len(opening) - 1)
        if overlap:
            descent[-overlap:, 7] = np.maximum(descent[-overlap:, 7], opening[1:overlap + 1, 7])
            opening[:, 7] = np.r_[opening[overlap:, 7], np.repeat(opening[-1, 7], overlap)]
    elif recipe == "settle-before-open":
        to_open_end = policy.step + len(descent) + SETTLE_STEPS + len(opening)
        if to_open_end <= RULES["max_steps"]:
            phases[PLACE] = np.concatenate((descent, np.repeat(descent[-1:], SETTLE_STEPS, axis=0)))
    elif recipe != "reference":
        raise ValueError("unknown placement phase timing recipe")


class PhaseTimingPolicy(PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.phase_timing = {(item.column, item.batch): item.recipe for item in corrections}
        if len(self.phase_timing) != len(corrections):
            raise ValueError("duplicate placement phase timing")
        super().__init__(checkpoint, focus, scale)

    def act(self, observation, task=""):
        if self.actions is not None and self.cursor is not None and self.cursor[1:] == (PLACE, 0):
            batch = self.cursor[0]
            recipe = self.phase_timing.get((task_column(task), batch))
            if recipe is not None:
                adjust(self, batch, recipe)
        return super().act(observation, task)


def fork(env, policy, task, batch, recipe, columns, column):
    branch = copy(env)
    branch.data = copy(env.data)
    controller = copy(policy)
    controller.actions = policy.actions.copy()
    controller.actions[batch] = [stage.copy() for stage in policy.actions[batch]]
    adjust(controller, batch, recipe)
    observation = branch._observation()
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated:
            break
    return placement_release.terminal(branch, info, columns, column, digest)


class TimingSamples(PlacementTrace):
    def __init__(self, policy, batch):
        self.probed_batch = batch
        super().__init__(policy)

    def reset(self, context=None):
        super().reset(context)
        self.task = ""
        self.trials = self.suffix = self.control = self.prefix = None

    def act(self, observation, task):
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.suffix is not None:
            self.suffix.update(placement_train.command_bytes(action, info))
            if self.safety or self.step == RULES["max_steps"]:
                self.control = placement_release.terminal(env, info, self.columns, self.column, self.suffix)
        policy, batch = self.policy, self.probed_batch
        if policy.batch != batch or policy.stage != TRANSFER or not policy.ended or self.safety or self.step >= RULES["max_steps"]:
            return
        self.prefix = {"step": self.step, "physics_sha256": placement_train.physics_digest(env),
                       "actions_sha256": self.stages[batch]["transfer"]["prefix_sha256"]}
        self.trials = {recipe: fork(env, policy, self.task, batch, recipe, self.columns, self.column) for recipe in RECIPES}
        self.suffix = hashlib.sha256()


def collect(checkpoint, focus, scale, batch, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = TimingSamples(PlacementPolicy(checkpoint, focus, scale), batch)
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "batch": batch, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "timing": [focus, scale], "score": result.score,
                "terminal_ids": np.flatnonzero(trace.target).tolist(), "control": trace.control, "trials": trace.trials,
                "prefix": trace.prefix, "wall_seconds": result.wall_seconds}
    finally:
        env.close()


def audit(job, row):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, focus, scale, batch, seed = job
    if (row["seed"] != seed or row["batch"] != batch or row["column"] != choose_column("random", seed)
            or row["profile"] != "advanced" or row["checkpoint_sha256"] != sha256(checkpoint)
            or row["timing"] != [focus, scale] or row["score"] != len(row["terminal_ids"])):
        raise ValueError("phase timing receipt differs from the training request")
    if row["trials"] is None and row["control"] is None and row["prefix"] is None:
        return
    if (row["prefix"] is None or set(row["trials"] or ()) != set(RECIPES) or row["control"] != row["trials"]["reference"]
            or row["control"]["counted_ids"] != row["terminal_ids"]
            or any(not row["prefix"]["step"] < trial["step"] <= RULES["max_steps"] for trial in row["trials"].values())):
        raise ValueError("phase timing fork failed replay parity or the step budget")


def training_plan(rows):
    plan = []
    for batch in (0, 1):
        failures, controls = {}, {}
        for row in rows:
            transfer = row["phase_geometry"][batch].get("transfer", {}).get("batches", ((), ()))[batch]
            if row["safety_reason"] or len(transfer) != 5:
                continue
            counted = set(row["terminal"]["counted_ids"])
            lost = sum(cookie["id"] not in counted for cookie in transfer)
            if lost:
                failures.setdefault(row["column"], []).append((-lost, row["seed"]))
            elif row["score"] == 10:
                controls.setdefault(row["column"], []).append(row["seed"])
        ordered = [sorted(entries) for entries in failures.values()]
        distinct = sorted(group[0] for group in ordered)
        remaining = sorted(entry for group in ordered for entry in group[1:])
        selected = [seed for _, seed in (distinct + remaining)[:MAX_FAILURES_PER_BATCH]]
        if not selected:
            continue
        relevant = sorted(min(controls[column]) for column in failures.keys() & controls.keys())
        fallback = sorted(set(seed for entries in controls.values() for seed in entries) - set(relevant))
        for seed in selected + (relevant + fallback)[:MAX_CONTROLS_PER_BATCH]:
            plan.append((batch, seed))
    return plan


def fit(rows):
    screened = []
    for batch in (0, 1):
        valid = [row for row in rows if row["batch"] == batch and row["trials"] is not None]
        columns = sorted({row["column"] for row in valid})
        for scope in [tuple(columns), *((column,) for column in columns if len(columns) > 1)]:
            if not scope:
                continue
            group = [row for row in valid if row["column"] in scope]
            for recipe in RECIPES[1:]:
                deltas, success, violations = [], 0, 0
                for row in group:
                    control, trial = row["control"], row["trials"][recipe]
                    deltas.append(len(trial["counted_ids"]) - len(control["counted_ids"]))
                    success += int(trial["success"]) - int(control["success"])
                    violations += bool(trial["wrong_source"] or trial["safety_reason"])
                screened.append({"batch": batch, "columns": list(scope), "recipe": recipe, "scenes": len(group),
                                 "controls": sum(len(row["control"]["counted_ids"]) == 10 for row in group),
                                 "improved": sum(delta > 0 for delta in deltas), "net": sum(deltas), "worst": min(deltas),
                                 "complete_delta": success, "violations": violations})
    eligible = [row for row in screened if row["controls"] and row["improved"] >= 2 and row["net"] > 0
                and row["worst"] >= -1 and row["complete_delta"] >= 0 and not row["violations"]]
    selected = max(eligible, key=lambda row: (row["net"], row["complete_delta"], row["improved"], -len(row["columns"])), default=None)
    model = () if selected is None else tuple(PhaseTiming(column, selected["batch"], selected["recipe"]) for column in selected["columns"])
    return model, {"screening": screened, "selected": selected, "corrections": encode(model)}


def run(root, reference, timing_run, sources, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and verified joint evaluator are required")
    development, confirmation = list(range(block, block + 12)), list(range(block + 100, block + 120))
    splits = {"scout": {"random": []}, "training": [], "development": development, "confirmation": {"random": confirmation}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    training = [reuse_training(source, checkpoint, timing, identity) for source in sources]
    rows = [row for group, _ in training for row in group]
    used = {row["seed"] for row in rows}
    if len(used) != len(rows) or used.intersection(development + confirmation):
        raise ValueError("phase timing training data must be unique and disjoint from evaluation")
    plan = training_plan(rows)
    splits["training"] = sorted(used)
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "probe_plan": [list(item) for item in plan],
                 "training_sources": [evidence for _, evidence in training], "recipes": list(RECIPES), "settle_steps": SETTLE_STEPS,
                 "code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "parity_sha256": sha256(parity), "target_mean_strictly_above": TARGET})
    print(f"Reserved {len(plan)} training probes with paired timing alternatives", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(checkpoint, *timing, batch, seed) for batch, seed in plan]
        samples = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: f"{job[-2]}/{job[-1]}", audit)
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"rows": samples})
        model, evidence = fit(samples)
        _record_once(root / "fit.json", {**evidence, "dataset_sha256": sha256(dataset)})
        print(f"Audited {len(samples)} training probes; selected {evidence['selected']}", flush=True)
        if not model:
            return {"status": "reference_retained_after_training", "selected": None}
        pairs = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                             {"reference": (), "selected": model}, policy_type=PhaseTimingPolicy)
        stats = {rule: compare([row[rule] for row in pairs["reference"]], [row[rule] for row in pairs["selected"]], model)
                 for rule in ("default", "strict-500")}
        _record_once(root / "development-selection.json", stats)
        if not all(passes(entry) for entry in stats.values()):
            return {"status": "reference_retained_after_development", "selected": evidence["selected"], "development": stats}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": confirmation, "corrections": encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        checked = confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, model, confirmation[:4],
                                   policy_type=PhaseTimingPolicy)
    primary = checked["default"]
    accepted = primary["mean"] > TARGET and all(passes(entry) for entry in checked.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation",
            "selected": evidence["selected"], "active_policy_above_target": accepted,
            "target_point_estimate": primary["mean"] > TARGET, "target_lower_bound": primary["mean_ci95"][0] > TARGET,
            "development": stats, "confirmation": checked}


def main():
    parser = argparse.ArgumentParser(description="Test contact timing interventions on placement failures from training.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed-block", type=int, default=124000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), [source.resolve() for source in args.source],
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
