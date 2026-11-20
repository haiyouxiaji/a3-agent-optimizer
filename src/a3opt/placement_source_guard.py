from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark as benchmark, placement_expert_landing as landing
from a3opt import placement_descent_agent as descent, placement_first_hover as first
from a3opt import placement_hover_agent as hover, placement_interbatch as interbatch
from a3opt import placement_release_recovery as recovery, placement_search
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import CLOSE, LIFT, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


OPTIONS = {"close3": (.003, True), "close5": (.005, True), "lift5": (.005, False)}
SECOND = 1


@dataclass(frozen=True)
class SourceGuard:
    column: int
    neighbor: int
    recipe: str
    batch = SECOND

    def __post_init__(self):
        if self.column not in range(1, 5) or self.neighbor not in range(1, 5) or abs(self.column - self.neighbor) != 1:
            raise ValueError("expected an adjacent source column")
        if self.recipe not in OPTIONS:
            raise ValueError("unknown source separation recipe")

    def record(self):
        return [self.column, self.neighbor, self.recipe]


def separate(phases, guard):
    offset, during_close = OPTIONS[guard.recipe]
    shift = np.array([np.sign(guard.column - guard.neighbor) * offset, 0., 0.])
    solver = nominal_kinematics()
    for stage in (CLOSE, LIFT):
        actions = phases[stage]
        progress = np.linspace(0., 1., len(actions))
        weights = _ramp(2. * progress) if stage == CLOSE else 1. - _ramp((progress - .35) / .55)
        if not during_close:
            weights *= 0. if stage == CLOSE else _ramp(progress / .2)
        for action, weight in zip(actions, weights, strict=True):
            if weight:
                action[:7] = solver.shift(action[:7], weight * shift)


class SourceGuardPolicy(landing.ExpertLandingPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent, late_recovery):
        guards = [item for item in corrections if isinstance(item, SourceGuard)]
        if len(guards) > 1:
            raise ValueError("expected at most one source guard")
        self.guard = guards[0] if guards else None
        super().__init__(checkpoint, focus, scale, tuple(item for item in corrections if isinstance(item, landing.Landing)),
                         incumbent=incumbent, late_recovery=late_recovery)

    def act(self, observation, task=""):
        if (self.guard is not None and self.actions is not None and self.cursor == (SECOND, CLOSE, 0)
                and not self.retry_remaining and task_column(task) == self.guard.column):
            separate(self.actions[SECOND], self.guard)
        return super().act(observation, task)


class SourceTrials(benchmark.CombinedTrace):
    def __init__(self, policy, options):
        super().__init__(policy)
        self.options = options

    def bind_env(self, env):
        super().bind_env(env)
        self.env = env

    def reset(self, context=None):
        super().reset(context)
        self.branches = None

    def branch(self, observation, task, guard):
        env = copy(self.env)
        env.data = copy(self.env.data)
        controller = copy(self.policy)
        controller.actions = [[phase.copy() for phase in batch] for batch in self.policy.actions]
        controller.retries = self.policy.retries.copy()
        controller.guard = guard
        trace = benchmark.CombinedTrace(controller)
        excluded = {"policy", "env", "options", "branches", "cameras", "digest", "applied_digest"}
        trace.__dict__.update({key: deepcopy(value) for key, value in vars(self).items() if key not in excluded})
        trace.digest, trace.applied_digest = self.digest.copy(), self.applied_digest.copy()
        while trace.step < RULES["max_steps"]:
            action = trace.act(observation, task)
            next_observation, _, terminated, truncated, info = env.step(action)
            trace.add_frame(observation, action, info, env=env)
            observation = next_observation
            if terminated or truncated:
                break
        return interbatch.endpoints(trace)

    def act(self, observation, task):
        if (self.policy.actions is not None and self.policy.cursor == (SECOND, CLOSE, 0)
                and not self.policy.retry_remaining and self.branches is None):
            self.branches = {guard.recipe: self.branch(observation, task, guard) for guard in self.options}
        return super().act(observation, task)


def collect(checkpoint, timing, constructor, options, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = SourceTrials(constructor(checkpoint, *timing), options)
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "reference": interbatch.endpoints(trace),
                "candidates": trace.branches}
    finally:
        env.close()


def audit_trial(job, row, *, archive):
    checkpoint, _, _, options, seed = job
    initial = archive[seed]
    expected = {rule: {key: initial[rule][key] for key in interbatch.ENDPOINTS} for rule in initial}
    if (row["seed"] != seed or row["column"] != options[0].column or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or row["reference"] != expected or row["candidates"] is None
            or set(row["candidates"]) != {item.recipe for item in options}):
        raise ValueError(f"source-guard collection differs from the archived control for seed {seed}")


def assess(pairs):
    minimum_gain = 0
    regressions = wrong = safety = completion_losses = 0
    for old, new in pairs:
        gains = [new[rule]["requested_score"] - old[rule]["requested_score"] for rule in old]
        minimum_gain += min(gains)
        regressions += min(gains) < 0
        wrong += any(row["score"] != row["requested_score"] for row in new.values())
        safety += any(row["safety_reason"] is not None for row in new.values())
        completion_losses += any(old[rule]["success"] and not new[rule]["success"] for rule in old)
    return {"net_gain": minimum_gain, "regressions": regressions, "wrong_source": wrong,
            "safety_stops": safety, "completion_losses": completion_losses}


def eligible(stats):
    return not any(stats[key] for key in ("regressions", "wrong_source", "safety_stops", "completion_losses"))


def scope_of(rows):
    incidence = {}
    for episode in rows:
        row = episode["strict-500"]
        column = row["column"]
        for neighbor in range(1, 5):
            excess = row["by_column"][neighbor - 1]
            if abs(column - neighbor) == 1 and excess:
                incidence.setdefault((column, neighbor), []).append(row["seed"])
    return max(incidence.items(), key=lambda item: (len(item[1]), item[0]), default=None)


def load_parent(source):
    lock_path, protocol_path = source / "confirmation-lock.json", source / "protocol.json"
    rows_path = source / "confirmation" / "rollouts" / "results.json"
    lock, protocol, paired = map(read_json, (lock_path, protocol_path, rows_path))
    checkpoint, timing = Path(lock["checkpoint"]), tuple(lock["timing"])
    if (protocol["environment"] != simulator_identity() or (protocol["rules"], protocol["strict_rules"]) != (RULES, STRICT_RULES)
            or lock["protocol_sha256"] != sha256(protocol_path) or lock["checkpoint_sha256"] != sha256(checkpoint)
            or protocol["code_sha256"] != sha256(Path(landing.__file__)) or protocol["dependency_sha256"] != landing.dependencies()
            or lock["seeds"] != protocol["schedule"]["confirmation"]):
        raise ValueError("source-guard learning requires the frozen parent comparison")
    incumbent = tuple(descent.Descent(*item) for item in lock["incumbent"])
    late_recovery = tuple(recovery.Recovery(*item[2:]) for item in lock["late_recovery"])
    release_anchor = tuple(landing.Landing(column, batch, tuple(anchor), tuple(center), tuple(map(tuple, coefficients)), tuple(axes))
                           for column, batch, anchor, center, coefficients, axes in lock["corrections"])
    controls = {episode["default"]["seed"]: episode for episode in paired["reference"]}
    for seed, episode in zip(lock["seeds"], paired["reference"], strict=True):
        benchmark.check((checkpoint, *timing, seed, ()), episode)
    for seed, episode in zip(lock["seeds"], paired["selected"], strict=True):
        benchmark.check((checkpoint, *timing, seed, release_anchor), episode)
    recorded = read_json(source / "result.json")
    if hover.pair_stats(paired, release_anchor) != recorded["confirmation"]:
        raise ValueError("source-guard learning found a misstated parent panel")
    policy_type = partial(SourceGuardPolicy, incumbent=incumbent, late_recovery=late_recovery)
    return checkpoint, timing, policy_type, release_anchor, controls, lock, sha256(lock_path), sha256(rows_path)


def run(source, root, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected an aligned seed block and a positive worker budget")
    checkpoint, timing, constructor, release_anchor, controls, parent, lock_hash, rows_hash = load_parent(source)
    scope = scope_of(controls.values())
    if scope is None:
        return {"status": "reference_retained_without_adjacent_source_errors"}
    (column, neighbor), failures = scope
    healthy = [seed for seed, episode in controls.items() if episode["strict-500"]["column"] == column
               and episode["strict-500"]["success"] and seed not in failures]
    if len(healthy) < 4:
        return {"status": "reference_retained_without_source_guard_controls"}
    training = failures + [healthy[index] for index in (0, len(healthy) // 3, 2 * len(healthy) // 3, len(healthy) - 1)]
    development = [seed for seed in range(block, block + 32) if choose_column("random", seed) == column]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    archives = [*root.parent.glob("*/protocol.json"), *root.parent.glob("*/replication-plan.json"),
                *checkpoint.parent.parent.glob("*/*protocol.json")]
    if any(reserved & hover.scheduled_seeds(read_json(path).get("schedule", {})) for path in archives if path.parent != root):
        raise ValueError("source guard requires unused development and confirmation seeds")
    protocol = root / "protocol.json"
    _record_once(protocol, {"environment": simulator_identity(), "rules": RULES, "strict_rules": STRICT_RULES,
                 "checkpoint_sha256": sha256(checkpoint), "code_sha256": sha256(Path(__file__)),
                 "parent_lock_sha256": lock_hash, "parent_rollouts_sha256": rows_hash,
                 "incumbent": parent["incumbent"], "late_recovery": parent["late_recovery"], "timing": list(timing),
                 "scope": [column, neighbor], "failures": failures, "options": {name: list(values) for name, values in OPTIONS.items()},
                 "schedule": {"training": training, "development": development, "confirmation": confirmation}})
    options = tuple(SourceGuard(column, neighbor, recipe) for recipe in OPTIONS)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent is collecting {len(training)} paired source-separation training scenes", flush=True)
        jobs = [(checkpoint, timing, constructor, options, seed) for seed in training]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]),
                     partial(audit_trial, archive={seed: controls[seed] for seed in training}))
        data_path = root / "training" / "dataset.json"
        _record_once(data_path, {"rows": rows})
        scores = {option.recipe: assess((row["reference"], row["candidates"][option.recipe]) for row in rows) for option in options}
        choice = max((option for option in options if eligible(scores[option.recipe])),
                     key=lambda option: (scores[option.recipe]["net_gain"], -OPTIONS[option.recipe][0]), default=None)
        fit_path = root / "fit.json"
        _record_once(fit_path, {"dataset_sha256": sha256(data_path), "training": scores, "guard": choice.record() if choice else None})
        if choice is None:
            return {"status": "reference_retained_after_source_guard_training", "training": scores}
        print(f"Agent selected {choice.recipe} for {len(development)} fresh source-column pairs", flush=True)
        paired = benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                    {"reference": (), "selected": (choice,)}, policy_type=constructor)
        development_scores = assess(zip(paired["reference"], paired["selected"], strict=True))
        selection_path = root / "development-selection.json"
        _record_once(selection_path, development_scores)
        if not eligible(development_scores):
            return {"status": "reference_retained_after_source_guard_development", "development": development_scores}
        candidate = (*release_anchor, choice)
        records = [item.record() for item in candidate]
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection_path),
                     "incumbent": parent["incumbent"], "late_recovery": parent["late_recovery"], "timing": list(timing),
                     "seeds": confirmation, "corrections": records})
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation, candidate,
                                                   confirmation[:4], policy_type=constructor,
                                                   compare_rows=partial(placement_search.compare, prefix_batch=0))
    if any(stats["scored_pairs_by_column"] != dict.fromkeys(map(str, range(1, 5)), 16) for stats in final.values()):
        raise ValueError("source guard confirmation requires four equally represented columns")
    status = "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_source_guard_confirmation"
    return {"status": status, "confirmation": final, "development": development_scores, "corrections": records,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Learn a bounded grasp separation correction from observed adjacent-column carryover.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=170000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(args.source.resolve(), root, args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
