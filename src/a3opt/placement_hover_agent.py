from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_descent_agent, placement_release, placement_search, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import PLACE, TRANSFER, task_column
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementTrace
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


COLUMN = 2
BATCH = 1
SCOUT_SCENES = 32
DEVELOPMENT_SCENES = 20
CONFIRMATION_SCENES = 64
MAX_PER_GROUP = 6
OFFSETS = {"rise-12mm": (0., 0., .012), "rise-20mm": (0., 0., .020), "rise-28mm": (0., 0., .028),
           "rise-20mm-in": (-.005, 0., .020), "rise-20mm-out": (.005, 0., .020)}
MATCH_FIELDS = ("steps", "score", "target_cookie_ids", "success", "safety_reason", "action_sha256", "applied_action_sha256")


@dataclass(frozen=True)
class Hover:
    recipe: str
    column = COLUMN
    batch = BATCH

    def __post_init__(self) -> None:
        if self.recipe not in OFFSETS:
            raise ValueError("unknown hover placement recipe")

    def record(self) -> list:
        return [self.column, self.batch, self.recipe]


class HoverPolicy(placement_descent_agent.DescentPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        if len(corrections) > 1:
            raise ValueError("expected a single hover placement recipe")
        super().__init__(checkpoint, focus, scale, incumbent)
        self.hover = corrections[0] if corrections else None

    def act(self, observation, task=""):
        if self.hover is not None and self.actions is not None and self.cursor == (BATCH, PLACE, 0) and task_column(task) == COLUMN:
            self.shift_placement(BATCH, np.asarray(OFFSETS[self.hover.recipe]))
        return super().act(observation, task)


def schedule(block, choose_column):
    if block < 0 or block % 4:
        raise ValueError("seed block must start on a four-column boundary")

    def focused(start, count):
        seeds = range(start, start + 4 * count)
        selected = [seed for seed in seeds if choose_column("random", seed) == COLUMN]
        if len(selected) != count:
            raise ValueError("expected one active-column scene in every block of four")
        return selected

    return {"scout": focused(block, SCOUT_SCENES), "development": focused(block + 200, DEVELOPMENT_SCENES),
            "confirmation": list(range(block + 400, block + 400 + CONFIRMATION_SCENES))}


def scheduled_seeds(value):
    if isinstance(value, dict):
        value = value.values()
    elif not isinstance(value, (tuple, list)):
        return {value} if type(value) is int else set()
    return {seed for item in value for seed in scheduled_seeds(item)}


def plan_cases(rows):
    failures, controls = [], []
    for pair in rows:
        row = pair["strict-500"]
        if (row["column"] != COLUMN or row["safety_reason"] or row["score"] != row["requested_score"]):
            continue
        stages = row["phase_geometry"][BATCH]
        transferred = stages.get("transfer", {}).get("batches", [[], []])[BATCH]
        if len(transferred) != 5 or min(cookie["abs_vertical_axis"] for cookie in transferred) < .98:
            continue
        missed = 5 - sum(cookie["scored"] for cookie in row["physical_terminal"]["batches"][BATCH])
        tilt = min((cookie["abs_vertical_axis"] for stage in ("descent", "open") if stage in stages
                    for cookie in stages[stage]["batches"][BATCH]), default=1.)
        if missed and tilt < .97:
            failures.append((-missed, tilt, row["seed"]))
        elif row["requested_score"] == 10:
            controls.append((not row["success"], row["seed"]))
    return {"failures": [seed for _, _, seed in sorted(failures)[:MAX_PER_GROUP]],
            "controls": [seed for _, seed in sorted(controls)[:MAX_PER_GROUP]]}


def fork(env, policy, task, columns, recipe):
    branch = copy(env)
    branch.data = copy(env.data)
    controller = copy(policy)
    if recipe is not None:
        controller.actions = policy.actions.copy()
        controller.actions[BATCH] = [stage.copy() for stage in policy.actions[BATCH]]
        controller.hover = Hover(recipe)
    observation = branch._observation()
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated:
            break
    return placement_release.terminal(branch, info, columns, COLUMN, digest)


class HoverTrace(PlacementTrace):
    def reset(self, context=None):
        super().reset(context)
        self.task = ""
        self.trials = self.suffix = self.continuation = None

    def act(self, observation, task):
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.suffix is not None:
            self.suffix.update(placement_train.command_bytes(action, info))
            if self.safety or self.step == RULES["max_steps"]:
                self.continuation = placement_release.terminal(env, info, self.columns, COLUMN, self.suffix)
        if (self.column == COLUMN and self.policy.batch == BATCH and self.policy.stage == TRANSFER
                and self.policy.ended and not self.safety and self.step < RULES["max_steps"]):
            self.trials = {"reference": fork(env, self.policy, self.task, self.columns, None),
                           **{name: fork(env, self.policy, self.task, self.columns, name) for name in OFFSETS}}
            self.suffix = hashlib.sha256()


def collect(checkpoint, timing, policy_type, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = HoverTrace(policy_type(checkpoint, *timing))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": result.seed, "column": result.source_column, "steps": result.steps,
                "profile": result.randomization["profile"], "score": result.score, "success": result.success,
                "target_cookie_ids": np.flatnonzero(trace.target).tolist(), "safety_reason": trace.safety,
                "action_sha256": trace.digest.hexdigest(), "applied_action_sha256": trace.applied_digest.hexdigest(),
                "trials": trace.trials, "continuation": trace.continuation, "wall_seconds": result.wall_seconds}
    finally:
        env.close()


def audit_training(job, row, *, scout):
    seed = job[-1]
    expected = scout[seed]
    if (row["seed"] != seed or row["column"] != COLUMN or row["profile"] != "advanced"
            or any(row[key] != expected[key] for key in MATCH_FIELDS)):
        raise ValueError("training scene does not replay the frozen scout reference")
    if (row["trials"] is None or set(row["trials"]) != {"reference", *OFFSETS}
            or row["trials"]["reference"] != row["continuation"]
            or row["continuation"]["counted_ids"] != row["target_cookie_ids"]):
        raise ValueError("unchanged fork does not reproduce the reference suffix")


def fit(rows, controls):
    metrics = {}
    for recipe in OFFSETS:
        changes, unsafe, complete_delta = {}, 0, 0
        for row in rows:
            control, candidate = (row["trials"][key] for key in ("reference", recipe))
            changes[row["seed"]] = len(candidate["counted_ids"]) - len(control["counted_ids"])
            unsafe += bool(candidate["safety_reason"] or candidate["wrong_source"])
            complete_delta += int(candidate["success"]) - int(control["success"])
        improved = sum(delta > 0 for delta in changes.values())
        gain = sum(changes.values())
        losses = sum(delta < 0 for delta in changes.values())
        eligible = (improved >= 2 and gain > 0 and complete_delta >= 0 and not unsafe
                    and all(delta > -placement_search.MAJOR_LOSS for delta in changes.values())
                    and all(changes[seed] >= 0 for seed in controls))
        metrics[recipe] = {"changes": {str(seed): delta for seed, delta in changes.items()}, "improved": improved,
                           "gain": gain, "losses": losses, "complete_delta": complete_delta, "unsafe": unsafe, "eligible": eligible}
    selected = max((name for name, row in metrics.items() if row["eligible"]), default=None,
                   key=lambda name: (metrics[name]["gain"], metrics[name]["complete_delta"], -metrics[name]["losses"]))
    return {"selected": selected, "candidates": metrics}


def pair_stats(paired, model):
    return {rule: placement_search.compare([row[rule] for row in paired["reference"]],
                                          [row[rule] for row in paired["selected"]], model)
            for rule in ("default", "strict-500")}


def run(root, release, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    release_path = release / "controller.json"
    frozen = read_json(release_path)
    checkpoint = release / frozen["checkpoint"]
    environment = simulator_identity()
    published_protocol = release / "confirmation-protocol.json"
    if (sha256(checkpoint) != frozen["checkpoint_sha256"] or sha256(published_protocol) != frozen["confirmation_protocol_sha256"]
            or sha256(Path(placement_descent_agent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]
            or environment["sim_revision"] != frozen["sim_revision"]):
        raise ValueError("the installed reference differs from the published controller")
    incumbent = tuple(placement_descent_agent.Descent(*record) for record in frozen["corrections"])
    if any(item.column == COLUMN for item in incumbent):
        raise ValueError("the hover search requires an independent active column")
    timing = tuple(frozen["timing"])
    policy_type = partial(HoverPolicy, incumbent=incumbent)
    splits = schedule(block, choose_column)
    historical = [published_protocol, release / "replication-protocol.json",
                  *(path for path in root.parent.glob("*/protocol.json") if path.parent != root)]
    for path in historical:
        if scheduled_seeds(splits) & scheduled_seeds(read_json(path).get("schedule", {})):
            raise ValueError(f"new seed block overlaps {path}")
    protocol = root / "protocol.json"
    _record_once(protocol, {"schedule": splits, "profile": "advanced", "environment": environment,
                            "rules": RULES, "strict_rules": STRICT_RULES, "reference_sha256": sha256(release_path),
                            "reference_checkpoint_sha256": sha256(checkpoint), "timing": list(timing),
                            "incumbent": frozen["corrections"], "code_sha256": sha256(Path(__file__)),
                            "selection_code_sha256": sha256(Path(placement_search.__file__)),
                            "offsets_m": {key: list(value) for key, value in OFFSETS.items()},
                            "confirmation_requires": ["course_mean_above_9", "positive_lower_bound_on_paired_gain_both_rules",
                                                      "no_major_losses", "no_safety_or_wrong_source", "no_success_count_regression"]})
    print(f"Scouting {len(splits['scout'])} fresh column-{COLUMN} scenes", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scout = placement_benchmark.evaluate(pool, root / "scout", checkpoint, timing, splits["scout"],
                                             {"reference": ()}, policy_type=policy_type)["reference"]
        cases = plan_cases(scout)
        _record_once(root / "acquisition-plan.json", {**cases, "scout_sha256": sha256(root / "scout" / "results.json")})
        if min(map(len, cases.values())) < 2:
            return {"status": "reference_retained_without_training_support", "cases": cases}
        print(f"Forking {len(cases['failures'])} failures and {len(cases['controls'])} scored controls", flush=True)
        expected = {pair["strict-500"]["seed"]: pair["strict-500"] for pair in scout}
        jobs = [(checkpoint, timing, policy_type, seed) for seed in cases["failures"] + cases["controls"]]
        samples = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]),
                        partial(audit_training, scout=expected))
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"rows": samples})
        fitted = fit(samples, cases["controls"])
        _record_once(root / "fit.json", {**fitted, "dataset_sha256": sha256(dataset)})
        if fitted["selected"] is None:
            return {"status": "reference_retained_after_training", "fit": fitted}
        model = (Hover(fitted["selected"]),)
        print(f"Evaluating {fitted['selected']} on {len(splits['development'])} development pairs", flush=True)
        trials = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, splits["development"],
                                               {"reference": (), "selected": model}, policy_type=policy_type)
        development = pair_stats(trials, model)
        selection = root / "development-selection.json"
        _record_once(selection, development)
        if not all(placement_search.passes(stats) for stats in development.values()):
            return {"status": "reference_retained_after_development", "candidate": fitted["selected"], "development": development}
        seeds = splits["confirmation"]
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "corrections": [model[0].record()], "seeds": seeds,
                     "protocol_sha256": sha256(protocol), "selection_sha256": sha256(selection),
                     "fit_sha256": sha256(root / "fit.json")})
        print(f"Frozen {fitted['selected']} before {len(seeds)} four-column confirmation pairs", flush=True)
        confirmation = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, seeds,
                                                         model, seeds[:4], policy_type=policy_type)
    expected_counts = Counter({str(column): len(seeds) // 4 for column in range(1, 5)})
    if any(Counter(stats["scored_pairs_by_column"]) != expected_counts for stats in confirmation.values()):
        raise ValueError("confirmation is not balanced across all four source columns")
    primary = confirmation["default"]
    accepted = primary["mean"] > placement_search.TARGET and all(
        placement_search.passes(stats) and stats["paired_count_ci95"][0] > 0 for stats in confirmation.values())
    return {"status": "candidate_promoted" if accepted else "reference_retained_after_confirmation",
            "candidate": fitted["selected"], "development": development, "confirmation": confirmation,
            "target_lower_bound": primary["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Learn a bounded second-batch hover release above the frozen advanced policy.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=140000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
