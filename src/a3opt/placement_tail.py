from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import TRANSFER
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_agent import reference_setup
from a3opt.placement_search import MAJOR_LOSS, TARGET, compare, confirm_combined, encode, passes
from a3opt.placement_sim import Correction, PlacementPolicy, PlacementTrace, projected_tool_x
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES


OFFSETS = {"reference": (0., 0., 0.), "out3": (-.003, 0., 0.), "out6": (-.006, 0., 0.),
           "in6": (.006, 0., 0.), "raise6": (0., 0., .006), "raise12": (0., 0., .012),
           "out6_raise6": (-.006, 0., .006), "in6_raise6": (.006, 0., .006),
           "forward6": (0., .006, 0.), "back6": (0., -.006, 0.)}


def terminal_branch(env, policy: PlacementPolicy, task: str, offset: tuple[float, float, float]) -> dict:
    branch = copy(env)
    branch.data = copy(env.data)
    controller = copy(policy)
    if any(offset):
        controller.actions = [policy.actions[0], [phase.copy() for phase in policy.actions[1]]]
        controller.shift_placement(1, np.asarray(offset))
    observation = branch._observation()
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = branch.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated:
            break
    return placement_train.outcome(branch, info, np.empty(0, dtype=int), digest.hexdigest())


class TailTrace(PlacementTrace):
    def reset(self, context=None) -> None:
        super().reset(context)
        self.task = ""
        self.fork = self.suffix_digest = self.continuation = None

    def act(self, observation, task):
        self.task = task
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        if self.suffix_digest is not None:
            self.suffix_digest.update(placement_train.command_bytes(action, info))
            if self.step == RULES["max_steps"] or self.safety:
                self.continuation = placement_train.outcome(env, info, np.empty(0, dtype=int),
                                                            self.suffix_digest.hexdigest())
        if (self.policy.batch == 1 and self.policy.stage == TRANSFER and self.policy.ended
                and not self.safety and self.step < RULES["max_steps"]):
            branches = {}
            for name, offset in OFFSETS.items():
                outcome = terminal_branch(env, self.policy, self.task, offset)
                sources = self.columns[outcome["counted_ids"]]
                branches[name] = {"physics": outcome, "wrong_source": int(np.count_nonzero(sources != self.column))}
            self.fork = {"prefix_step": self.step, "prefix_sha256": self.stages[1]["transfer"]["prefix_sha256"],
                         "first_corrected": self.policy.first_corrected, "outcomes": branches}
            self.suffix_digest = hashlib.sha256()


def collect(checkpoint: Path, focus: int, scale: float, seed: int, first: tuple[Correction, ...]) -> dict:
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = TailTrace(PlacementPolicy(checkpoint, focus, scale, first))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        if trace.fork is not None:
            observed = trace.measurements[1]
            trace.fork["tool_x_m"] = projected_tool_x(observed["eef_xyz_m"], observed["predicted_pose"])
        observed = trace.measurements[0]
        first_x = None if observed is None else projected_tool_x(observed["eef_xyz_m"], observed["predicted_pose"])
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "timing": [focus, scale], "first": encode(first),
                "first_tool_x_m": first_x, "steps": result.steps, "score": result.score,
                "target_cookie_ids": np.flatnonzero(trace.target).tolist(),
                "wrong_source": int(np.count_nonzero(trace.columns[trace.target] != result.source_column)),
                "safety_reason": trace.safety, "fork": trace.fork, "continuation": trace.continuation,
                "wall_seconds": result.wall_seconds}
    finally:
        env.close()


def anchor_applies(row: dict, first: tuple[Correction, ...]) -> bool:
    tool = row["first_tool_x_m"]
    return tool is not None and any(item.column == row["column"] and tool >= item.minimum_tool_x_m for item in first)


def collect_pair(checkpoint: Path, focus: int, scale: float, seed: int, first: tuple[Correction, ...]) -> dict:
    reference = collect(checkpoint, focus, scale, seed, ())
    anchor = collect(checkpoint, focus, scale, seed, first) if anchor_applies(reference, first) else None
    return {"reference": reference, "anchor": anchor}


def audit_row(checkpoint: Path, timing, seed: int, first: tuple[Correction, ...], row: dict) -> None:
    from a3_dual_arm_sim.sim.randomization import choose_column

    if (row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != list(timing)
            or row["first"] != encode(first) or row["seed"] != seed or row["profile"] != "advanced"
            or row["column"] != choose_column("random", seed) or row["score"] != len(row["target_cookie_ids"])):
        raise ValueError("training tail receipt disagrees with its physical episode request")
    fork, continuation = row["fork"], row["continuation"]
    if fork is None:
        if continuation is not None:
            raise ValueError("unexpected continuation without a training fork")
        return
    if (set(fork["outcomes"]) != set(OFFSETS) or fork["outcomes"]["reference"]["physics"] != continuation
            or continuation["counted_ids"] != row["target_cookie_ids"] or not np.isfinite(fork["tool_x_m"])):
        raise ValueError("zero-offset training fork did not reproduce its control episode")
    for branch in fork["outcomes"].values():
        if not fork["prefix_step"] < branch["physics"]["step"] <= RULES["max_steps"]:
            raise ValueError("training fork exceeded the step budget")


def audit_pair(job: tuple, pair: dict) -> None:
    checkpoint, focus, scale, seed, first = job
    reference, anchor = pair["reference"], pair["anchor"]
    audit_row(checkpoint, (focus, scale), seed, (), reference)
    if (anchor is not None) != anchor_applies(reference, first):
        raise ValueError("missing replay for a state-changing first correction")
    if anchor is not None:
        audit_row(checkpoint, (focus, scale), seed, first, anchor)
        if reference["first_tool_x_m"] != anchor["first_tool_x_m"]:
            raise ValueError("the first correction changed its own input")


def fit_column(column: int, pairs: list[dict], first: Correction | None) -> tuple[tuple[Correction, ...], dict]:
    choices = []
    for retain_first in ((False, True) if first is not None else (False,)):
        prefix = (first,) if retain_first else ()
        controls = [pair["reference"] for pair in pairs]
        rows = [(pair["anchor"] or pair["reference"]) if retain_first else pair["reference"] for pair in pairs]
        tools = [row["fork"]["tool_x_m"] for row in rows if row["fork"] is not None]
        thresholds = [None, *map(float, np.unique(np.quantile(tools, [.25, .5, .75])))] if tools else [None]
        for option, offset in OFFSETS.items():
            for threshold in thresholds if option != "reference" else [None]:
                model = prefix + ((Correction(column, 1, offset, threshold),) if option != "reference" else ())
                changes, rejected, activations = [], 0, 0
                for control, row in zip(controls, rows, strict=True):
                    fork = row["fork"]
                    triggered = (option != "reference" and fork is not None
                                 and (threshold is None or fork["tool_x_m"] >= threshold))
                    branch = fork["outcomes"][option] if triggered else None
                    outcome = branch["physics"] if branch else row["continuation"]
                    wrong = branch["wrong_source"] if branch else row["wrong_source"]
                    count = len(outcome["counted_ids"]) if outcome else row["score"]
                    rejected += bool((outcome or row)["safety_reason"]) or bool(wrong)
                    if outcome and control["continuation"]:
                        rejected += outcome["source_remaining"] != control["continuation"]["source_remaining"]
                    changes.append(count - control["score"])
                    activations += triggered or bool(fork and fork["first_corrected"])
                stats = {"count_delta": sum(changes), "gains": sum(delta > 0 for delta in changes),
                         "losses": sum(delta < 0 for delta in changes),
                         "major_losses": sum(delta <= -MAJOR_LOSS for delta in changes),
                         "rejected_safety_or_source": rejected, "activations": activations, "corrections": encode(model)}
                choices.append((model, stats))
    eligible = [(model, stats) for model, stats in choices if not model or (
        stats["gains"] >= 2 and stats["count_delta"] > 0 and not (stats["major_losses"] or stats["rejected_safety_or_source"]))]
    winner, _ = max(eligible, key=lambda choice: (choice[1]["count_delta"], -choice[1]["losses"],
                                                 -choice[1]["activations"], -len(choice[0])))
    return winner, {"selected": encode(winner), "trials": [stats for _, stats in choices]}


def run(root: Path, reference: Path, timing_run: Path, prior_run: Path, parity: Path, workers: int, block: int) -> dict:
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and native evaluation audit are required")
    splits = {"scout": {"random": []}, "training": list(range(block, block + 24)),
              "development": list(range(block + 100, block + 112)),
              "confirmation": {"random": list(range(block + 200, block + 220))}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    frozen = prior_run / "fit.json"
    first = tuple(Correction(column, batch, tuple(xyz), *limits)
                  for column, batch, xyz, *limits in read_json(frozen)["candidates"]["batch"])
    if any(item.batch != 0 or item.minimum_tool_x_m is None for item in first):
        raise ValueError("training anchor must be a gated first-batch policy")
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "anchor_fit_sha256": sha256(frozen),
                 "anchor": encode(first), "parity_sha256": sha256(parity),
                 "offsets_m": {name: list(offset) for name, offset in OFFSETS.items()},
                 "training_code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "primary_target": TARGET, "training_minimum_improved_scenes": 2})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(checkpoint, *timing, seed, first) for seed in splits["training"]]
        training = _jobs(pool, collect_pair, jobs, root / "training" / "episodes", lambda job: str(job[-2]), audit_pair)
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"pairs": training})
        winners, statistics = {}, {}
        for column in range(1, 5):
            pairs = [pair for pair in training if pair["reference"]["column"] == column]
            item = next((item for item in first if item.column == column), None)
            winners[column], statistics[str(column)] = fit_column(column, pairs, item)
        model = tuple(item for winner in winners.values() for item in winner)
        _record_once(root / "fit.json", {"dataset_sha256": sha256(dataset), "columns": statistics, "corrections": encode(model)})
        print(f"Training finished with {len(training)} paired seeds and {len(model)} corrections", flush=True)
        if not model:
            return {"status": "reference_retained_after_training"}
        runs = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, splits["development"],
                                            {"reference": (), "selected": model})
        development = {rule: compare([row[rule] for row in runs["reference"]], [row[rule] for row in runs["selected"]], model)
                       for rule in ("default", "strict-500")}
        _record_once(root / "development-selection.json", development)
        if not all(passes(stats) for stats in development.values()):
            return {"status": "reference_retained_after_development", "development": development}
        seeds = splits["confirmation"]["random"]
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": seeds, "corrections": encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        confirmation = confirm_combined(pool, root / "confirmation", checkpoint, timing, seeds, model, seeds[:4])
    confirmed = all(passes(stats) for stats in confirmation.values())
    above = confirmation["default"]["mean"] > TARGET
    return {"status": "candidate_promoted" if confirmed else "reference_retained_after_confirmation",
            "active_policy_above_target": confirmed and above, "confirmation": confirmation, "development": development}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a terminal-scored policy on states visited after first-batch placement.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--first-fit", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-block", type=int, default=112000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), args.first_fit.resolve(),
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
