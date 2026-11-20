from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_clearance_feedback as clearance, placement_descent_agent as descent
from a3opt import placement_first_hover as first, placement_hover_agent as hover, placement_routed_agent as routed
from a3opt import placement_search, placement_staged_descent as staged
from a3opt.advanced_agent import _jobs, _npz, _record_once
from a3opt.backends.adaptive_phase import PLACE, task_column
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_sim import PlacementPolicy
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES


COLUMN, BATCH = 1, 0
MODES = ("plan", "pose", "force", "edges")
METRICS = ("default", "strict-500")


@dataclass(frozen=True)
class StagedGate:
    entry: staged.StagedEntry
    estimator: clearance.WallModel
    threshold_m: float
    column = COLUMN
    batch = BATCH

    def __post_init__(self):
        if (self.entry.column, self.entry.batch) != (self.column, self.batch) or not np.isfinite(self.threshold_m):
            raise ValueError("invalid staged placement gate")

    def record(self):
        return [self.entry.record(), self.estimator.record(), self.threshold_m]


class StagedGatePolicy(descent.DescentPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), *, incumbent):
        if len(corrections) > 1:
            raise ValueError("expected one staged entry gate")
        self.gate = corrections[0] if corrections else None
        super().__init__(checkpoint, focus, scale, incumbent)

    def pending(self):
        return (self.gate is not None and self.actions is not None and self.cursor == (BATCH, PLACE, 0)
                and not self.retry_remaining)

    @property
    def camera_names(self):
        if self.pending() and self.gate.estimator.mode == "edges":
            return ("left_wrist",)
        return super().camera_names

    def act(self, observation, task=""):
        if self.pending() and task_column(task) == COLUMN:
            values = clearance.readings(self, observation, BATCH)
            if self.gate.estimator.mode == "edges":
                frame = observation.get(clearance.wrist.WRIST)
                if frame is None or not np.any(frame):
                    return super().act(observation, task)
                values = np.r_[values, clearance.wrist_edges(frame)]
            if np.isfinite(values).all() and self.gate.estimator.predict(values) <= self.gate.threshold_m:
                if staged.replan(self, observation, self.gate.entry):
                    return PlacementPolicy.act(self, observation, task)
        return super().act(observation, task)


class FirstTransferTrace(clearance.TransferTrace):
    def add_frame(self, observation, action, info, **context):
        super().add_frame(observation, action, info, **context)
        self.finished = bool(self.samples)


def collect(checkpoint, timing, incumbent, shards, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = FirstTransferTrace(descent.DescentPolicy(checkpoint, *timing, incumbent), clearance.wrist_edges)
        score = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        shard = shards / f"{seed}.npz"
        _npz(shard, **trace.frames)
        return {"seed": seed, "column": score.source_column, "profile": score.randomization["profile"],
                "sample": trace.samples[0] if trace.samples else None, "steps": score.steps,
                "shard_sha256": sha256(shard), "checkpoint_sha256": sha256(checkpoint)}
    finally:
        env.close()


def audit(job, row):
    checkpoint, _, _, shards, seed = job
    sample = row["sample"]
    shard = shards / f"{seed}.npz"
    if (row["seed"] != seed or row["column"] != COLUMN or row["profile"] != "advanced" or sample is None
            or sample["step"] != row["steps"] or sample["batch"] != BATCH or row["shard_sha256"] != sha256(shard)
            or row["checkpoint_sha256"] != sha256(checkpoint) or len(sample["features"]) != clearance.FEATURES["edges"]
            or not np.isfinite(sample["features"]).all()):
        raise ValueError("staged intervention sample differs from the reserved scene")
    with np.load(shard, allow_pickle=False) as frames:
        if frames.files != [str(BATCH)] or not np.array_equal(sample["features"][-50:], clearance.wrist_edges(frames[str(BATCH)])):
            raise ValueError("staged intervention image and features do not match")


def previous_training(source, checkpoint, timing, environment):
    protocol = read_json(source / "protocol.json")
    fit = read_json(source / "fit.json")
    controls = source / "training" / "dataset.json"
    variants = source / "training" / str(COLUMN) / "results.json"
    group = read_json(controls)["groups"][str(COLUMN)]
    model = tuple(staged.StagedEntry(*record) for record in fit["corrections"])
    entry, = [item for item in model if (item.column, item.batch) == (COLUMN, BATCH)]
    if (protocol["environment"] != environment or protocol["rules"] != RULES or protocol["strict_rules"] != STRICT_RULES
            or protocol["code_sha256"] != sha256(Path(staged.__file__)) or protocol["timing"] != list(timing)
            or fit["dataset_sha256"] != sha256(controls) or group["trials_sha256"] != sha256(variants)):
        raise ValueError("the staged entry does not match its archived training run")
    training = {"reference": group["reference"], "selected": read_json(variants)[entry.recipe]}
    if hover.pair_stats(training, (entry,)) != fit["decisions"][str(COLUMN)]["metrics"][entry.recipe]:
        raise ValueError("staged training receipts differ from the fit")
    development = source / "development" / "results.json"
    rows = read_json(development)
    if hover.pair_stats(rows, model) != read_json(source / "result.json")["development"]:
        raise ValueError("staged development receipts differ from the evaluation")
    records = {}
    for panel, pairs, corrections in (("staged-training", training, (entry,)), ("staged-development", rows, model)):
        for control, trial in zip(pairs["reference"], pairs["selected"], strict=True):
            reference = control["strict-500"]
            if reference["column"] != COLUMN:
                continue
            seed = reference["seed"]
            placement_benchmark.check((checkpoint, *timing, seed, corrections), trial)
            prefix = reference["phase_geometry"][BATCH].get("transfer")
            if seed in records or prefix is None:
                raise ValueError("expected a unique staged prefix for each training scene")
            records[seed] = {"seed": seed, "panel": panel, "prefix_sha256": prefix["prefix_sha256"],
                             "delta": [trial[rule]["requested_score"] - control[rule]["requested_score"] for rule in METRICS],
                             "complete_delta": [int(trial[rule]["success"]) - int(control[rule]["success"]) for rule in METRICS],
                             "unsafe": any(trial[rule]["safety_reason"] or trial[rule]["score"] != trial[rule]["requested_score"]
                                           for rule in METRICS)}
    return entry, records, {"training": sha256(controls), "variants": sha256(variants), "development": sha256(development)}


def metrics(rows):
    return {"gain": [sum(row["delta"][index] for row in rows) for index in range(2)],
            "complete_gain": [sum(row["complete_delta"][index] for row in rows) for index in range(2)],
            "helped": sum(min(row["delta"]) > 0 for row in rows), "harmed": sum(min(row["delta"]) < 0 for row in rows),
            "major_losses": sum(min(row["delta"]) <= -placement_search.MAJOR_LOSS for row in rows),
            "unsafe": sum(row["unsafe"] for row in rows), "activations": len(rows)}


def admissible(summary):
    return (min(summary["gain"]) > 0 and summary["helped"] >= 2 and min(summary["complete_gain"]) >= 0
            and not (summary["major_losses"] or summary["unsafe"]))


def threshold(rows, mode):
    estimates = sorted({row["estimates_m"][mode] for row in rows})
    options = []
    for low, high in zip(estimates, estimates[1:]):
        cutoff = (low + high) / 2.
        changed = [row for row in rows if row["estimates_m"][mode] <= cutoff]
        result = metrics(changed)
        if admissible(result):
            rank = (min(result["gain"]), min(result["complete_gain"]), -result["harmed"], -result["activations"])
            options.append((rank, cutoff))
    return max(options)[1] if options else None


def fit(rows, models, entry):
    always = metrics(rows)
    checks, candidates = {}, []
    for mode, estimator in models.items():
        observed = []
        for row in rows:
            cutoff = threshold([other for other in rows if other["seed"] != row["seed"]], mode)
            if cutoff is not None and row["estimates_m"][mode] <= cutoff:
                observed.append(row)
        heldout = metrics(observed)
        cutoff = threshold(rows, mode)
        admitted = cutoff is not None and admissible(heldout) and heldout["harmed"] < always["harmed"]
        checks[mode] = {"threshold_m": cutoff, "leave_one_scene_out": heldout, "admitted": admitted}
        if admitted:
            rank = (min(heldout["gain"]), min(heldout["complete_gain"]), -heldout["harmed"], -clearance.FEATURES[mode])
            candidates.append((rank, StagedGate(entry, estimator, cutoff)))
    selected = max(candidates, key=lambda candidate: candidate[0])[1] if candidates else None
    return selected, {"always_staged": always, "checks": checks, "corrections": [selected.record()] if selected else []}


def run(root, release, source, rgb_archive, block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    if block < 0 or block % 4 or workers < 1:
        raise ValueError("expected an aligned seed block and a positive worker budget")
    manifest = release / "controller.json"
    frozen = read_json(manifest)
    checkpoint, timing = release / frozen["checkpoint"], tuple(frozen["timing"])
    incumbent = tuple(descent.Descent(*item) for item in frozen["corrections"])
    environment = simulator_identity()
    if (environment["sim_revision"] != frozen["sim_revision"] or sha256(checkpoint) != frozen["checkpoint_sha256"]
            or sha256(Path(descent.__file__)) != frozen["policy_code_sha256"]
            or sha256(Path(placement_benchmark.__file__)) != frozen["evaluator_sha256"]):
        raise ValueError("staged gating requires the frozen published controller")
    entry, history, sources = previous_training(source, checkpoint, timing, environment)
    sensor_rows, sensor_sources = clearance.archived_training(rgb_archive, checkpoint, timing, environment)
    development = [seed for seed in range(block, block + 96) if choose_column("random", seed) == COLUMN]
    confirmation = list(range(block + 400, block + 464))
    reserved = set(development + confirmation)
    seen = set(history) | {row["seed"] for row in sensor_rows}
    if reserved & seen or any(path.parent != root and reserved & hover.scheduled_seeds(read_json(path).get("schedule", {}))
                              for path in root.parent.glob("*/protocol.json")):
        raise ValueError("staged gating requires untouched development and confirmation scenes")
    protocol = root / "protocol.json"
    _record_once(protocol, {"environment": environment, "incumbent": frozen["corrections"], "timing": list(timing),
                 "rules": RULES, "strict_rules": STRICT_RULES, "reference_sha256": sha256(manifest),
                 "code_sha256": sha256(Path(__file__)), "source": sources, "sensor_source": sensor_sources, "recipe": entry.record(),
                 "schedule": {"training": sorted(seen), "interventions": list(history), "development": development,
                              "confirmation": confirmation}, "estimator_modes": list(MODES), "confirmation_requires": routed.REQUIREMENTS})
    policy_type = partial(StagedGatePolicy, incumbent=incumbent)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        print(f"Agent reacquiring first-transfer observations for {len(history)} staged intervention pairs", flush=True)
        saved = {row["seed"]: row["samples"][0] for row in sensor_rows if row["samples"]}
        jobs = [(checkpoint, timing, incumbent, root / "training" / "frames", seed) for seed in history if seed not in saved]
        acquired = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]), audit)
        observed = {**saved, **{row["seed"]: row["sample"] for row in acquired}}
        models = {mode: clearance.fit([row for row in sensor_rows if row["seed"] not in history], mode) for mode in MODES}
        points = []
        for seed, target in history.items():
            sample = observed[seed]
            if sample["prefix_sha256"] != target["prefix_sha256"]:
                raise ValueError("the observable training sample did not reproduce its staged prefix")
            estimates = {mode: estimator.predict(sample["features"]) for mode, estimator in models.items()}
            points.append({**target, "features": sample["features"], "estimates_m": estimates})
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"interventions": points, "acquired": acquired, "estimators": [model.record() for model in models.values()]})
        gate, search = fit(points, models, entry)
        fit_path = root / "fit.json"
        _record_once(fit_path, {**search, "dataset_sha256": sha256(dataset)})
        if gate is None:
            return {"status": "reference_retained_after_staged_gate_training", **search}
        correction = (gate,)
        print(f"Agent froze a staged gate for {len(development)} new development pairs", flush=True)
        paired = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, development,
                                              {"reference": (), "selected": correction}, policy_type=policy_type)
        scores = hover.pair_stats(paired, correction)
        selection = root / "development-selection.json"
        _record_once(selection, scores)
        if not all(placement_search.passes(stats) for stats in scores.values()):
            return {"status": "reference_retained_after_staged_gate_development", "development": scores, **search}
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "incumbent": frozen["corrections"], "seeds": confirmation, "corrections": search["corrections"],
                     "protocol_sha256": sha256(protocol), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)})
        print(f"Agent started {len(confirmation)} independent four-column pairs", flush=True)
        final = placement_search.confirm_combined(pool, root / "confirmation", checkpoint, timing, confirmation,
                                                   correction, confirmation[:4], policy_type=policy_type)
    if any(stats["scored_pairs_by_column"] != {str(column): 16 for column in range(1, 5)} for stats in final.values()):
        raise ValueError("confirmation requires four-column balance")
    return {"status": "candidate_promoted" if first.supports_confirmation(final) else "reference_retained_after_staged_gate_confirmation",
            "development": scores, "confirmation": final, **search,
            "target_lower_bound": final["default"]["mean_ci95"][0] > placement_search.TARGET}


def main():
    parser = argparse.ArgumentParser(description="Learn when to apply a staged descent from matched intervention receipts.")
    parser.add_argument("--release", type=Path, default=Path("evidence/advanced-20261120"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rgb-archive", type=Path, default=Path("evidence/feedback-20261120"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=156000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.release.resolve(), args.source.resolve(), args.rgb_archive.resolve(), args.seed_block, args.workers)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
