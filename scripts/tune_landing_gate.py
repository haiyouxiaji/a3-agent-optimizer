from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from inspect import getfile
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark as benchmark, placement_contact_feedback as contact
from a3opt import placement_descent_agent as descent, placement_expert_landing as landing, placement_search
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import PLACE, task_column
from a3opt.backends.release_residual import nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.refinement_eval import read_json, sha256, simulator_identity
from a3opt.release_timing_agent import STRICT_RULES

from audit_placement_confirmation import scheduled_seeds
from confirm_expert_grasp import audit, measurements, resume, verify_receipts
from confirm_source_guard import compare, inputs
from tune_landing_path import LandingPath, LandingPathPolicy, no_regression, sample


FEATURES = ("tray_yaw", "tool_tray_x", "tool_tray_y", "command_error_x", "command_error_y", "command_error_z",
            "anchor_displacement_x", "anchor_displacement_y", "lateral_force_change", "vertical_force_change")
RECIPE = "align-before-lowering"


@dataclass(frozen=True)
class Gate:
    column: int
    batch: int
    feature: int
    threshold: float
    above: bool

    def __post_init__(self):
        if (self.column not in range(1, 5) or self.batch not in (0, 1) or self.feature not in range(len(FEATURES))
                or not np.isfinite(self.threshold) or type(self.above) is not bool):
            raise ValueError("invalid landing path gate")

    def activates(self, values):
        return values is not None and all(np.isfinite(values)) and (values[self.feature] >= self.threshold) == self.above

    def record(self):
        return [self.column, self.batch, self.feature, self.threshold, self.above]


class GatedLandingPolicy(LandingPathPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=(), **options):
        gates = [item for item in corrections if isinstance(item, Gate)]
        if len(gates) > 1:
            raise ValueError("expected at most one landing gate")
        self.gate = gates[0] if gates else None
        super().__init__(checkpoint, focus, scale, tuple(item for item in corrections if not isinstance(item, Gate)), **options)
        if self.gate and (self.landing is None or (self.gate.column, self.gate.batch) != (self.landing.column, self.landing.batch)):
            raise ValueError("landing gate scope differs from the frozen anchor")

    def reset(self, context=None):
        super().reset(context)
        self.descent_force = None

    def signals(self, observation, task):
        anchor = self.landing
        if (anchor is None or self.actions is None or self.cursor is None or self.retry_remaining
                or self.cursor[:2] != (anchor.batch, PLACE) or task_column(task) != anchor.column):
            return None
        frame = self.cursor[2]
        trajectory = self.actions[anchor.batch][PLACE]
        if frame == 0:
            self.descent_force = contact.force_observation(observation)[:3].copy()
        if frame != max(1, len(trajectory) // 4) or self.descent_force is None:
            return None
        solver = nominal_kinematics()
        tool = np.asarray(observation["observation.eef_pose"][:3])
        commanded = descent.tool_xyz(solver, trajectory[frame - 1, :7])
        target = descent.tool_xyz(solver, trajectory[-1, :7])
        point = landing.local_point(tool, self.pose)
        error = point - landing.local_point(commanded, self.pose)
        offset = anchor.target(self.pose) - landing.local_point(target, self.pose)
        change = contact.force_observation(observation)[:3] - self.descent_force
        lateral, _ = change[:2] @ landing.tray_basis(self.pose)
        return [float(value) for value in (self.pose[4], *point[:2], *error, *offset[:2], lateral, change[2])]

    def act(self, observation, task=""):
        if self.gate is not None and self.gate.activates(self.signals(observation, task)):
            self.path = LandingPath(self.gate.column, self.gate.batch, RECIPE)
            try:
                return super().act(observation, task)
            finally:
                self.path = None
        return super().act(observation, task)


class ProbeComplete(Exception):
    pass


class LandingProbe(benchmark.CombinedTrace):
    def act(self, observation, task):
        values = self.policy.signals(observation, task)
        if values is not None:
            batch = self.policy.landing.batch
            raise ProbeComplete({"features": values, "step": self.step, "estimate": self.policy.pose.tolist(),
                                 "transfer": self.stages[batch]["transfer"], "inputs_sha256": self.digest.hexdigest()})
        return super().act(observation, task)


def collect(checkpoint, timing, constructor, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **RULES)
    env = engine.create_env(render_cameras=False)
    try:
        trace = LandingProbe(constructor(checkpoint, *timing))
        try:
            engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        except ProbeComplete as ready:
            return {"seed": seed, "column": trace.column, "checkpoint_sha256": sha256(checkpoint), **ready.args[0]}
    finally:
        env.close()
    raise ValueError(f"scene {seed} did not reach the landing decision")


def audit_probe(job, row, *, previous, column, batch):
    checkpoint, _, _, seed = job
    expected = previous[seed]["strict-500"]
    transfer = expected["phase_geometry"][batch]["transfer"]
    if ((row["seed"], row["column"], row["checkpoint_sha256"]) != (seed, column, sha256(checkpoint))
            or row["estimate"] != expected["estimated_pose"] or row["transfer"] != transfer
            or not transfer["step"] < row["step"] < expected["phase_geometry"][batch]["descent"]["step"]
            or len(row["features"]) != len(FEATURES) or not all(np.isfinite(row["features"]))):
        raise ValueError(f"landing observations disagree with the frozen prefix for seed {seed}")


def evaluate(rows, gate):
    active = [row for row in rows if gate and gate.activates(row["features"])]
    gain = sum(row["count_gain"] for row in active)
    losses = sum(row["count_gain"] < 0 or row["completion_loss"] or row["unsafe"] for row in active)
    return {"net_gain": gain, "losses": losses, "activations": len(active),
            "improvements": sum(row["count_gain"] > 0 for row in active)}


def fit(rows, column, batch):
    choices = []
    for feature in range(len(FEATURES)):
        values = sorted({row["features"][feature] for row in rows})
        for low, high in zip(values, values[1:]):
            for above in (True, False):
                gate = Gate(column, batch, feature, (low + high) / 2, above)
                report = evaluate(rows, gate)
                if not report["losses"] and report["improvements"] >= 2:
                    choices.append((report["net_gain"], -report["activations"], gate))
    return max(choices, key=lambda item: item[:2])[-1] if choices else None


def feedback(old, new):
    return {"count_gain": min(new[rule]["requested_score"] - old[rule]["requested_score"] for rule in old),
            "completion_loss": any(old[rule]["success"] and not new[rule]["success"] for rule in old),
            "unsafe": any(row["safety_reason"] or row["score"] != row["requested_score"] for row in new.values())}


def run(path_run, source, parent, output, seed_block, workers):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, timing, constructor, models, lock, _, _ = inputs(source, parent)
    anchor = next(item for item in models if isinstance(item, landing.Landing))
    recipe = (LandingPath(anchor.column, anchor.batch, RECIPE),)
    path_protocol = read_json(path_run / "protocol.json")
    path_fit = read_json(path_run / "fit.json")
    if (path_protocol["source_lock_sha256"] != sha256(source / "confirmation-lock.json")
            or path_protocol["code_sha256"] != sha256(Path(getfile(LandingPathPolicy)))
            or path_protocol["evaluator_sha256"] != sha256(Path(benchmark.__file__))
            or path_protocol["controller_sha256"] != sha256(checkpoint)
            or path_protocol["environment"] != simulator_identity()
            or (path_protocol["rules"], path_protocol["strict_rules"]) != (RULES, STRICT_RULES)):
        raise ValueError("feedback was collected with a different release controller or path implementation")
    if path_fit["choice"] != RECIPE:
        raise ValueError("expected an independently developed alignment candidate")
    history, archives = {}, {}
    for name, panel in (("parent", parent), ("source", source)):
        controls = panel / "confirmation/rollouts/results.json"
        old = sample(read_json(controls), anchor)
        folder = path_run / "training" / name
        trials = folder / "results.json"
        if sha256(trials) != path_fit["training_receipt_sha256"][name]:
            raise ValueError("landing path training archive differs from its fit")
        history[name] = {"reference": old, "selected": read_json(trials)[RECIPE]}
        for seed, row in zip(path_protocol["schedule"]["training"][name], history[name]["selected"], strict=True):
            benchmark.check((checkpoint, *timing, seed, recipe), row)
        if measurements(history[name], recipe, compare_rows=compare) != path_fit["trials"][name][RECIPE]:
            raise ValueError("landing path training summary disagrees with the paired feedback")
        archives[name] = {"control_sha256": sha256(controls), "trial_sha256": sha256(trials)}
    folder = path_run / "development"
    trials = folder / "results.json"
    history["development"] = read_json(trials)
    verify_receipts(folder, checkpoint, timing, path_protocol["schedule"]["development"], recipe, history["development"])
    if measurements(history["development"], recipe, compare_rows=compare) != read_json(path_run / "development-selection.json"):
        raise ValueError("landing path development summary disagrees with the paired feedback")
    archives["development"] = {"trial_sha256": sha256(trials)}
    by_group = {name: [episode["default"]["seed"] for episode in panel["reference"]] for name, panel in history.items()}
    previous = {episode["default"]["seed"]: episode for panel in history.values() for episode in panel["reference"]}
    if len(previous) != sum(map(len, by_group.values())):
        raise ValueError("feedback groups overlap")
    development = [seed for seed in range(seed_block, seed_block + 64) if choose_column("random", seed) == anchor.column]
    confirmation = list(range(seed_block + 600, seed_block + 728))
    reserved = {*development, *confirmation}
    protocols = [*output.parent.glob("*/protocol.json"), *output.parent.glob("*/replication-plan.json")]
    if (reserved & previous.keys() or any(reserved & scheduled_seeds(read_json(path).get("schedule", {}))
                                        for path in protocols if path.parent != output)):
        raise ValueError("landing gate requires new development and confirmation scenes")
    protocol_path = output / "protocol.json"
    _record_once(protocol_path, {"environment": simulator_identity(), "rules": RULES, "strict_rules": STRICT_RULES,
                 "code_sha256": sha256(Path(__file__)), "checkpoint_sha256": sha256(checkpoint),
                 "source_lock_sha256": sha256(source / "confirmation-lock.json"), "path_protocol_sha256": sha256(path_run / "protocol.json"),
                 "archive_sha256": archives, "features": list(FEATURES), "scope": [anchor.column, anchor.batch],
                 "schedule": {"training": by_group, "development": development, "confirmation": confirmation}})
    baseline_type = partial(GatedLandingPolicy, baseline=models, **constructor.keywords)
    candidate_type = partial(GatedLandingPolicy, **constructor.keywords)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(checkpoint, timing, baseline_type, seed) for seed in previous]
        print(f"Agent is replaying {len(jobs)} paired decision prefixes across {len(history)} groups", flush=True)
        captures = _jobs(pool, collect, jobs, output / "training" / "prefixes", lambda job: str(job[-1]),
                         partial(audit_probe, previous=previous, column=anchor.column, batch=anchor.batch))
        by_seed = {row["seed"]: row for row in captures}
        data = {name: [{**by_seed[old["default"]["seed"]], **feedback(old, new)}
                       for old, new in zip(panel["reference"], panel["selected"], strict=True)] for name, panel in history.items()}
        dataset = output / "training" / "dataset.json"
        _record_once(dataset, {"features": list(FEATURES), "groups": data})
        cv = {}
        for name, group in data.items():
            training = [row for other, rows in data.items() if other != name for row in rows]
            gate = fit(training, anchor.column, anchor.batch)
            cv[name] = {"gate": gate.record() if gate else None, **evaluate(group, gate)}
        learned = fit([row for rows in data.values() for row in rows], anchor.column, anchor.batch)
        accepted = bool(learned and all(row["gate"] and not row["losses"] for row in cv.values())
                        and sum(row["net_gain"] > 0 for row in cv.values()) >= 2)
        fit_path = output / "fit.json"
        _record_once(fit_path, {"held_out": cv, "admitted": accepted, "gate": learned.record() if learned else None,
                                "dataset_sha256": sha256(dataset)})
        if not accepted:
            return {"status": "reference_retained_after_landing_gate_fit", "held_out": cv}
        print(f"Agent selected {FEATURES[learned.feature]} and is evaluating {len(development)} new development pairs", flush=True)
        paired = benchmark.evaluate(pool, output / "development", checkpoint, timing, development,
                                    {"reference": (), "selected": (learned,)}, policy_type=baseline_type)
        metrics = measurements(paired, (learned,), compare_rows=compare)
        selection = output / "development-selection.json"
        _record_once(selection, metrics)
        if not all(no_regression(row) and row["paired_count_delta"] > 0 for row in metrics.values()):
            return {"status": "reference_retained_after_landing_gate_development", "development": metrics}
        corrections = (*models, learned)
        frozen = {**lock, "seeds": confirmation, "corrections": [item.record() for item in corrections],
                  "protocol_sha256": sha256(protocol_path), "fit_sha256": sha256(fit_path), "selection_sha256": sha256(selection)}
        _record_once(output / "confirmation-lock.json", frozen)
        parity_seeds = sorted({development[0], metrics["default"]["changes"][0]["seed"]})
        parity = benchmark.evaluate(pool, output / "calibration", checkpoint, timing, parity_seeds,
                                    {"reference": (), "selected": corrections}, policy_type=candidate_type)
    if not (output / "result.json").is_file():
        resume(output, checkpoint, timing, candidate_type, corrections, frozen, parity, metrics, workers,
               compare_rows=compare, parity_seeds=parity_seeds)
    return audit(output, checkpoint, timing, corrections, frozen, metrics, compare_rows=compare,
                 retained_status="reference_retained_after_landing_gate_confirmation")


def main():
    parser = argparse.ArgumentParser(description="Fit a landing path gate using replays of public predecision observations.")
    parser.add_argument("path_run", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-block", type=int, default=175000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    path_run, source, parent, output = (path.resolve() for path in (args.path_run, args.source, args.parent, args.out))
    if output in (path_run, source, parent) or args.seed_block < 0 or args.seed_block % 4 or args.workers < 1:
        parser.error("expected separate output, an aligned seed block, and a positive worker budget")
    with _run_lock(output):
        result = run(path_run, source, parent, output, args.seed_block, args.workers)
        if "confirmation" not in result:
            _record_once(output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
