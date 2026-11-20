from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from a3opt import placement_benchmark, placement_sim, placement_train
from a3opt.advanced_agent import _jobs, _record_once
from a3opt.backends.adaptive_phase import OPEN, PLACE, RETRACT, task_column
from a3opt.backends.release_residual import _ramp, nominal_kinematics
from a3opt.cli import _run_lock
from a3opt.difficulty_eval import RULES
from a3opt.placement_agent import reference_setup
from a3opt.placement_search import TARGET, best_option, branch_stats, compare, confirm_combined, encode, passes, reuse_training
from a3opt.refinement_eval import read_json, sha256
from a3opt.release_timing_agent import STRICT_RULES


SETTINGS = {"light": (2., .003), "firm": (5., .003), "lift": (5., .006)}
DESCENT_FRACTION = .75


@dataclass(frozen=True)
class ContactCorrection:
    column: int
    batch: int
    force_n: float
    lift_m: float

    def __post_init__(self) -> None:
        if (self.column not in range(1, 5) or self.batch not in (0, 1)
                or (self.force_n, self.lift_m) not in SETTINGS.values()):
            raise ValueError("invalid wrist-triggered placement correction")

    def record(self) -> list:
        return [self.column, self.batch, self.force_n, self.lift_m]


class ContactMonitor:
    def __init__(self):
        self.start_force = None
        self.previous_delta = 0.

    def update(self, observation: dict, frame: int, length: int) -> float:
        force = np.asarray(observation["observation.force"][:3], dtype=float)
        if frame == 0:
            self.start_force = force.copy()
            self.previous_delta = 0.
        delta = float(np.linalg.norm(force - self.start_force))
        sustained = min(delta, self.previous_delta)
        self.previous_delta = delta
        return sustained if frame >= DESCENT_FRACTION * length else 0.


def redirect(policy: placement_sim.PlacementPolicy, observation: dict, batch: int, frame: int, lift: float) -> None:
    phases = policy.actions[batch]
    descent, opening, retract = (phases[index] for index in (PLACE, OPEN, RETRACT))
    previous = descent[frame - 1, :7].copy()
    measured = np.asarray(observation["observation.state"][:7], dtype=float)
    release = nominal_kinematics().shift(measured, np.array([0., 0., lift]))
    ramp = _ramp(np.arange(1, len(descent) - frame + 1) / (len(descent) - frame))[:, None]
    descent[frame:, :7] = previous + ramp * (release - previous)
    difference = release - opening[-1, :7]
    opening[:, :7] = release
    retreat_ramp = 1. - _ramp(np.arange(1, len(retract) + 1) / len(retract))[:, None]
    retract[:, :7] += retreat_ramp * difference


class ContactPolicy(placement_sim.PlacementPolicy):
    def __init__(self, checkpoint, focus, scale, corrections=()):
        self.contact = {(item.column, item.batch): item for item in corrections}
        if len(self.contact) != len(corrections):
            raise ValueError("duplicate contact correction")
        super().__init__(checkpoint, focus, scale)

    def reset(self, context=None):
        super().reset(context)
        self.monitor = ContactMonitor()
        self.redirected = set()

    def act(self, observation, task=""):
        if self.actions is not None and self.cursor is not None and self.cursor[1] == PLACE:
            batch, _, frame = self.cursor
            item = self.contact.get((task_column(task), batch))
            if item is not None and batch not in self.redirected:
                loading = self.monitor.update(observation, frame, len(self.actions[batch][PLACE]))
                if loading >= item.force_n:
                    redirect(self, observation, batch, frame, item.lift_m)
                    self.redirected.add(batch)
        return super().act(observation, task)


def branch(env, reference, observation, task, ids, lift_m):
    state = copy(env)
    state.data = copy(env.data)
    controller = copy(reference)
    batch, _, frame = controller.cursor
    if lift_m:
        controller.actions = reference.actions.copy()
        controller.actions[batch] = [phase.copy() for phase in reference.actions[batch]]
        redirect(controller, observation, batch, frame, lift_m)
    digest = hashlib.sha256()
    while controller.step < RULES["max_steps"]:
        action = controller.act(observation, task)
        observation, _, terminated, truncated, info = state.step(action)
        digest.update(placement_train.command_bytes(action, info))
        if terminated or truncated or (batch == 0 and controller.stage == RETRACT and controller.ended):
            break
    return placement_train.outcome(state, info, ids, digest.hexdigest())


class ContactTrace(placement_sim.PlacementTrace):
    def bind_env(self, env):
        super().bind_env(env)
        self.env = env

    def reset(self, context=None):
        super().reset(context)
        self.monitor = ContactMonitor()
        self.forks = [{}, {}]
        self.peaks = [0., 0.]
        self.actual = [None, None]
        self.digests = [None, None]
        self.replays = [None, None]

    def act(self, observation, task):
        cursor = self.policy.cursor if self.policy.actions is not None else None
        if cursor is not None and cursor[1] == PLACE:
            batch, _, frame = cursor
            loading = self.monitor.update(observation, frame, len(self.policy.actions[batch][PLACE]))
            self.peaks[batch] = max(self.peaks[batch], loading)
            forks = self.forks[batch]
            ids = self.lifted_ids[batch]
            for name, (threshold, lift) in SETTINGS.items():
                if name not in forks and loading >= threshold:
                    if self.digests[batch] is None:
                        self.digests[batch] = hashlib.sha256()
                        self.replays[batch] = branch(self.env, self.policy, observation, task, ids, None)
                    forks[name] = {"step": self.step, "force_n": loading,
                                   "outcome": branch(self.env, self.policy, observation, task, ids, lift)}
        return super().act(observation, task)

    def add_frame(self, observation, action, info, *, env, **context):
        super().add_frame(observation, action, info, env=env, **context)
        batch = self.policy.batch
        digest = self.digests[batch]
        if digest is not None:
            digest.update(placement_train.command_bytes(action, info))
        final = bool(self.safety or self.step == RULES["max_steps"]) or (
            batch == 0 and self.policy.stage == RETRACT and self.policy.ended)
        if final:
            self.actual[batch] = placement_train.outcome(env, info, self.lifted_ids[batch], digest.hexdigest() if digest else "")


def collect(checkpoint, focus, scale, seed):
    import cv2
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    cv2.setNumThreads(1)
    engine = make_cookie_benchmark(profile="advanced", source_column="random", **STRICT_RULES)
    env = engine.create_env(render_cameras=False)
    try:
        env.task_config = replace(env.task_config, terminate_on_success=False)
        trace = ContactTrace(placement_sim.PlacementPolicy(checkpoint, focus, scale))
        result = engine.run_episode(trace, seed=seed, env=env, diagnostic=trace)
        return {"seed": seed, "column": result.source_column, "profile": result.randomization["profile"],
                "checkpoint_sha256": sha256(checkpoint), "timing": [focus, scale], "peaks_n": trace.peaks,
                "actual": trace.actual, "replays": trace.replays, "forks": trace.forks,
                "score": result.score, "wall_seconds": result.wall_seconds}
    finally:
        env.close()


def audit(job, row):
    from a3_dual_arm_sim.sim.randomization import choose_column

    checkpoint, focus, scale, seed = job
    if (row["seed"] != seed or row["column"] != choose_column("random", seed) or row["profile"] != "advanced"
            or row["checkpoint_sha256"] != sha256(checkpoint) or row["timing"] != [focus, scale]):
        raise ValueError("contact collection request differs from its replay receipt")
    for peak, forks, actual, replay in zip(row["peaks_n"], row["forks"], row["actual"], row["replays"], strict=True):
        expected = {name for name, (threshold, _) in SETTINGS.items() if peak >= threshold}
        if set(forks) != expected or (forks and actual != replay):
            raise ValueError("the wrist-triggered forks did not reproduce the untreated continuation")


def fit(rows):
    model, history = [], {}
    for batch in range(2):
        trials = [{"outcomes": {"reference": row["actual"][batch], **{
            name: row["forks"][batch].get(name, {}).get("outcome", row["actual"][batch]) for name in SETTINGS}}}
                  for row in rows if row["actual"][batch] is not None]
        stats = {name: branch_stats(trials, name) for name in ("reference", *SETTINGS)}
        selected = best_option(stats, minimum_gains=2)
        if selected != "reference":
            model.extend(ContactCorrection(column, batch, *SETTINGS[selected]) for column in range(1, 5))
        history[str(batch)] = {"options": stats, "selected": selected}
    return tuple(model), history


def run(root, reference, timing_run, sources, parity, workers, block):
    if block % 4 or read_json(parity)["native_receipts_matched"] < 10:
        raise ValueError("a balanced seed block and native evaluation audit are required")
    splits = {"scout": {"random": []}, "training": [], "development": list(range(block, block + 12)),
              "confirmation": {"random": list(range(block + 100, block + 120))}}
    checkpoint, timing, identity = reference_setup(reference, timing_run, splits)
    data = [reuse_training(source, checkpoint, timing, identity) for source in sources]
    seeds = []
    for column in range(1, 5):
        rows = sorted((row for group, _ in data for row in group if row["column"] == column),
                      key=lambda row: (row["score"], row["seed"]))
        shortfalls = [row["seed"] for row in rows if row["score"] < 10 and not row["safety_reason"]]
        controls = [row["seed"] for row in rows if row["score"] == 10 and not row["safety_reason"]]
        seeds.extend(shortfalls[:2] + controls[:1] if shortfalls else controls[:2])
    if len(seeds) != len(set(seeds)) or set(seeds).intersection(splits["development"] + splits["confirmation"]["random"]):
        raise ValueError("sampling requires distinct training and evaluation scenes")
    splits["training"] = seeds
    _record_once(root / "protocol.json", {**identity, "schedule": splits, "sources": [evidence for _, evidence in data],
                 "native_parity_sha256": sha256(parity), "contact_settings": {name: list(option) for name, option in SETTINGS.items()},
                 "descent_fraction": DESCENT_FRACTION, "minimum_training_gains": 2,
                 "training_code_sha256": sha256(Path(__file__)), "evaluator_sha256": sha256(Path(placement_benchmark.__file__)),
                 "target": TARGET})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(checkpoint, *timing, seed) for seed in seeds]
        rows = _jobs(pool, collect, jobs, root / "training" / "episodes", lambda job: str(job[-1]), audit)
        dataset = root / "training" / "dataset.json"
        _record_once(dataset, {"rows": rows})
        model, evidence = fit(rows)
        _record_once(root / "fit.json", {"dataset_sha256": sha256(dataset), "batches": evidence, "corrections": encode(model)})
        print(f"Collected {len(rows)} contact-training receipts and selected {len(model)} corrections", flush=True)
        if not model:
            return {"status": "reference_retained_after_training"}
        rollouts = placement_benchmark.evaluate(pool, root / "development", checkpoint, timing, splits["development"],
                                                {"reference": (), "selected": model}, policy_type=ContactPolicy)
        development = {rule: compare([row[rule] for row in rollouts["reference"]], [row[rule] for row in rollouts["selected"]], model)
                       for rule in ("default", "strict-500")}
        _record_once(root / "development-selection.json", development)
        if not all(passes(stats) for stats in development.values()):
            return {"status": "reference_retained_after_development", "development": development}
        seeds = splits["confirmation"]["random"]
        _record_once(root / "confirmation-lock.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "timing": list(timing), "seeds": seeds, "corrections": encode(model),
                     "protocol_sha256": sha256(root / "protocol.json"), "fit_sha256": sha256(root / "fit.json"),
                     "selection_sha256": sha256(root / "development-selection.json")})
        confirmation = confirm_combined(pool, root / "confirmation", checkpoint, timing, seeds, model, seeds[:4], policy_type=ContactPolicy)
    confirmed = all(passes(stats) for stats in confirmation.values())
    return {"status": "candidate_promoted" if confirmed else "reference_retained_after_confirmation",
            "active_policy_above_target": confirmed and confirmation["default"]["mean"] > TARGET,
            "development": development, "confirmation": confirmation}


def upstream_policy():
    lock = read_json(Path(os.environ["A3OPT_CONTACT_LOCK"]))
    checkpoint = Path(lock["checkpoint"])
    if sha256(checkpoint) != lock["checkpoint_sha256"]:
        raise ValueError("contact policy requires the frozen checkpoint")
    return ContactPolicy(checkpoint, *lock["timing"], tuple(ContactCorrection(*item) for item in lock["corrections"]))


def main():
    parser = argparse.ArgumentParser(description="Train and confirm wrist-triggered compliant placement on actual MuJoCo trajectories.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-block", type=int, default=113000)
    args = parser.parse_args()
    root = args.out.resolve()
    with _run_lock(root):
        result = run(root, args.reference.resolve(), args.timing.resolve(), [source.resolve() for source in args.source],
                     args.parity.resolve(), args.workers, args.seed_block)
        _record_once(root / "result.json", result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
