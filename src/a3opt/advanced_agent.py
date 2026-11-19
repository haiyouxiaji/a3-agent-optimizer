from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from a3opt.advanced_sim import collect_expert, evaluate_episode, probe_training_scenes
from a3opt.backends.advanced_trajectory import POSE_SCALE, AdvancedTrajectoryPolicy, align_batch, fit_pose
from a3opt.refinement_eval import read_json, sha256, simulator_identity, wilson_interval, write_json


MILESTONE = 5.0
TARGET = 7.5
SUITES = ("1", "random")


@dataclass(frozen=True)
class AdvancedBudget:
    seed_block: int = 40000
    calibration: int = 640
    initial_attempts: int = 40
    feedback_attempts: int = 24
    rounds: int = 4

    def validate(self) -> None:
        if (self.seed_block < 10000 or self.seed_block % 4000 or not 40 <= self.calibration <= 1800
                or not 8 <= self.initial_attempts <= self.calibration or self.feedback_attempts < 4
                or self.rounds < 1 or self.initial_attempts + (self.rounds - 1) * self.feedback_attempts > self.calibration):
            raise ValueError("invalid advanced search budget or seed block")

    def seeds(self, split: str, suite: str = "1") -> tuple[int, ...]:
        if split == "train":
            return tuple(range(self.seed_block, self.seed_block + self.calibration))
        offset = {"development": 2000, "confirmation": 2400}[split] + (100 if suite == "random" else 0)
        count = 20 if split == "confirmation" else (8 if suite == "random" else 6)
        return tuple(range(self.seed_block + offset, self.seed_block + offset + count))


def _record_once(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"existing artifact does not match this run: {path}")
    else:
        write_json(path, value)


def _npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _jobs(pool, function: Callable, jobs: list[tuple], cache: Path, key: Callable, audit: Callable) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    rows, futures = {}, {}
    for inputs in jobs:
        name = key(inputs)
        path = cache / f"{name}.json"
        if path.exists():
            row = read_json(path)
            audit(inputs, row)
            rows[name] = row
        else:
            futures[pool.submit(function, *inputs)] = (inputs, name, path)
    errors = []
    for job in as_completed(futures):
        inputs, name, path = futures[job]
        try:
            row = job.result()
            audit(inputs, row)
            write_json(path, row)
            rows[name] = row
        except Exception as error:
            errors.append((name, error))
            write_json(cache / f"{name}.error.json", {"error": type(error).__name__, "detail": str(error)})
    if errors:
        name, error = errors[0]
        raise RuntimeError(f"{len(errors)} worker results failed, first failure at {cache / name}") from error
    return [rows[key(inputs)] for inputs in jobs]


def sample_plan(scenes: dict, attempts: list[dict], feedback: dict, count: int) -> list[dict]:
    attempted = {row["seed"] for row in attempts}
    pool = sorted(set(scenes) - attempted)
    if len(pool) < count:
        raise ValueError("training pool has fewer unseen seeds than the requested budget")
    poses = np.stack([scenes[seed][0] for seed in pool]) / POSE_SCALE
    colors = np.asarray([scenes[seed][1] for seed in pool])
    previous = {column: [row for row in attempts if row["column"] == column] for column in range(1, 5)}
    failures = [row for result in feedback.values() for row in result["rows"] if row["requested_score"] < 10]
    weights = np.zeros((4, len(pool)))
    difficulty = np.zeros(4)
    for column in range(1, 5):
        examples = [row for row in failures if row["column"] == column]
        difficulty[column - 1] = np.mean([1 - row["requested_score"] / 10 for row in examples]) if examples else 0.
        for row in examples:
            distance = np.linalg.norm(poses - np.asarray(row["initial_pose"]) / POSE_SCALE, axis=1)
            weights[column - 1] = np.maximum(weights[column - 1], np.exp(-.55 * distance**2) * (
                1 - row["requested_score"] / 10
            ))
    order = [1, 2, 1, 3, 1, 4, 1, 2, 3, 4]
    new_counts = Counter()
    known = {column: [np.asarray(row["pose"]) / POSE_SCALE for row in previous[column] if row["batches"]]
             for column in previous}
    color_counts = {column: Counter(row["color"] for row in previous[column]) for column in previous}
    available = np.ones(len(pool), dtype=bool)
    plan = []
    for index in range(count):
        column = order[index % len(order)]
        if feedback and index % 5 == 4:
            column = max(previous, key=lambda item: (difficulty[item - 1] / (new_counts[item] + 1), -item))
        prior = known[column]
        coverage = np.min(np.linalg.norm(poses[:, None] - prior, axis=2), axis=1) if prior else np.linalg.norm(
            poses - np.median(poses, axis=0), axis=1
        )
        if not prior:
            coverage = -coverage
        balance = np.asarray([.18 / (1 + color_counts[column][color]) for color in colors])
        merit = coverage + weights[column - 1] + balance
        selected = int(np.argmax(np.where(available, merit, -np.inf)))
        seed = pool[selected]
        plan.append({"seed": seed, "column": column, "color": scenes[seed][1],
                     "coverage": float(coverage[selected]), "failure_feedback": float(weights[column - 1, selected])})
        known[column].append(poses[selected])
        color_counts[column][scenes[seed][1]] += 1
        new_counts[column] += 1
        available[selected] = False
    return plan


def _audit_expert(inputs: tuple, row: dict) -> None:
    seed, column, root = inputs
    if (row["seed"], row["column"]) != (seed, column) or row["batches"] not in ([], [0], [0, 1]):
        raise ValueError("attempt metadata does not match the collection plan")
    if row["batches"]:
        if row["shard"] != f"{column}-{seed}.npz" or sha256(root / row["shard"]) != row["sha256"]:
            raise ValueError("expert shard is missing or differs from its receipt")
    elif row["shard"] is not None:
        raise ValueError("rejected attempt retained an expert shard")


def _choose_hyperparameters(incumbent: dict | None, round_index: int) -> list[dict]:
    if incumbent is None:
        return [{"mode": "factored", "neighbors": 20, "ridge": .015},
                {"mode": "joint", "neighbors": 24, "ridge": .035}]
    parent = incumbent["parameters"]
    if round_index % 2 == 0:
        variation = {**parent, "neighbors": max(6, parent["neighbors"] // 2), "ridge": parent["ridge"] / 2}
    else:
        variation = {**parent, "mode": "joint" if parent["mode"] == "factored" else "factored",
                     "neighbors": min(36, parent["neighbors"] + 4), "ridge": parent["ridge"] * 1.5}
    return [parent, variation]


def _actions_dataset(attempts: list[dict], root: Path) -> dict:
    data = {name: [] for name in ("poses", "traces", "lengths", "columns", "batches", "seeds")}
    for row in attempts:
        if not row["batches"]:
            continue
        with np.load(root / row["shard"], allow_pickle=False) as archive:
            if int(archive["seed"]) != row["seed"] or int(archive["column"]) != row["column"]:
                raise ValueError("expert shard provenance does not match its receipt")
            for batch in row["batches"]:
                data["poses"].append(archive["pose"])
                data["traces"].append(archive[f"trace_{batch}"])
                data["lengths"].append(archive[f"lengths_{batch}"])
                data["columns"].append(row["column"])
                data["batches"].append(batch)
                data["seeds"].append(row["seed"])
    return {name: np.asarray(values) for name, values in data.items()}


def _audit_evaluation(inputs: tuple, row: dict) -> None:
    from a3_dual_arm_sim.sim.randomization import choose_column

    _, suite, seed = inputs
    expected = choose_column(suite, seed)
    by_column = row["by_column"]
    if (row["seed"] != seed or row["column"] != expected or not 1 <= row["steps"] <= 500
            or len(by_column) != 4 or any(type(count) is not int or count < 0 for count in by_column)
            or not 0 <= row["score"] <= 10 or sum(by_column) != row["score"]
            or by_column[expected - 1] != row["requested_score"] or type(row["success"]) is not bool
            or row["success"] and (row["score"] != 10 or row["source_remaining"] != 70 or row["safety_reason"])):
        raise ValueError(f"invalid evaluation evidence: {suite}, seed {seed}")


def summarize(rows: list[dict]) -> dict:
    groups = {column: [row for row in rows if row["column"] == column] for column in range(1, 5)}
    groups = {column: group for column, group in groups.items() if group}
    requested = np.asarray([row["requested_score"] for row in rows])
    rng = np.random.default_rng(0)
    bootstraps = np.zeros(10000)
    for group in groups.values():
        scores = [row["requested_score"] for row in group]
        bootstraps += rng.choice(scores, size=(len(bootstraps), len(scores)), replace=True).sum(axis=1)
    successes = sum(row["success"] and row["requested_score"] == 10 for row in rows)
    histogram = Counter(int(value) for value in requested)
    return {"episodes": len(rows), "mean_score": float(np.mean([row["score"] for row in rows])),
            "mean_requested_score": float(requested.mean()), "requested_distribution": dict(sorted(histogram.items())),
            "requested_mean_bootstrap_95": np.quantile(bootstraps / len(rows), [.025, .975]).tolist(),
            "full_successes": successes, "full_success_wilson_95": wilson_interval(successes, len(rows)),
            "mean_pose_mae_mm_mrad": (np.abs(np.asarray([row["estimated_pose"] for row in rows])
                                          - np.asarray([row["initial_pose"] for row in rows])).mean(axis=0) * 1000).tolist(),
            "by_requested_column": {column: {"n": len(group), "mean_requested_score": float(np.mean(
                [row["requested_score"] for row in group]))} for column, group in groups.items()}}


def _evaluate(pool, checkpoint: Path, root: Path, budget: AdvancedBudget, split: str) -> dict:
    output = {}
    _record_once(root / "request.json", {"checkpoint_sha256": sha256(checkpoint), "split": split,
                                         "seeds": {suite: list(budget.seeds(split, suite)) for suite in SUITES}})
    for suite in SUITES:
        seeds = budget.seeds(split, suite)
        jobs = [(checkpoint, suite, seed) for seed in seeds]
        rows = _jobs(pool, evaluate_episode, jobs, root / suite, lambda inputs: str(inputs[2]), _audit_evaluation)
        output[suite] = {"metrics": summarize(rows), "rows": rows}
    write_json(root / "results.json", output)
    return output


def _rank(results: dict) -> tuple[float, float, int]:
    means = [item["metrics"]["mean_requested_score"] for item in results.values()]
    return min(means), sum(means), sum(item["metrics"]["full_successes"] for item in results.values())


def _calibrate(pool, path: Path, train_seeds: tuple[int, ...]) -> tuple[dict, dict]:
    if not path.exists():
        chunks = [train_seeds[start:start + 12] for start in range(0, len(train_seeds), 12)]
        batches = pool.map(probe_training_scenes, chunks)
        rows = [row for batch in batches for row in batch]
        seeds, outlines, silhouettes, labels, colors = zip(*rows)
        _npz(path, seeds=seeds, outlines=outlines, silhouettes=silhouettes, labels=labels, colors=colors)
    with np.load(path, allow_pickle=False) as data:
        if data["seeds"].tolist() != list(train_seeds):
            raise ValueError("calibration contains seeds outside the locked training pool")
        model, report = fit_pose(data["outlines"], data["silhouettes"], data["labels"], data["seeds"])
        scenes = {int(seed): (pose, str(color)) for seed, pose, color in zip(
            data["seeds"], data["labels"], data["colors"], strict=True
        )}
    write_json(path.with_suffix(".json"), {**report, "dataset_sha256": sha256(path)})
    return model, scenes


def _recover_first_round(source: Path, destination: Path, plan: list[dict], scenes: dict) -> None:
    prior = read_json(source / "round-01/plan.json")["attempts"]
    if prior != plan:
        raise ValueError("interrupted collection used a different first-round sampling plan")
    destination.mkdir(exist_ok=True)
    for item in plan:
        seed, column = item["seed"], item["column"]
        name = f"{column}-{seed}"
        shard, receipt = source / "expert" / f"{name}.npz", source / "expert" / f"{name}.json"
        new_receipt = destination / receipt.name
        if new_receipt.exists():
            continue
        if receipt.is_file():
            row = read_json(receipt)
            _audit_expert((seed, column, shard.parent), row)
        elif shard.is_file():
            with np.load(shard, allow_pickle=False) as archive:
                if (int(archive["seed"]) != seed or int(archive["column"]) != column
                        or not np.allclose(archive["pose"], scenes[seed][0], atol=5e-7, rtol=0)):
                    raise ValueError(f"interrupted shard {shard} has incorrect scene provenance")
                batches = [index for index in (0, 1) if f"trace_{index}" in archive]
                for batch in batches:
                    trace, lengths = align_batch(archive[f"actions_{batch}"], archive[f"phases_{batch}"])
                    if not np.array_equal(trace, archive[f"trace_{batch}"]) or not np.array_equal(
                        lengths, archive[f"lengths_{batch}"]
                    ):
                        raise ValueError(f"interrupted shard {shard} has inconsistently aligned actions")
                if batches not in ([0], [0, 1]):
                    raise ValueError(f"interrupted shard {shard} has missing verified batches")
                row = {"seed": seed, "column": column, "batches": batches, "color": item["color"],
                       "pose": archive["pose"].tolist(), "shard": shard.name, "sha256": sha256(shard),
                       "recovered_from_interrupted_collection": str(shard), "episode_receipt_available": False}
        else:
            continue
        if row["batches"]:
            shutil.copyfile(shard, destination / shard.name)
        _audit_expert((seed, column, destination), row)
        write_json(new_receipt, row)


def run_agent(root: Path, budget: AdvancedBudget, workers: int, recover_from: Path | None = None) -> dict:
    budget.validate()
    if workers < 1:
        raise ValueError("workers must be positive")
    root = root.resolve()
    recover_from = recover_from.resolve() if recover_from is not None else None
    package = Path(__file__).parent
    protocol = {"budget": asdict(budget), "milestone_at_least": MILESTONE, "target_above": TARGET,
                "task": "2x5", "profile": "advanced", "prompt_style": "task", "max_steps": 500,
                "environment": simulator_identity(), "code_sha256": {name: sha256(package / name) for name in (
                    "advanced_agent.py", "advanced_sim.py", "backends/advanced_trajectory.py",
                    "backends/trajectory.py", "refinement_eval.py")}}
    if recover_from is not None:
        prior = read_json(recover_from / "protocol.json")
        if (root == recover_from or (recover_from / "confirmation-lock.json").exists()
                or any(prior[field] != protocol[field] for field in ("environment", "task", "profile", "budget"))
                or prior["code_sha256"]["backends/advanced_trajectory.py"] != protocol["code_sha256"]["backends/advanced_trajectory.py"]):
            raise ValueError("recovery requires an interrupted run with the same simulation and training protocol")
        protocol["recover_training_from"] = str(recover_from)
    _record_once(root / "protocol.json", protocol)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    root.mkdir(parents=True, exist_ok=True)
    attempts, incumbent, feedback = [], None, {}
    collection = root / "expert"
    calibration = root / "calibration.npz"
    if recover_from is not None and not calibration.exists():
        previous = recover_from / "calibration.npz"
        if sha256(previous) != read_json(previous.with_suffix(".json"))["dataset_sha256"]:
            raise ValueError("interrupted run calibration does not match its recorded checksum")
        temporary = calibration.with_suffix(".tmp")
        shutil.copyfile(previous, temporary)
        temporary.replace(calibration)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        model, scenes = _calibrate(pool, calibration, budget.seeds("train"))
        print(f"Agent calibrated on {len(scenes)} train-only scenes", flush=True)
        for index in range(1, budget.rounds + 1):
            directory = root / f"round-{index:02d}"
            count = budget.feedback_attempts if index > 1 else budget.initial_attempts
            plan = sample_plan(scenes, attempts, feedback, count)
            _record_once(directory / "plan.json", {"parent": incumbent["checkpoint"] if incumbent else None,
                                                   "feedback_split": "development", "attempts": plan})
            if index == 1 and recover_from is not None:
                _recover_first_round(recover_from, collection, plan, scenes)
            print(f"Agent round {index}: collecting {len(plan)} expert attempts", flush=True)
            jobs = [(item["seed"], item["column"], collection) for item in plan]
            attempts += _jobs(pool, collect_expert, jobs, collection,
                              lambda inputs: f"{inputs[1]}-{inputs[0]}", _audit_expert)
            dataset = _actions_dataset(attempts, collection)
            coverage = {str(col): [int(((dataset["columns"] == col) & (dataset["batches"] == batch)).sum())
                                   for batch in range(2)] for col in range(1, 5)}
            manifest = {"attempts": len(attempts), "coverage": coverage, "records": attempts}
            _record_once(directory / "dataset.json", manifest)
            print(f"Agent round {index}: retained batches per column {coverage}", flush=True)
            if any(batches[0] < 2 for batches in coverage.values()):
                continue
            options = _choose_hyperparameters(incumbent, index)
            candidates = []
            for slot, parameters in enumerate(options, 1):
                candidate = directory / f"candidate-{slot}"
                checkpoint = candidate / "policy.npz"
                _record_once(candidate / "training.json", {"parameters": parameters,
                    "dataset_sha256": sha256(directory / "dataset.json"),
                    "calibration_sha256": sha256(root / "calibration.npz")})
                if not checkpoint.exists():
                    _npz(checkpoint, format=AdvancedTrajectoryPolicy.name, **model, **dataset, **parameters)
                dev = _evaluate(pool, checkpoint, candidate / "development", budget, "development")
                item = {"checkpoint": str(checkpoint), "parameters": parameters, "rank": list(_rank(dev)),
                        "development": str(candidate / "development/results.json")}
                candidates.append(item)
                print(f"Agent round {index} candidate {slot}: " + json.dumps({
                    suite: info["metrics"]["mean_requested_score"] for suite, info in dev.items()}), flush=True)
            best = max(candidates, key=lambda item: item["rank"])
            if incumbent is None or best["rank"] > incumbent["rank"]:
                incumbent = best
            feedback = read_json(Path(incumbent["development"]))
            _record_once(directory / "decision.json", {"considered": candidates, "selected": incumbent})
            if incumbent["rank"][0] > TARGET:
                break
        if incumbent is None:
            result = {"status": "insufficient_expert_coverage", "collection_attempts": len(attempts),
                      "batch_coverage": coverage}
            write_json(root / "result.json", result)
            return result
        checkpoint = Path(incumbent["checkpoint"])
        _record_once(root / "confirmation-lock.json", {"selected": incumbent, "checkpoint_sha256": sha256(checkpoint),
                     "protocol_sha256": sha256(root / "protocol.json"),
                     "seeds": {suite: list(budget.seeds("confirmation", suite)) for suite in SUITES}})
        print("Agent froze the development winner and started confirmation", flush=True)
        final = _evaluate(pool, checkpoint, root / "confirmation", budget, "confirmation")
    result = {"status": "confirmed", "checkpoint": incumbent["checkpoint"], "collection_attempts": len(attempts),
              "development": {suite: item["metrics"] for suite, item in feedback.items()},
              "confirmation": {suite: item["metrics"] for suite, item in final.items()},
              "milestone_met": _rank(final)[0] >= MILESTONE, "target_met": _rank(final)[0] > TARGET}
    write_json(root / "result.json", result)
    return result
