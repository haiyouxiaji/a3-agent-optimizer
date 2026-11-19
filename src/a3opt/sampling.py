from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from a3opt.backends.trajectory import FRONT, JOINTS, layout_features, split_trajectory
from a3opt.refinement_eval import (
    DEVELOPMENT_SEEDS,
    MAX_STEPS,
    TRAIN_POOL,
    make_benchmark,
    read_json,
    render_frames,
    sha256,
    validate_rows,
    write_json,
)


class LayoutProbe:
    action_mode = "joint_position"
    finished = True

    def bind_env(self, env: Any) -> None:
        self.env = env

    def act(self, observation: dict, task: str = "") -> np.ndarray:
        self.features = layout_features(render_frames(self.env, ("front",))[FRONT])
        return observation["observation.state"]


def probe_layouts(seeds: tuple[int, ...]) -> dict[int, list[float]]:
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(TRAIN_POOL):
        raise ValueError("layout probing requires unique seeds from the training pool")
    benchmark, probe = make_benchmark(), LayoutProbe()
    env = benchmark.create_env(render_cameras=False)
    try:
        layouts = {}
        for seed in seeds:
            benchmark.run_episode(probe, seed=seed, env=env)
            layouts[seed] = probe.features.tolist()
        return layouts
    finally:
        env.close()


def choose_layouts(
    layouts: dict[int, list[float]],
    training: np.ndarray,
    development: list[dict],
    count: int,
    *,
    method: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    seeds = tuple(sorted(layouts))
    if not 1 <= count <= len(seeds) or not set(seeds) <= set(TRAIN_POOL):
        raise ValueError("sampling budget exceeds the available training pool")
    if method == "uniform":
        draw = np.random.default_rng(random_seed).choice(seeds, count, replace=False)
        return [{"seed": int(seed), "layout": layouts[int(seed)], "priority": None} for seed in draw]
    if method != "feedback":
        raise ValueError(f"unknown sampling method {method}")
    validate_rows(development, DEVELOPMENT_SEEDS)
    scale = np.maximum(training.std(axis=0), [0.2, 0.2, 0.2, 0.2, 0.30])
    pool = np.array([layouts[seed] for seed in seeds]) / scale
    known = training / scale
    dev = np.array([row["layout_features"] for row in development]) / scale
    deficit = np.array([(10 - row["score"] + (not row["success"])) / 11 for row in development])
    dev_distances = np.linalg.norm(pool[:, None] - dev, axis=-1)
    failure_weight = np.exp(-0.5 * dev_distances**2) @ deficit
    selected = []
    for _ in range(count):
        coverage = np.linalg.norm(pool[:, None] - known, axis=-1).min(axis=1)
        values = coverage / (1 + coverage) + failure_weight
        for choice in selected:
            values[choice["index"]] = -np.inf
        index = int(values.argmax())
        selected.append({"index": index, "seed": seeds[index], "layout": layouts[seeds[index]],
                         "priority": float(values[index]), "coverage": float(coverage[index]),
                         "failure_weight": float(failure_weight[index])})
        known = np.concatenate((known, pool[index:index + 1]))
    return [{key: value for key, value in choice.items() if key != "index"} for choice in selected]


class TrajectoryRecorder:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.complete = False

    def start_episode(self, context: Any, controller_type: str) -> None:
        self.frame = render_frames(self.env, ("front",))[FRONT]
        self.context = context
        self.controller = controller_type

    def add_frame(self, observation: dict, action: np.ndarray) -> None:
        self.states.append(np.asarray(observation["observation.state"], dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())

    def finish_episode(self, success: bool) -> None:
        self.complete = success

    def discard_episode(self) -> None:
        self.complete = False


def record_episode(seed: int, directory: Path) -> dict[str, Any]:
    if seed not in TRAIN_POOL:
        raise ValueError(f"seed {seed} is outside the training pool")
    started = time.perf_counter()
    benchmark = make_benchmark()
    env = benchmark.create_env(render_cameras=False)
    recorder = TrajectoryRecorder(env)
    try:
        result = benchmark.run_episode("expert", seed=seed, env=env, recorder=recorder)
        row = result.to_dict()
        report = {"seed": seed, "score": result.score, "success": result.success, "steps": result.steps,
                  "failure_reason": result.failure_reason, "dataset_rejection": None,
                  "controller": recorder.controller, "action_mode": "applied_joint_position",
                  "elapsed_seconds": time.perf_counter() - started, "shard": None}
        if not recorder.complete or not row["success"] or row["cookies_in_target"] != 10 or row["cookies_in_source"] != 70:
            report["dataset_rejection"] = "expert episode did not satisfy the success contract"
            return report
        actions, states = np.asarray(recorder.actions), np.asarray(recorder.states)
        try:
            split_trajectory(actions)
            if actions.shape != states.shape or not np.all(np.isfinite(states)) or len(actions) != result.steps:
                raise ValueError("invalid state/action alignment")
            layout = layout_features(recorder.frame)
        except ValueError as exc:
            report["dataset_rejection"] = str(exc)
            return report
        output = directory / f"{seed}.npz"
        temporary = directory / f".{seed}.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, seed=seed, front=recorder.frame, layout=layout,
                                state=states, action=actions)
        temporary.replace(output)
        report.update(shard=output.name, sha256=sha256(output), layout=layout.tolist())
        return report
    finally:
        env.close()


def collect(plan: list[dict], directory: Path, workers: int) -> list[dict]:
    seeds = tuple(int(item["seed"]) for item in plan)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(TRAIN_POOL):
        raise ValueError("collection needs unique training seeds")
    directory.mkdir(parents=True, exist_ok=True)
    reports = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for seed in seeds:
            path = directory / f"{seed}.json"
            if path.exists():
                reports[seed] = read_json(path)
            else:
                pending[pool.submit(record_episode, seed, directory)] = (seed, path)
        for future in as_completed(pending):
            seed, path = pending[future]
            report = future.result()
            write_json(path, report)
            reports[seed] = report
            print(f"collection seed={seed} score={report['score']} kept={bool(report['shard'])}", flush=True)
    for choice in plan:
        report = reports[choice["seed"]]
        if report["shard"] and not np.array_equal(report["layout"], choice["layout"]):
            raise ValueError(f"initial render differs from pre-collection probe for seed {choice['seed']}")
    return [reports[seed] for seed in seeds]


def augment(
    base: tuple[np.ndarray, list[np.ndarray], np.ndarray], reports: list[dict], directory: Path
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    features, actions, seeds = base
    extra_features, extra_actions, extra_seeds = [], [], []
    for report in reports:
        if not report["shard"]:
            continue
        seed = report["seed"]
        path = directory / f"{seed}.npz"
        if (seed not in TRAIN_POOL or seed in seeds or seed in extra_seeds or path.name != report["shard"]
                or sha256(path) != report["sha256"] or not report["success"] or report["score"] != 10):
            raise ValueError(f"invalid dataset shard or duplicate seed {seed}")
        with np.load(path, allow_pickle=False) as data:
            if int(data["seed"]) != seed:
                raise ValueError(f"shard seed mismatch for {seed}")
            trace, state, layout = data["action"], data["state"], data["layout"]
            if (trace.shape != state.shape or trace.shape != (report["steps"], JOINTS)
                    or len(trace) > MAX_STEPS or not np.all(np.isfinite(state))
                    or not np.array_equal(layout, layout_features(data["front"]))):
                raise ValueError(f"invalid state, action or layout for seed {seed}")
            split_trajectory(trace)
            extra_features.append(layout)
            extra_actions.append(trace)
        extra_seeds.append(seed)
    if not extra_seeds:
        return base
    return np.concatenate((features, extra_features)), [*actions, *extra_actions], np.r_[seeds, extra_seeds]
