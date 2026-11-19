from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

TASK = "2x5"
PROFILE = "basic"
SOURCE_COLUMN = "1"
MAX_STEPS = 600
GOAL_SCORE = 7.5
LEGACY_SEEDS = tuple(range(1000, 1060))
TRAIN_POOL = tuple(range(1200, 1400))
DEVELOPMENT_SEEDS = tuple(range(2000, 2006))
CONFIRMATION_SEEDS = tuple(range(3100, 3120))
SIM_REVISION = "5fa24970648f53a24c0b5038d19bf6a14e1be5b9"
CAMERAS = ("front", "left_wrist", "right_wrist")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def simulator_identity() -> dict[str, Any]:
    import a3_dual_arm_sim

    module_path = Path(a3_dual_arm_sim.__file__).resolve()
    root = module_path.parents[2]
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if revision != SIM_REVISION or dirty:
        raise ValueError(f"expected clean simulator checkout {SIM_REVISION}, got {revision}")
    versions = {}
    for package in ("mujoco", "numpy", "torch", "gymnasium"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package != "torch":
                raise
    return {
        "sim_revision": revision,
        "sim_config_sha256": {
            name: sha256(root / "configs" / name) for name in ("cookie_batch.yaml", "randomization.yaml")
        },
        "packages": versions,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "bc2_device": "cpu",
        "bc2_torch_threads": 1,
    }


def make_benchmark():
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    return make_cookie_benchmark(
        task=TASK, profile=PROFILE, source_column=SOURCE_COLUMN, max_steps=MAX_STEPS,
        success_hold_steps=5, slot_tolerance_m=(0.015, 0.005), wall_contact_tolerance_m=0.03,
    )


def render_frames(env: Any, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    prior = env.render_cameras
    env.render_cameras = True
    try:
        return {f"observation.images.{name}": env._render_camera(name) for name in names}
    finally:
        env.render_cameras = prior


@dataclass(frozen=True)
class PolicySpec:
    kind: str
    checkpoint: str
    horizon: int = 12

    def identity(self) -> dict[str, Any]:
        path = Path(self.checkpoint).resolve()
        files = (path / "model.pt", path / "spec.json") if self.kind == "bc2" else (path,)
        return {**asdict(self), "checkpoint": str(path), "sha256": {item.name: sha256(item) for item in files}}


class ObservationPolicy:
    action_mode = "joint_position"
    finished = False

    def __init__(self, policy: Any, eager: bool):
        self.policy = policy
        self.eager = eager

    def bind_env(self, env: Any) -> None:
        self.env = env

    def reset(self, context: Any = None) -> None:
        self.policy.reset()
        self.digest = hashlib.sha256()
        self.first_batch_step = None
        self.target_peak = 0
        self.initial_layout = None
        self.step_count = 0

    def act(self, observation: dict, task: str = "") -> np.ndarray:
        names = self.policy.camera_names
        if self.step_count == 0 and "front" not in names:
            names = (*names, "front")
        if not self.eager and names:
            observation = {**observation, **render_frames(self.env, names)}
        if self.step_count == 0:
            from a3opt.backends.trajectory import FRONT, layout_features

            self.initial_layout = layout_features(observation[FRONT])
        action = self.policy.act(observation, task)
        if action.shape != (16,) or not np.all(np.isfinite(action)):
            raise ValueError("policy returned an invalid joint-position action")
        self.digest.update(np.asarray(action, dtype="<f8").tobytes())
        self.step_count += 1
        return action

    def add_frame(self, observation: dict, action: np.ndarray, info: dict, **metadata: Any) -> None:
        count = int(info["cookies_in_target"])
        self.target_peak = max(self.target_peak, count)
        if self.first_batch_step is None and count >= 5:
            self.first_batch_step = self.step_count


def run_episode(spec: PolicySpec, seed: int, eager: bool = False) -> dict[str, Any]:
    if spec.kind == "bc2":
        import torch

        from a3opt.backends.bc2 import BC2Policy

        torch.set_num_threads(1)
        policy = BC2Policy(spec.checkpoint, device="cpu", execution_horizon=spec.horizon)
    elif spec.kind == "trajectory":
        from a3opt.backends.trajectory import TrajectoryPolicy

        policy = TrajectoryPolicy(spec.checkpoint)
    else:
        raise ValueError(f"unknown policy kind {spec.kind}")
    benchmark = make_benchmark()
    env = benchmark.create_env(render_cameras=eager)
    wrapper = ObservationPolicy(policy, eager)
    try:
        result = benchmark.run_episode(wrapper, seed=seed, env=env, diagnostic=wrapper)
        return {**result.to_dict(), "action_sha256": wrapper.digest.hexdigest(),
                "first_batch_step": wrapper.first_batch_step, "target_peak": wrapper.target_peak,
                "layout_features": wrapper.initial_layout.tolist()}
    finally:
        env.close()
        close = getattr(policy, "close", None)
        if close is not None:
            close()


def validate_rows(rows: list[dict], seeds: tuple[int, ...]) -> None:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected a nonempty and unique seed list")
    if len(rows) != len(seeds) or {row["seed"] for row in rows} != set(seeds):
        raise ValueError("missing, duplicate or unexpected evaluation episodes")
    for row in rows:
        if (type(row["score"]) is not int or not 0 <= row["score"] <= 10
                or type(row["steps"]) is not int or not 1 <= row["steps"] <= MAX_STEPS
                or type(row["success"]) is not bool):
            raise ValueError(f"invalid evaluation result for seed {row['seed']}")
        if row["score"] != row["cookies_in_target"] or row["max_score"] != 10 or row["source_column"] != 1:
            raise ValueError(f"scoring contract mismatch for seed {row['seed']}")
        if row["success"] and (row["score"] != 10 or row["cookies_in_source"] != 70):
            raise ValueError(f"invalid success label for seed {row['seed']}")


def bootstrap_mean(values: np.ndarray) -> list[float]:
    samples = np.random.default_rng(0).choice(values, size=(10000, len(values)), replace=True)
    return np.quantile(samples.mean(axis=1), [0.025, 0.975]).tolist()


def wilson_interval(successes: int, count: int) -> list[float]:
    z_value = 1.959963984540054
    rate = successes / count
    variance = z_value**2 / count
    center = (rate + variance / 2) / (1 + variance)
    margin = z_value * math.sqrt((rate * (1 - rate) + variance / 4) / count) / (1 + variance)
    return [max(0, center - margin), min(1, center + margin)]


def summarize(rows: list[dict], seeds: tuple[int, ...]) -> dict[str, Any]:
    validate_rows(rows, seeds)
    scores = np.array([row["score"] for row in rows])
    successes = sum(row["success"] for row in rows)
    return {
        "episodes": len(seeds), "mean_score": float(scores.mean()),
        "mean_score_bootstrap_95": bootstrap_mean(scores),
        "successes": successes, "success_rate": successes / len(seeds),
        "success_rate_wilson_95": wilson_interval(successes, len(seeds)),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "score_distribution": {int(score): int((scores == score).sum()) for score in np.unique(scores)},
    }


def evaluate(specs: dict[str, PolicySpec], seeds: tuple[int, ...], directory: Path, workers: int, *, eager: bool = False):
    if not seeds or len(set(seeds)) != len(seeds) or not specs or workers < 1:
        raise ValueError("evaluation needs policies, unique seeds and at least one worker")
    receipt_path = directory / "request.json"
    receipt = {
        "environment": simulator_identity(), "seeds": list(seeds), "task": TASK, "profile": PROFILE,
        "source_column": SOURCE_COLUMN, "max_steps": MAX_STEPS, "camera_mode": "eager" if eager else "on_demand",
        "hold_steps": 5, "slot_tolerance_m": [0.015, 0.005], "wall_tolerance_m": 0.03,
        "policies": {name: policy.identity() for name, policy in specs.items()},
    }
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise ValueError(f"evaluation request changed in {directory}")
    write_json(receipt_path, receipt)
    results = {name: {} for name in specs}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for seed in seeds:
            for name, spec in specs.items():
                path = directory / name / f"{seed}.json"
                if path.exists():
                    row = read_json(path)
                    validate_rows([row], (seed,))
                    results[name][seed] = row
                else:
                    pending[pool.submit(run_episode, spec, seed, eager)] = (name, seed, path)
        for future in as_completed(pending):
            name, seed, path = pending[future]
            row = future.result()
            validate_rows([row], (seed,))
            write_json(path, row)
            results[name][seed] = row
            print(f"{directory.name} {name} seed={seed} score={row['score']} success={row['success']} steps={row['steps']}", flush=True)
    if {name: spec.identity() for name, spec in specs.items()} != receipt["policies"]:
        raise ValueError("evaluation checkpoint changed during execution")
    summary = {}
    for name, episodes in results.items():
        rows = [episodes[seed] for seed in seeds]
        summary[name] = {**summarize(rows, seeds), "rows": rows}
    write_json(directory / "results.json", summary)
    return summary


def evaluation_code() -> dict[str, str]:
    package = Path(__file__).parent
    names = ("refinement_eval.py", "backends/bc2.py", "backends/trajectory.py")
    return {name: sha256(package / name) for name in names}


def freeze_confirmation(
    directory: Path, development_dir: Path, reference_name: str, candidate_name: str,
) -> Path:
    path = directory / "confirmation-lock.json"
    if path.exists():
        raise FileExistsError(f"confirmation already frozen at {path}")
    development_request = development_dir / "request.json"
    development_results = development_dir / "results.json"
    request = read_json(development_request)
    results = read_json(development_results)
    if tuple(request["seeds"]) != DEVELOPMENT_SEEDS or request["environment"] != simulator_identity():
        raise ValueError("confirmation must follow the standard development protocol")
    if reference_name == candidate_name:
        raise ValueError("choose distinct reference and candidate policies")
    policies = {}
    for label, name in (("reference", reference_name), ("candidate", candidate_name)):
        item = request["policies"][name]
        spec = PolicySpec(item["kind"], item["checkpoint"], item["horizon"])
        policies[label] = spec.identity()
        if policies[label] != item:
            raise ValueError(f"{label} artifact changed since development evaluation")
        validate_rows(results[name]["rows"], DEVELOPMENT_SEEDS)
    payload = {
        "task": TASK, "profile": PROFILE, "source_column": SOURCE_COLUMN, "max_steps": MAX_STEPS,
        "seeds": list(CONFIRMATION_SEEDS), "goal_score_strictly_above": GOAL_SCORE,
        "camera_mode": "on_demand", "environment": simulator_identity(), "code_sha256": evaluation_code(),
        "policies": policies, "development_names": {"reference": reference_name, "candidate": candidate_name},
        "development_evidence": {
            str(item.resolve()): sha256(item) for item in (development_request, development_results)
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def confirm(lock_path: Path, workers: int) -> dict[str, Any]:
    lock = read_json(lock_path)
    if (tuple(lock["seeds"]) != CONFIRMATION_SEEDS or lock["goal_score_strictly_above"] != GOAL_SCORE
            or lock["environment"] != simulator_identity() or lock["code_sha256"] != evaluation_code()
            or (lock["task"], lock["profile"], lock["source_column"], lock["max_steps"], lock["camera_mode"])
            != (TASK, PROFILE, SOURCE_COLUMN, MAX_STEPS, "on_demand")
            or any(sha256(Path(path)) != digest for path, digest in lock["development_evidence"].items())):
        raise ValueError("confirmation protocol or development evidence changed after freeze")
    if set(lock["policies"]) != {"reference", "candidate"}:
        raise ValueError("confirmation requires one locked reference and one locked candidate")
    specs = {name: PolicySpec(item["kind"], item["checkpoint"], item["horizon"])
             for name, item in lock["policies"].items()}
    if {name: spec.identity() for name, spec in specs.items()} != lock["policies"]:
        raise ValueError("confirmation policy changed after freeze")
    directory = lock_path.parent / "confirmation"
    results = evaluate(specs, CONFIRMATION_SEEDS, directory, workers)
    reference, candidate = results["reference"], results["candidate"]
    gains = np.array([after["score"] - before["score"] for before, after in zip(reference["rows"], candidate["rows"])])
    verdict = {
        "lock_sha256": sha256(lock_path), "threshold": GOAL_SCORE,
        "mean_score": candidate["mean_score"], "mean_score_bootstrap_95": candidate["mean_score_bootstrap_95"],
        "point_estimate_above_threshold": candidate["mean_score"] > GOAL_SCORE,
        "bootstrap_lower_bound_above_threshold": candidate["mean_score_bootstrap_95"][0] > GOAL_SCORE,
        "paired_mean_gain": float(gains.mean()), "paired_gain_bootstrap_95": bootstrap_mean(gains),
        "paired_wins": int((gains > 0).sum()), "paired_ties": int((gains == 0).sum()),
        "paired_losses": int((gains < 0).sum()),
    }
    write_json(directory / "verdict.json", verdict)
    return verdict
