from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from a3opt.backends.trajectory import fit_trajectory, load_legacy_demos
from a3opt.refinement_eval import (
    DEVELOPMENT_SEEDS,
    LEGACY_SEEDS,
    TRAIN_POOL,
    PolicySpec,
    evaluate,
    evaluation_code,
    read_json,
    sha256,
    simulator_identity,
    write_json,
)
from a3opt.sampling import augment, choose_layouts, collect, probe_layouts


@dataclass(frozen=True)
class RefinementConfig:
    dataset: Path
    reference: Path
    out: Path
    initial_episodes: int | None = None
    sample_episodes: int = 8
    pool_size: int = 48
    rounds: int = 1
    workers: int = 8


def _record_once(path: Path, payload: dict) -> None:
    if path.exists() and read_json(path) != payload:
        raise ValueError(f"inputs changed for {path}")
    if not path.exists():
        write_json(path, payload)


def _fit(directory: Path, data: tuple, provenance: dict) -> PolicySpec:
    features, actions, seeds = data
    checkpoint = directory / "policy.npz"
    expected = {
        "provenance": provenance, "training_seeds": [int(seed) for seed in seeds],
        "episodes": len(seeds), "frames": sum(len(trace) for trace in actions),
        "learner": {"name": "local_trajectory_regression", "neighbors": 18, "ridge": 0.02},
    }
    _record_once(directory / "dataset-manifest.json", expected)
    fitted = directory / "training.json"
    if not fitted.exists():
        started = time.perf_counter()
        dataset_path = directory / "dataset.npz"
        if not dataset_path.exists():
            temporary = directory / ".dataset.tmp"
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, layout=features, seeds=seeds, actions=np.concatenate(actions),
                                    offsets=np.r_[0, np.cumsum([len(trace) for trace in actions])])
            temporary.replace(dataset_path)
        if not checkpoint.exists():
            temporary = directory / ".policy.tmp"
            if temporary.exists():
                temporary.unlink()
            fit_trajectory(temporary, features, actions, seeds)
            temporary.replace(checkpoint)
        write_json(fitted, {
            "dataset_sha256": sha256(dataset_path), "checkpoint_sha256": sha256(checkpoint),
            "manifest_sha256": sha256(directory / "dataset-manifest.json"),
            "fit_wall_seconds": time.perf_counter() - started,
        })
    else:
        receipt = read_json(fitted)
        for key, path in (("dataset_sha256", directory / "dataset.npz"), ("checkpoint_sha256", checkpoint),
                          ("manifest_sha256", directory / "dataset-manifest.json")):
            if receipt[key] != sha256(path):
                raise ValueError(f"training artifact changed for {path}")
    return PolicySpec("trajectory", str(checkpoint))


def _promotable(candidate: dict, incumbent: dict) -> bool:
    gain = candidate["mean_score"] - incumbent["mean_score"]
    success_gain = candidate["success_rate"] - incumbent["success_rate"]
    same_seeds = [row["seed"] for row in candidate["rows"]] == [row["seed"] for row in incumbent["rows"]]
    if not same_seeds:
        raise ValueError("promotion requires the same development episodes")
    fraction = np.mean([new["score"] >= old["score"] for new, old in zip(candidate["rows"], incumbent["rows"])])
    return gain >= 0 and success_gain >= 0 and (gain > 0 or success_gain > 0) and fraction >= 0.7


def develop(config: RefinementConfig) -> dict[str, Any]:
    if not 1 <= config.pool_size <= len(TRAIN_POOL) or config.rounds < 1 or config.sample_episodes < 1:
        raise ValueError("sampling needs a valid training pool and positive round and episode budgets")
    if 2 * config.rounds * config.sample_episodes > config.pool_size or config.workers < 1:
        raise ValueError("the pool must cover both arms and all rounds, with at least one worker")
    if config.initial_episodes is not None and config.initial_episodes < 1:
        raise ValueError("initial episode count must be positive")
    root = config.out.resolve()
    dataset = config.dataset.resolve()
    metadata = dataset.with_suffix(".meta.json")
    reference = PolicySpec("bc2", str(config.reference.resolve()))
    provenance = {str(dataset): sha256(dataset), str(metadata): sha256(metadata)}
    parameters = {key: value for key, value in asdict(config).items()
                  if key not in ("dataset", "reference", "out", "workers")}
    _record_once(root / "development-protocol.json", {
        "input_sha256": provenance, "reference": reference.identity(), "environment": simulator_identity(),
        "code_sha256": {**evaluation_code(), "refinement.py": sha256(Path(__file__)),
                        "sampling.py": sha256(Path(__file__).with_name("sampling.py"))},
        "development_seeds": list(DEVELOPMENT_SEEDS), "training_pool": list(TRAIN_POOL[:config.pool_size]),
        "budget": parameters, "promotion_non_worse_fraction": 0.7,
    })
    features, actions, seeds = load_legacy_demos(dataset, set(LEGACY_SEEDS))
    legacy_reports = read_json(metadata)["episode_reports"]
    valid_seeds = {row["seed"] for row in legacy_reports if row["success"] and row["score"] == 10}
    if not set(seeds) <= valid_seeds:
        raise ValueError("legacy dataset contains episodes without a complete expert demonstration")
    if config.initial_episodes is not None and config.initial_episodes > len(seeds):
        raise ValueError("requested initial set is larger than the legacy dataset")
    count = config.initial_episodes or len(seeds)
    order = np.argsort(seeds, kind="stable")[:count]
    base = features[order], [actions[index] for index in order], seeds[order]
    initial = root / "initial"
    incumbent_spec = _fit(initial / "base", base, {"legacy_sha256": provenance, "new_shards": []})
    incumbent_name, incumbent_dir = "base", initial / "evaluation"
    initial_results = evaluate({"bc2": reference, "base": incumbent_spec}, DEVELOPMENT_SEEDS, incumbent_dir, config.workers)
    incumbent_data, incumbent_metrics = base, initial_results["base"]
    incumbent_manifest = initial / "base" / "dataset-manifest.json"
    layout_path = root / "training-layouts.json"
    if not layout_path.exists():
        write_json(layout_path, probe_layouts(TRAIN_POOL[:config.pool_size]))
    layouts = {int(seed): layout for seed, layout in read_json(layout_path).items()}
    if set(layouts) != set(TRAIN_POOL[:config.pool_size]):
        raise ValueError("training layout cache does not match the predeclared pool")
    episode_dir = root / "episodes"
    decision_log = []
    for round_number in range(1, config.rounds + 1):
        round_dir = root / f"round-{round_number:02d}"
        feedback_source = "bc2" if incumbent_metrics["success_rate"] == 1 else incumbent_name
        diagnostics = initial_results["bc2"] if feedback_source == "bc2" else incumbent_metrics
        plans = {method: choose_layouts(layouts, incumbent_data[0], diagnostics["rows"], config.sample_episodes,
                                        method=method, random_seed=round_number - 1)
                 for method in ("uniform", "feedback")}
        _record_once(round_dir / "plan.json", {
            "parent": incumbent_name, "feedback_source": feedback_source,
            "parent_manifest_sha256": sha256(incumbent_manifest), "diagnostic_rows": diagnostics["rows"], "plans": plans,
        })
        union = {choice["seed"]: choice for plan in plans.values() for choice in plan}
        reports = {row["seed"]: row for row in collect(list(union.values()), episode_dir, config.workers)}
        branches = {"bc2": reference}
        datasets, counts = {}, {}
        for method, plan in plans.items():
            records = [reports[choice["seed"]] for choice in plan]
            datasets[method] = augment(incumbent_data, records, episode_dir)
            counts[method] = len(datasets[method][2])
            branches[method] = _fit(round_dir / method, datasets[method], {
                "parent_manifest": str(incumbent_manifest), "parent_manifest_sha256": sha256(incumbent_manifest),
                "new_shards": records,
            })
        branch_dir = round_dir / "evaluation"
        results = evaluate(branches, DEVELOPMENT_SEEDS, branch_dir, config.workers)
        eligible = [method for method in plans if _promotable(results[method], incumbent_metrics)]
        selected = max(eligible, key=lambda method: (results[method]["mean_score"],
                       results[method]["success_rate"], -counts[method])) if eligible else None
        decision = {"round": round_number, "actual_rollouts": len(union), "parent": incumbent_name,
                    "feedback_source": feedback_source, "training_episode_counts": counts,
                    "eligible": eligible, "selected": selected}
        write_json(round_dir / "decision.json", decision)
        decision_log.append(decision)
        if selected is not None:
            incumbent_name, incumbent_dir = selected, branch_dir
            incumbent_data, incumbent_metrics = datasets[selected], results[selected]
            incumbent_manifest = round_dir / selected / "dataset-manifest.json"
        for seed in union:
            del layouts[seed]
    summary = {"selected_policy": incumbent_name, "development_dir": str(incumbent_dir),
               "reference_name": "bc2", "mean_score": incumbent_metrics["mean_score"],
               "success_rate": incumbent_metrics["success_rate"], "dataset_manifest": str(incumbent_manifest),
               "rounds": decision_log}
    write_json(root / "development-summary.json", summary)
    return summary
