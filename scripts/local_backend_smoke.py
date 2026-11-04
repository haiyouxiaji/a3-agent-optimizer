from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SIM_SRC = Path(
    os.environ.get("A3_DUAL_ARM_SIM_SRC", str(ROOT.parent / "a3_dual_arm_sim" / "src"))
)
if SIM_SRC.is_dir() and str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))

from a3opt.backends.local_sim import (
    DEFAULT_DATASET_PATH,
    LocalSimBackend,
    evaluate_checkpoint,
    load_policy,
    record_dataset,
    train_candidate,
)
from a3opt.contracts import CandidateSpec, ExperimentBudget, ExperimentOutcome, now_iso

LOG_FILE: Path | None = None


def log(message: str) -> None:
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    target = LOG_FILE
    if target is not None:
        with target.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def write_json(path: Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def default_candidates() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            candidate_id="tiny_cnn-concat-chunk-k20-lr0.001",
            encoder="tiny_cnn",
            fusion="concat",
            action_head="chunk",
            chunk_size=20,
            lr=1e-3,
            image_size=64,
            state_dim=16,
            seed=0,
        ),
        CandidateSpec(
            candidate_id="tiny_cnn-film-chunk-k20-lr0.001",
            encoder="tiny_cnn",
            fusion="film",
            action_head="chunk",
            chunk_size=20,
            lr=1e-3,
            image_size=64,
            state_dim=16,
            seed=0,
        ),
        CandidateSpec(
            candidate_id="state_mlp-concat-chunk-k1-lr0.001",
            encoder="state_mlp",
            fusion="concat",
            action_head="chunk",
            chunk_size=1,
            lr=1e-3,
            image_size=0,
            state_dim=16,
            seed=0,
        ),
    ]


def parse_seeds(text: str) -> list[int]:
    seeds = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real a3opt local_sim pipeline: expert collection, behavior cloning, closed-loop evaluation"
    )
    parser.add_argument(
        "--stage",
        choices=("record", "train", "eval", "all", "backend", "cli", "supplement"),
        default="all",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts" / "real_runs"))
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--collect-max-steps", type=int, default=600)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--task", default="2x5")
    parser.add_argument("--profile", default="basic")
    parser.add_argument("--source-column", default="1")
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-seeds", type=parse_seeds, default=[2100, 2101])
    parser.add_argument("--eval-max-steps", type=int, default=120)
    parser.add_argument("--supplement-max-steps", type=int, default=500)
    parser.add_argument("--backend-steps", type=int, default=20)
    parser.add_argument("--backend-eval-steps", type=int, default=10)
    parser.add_argument("--cli-seed", type=int, default=2200)
    parser.add_argument("--cli-max-steps", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    return parser


def dataset_seeds(args: argparse.Namespace) -> list[int]:
    return [int(args.seed_start) + index for index in range(int(args.episodes))]


def record_stage(args: argparse.Namespace) -> dict[str, Any]:
    seeds = dataset_seeds(args)
    if len(seeds) < 1:
        raise ValueError("--episodes must be positive")
    log(
        f"recording expert dataset task={args.task} profile={args.profile} "
        f"source_column={args.source_column} seeds={seeds} max_steps={args.collect_max_steps}"
    )

    def on_episode(index: int, total: int, seed: int, report: dict[str, Any]) -> None:
        log(
            f"episode {index + 1}/{total} seed={seed} score={report['score']:.0f}/10 "
            f"success={report['success']} steps={report['steps']} wall={report['wall_seconds']:.2f}s"
        )

    meta = record_dataset(
        {
            "dataset_path": str(args.dataset),
            "seeds": seeds,
            "task": args.task,
            "profile": args.profile,
            "source_column": str(args.source_column),
            "max_steps": int(args.collect_max_steps),
            "image_size": int(args.image_size),
        },
        on_episode=on_episode,
    )
    log(
        f"dataset saved path={meta['dataset_path']} frames={meta['frames']} "
        f"episodes={meta['episodes']} recorded_seeds={meta['recorded_seeds']} "
        f"wall={meta['wall_seconds']:.2f}s"
    )
    return meta


def train_stage(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = default_candidates()
    entries: list[dict[str, Any]] = []
    for candidate in candidates:
        checkpoint_dir = Path(args.out_dir) / "candidates" / candidate.candidate_id / "checkpoint"
        budget = ExperimentBudget(max_train_steps=int(args.train_steps), batch_size=int(args.batch_size))
        log(
            f"training {candidate.candidate_id} steps={args.train_steps} batch={args.batch_size} "
            f"lr={candidate.lr} image_size={candidate.image_size}"
        )
        result = train_candidate(candidate, args.dataset, budget, checkpoint_dir)
        entry: dict[str, Any] = {
            "candidate": candidate.to_dict(),
            "train": result.to_dict(),
            "checkpoint_dir": str(checkpoint_dir),
            "spec": None,
        }
        spec_path = checkpoint_dir / "spec.json"
        if spec_path.is_file():
            entry["spec"] = json.loads(spec_path.read_text(encoding="utf-8"))
        if result.status == "ok":
            log(
                f"trained {candidate.candidate_id} first_loss={result.metrics.get('first_loss'):.6f} "
                f"final_loss={result.metrics.get('final_loss'):.6f} steps={result.train_steps} "
                f"wall={result.wall_seconds:.2f}s artifact={result.artifact}"
            )
        else:
            log(f"training failed {candidate.candidate_id}: {result.error}")
        entries.append(entry)
    return entries


def load_train_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for candidate in default_candidates():
        checkpoint_dir = Path(args.out_dir) / "candidates" / candidate.candidate_id / "checkpoint"
        spec_path = checkpoint_dir / "spec.json"
        if not spec_path.is_file():
            log(f"missing checkpoint spec for {candidate.candidate_id}: {spec_path}")
            entries.append(
                {
                    "candidate": candidate.to_dict(),
                    "train": {"status": "failed", "error": "missing checkpoint"},
                    "checkpoint_dir": str(checkpoint_dir),
                    "spec": None,
                }
            )
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "candidate": candidate.to_dict(),
                "train": {
                    "status": "ok",
                    "train_steps": int(spec.get("train_steps", 0)),
                    "metrics": {
                        "first_loss": spec.get("first_loss"),
                        "final_loss": spec.get("final_loss"),
                        "loss_last_10_mean": spec.get("loss_last_10_mean"),
                    },
                    "artifact": str(checkpoint_dir / "model.pt"),
                    "wall_seconds": float(spec.get("wall_seconds", 0.0)),
                    "error": "",
                },
                "checkpoint_dir": str(checkpoint_dir),
                "spec": spec,
            }
        )
    return entries


def eval_stage(args: argparse.Namespace, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for entry in entries:
        train = entry["train"]
        if train.get("status") != "ok":
            entry["eval"] = {
                "status": "skipped",
                "error": train.get("error", "training was not completed"),
                "episodes": [],
                "mean_score": None,
                "success_rate": None,
                "wall_seconds": 0.0,
            }
            continue
        candidate = CandidateSpec.from_dict(entry["candidate"])
        eval_dir = Path(args.out_dir) / "candidates" / candidate.candidate_id / "eval"
        log(
            f"evaluating {candidate.candidate_id} seeds={args.eval_seeds} "
            f"max_steps={args.eval_max_steps} checkpoint={entry['checkpoint_dir']}"
        )
        result = evaluate_checkpoint(
            candidate,
            entry["checkpoint_dir"],
            args.eval_seeds,
            int(args.eval_max_steps),
            eval_dir,
        )
        entry["eval"] = result.to_dict()
        if result.status == "ok":
            scores = [row.score for row in result.episodes]
            steps = [row.steps for row in result.episodes]
            log(
                f"evaluated {candidate.candidate_id} mean_score={result.mean_score:.3f} "
                f"success_rate={result.success_rate:.3f} scores={scores} steps={steps} "
                f"wall={result.wall_seconds:.2f}s"
            )
        else:
            log(f"evaluation failed {candidate.candidate_id}: {result.error}")
    return entries


def backend_stage(args: argparse.Namespace) -> dict[str, Any]:
    backend = LocalSimBackend(
        {
            "dataset_path": str(args.dataset),
            "experiment_root": str(Path(args.out_dir) / "backend_smoke"),
            "seeds": dataset_seeds(args),
            "collect_max_steps": int(args.collect_max_steps),
            "task": args.task,
            "profile": args.profile,
            "source_column": str(args.source_column),
            "device": "cpu",
        }
    )
    available, reason = backend.available()
    log(f"backend available={available} reason={reason}")
    candidate = CandidateSpec(
        candidate_id="backend_smoke-state_mlp",
        encoder="state_mlp",
        fusion="concat",
        action_head="chunk",
        chunk_size=1,
        lr=1e-3,
        image_size=0,
        state_dim=16,
        seed=0,
    )
    outcome = backend.run(
        candidate,
        ExperimentBudget(max_train_steps=int(args.backend_steps), batch_size=16),
        [int(args.eval_seeds[0])],
        int(args.backend_eval_steps),
    )
    payload = outcome.to_dict()
    roundtrip = ExperimentOutcome.from_dict(json.loads(json.dumps(payload)))
    roundtrip_ok = roundtrip.candidate == candidate and roundtrip.status == payload["status"]
    log(
        f"backend.run status={payload['status']} mean_score={payload['eval']['mean_score']} "
        f"train_steps={payload['train']['train_steps']} roundtrip_ok={roundtrip_ok}"
    )
    return {
        "available": [bool(available), reason],
        "candidate": candidate.to_dict(),
        "outcome": payload,
        "roundtrip_ok": bool(roundtrip_ok),
    }


def cli_stage(args: argparse.Namespace) -> dict[str, Any]:
    candidate = default_candidates()[0]
    checkpoint_dir = Path(args.out_dir) / "candidates" / candidate.candidate_id / "checkpoint"
    output_path = Path(args.out_dir) / "policy_cli_result.json"
    python_path = [str(SRC), str(SIM_SRC)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_path.append(existing)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    environment["A3OPT_CHECKPOINT"] = str(checkpoint_dir)
    command = [
        sys.executable,
        "-m",
        "a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli",
        "--policy",
        "a3opt.backends.policy_plugin:create_policy",
        "--episodes",
        "1",
        "--seed-start",
        str(int(args.cli_seed)),
        "--max-steps",
        str(int(args.cli_max_steps)),
        "--workers",
        "1",
        "--task",
        args.task,
        "--profile",
        args.profile,
        "--source-column",
        str(args.source_column),
        "--output",
        str(output_path),
    ]
    log(f"policy plugin CLI command: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
        env=environment,
    )
    log_path = Path(args.out_dir) / "policy_cli.log"
    log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
    parsed: Any = None
    if output_path.is_file():
        try:
            parsed = json.loads(output_path.read_text(encoding="utf-8"))
        except ValueError:
            parsed = None
    summary = parsed.get("summary") if isinstance(parsed, dict) else None
    log(f"policy plugin CLI returncode={completed.returncode} summary={summary}")
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "output_json": str(output_path),
        "log": str(log_path),
        "summary": summary,
        "checkpoint": str(checkpoint_dir),
        "stdout_tail": "\n".join((completed.stdout or "").splitlines()[-25:]),
    }


def dataset_meta(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    meta_path = dataset_path.with_suffix(".meta.json")
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    if not dataset_path.is_file():
        return {"dataset_path": str(dataset_path), "exists": False}
    with np.load(dataset_path) as data:
        seeds = np.asarray(data["seeds"], dtype=np.int64)
        episodes = int(np.unique(seeds).size)
        front = np.asarray(data["front"])
        return {
            "dataset_path": str(dataset_path),
            "exists": True,
            "frames": int(front.shape[0]),
            "episodes": episodes,
            "seeds": sorted(int(value) for value in np.unique(seeds).tolist()),
            "image_size": int(front.shape[1]),
            "created_at": None,
        }


def offline_action_error(
    entry: dict[str, Any],
    front: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
) -> float | None:
    try:
        policy = load_policy(entry["checkpoint_dir"])
    except (OSError, ValueError):
        return None
    model = policy.model
    images = torch.from_numpy(front)
    states = torch.from_numpy(state)
    predictions = []
    with torch.no_grad():
        for start in range(0, front.shape[0], 256):
            predictions.append(model(images[start : start + 256], states[start : start + 256]).numpy())
    policy.close()
    predicted = np.concatenate(predictions, axis=0)[:, 0, :]
    return round(float(np.abs(predicted - action).mean()), 6)


def expert_baseline(args: argparse.Namespace, max_steps: int, seeds: list[int]) -> dict[str, Any]:
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark

    benchmark = make_cookie_benchmark(
        task=args.task,
        profile=args.profile,
        source_column=str(args.source_column),
        max_steps=int(max_steps),
    )
    episodes: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        score = benchmark.run_episode(policy="same_column", seed=int(seed), episode_idx=index + 1)
        episodes.append(
            {
                "seed": int(seed),
                "score": float(score.score),
                "success": bool(score.success),
                "steps": int(score.steps),
                "cookies_in_target": int(score.cookies_in_target),
                "cookies_in_source": int(score.cookies_in_source),
                "wall_seconds": float(score.wall_seconds),
            }
        )
        log(
            f"expert baseline max_steps={max_steps} seed={seed} score={score.score} "
            f"success={score.success} steps={score.steps} source={score.cookies_in_source} "
            f"wall={score.wall_seconds:.2f}s"
        )
    mean_score = sum(row["score"] for row in episodes) / len(episodes) if episodes else 0.0
    return {
        "policy": "same_column",
        "max_steps": int(max_steps),
        "mean_score": round(mean_score, 4),
        "episodes": episodes,
    }


def supplement_stage(args: argparse.Namespace) -> dict[str, Any]:
    max_steps = int(args.supplement_max_steps)
    seeds = [int(seed) for seed in args.eval_seeds]
    log(f"supplementary long-horizon evaluation max_steps={max_steps} seeds={seeds}")
    with np.load(args.dataset) as data:
        front = np.asarray(data["front"], dtype=np.uint8)
        state = np.asarray(data["state"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
    entries = load_train_entries(args)
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        candidate_id = entry["candidate"]["candidate_id"]
        if entry["train"].get("status") != "ok":
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "status": "skipped",
                    "error": entry["train"].get("error", ""),
                }
            )
            continue
        candidate = CandidateSpec.from_dict(entry["candidate"])
        out_dir = Path(args.out_dir) / "candidates" / candidate_id / "eval_long"
        result = evaluate_checkpoint(candidate, entry["checkpoint_dir"], seeds, max_steps, out_dir)
        row = {
            "candidate_id": candidate_id,
            "status": result.status,
            "max_steps": max_steps,
            "mean_score": round(result.mean_score, 4),
            "success_rate": round(result.success_rate, 4),
            "episodes": [episode.to_dict() for episode in result.episodes],
            "wall_seconds": round(result.wall_seconds, 3),
            "offline_action_mae": offline_action_error(entry, front, state, action),
        }
        candidates.append(row)
        log(
            f"supplementary {candidate_id} status={row['status']} mean_score={row['mean_score']} "
            f"scores={[episode['score'] for episode in row['episodes']]} "
            f"wall={row['wall_seconds']}s offline_action_mae={row['offline_action_mae']}"
        )
    expert: list[dict[str, Any]] = []
    probe_steps = sorted({120, 200, max_steps}) if max_steps > 200 else [max_steps]
    for steps in probe_steps:
        probe_seeds = seeds if steps == max_steps else seeds[:1]
        expert.append(expert_baseline(args, steps, probe_seeds))
    return {"max_steps": max_steps, "seeds": seeds, "candidates": candidates, "expert": expert}


def summarize(
    args: argparse.Namespace,
    dataset: dict[str, Any],
    entries: list[dict[str, Any]],
    backend_verification: dict[str, Any] | None,
    cli_verification: dict[str, Any] | None,
    supplementary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        raw_candidate = entry["candidate"]
        train = entry.get("train") or {}
        evaluation = entry.get("eval") or {}
        metrics = train.get("metrics") or {}
        spec = entry.get("spec") or {}
        candidates.append(
            {
                "candidate_id": raw_candidate["candidate_id"],
                "encoder": raw_candidate["encoder"],
                "fusion": raw_candidate["fusion"],
                "action_head": raw_candidate["action_head"],
                "chunk_size": raw_candidate["chunk_size"],
                "lr": raw_candidate["lr"],
                "image_size": raw_candidate["image_size"],
                "seed": raw_candidate["seed"],
                "status": evaluation.get("status") or train.get("status"),
                "first_loss": metrics.get("first_loss"),
                "final_loss": metrics.get("final_loss"),
                "loss_last_10_mean": metrics.get("loss_last_10_mean"),
                "train_steps": train.get("train_steps"),
                "train_wall_seconds": train.get("wall_seconds"),
                "device": spec.get("device"),
                "mean_score": evaluation.get("mean_score"),
                "success_rate": evaluation.get("success_rate"),
                "eval_episodes": evaluation.get("episodes"),
                "eval_wall_seconds": evaluation.get("wall_seconds"),
                "total_wall_seconds": round(
                    float(train.get("wall_seconds") or 0.0)
                    + float(evaluation.get("wall_seconds") or 0.0),
                    3,
                ),
                "checkpoint": entry.get("checkpoint_dir"),
            }
        )
    scored = [
        row for row in candidates if row["status"] == "ok" and row["mean_score"] is not None
    ]
    best = max(scored, key=lambda row: float(row["mean_score"])) if scored else None
    train_wall = sum(float(row["train_wall_seconds"] or 0.0) for row in candidates)
    eval_wall = sum(float(row["eval_wall_seconds"] or 0.0) for row in candidates)
    mean_scores = [float(row["mean_score"]) for row in scored]
    return {
        "created_at": now_iso(),
        "backend": "local_sim",
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "mujoco": package_version("mujoco"),
            "numpy": np.__version__,
        },
        "protocol": {
            "task": args.task,
            "profile": args.profile,
            "source_column": str(args.source_column),
            "dataset_seeds": dataset_seeds(args),
            "train_steps": int(args.train_steps),
            "batch_size": int(args.batch_size),
            "eval_seeds": [int(seed) for seed in args.eval_seeds],
            "eval_max_steps": int(args.eval_max_steps),
            "collect_max_steps": int(args.collect_max_steps),
        },
        "dataset": dataset,
        "candidates": candidates,
        "best_candidate": best["candidate_id"] if best else None,
        "best_mean_score": best["mean_score"] if best else None,
        "mean_of_candidate_mean_scores": (
            round(sum(mean_scores) / len(mean_scores), 4) if mean_scores else None
        ),
        "totals": {
            "train_wall_seconds": round(train_wall, 3),
            "eval_wall_seconds": round(eval_wall, 3),
            "wall_seconds": round(train_wall + eval_wall, 3),
        },
        "verification": {
            "backend_run": backend_verification,
            "policy_plugin_cli": cli_verification,
        },
        "supplementary": supplementary,
    }


def main() -> int:
    global LOG_FILE
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = out_dir / "run.log"
    log(
        f"local_sim real run stage={args.stage} out_dir={out_dir.resolve()} "
        f"python={sys.executable} sim_src={SIM_SRC}"
    )
    if args.stage == "record" and not args.force and Path(args.dataset).is_file():
        log(f"dataset already exists, refusing silent overwrite: {args.dataset}; pass --force")
        return 1
    dataset = dataset_meta(args)
    entries: list[dict[str, Any]] = []
    backend_verification: dict[str, Any] | None = None
    cli_verification: dict[str, Any] | None = None
    supplementary: dict[str, Any] | None = None
    if args.stage in ("record", "all"):
        if args.force or not Path(args.dataset).is_file():
            dataset = record_stage(args)
        else:
            log(f"dataset exists, reusing {args.dataset} frames={dataset.get('frames')}")
    if args.stage == "record":
        write_json(
            out_dir / "summary.json",
            {
                "created_at": now_iso(),
                "backend": "local_sim",
                "stage": "record",
                "command": " ".join(sys.argv),
                "protocol": {
                    "task": args.task,
                    "profile": args.profile,
                    "source_column": str(args.source_column),
                    "dataset_seeds": dataset_seeds(args),
                },
                "dataset": dataset,
            },
        )
        log(f"record stage done summary={out_dir / 'summary.json'}")
        return 0
    if args.stage in ("train", "all"):
        entries = train_stage(args)
    if args.stage == "eval":
        entries = load_train_entries(args)
    if args.stage in ("eval", "all"):
        entries = eval_stage(args, entries)
    if args.stage in ("backend", "all"):
        backend_verification = backend_stage(args)
    if args.stage in ("cli", "all"):
        cli_verification = cli_stage(args)
    if args.stage in ("supplement", "all"):
        supplementary = supplement_stage(args)
    if args.stage == "supplement":
        summary_path = out_dir / "summary.json"
        existing: dict[str, Any] = {}
        if summary_path.is_file():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        existing["dataset"] = dataset
        existing["supplementary"] = supplementary
        existing["updated_at"] = now_iso()
        existing["command"] = " ".join(sys.argv)
        write_json(summary_path, existing)
        log(f"supplementary evaluation merged into {summary_path}")
        return 0
    summary = summarize(
        args,
        dataset,
        entries,
        backend_verification,
        cli_verification,
        supplementary,
    )
    summary["stage"] = args.stage
    write_json(out_dir / "summary.json", summary)
    log(
        f"summary written {out_dir / 'summary.json'} candidates={len(summary['candidates'])} "
        f"best={summary['best_candidate']} best_mean_score={summary['best_mean_score']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
