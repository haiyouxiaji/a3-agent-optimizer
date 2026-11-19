from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from a3opt.acceptance import AcceptanceCriteria
from a3opt.backends.base import make_backend
from a3opt.contracts import ExperimentBudget, RunBudget, RunLayout
from a3opt.loop import LoopConfig, run_optimization
from a3opt.report import write_report
from a3opt.space import CandidateSpace


def _parse_seeds(text: str) -> tuple[int, ...]:
    seeds = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one evaluation seed is required")
    return tuple(seeds)


def _default_run_id() -> str:
    return "run-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def _print_summary(result: dict) -> None:
    summary = result["summary"]
    print(f"run_dir: {result['run_dir']}")
    print(f"report: {result['report']}")
    print(f"experiments: {summary['experiments']}")
    print(f"best_candidate: {summary['best_candidate_id']}")
    print(f"best_mean_normalized: {summary['best_mean_score']}")
    print(f"accepted: {summary['accepted']}")
    print(f"rejected: {summary['rejected']}")
    budget = summary.get("budget", {})
    print(f"spent_wall_seconds: {budget.get('spent_wall_seconds')}")
    print(f"spent_train_steps: {budget.get('spent_train_steps')}")


def _cmd_plan(args: argparse.Namespace) -> int:
    space = CandidateSpace()
    payload = {
        "space": space.name,
        "grid_size": space.grid_size(),
        "staged": [
            {"stage": stage, "candidate": spec.to_dict(), "errors": space.validate(spec)}
            for stage, spec in space.staged_candidates()
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    dataset = str(getattr(args, "dataset", "") or "").strip()
    experiment_root = str(getattr(args, "experiment_root", "") or "").strip()
    if args.backend in ("local", "local_sim", "local_sim_v2"):
        backend_kwargs = {}
        if dataset:
            backend_kwargs["dataset_path"] = Path(dataset)
        if experiment_root:
            backend_kwargs["experiment_root"] = Path(experiment_root)
        backend = make_backend(args.backend, **backend_kwargs)
    else:
        backend = make_backend(args.backend)
    available, reason = backend.available()
    if not available:
        print(f"backend unavailable: {reason}", file=sys.stderr)
        return 2
    run_id = args.run_id or _default_run_id()
    config = LoopConfig(
        run_dir=Path(args.out) / run_id,
        backend=backend,
        experiments=int(args.experiments),
        eval_seeds=_parse_seeds(args.eval_seeds),
        max_eval_steps=args.max_eval_steps if args.max_eval_steps is not None else (120 if backend.name == "stub" else 600),
        experiment_budget=ExperimentBudget(
            max_train_steps=int(args.train_steps),
            batch_size=int(args.batch_size),
            wall_seconds_limit=float(args.experiment_wall_limit),
        ),
        run_budget=RunBudget(
            max_experiments=int(args.experiments),
            max_train_steps_total=int(args.train_steps) * int(args.experiments),
            max_wall_seconds=float(args.wall_limit),
            max_gpu_seconds=float(args.gpu_limit),
        ),
        criteria=AcceptanceCriteria(
            min_mean_gain=float(args.min_gain),
            min_non_worse_fraction=float(args.min_non_worse),
            max_cost_ratio=float(args.max_cost_ratio),
        ),
        exploration_constant=float(args.exploration),
        max_children=int(args.max_children),
        run_id=run_id,
    )
    result = run_optimization(config)
    _print_summary(result)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    summary_path = run_dir / "run.json"
    tree_path = run_dir / "tree.json"
    memory_path = run_dir / "memory.jsonl"
    if not summary_path.is_file() or not tree_path.is_file():
        print(f"missing run artifacts under {run_dir}", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in memory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if memory_path.is_file() else []
    layout = RunLayout(run_dir=run_dir)
    criteria = summary.get("config", {}).get("criteria", asdict(AcceptanceCriteria()))
    report = write_report(layout, summary, tree, records, criteria)
    print(f"report: {report}")
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    run_dir = Path(args.work_dir) / (_default_run_id() + "-selftest")
    config = LoopConfig(
        run_dir=run_dir,
        backend=make_backend("stub"),
        experiments=int(args.experiments),
        eval_seeds=(2000, 2001),
        max_eval_steps=120,
        experiment_budget=ExperimentBudget(max_train_steps=int(args.train_steps), batch_size=16),
        run_budget=RunBudget(max_experiments=int(args.experiments), max_train_steps_total=int(args.train_steps) * int(args.experiments)),
        criteria=AcceptanceCriteria(),
        run_id=run_dir.name,
    )
    result = run_optimization(config)
    _print_summary(result)
    return 0


@contextmanager
def _run_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(directory / ".run-lock.sqlite3", timeout=0)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"cannot acquire run lock in {directory}") from exc
        yield


def _cmd_refine(args: argparse.Namespace) -> int:
    from a3opt.refinement import RefinementConfig, develop

    config = RefinementConfig(
        dataset=Path(args.dataset), reference=Path(args.reference), out=Path(args.out),
        initial_episodes=args.initial_episodes, sample_episodes=args.sample_episodes,
        pool_size=args.pool_size, rounds=args.rounds, workers=args.workers,
    )
    with _run_lock(config.out):
        result = develop(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_freeze_confirmation(args: argparse.Namespace) -> int:
    from a3opt.refinement_eval import freeze_confirmation, read_json

    evidence = read_json(Path(args.selection))
    with _run_lock(Path(args.out)):
        lock = freeze_confirmation(Path(args.out), Path(evidence["development_dir"]),
                                   evidence["reference_name"], evidence["selected_policy"])
    print(lock.resolve())
    return 0


def _cmd_confirm(args: argparse.Namespace) -> int:
    from a3opt.refinement_eval import confirm, read_json

    lock = Path(args.lock)
    with _run_lock(lock.parent):
        verdict = confirm(lock, args.workers)
        if args.eager_replay:
            from a3opt.refinement_replay import replay_candidate

            replay = replay_candidate(lock, args.workers, read_json(lock.parent / "confirmation/results.json")["candidate"])
            verdict["eager_replay_exact_matches"] = len(replay)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0


def _cmd_advanced(args: argparse.Namespace) -> int:
    from a3opt.advanced_agent import AdvancedBudget, run_agent

    budget = AdvancedBudget(args.seed_block, args.calibration, args.initial_attempts, args.feedback_attempts, args.rounds)
    with _run_lock(Path(args.out).resolve()):
        result = run_agent(Path(args.out), budget, args.workers, args.recover_from)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("milestone_met") else 2


def _cmd_advanced_followup(args: argparse.Namespace) -> int:
    from a3opt.advanced_followup import FollowupBudget, run_followup

    budget = FollowupBudget(args.seed_block, args.calibration, args.initial_attempts, args.feedback_attempts, args.rounds)
    with _run_lock(args.out.resolve()):
        result = run_followup(args.out, args.reference, budget, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("milestone_met") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a3opt",
        description="Agent-driven experiment search for modular A3 robot policy optimization",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="print the staged candidate space")
    plan.set_defaults(func=_cmd_plan)

    run = sub.add_parser("run", help="run the optimization loop")
    run.add_argument("--backend", default="stub", choices=["stub", "local", "local_sim", "local_sim_v2", "a3_sim", "a3_sim_cli"])
    run.add_argument("--experiments", type=int, default=6)
    run.add_argument("--out", default="runs")
    run.add_argument("--run-id", default="")
    run.add_argument("--eval-seeds", default="2000,2001")
    run.add_argument("--dataset", default="")
    run.add_argument("--experiment-root", default="")
    run.add_argument("--max-eval-steps", type=int)
    run.add_argument("--train-steps", type=int, default=400)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--experiment-wall-limit", type=float, default=1800.0)
    run.add_argument("--wall-limit", type=float, default=10800.0)
    run.add_argument("--gpu-limit", type=float, default=0.0)
    run.add_argument("--exploration", type=float, default=0.7)
    run.add_argument("--max-children", type=int, default=3)
    run.add_argument("--min-gain", type=float, default=1.0)
    run.add_argument("--min-non-worse", type=float, default=0.7)
    run.add_argument("--max-cost-ratio", type=float, default=3.0)
    run.set_defaults(func=_cmd_run)

    report = sub.add_parser("report", help="regenerate the markdown report for a run")
    report.add_argument("--run", required=True)
    report.set_defaults(func=_cmd_report)

    selftest = sub.add_parser("selftest", help="run a deterministic synthetic optimization loop")
    selftest.add_argument("--experiments", type=int, default=6)
    selftest.add_argument("--train-steps", type=int, default=100)
    selftest.add_argument("--work-dir", default=str(Path(tempfile.gettempdir()) / "a3opt-selftest"))
    selftest.set_defaults(func=_cmd_selftest)

    refine = sub.add_parser("refine", help="collect and compare feedback-guided and uniform training data")
    refine.add_argument("--dataset", required=True)
    refine.add_argument("--reference", required=True)
    refine.add_argument("--out", required=True)
    refine.add_argument("--initial-episodes", type=int)
    refine.add_argument("--sample-episodes", type=int, default=8)
    refine.add_argument("--pool-size", type=int, default=48)
    refine.add_argument("--rounds", type=int, default=1)
    refine.add_argument("--workers", type=int, default=8)
    refine.set_defaults(func=_cmd_refine)

    freeze = sub.add_parser("freeze-confirmation", help="bind a development selection to the confirmation protocol")
    freeze.add_argument("--selection", required=True)
    freeze.add_argument("--out", required=True)
    freeze.set_defaults(func=_cmd_freeze_confirmation)

    confirm = sub.add_parser("confirm", help="evaluate one frozen policy pair on the reserved seed block")
    confirm.add_argument("--lock", required=True)
    confirm.add_argument("--workers", type=int, default=8)
    confirm.add_argument("--eager-replay", action="store_true", help="repeat the locked candidate with upstream camera rendering")
    confirm.set_defaults(func=_cmd_confirm)

    advanced = sub.add_parser("advanced", help="autonomous search and confirmation on the 2x5 advanced protocol")
    advanced.add_argument("--out", required=True)
    advanced.add_argument("--workers", type=int, default=8)
    advanced.add_argument("--seed-block", type=int, default=40000)
    advanced.add_argument("--calibration", type=int, default=640)
    advanced.add_argument("--initial-attempts", type=int, default=40)
    advanced.add_argument("--feedback-attempts", type=int, default=24)
    advanced.add_argument("--rounds", type=int, default=4)
    advanced.add_argument("--recover-from", type=Path, help="recover training artifacts from an interrupted run")
    advanced.set_defaults(func=_cmd_advanced)

    followup = sub.add_parser("advanced-followup", help="compare fine-grained trajectories on a fresh advanced split")
    followup.add_argument("--out", type=Path, required=True)
    followup.add_argument("--reference", type=Path, required=True)
    followup.add_argument("--workers", type=int, default=8)
    followup.add_argument("--seed-block", type=int, default=44000)
    followup.add_argument("--calibration", type=int, default=920)
    followup.add_argument("--initial-attempts", type=int, default=72)
    followup.add_argument("--feedback-attempts", type=int, default=32)
    followup.add_argument("--rounds", type=int, default=2)
    followup.set_defaults(func=_cmd_advanced_followup)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
