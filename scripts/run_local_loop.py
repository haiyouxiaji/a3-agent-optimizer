from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SIM_SRC = ROOT.parent / "a3_dual_arm_sim" / "src"
if SIM_SRC.is_dir() and str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))

from a3opt.acceptance import AcceptanceCriteria
from a3opt.backends.local_sim import DEFAULT_DATASET_PATH, LocalSimBackend
from a3opt.contracts import ExperimentBudget, RunBudget
from a3opt.loop import LoopConfig, run_optimization


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one evaluation seed is required")
    return tuple(seeds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the a3opt agent loop against the real local simulation backend"
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--experiment-root", default=str(ROOT / "artifacts" / "local_sim_loop"))
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"))
    parser.add_argument("--run-id", default="local-sim-loop")
    parser.add_argument("--experiments", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-seeds", type=parse_seeds, default=(2000, 2001))
    parser.add_argument("--max-eval-steps", type=int, default=120)
    parser.add_argument("--exploration", type=float, default=0.7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    backend = LocalSimBackend(
        {
            "dataset_path": str(args.dataset),
            "experiment_root": str(args.experiment_root),
            "device": "cpu",
        }
    )
    available, reason = backend.available()
    print(f"backend available={available} reason={reason}", flush=True)
    if not available:
        return 2
    experiment_budget = ExperimentBudget(
        max_train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        wall_seconds_limit=1800.0,
    )
    config = LoopConfig(
        run_dir=Path(args.runs_dir) / str(args.run_id),
        backend=backend,
        experiments=int(args.experiments),
        eval_seeds=tuple(args.eval_seeds),
        max_eval_steps=int(args.max_eval_steps),
        experiment_budget=experiment_budget,
        run_budget=RunBudget(
            max_experiments=int(args.experiments),
            max_train_steps_total=int(args.train_steps) * int(args.experiments),
            max_wall_seconds=10800.0,
        ),
        criteria=AcceptanceCriteria(),
        exploration_constant=float(args.exploration),
        run_id=str(args.run_id),
    )
    result = run_optimization(config)
    summary = result["summary"]
    print(json.dumps({key: summary[key] for key in ("run_id", "experiments", "best_candidate_id", "best_mean_score", "accepted", "rejected")}, indent=2))
    print(f"report: {result['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
