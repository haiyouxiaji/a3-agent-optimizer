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
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))

from a3opt.backends.bc2 import BC2Config, evaluate_checkpoint, train_candidate
from a3opt.contracts import CandidateSpec, ExperimentBudget, now_iso


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def build_candidate(name: str, chunk_size: int, lr: float, encoder: str = "tiny_cnn", fusion: str = "concat", image_size: int = 128) -> CandidateSpec:
    action_head = "chunk" if chunk_size > 1 else "mlp"
    image_size = image_size if encoder == "tiny_cnn" else 0
    return CandidateSpec(
        candidate_id=name,
        encoder=encoder,
        fusion=fusion,
        action_head=action_head,
        chunk_size=chunk_size,
        lr=lr,
        image_size=image_size,
        state_dim=16,
        seed=0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate one BC2 candidate on real A3 data")
    parser.add_argument("--dataset", default=str(ROOT / "artifacts" / "datasets" / "expert_2x5_multicam_20ep.npz"))
    parser.add_argument("--experiment-root", default=str(ROOT / "artifacts" / "local_sim_v2"))
    parser.add_argument("--name", default="bc2-baseline")
    parser.add_argument("--encoder", default="tiny_cnn", choices=["tiny_cnn", "state_mlp"])
    parser.add_argument("--fusion", default="concat", choices=["concat", "film"])
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-seeds", type=parse_seeds, default=[2000, 2001])
    parser.add_argument("--max-eval-steps", type=int, default=500)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidate = build_candidate(args.name, args.chunk_size, args.lr, args.encoder, args.fusion, args.image_size)
    config = BC2Config(
        dataset_path=Path(args.dataset),
        experiment_root=Path(args.experiment_root),
        execution_horizon=int(args.execution_horizon),
    )
    experiment_dir = Path(args.experiment_root) / candidate.candidate_id
    checkpoint_dir = experiment_dir / "checkpoint"
    budget = ExperimentBudget(max_train_steps=int(args.train_steps), batch_size=int(args.batch_size))
    print(f"[{now_iso()}] training {candidate.candidate_id} steps={args.train_steps}", flush=True)
    train = train_candidate(candidate, Path(args.dataset), budget, checkpoint_dir, config=config)
    print(json.dumps(train.to_dict(), indent=2, ensure_ascii=False)[:1200], flush=True)
    if train.status != "ok":
        return 1
    print(f"[{now_iso()}] evaluating seeds={args.eval_seeds} max_steps={args.max_eval_steps}", flush=True)
    evaluation = evaluate_checkpoint(
        candidate,
        checkpoint_dir,
        args.eval_seeds,
        int(args.max_eval_steps),
        experiment_dir / "eval",
        cfg=config,
    )
    print(json.dumps(evaluation.to_dict(), indent=2, ensure_ascii=False)[:1600], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
