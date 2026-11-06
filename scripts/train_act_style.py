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

from a3opt.backends.act_style import ACTStyleConfig, train_act_style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an ACT-style (ResNet18 + transformer chunk) policy on A3 demonstrations")
    parser.add_argument("--dataset", default=str(ROOT / "artifacts" / "datasets" / "expert_2x5_multicam_60ep.npz"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "act_style" / "act_60ep"))
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--save-freq", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ACTStyleConfig(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        chunk_size=int(args.chunk_size),
        execution_horizon=int(args.execution_horizon),
        d_model=int(args.d_model),
        layers=int(args.layers),
        heads=int(args.heads),
        latent_dim=int(args.latent_dim),
        kl_weight=float(args.kl_weight),
        image_size=int(args.image_size),
        seed=int(args.seed),
        device=args.device,
        pretrained_backbone=bool(args.pretrained_backbone),
        save_freq=int(args.save_freq),
    )
    print(f"training act-style dataset={config.dataset_path} steps={config.steps} batch={config.batch_size} device={args.device}", flush=True)
    report = train_act_style(config)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
