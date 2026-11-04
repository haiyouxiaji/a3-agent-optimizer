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

from a3opt.backends.bc2 import EnsemblePolicy
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an action-averaged ensemble of BC2 policies")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--execution-horizon", type=int, default=12)
    parser.add_argument("--seeds", type=parse_seeds, default=[2000, 2001, 2002, 2003, 2004, 2005])
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    bench = make_cookie_benchmark(task="2x5", profile="basic", source_column="1", max_steps=int(args.max_steps))
    policy = EnsemblePolicy(args.checkpoints, device="cpu", execution_horizon=int(args.execution_horizon))
    rows = []
    env = bench.create_env(render_cameras=True)
    try:
        for index, seed in enumerate(args.seeds):
            score = bench.run_episode(policy=policy, seed=int(seed), episode_idx=index + 1, env=env)
            rows.append({"seed": int(seed), "score": float(score.score), "success": bool(score.success), "steps": int(score.steps)})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    finally:
        env.close()
        policy.close()
    mean = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
    payload = {
        "checkpoints": [str(Path(item)) for item in args.checkpoints],
        "execution_horizon": int(args.execution_horizon),
        "max_steps": int(args.max_steps),
        "seeds": [int(seed) for seed in args.seeds],
        "mean_score": mean,
        "episodes": rows,
    }
    print(json.dumps({"mean_score": mean, "episodes": rows}, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
