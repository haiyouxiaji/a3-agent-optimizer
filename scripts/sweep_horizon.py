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

from a3opt.backends.bc2 import BC2Policy, TemporalEnsemblePolicy
from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def evaluate(checkpoint: Path, horizons: list[int], seeds: list[int], max_steps: int, device: str = "cpu", ensemble: bool = False, coefficient: float = 0.01) -> list[dict]:
    results: list[dict] = []
    for horizon in horizons:
        bench = make_cookie_benchmark(task="2x5", profile="basic", source_column="1", max_steps=int(max_steps))
        policy_factory = TemporalEnsemblePolicy if ensemble else BC2Policy
        policy = policy_factory(checkpoint, device=device, execution_horizon=int(horizon), coefficient=coefficient) if ensemble else policy_factory(checkpoint, device=device, execution_horizon=int(horizon))
        rows = []
        env = bench.create_env(render_cameras=True)
        try:
            for index, seed in enumerate(seeds):
                score = bench.run_episode(policy=policy, seed=int(seed), episode_idx=index + 1, env=env)
                rows.append({"seed": int(seed), "score": float(score.score), "success": bool(score.success), "steps": int(score.steps)})
        finally:
            env.close()
            policy.close()
        mean = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
        record = {"horizon": int(horizon), "max_steps": int(max_steps), "ensemble": bool(ensemble), "coefficient": float(coefficient), "mean_score": mean, "episodes": rows}
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep execution horizons for a BC2 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--horizons", default="4,8,12,20")
    parser.add_argument("--seeds", type=parse_seeds, default=[2000, 2001])
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--out", default="")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--coefficient", type=float, default=0.01)
    args = parser.parse_args()
    horizons = [int(part) for part in str(args.horizons).split(",") if part.strip()]
    results = evaluate(Path(args.checkpoint), horizons, args.seeds, int(args.max_steps), ensemble=bool(args.ensemble), coefficient=float(args.coefficient))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
