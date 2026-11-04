from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from a3opt.contracts import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_TIMEOUT,
    CandidateSpec,
    EpisodeRow,
    EvalResult,
    ExperimentBudget,
    ExperimentOutcome,
    TrainResult,
)


@dataclass
class A3SimCliConfig:
    sim_root: Path = Path("../a3_dual_arm_sim")
    dataset_root: Path = Path("datasets/a3_front_close_left_100")
    output_root: Path = Path("artifacts/a3_sim_cli")
    python: str = sys.executable
    train_cli: str = "a3-sim"
    benchmark_module: str = "a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli"
    device: str = "cpu"
    dry_run: bool = True
    timeout_seconds: float = 7200.0
    eval_max_steps: int = 200
    task: str = "2x5"
    extra: dict[str, Any] = field(default_factory=dict)


def build_train_command(config: A3SimCliConfig, candidate: CandidateSpec, budget: ExperimentBudget, output: Path) -> list[str]:
    chunk = max(1, int(candidate.chunk_size))
    command = [
        config.train_cli,
        "train-act",
        "--root",
        str(config.dataset_root),
        "--output",
        str(output),
        "--steps",
        str(int(budget.max_train_steps)),
        "--batch-size",
        str(int(budget.batch_size)),
        "--chunk-size",
        str(chunk),
        "--n-action-steps",
        str(min(chunk, 50)),
        "--lr",
        str(candidate.lr),
        "--seed",
        str(candidate.seed),
        "--num-workers",
        "0",
        "--device",
        config.device,
    ]
    if candidate.image_size <= 0:
        command.append("--no-pretrained-backbone")
    return command


def build_eval_command(config: A3SimCliConfig, checkpoint: Path, seeds: Sequence[int], max_steps: int) -> list[str]:
    return [
        config.python,
        "-m",
        config.benchmark_module,
        "--policy",
        f"act:{checkpoint}",
        "--dataset-root",
        str(config.dataset_root),
        "--task",
        config.task,
        "--episodes",
        str(len(seeds)),
        "--seed-start",
        str(int(seeds[0])),
        "--max-steps",
        str(int(max_steps)),
        "--workers",
        "1",
        "--device",
        config.device,
    ]


class A3SimCliBackend:
    name = "a3_sim_cli"

    def __init__(self, config: A3SimCliConfig | None = None):
        self.config = config or A3SimCliConfig()

    def available(self) -> tuple[bool, str]:
        root = Path(self.config.sim_root).expanduser()
        if not root.is_dir():
            return False, f"a3 simulation repository not found: {root}"
        if self.config.dry_run:
            return True, "dry-run mode; commands are written but not executed"
        return True, "a3 simulation CLI backend; requires lerobot and the upstream repository"

    def run(
        self,
        candidate: CandidateSpec,
        train_budget: ExperimentBudget,
        eval_seeds: Sequence[int],
        max_eval_steps: int,
    ) -> ExperimentOutcome:
        root = Path(self.config.sim_root).expanduser()
        out_dir = Path(self.config.output_root) / candidate.candidate_id
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = out_dir / "checkpoint"
        train_command = build_train_command(self.config, candidate, train_budget, checkpoint)
        eval_command = build_eval_command(self.config, checkpoint, eval_seeds, max_eval_steps)
        commands = {"train": train_command, "evaluate": eval_command}
        (out_dir / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
        if self.config.dry_run:
            train = TrainResult(
                status=STATUS_OK,
                train_steps=int(train_budget.max_train_steps),
                metrics={"planned_steps": float(train_budget.max_train_steps)},
                artifact=str(checkpoint),
                error="dry_run",
            )
            evaluation = EvalResult(status=STATUS_OK, episodes=[], max_score=10.0, error="dry_run")
            return ExperimentOutcome(candidate=candidate, train=train, eval=evaluation, status=STATUS_OK)
        try:
            train_run = subprocess.run(
                train_command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return self._failed(candidate, f"training timed out: {exc}", out_dir)
        (out_dir / "train.log").write_text((train_run.stdout or "") + (train_run.stderr or ""), encoding="utf-8")
        if train_run.returncode != 0:
            return self._failed(candidate, f"training exited with {train_run.returncode}", out_dir)
        try:
            eval_run = subprocess.run(
                eval_command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return self._failed(candidate, f"evaluation timed out: {exc}", out_dir)
        (out_dir / "eval.log").write_text((eval_run.stdout or "") + (eval_run.stderr or ""), encoding="utf-8")
        rows = parse_eval_episodes(eval_run.stdout or "")
        evaluation = EvalResult(status=STATUS_OK if rows else STATUS_FAILED, episodes=rows, max_score=10.0)
        train = TrainResult(
            status=STATUS_OK,
            train_steps=int(train_budget.max_train_steps),
            metrics={},
            artifact=str(checkpoint),
        )
        return ExperimentOutcome(candidate=candidate, train=train, eval=evaluation, status=STATUS_OK if rows else STATUS_FAILED)

    def _failed(self, candidate: CandidateSpec, error: str, out_dir: Path) -> ExperimentOutcome:
        (out_dir / "failure.txt").write_text(error, encoding="utf-8")
        return ExperimentOutcome(
            candidate=candidate,
            train=TrainResult(status=STATUS_FAILED, error=error),
            eval=EvalResult(status=STATUS_FAILED, error=error),
            status=STATUS_FAILED,
            error=error,
        )


def parse_eval_episodes(output: str) -> list[EpisodeRow]:
    rows: list[EpisodeRow] = []
    for line in output.splitlines():
        if "Seed" not in line or "Score:" not in line:
            continue
        try:
            seed = int(line.split("Seed", 1)[1].split("|")[0].strip())
            score = float(line.split("Score:", 1)[1].split("/")[0].strip())
            steps = int(line.split("Steps:", 1)[1].split("|")[0].strip())
            success = "SUCCESS" in line
        except (IndexError, ValueError):
            continue
        rows.append(EpisodeRow(seed=seed, score=score, success=success, steps=steps))
    return rows


TIMEOUT_MARKER = STATUS_TIMEOUT
