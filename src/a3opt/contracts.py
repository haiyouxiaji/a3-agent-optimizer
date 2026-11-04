from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
STATUS_SKIPPED = "skipped"

ENCODERS = ("tiny_cnn", "state_mlp")
FUSIONS = ("concat", "film")
ACTION_HEADS = ("mlp", "chunk")
CHUNK_SIZES = (1, 20, 50)
IMAGE_SIZES = (0, 64)
LEARNING_RATES = (1e-3, 3e-3)
MAX_SCORE_2X5 = 10.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    encoder: str = "tiny_cnn"
    fusion: str = "concat"
    action_head: str = "chunk"
    chunk_size: int = 20
    lr: float = 1e-3
    image_size: int = 64
    state_dim: int = 16
    seed: int = 0
    family: str = "bc_policy"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "encoder": self.encoder,
            "fusion": self.fusion,
            "action_head": self.action_head,
            "chunk_size": self.chunk_size,
            "lr": self.lr,
            "image_size": self.image_size,
            "state_dim": self.state_dim,
            "seed": self.seed,
            "family": self.family,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateSpec":
        return cls(
            candidate_id=str(data["candidate_id"]),
            encoder=str(data.get("encoder", "tiny_cnn")),
            fusion=str(data.get("fusion", "concat")),
            action_head=str(data.get("action_head", "chunk")),
            chunk_size=int(data.get("chunk_size", 20)),
            lr=float(data.get("lr", 1e-3)),
            image_size=int(data.get("image_size", 64)),
            state_dim=int(data.get("state_dim", 16)),
            seed=int(data.get("seed", 0)),
            family=str(data.get("family", "bc_policy")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class ExperimentBudget:
    max_train_steps: int = 400
    batch_size: int = 64
    wall_seconds_limit: float = 900.0
    gpu_seconds_limit: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_train_steps": self.max_train_steps,
            "batch_size": self.batch_size,
            "wall_seconds_limit": self.wall_seconds_limit,
            "gpu_seconds_limit": self.gpu_seconds_limit,
        }


@dataclass
class RunBudget:
    max_experiments: int = 8
    max_train_steps_total: int = 4800
    max_wall_seconds: float = 10800.0
    max_gpu_seconds: float = 0.0
    spent_experiments: int = 0
    spent_train_steps: int = 0
    spent_wall_seconds: float = 0.0
    spent_gpu_seconds: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def can_afford(self, experiment: ExperimentBudget) -> bool:
        if self.spent_experiments >= self.max_experiments:
            return False
        if self.spent_train_steps + experiment.max_train_steps > self.max_train_steps_total:
            return False
        if self.max_wall_seconds > 0 and self.spent_wall_seconds >= self.max_wall_seconds:
            return False
        if self.max_gpu_seconds > 0 and self.spent_gpu_seconds >= self.max_gpu_seconds:
            return False
        return True

    def record(self, outcome: "ExperimentOutcome") -> None:
        self.spent_experiments += 1
        self.spent_train_steps += outcome.train.train_steps
        self.spent_wall_seconds += outcome.cost_wall_seconds
        self.spent_gpu_seconds += outcome.cost_gpu_seconds
        self.history.append(
            {
                "candidate_id": outcome.candidate.candidate_id,
                "status": outcome.status,
                "train_steps": outcome.train.train_steps,
                "wall_seconds": round(outcome.cost_wall_seconds, 3),
                "gpu_seconds": round(outcome.cost_gpu_seconds, 3),
                "mean_score": round(outcome.eval.mean_score, 3),
            }
        )

    def summary(self) -> dict[str, Any]:
        return {
            "max_experiments": self.max_experiments,
            "max_train_steps_total": self.max_train_steps_total,
            "max_wall_seconds": self.max_wall_seconds,
            "max_gpu_seconds": self.max_gpu_seconds,
            "spent_experiments": self.spent_experiments,
            "spent_train_steps": self.spent_train_steps,
            "spent_wall_seconds": round(self.spent_wall_seconds, 3),
            "spent_gpu_seconds": round(self.spent_gpu_seconds, 3),
            "exhausted": not self.can_afford(ExperimentBudget()),
        }


@dataclass
class TrainResult:
    status: str = STATUS_OK
    train_steps: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    artifact: str = ""
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "train_steps": self.train_steps,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "artifact": self.artifact,
            "wall_seconds": round(self.wall_seconds, 3),
            "gpu_seconds": round(self.gpu_seconds, 3),
            "error": self.error,
        }


@dataclass
class EpisodeRow:
    seed: int
    score: float
    success: bool
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "score": float(self.score),
            "success": bool(self.success),
            "steps": int(self.steps),
        }


@dataclass
class EvalResult:
    status: str = STATUS_OK
    episodes: list[EpisodeRow] = field(default_factory=list)
    max_score: float = MAX_SCORE_2X5
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    error: str = ""

    @property
    def mean_score(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(row.score for row in self.episodes) / len(self.episodes)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for row in self.episodes if row.success) / len(self.episodes)

    def scores_by_seed(self) -> dict[int, float]:
        return {row.seed: float(row.score) for row in self.episodes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "max_score": self.max_score,
            "mean_score": round(self.mean_score, 4),
            "success_rate": round(self.success_rate, 4),
            "wall_seconds": round(self.wall_seconds, 3),
            "gpu_seconds": round(self.gpu_seconds, 3),
            "error": self.error,
            "episodes": [row.to_dict() for row in self.episodes],
        }


@dataclass
class ExperimentOutcome:
    candidate: CandidateSpec
    train: TrainResult
    eval: EvalResult
    status: str = STATUS_OK
    error: str = ""

    @property
    def cost_wall_seconds(self) -> float:
        return float(self.train.wall_seconds + self.eval.wall_seconds)

    @property
    def cost_gpu_seconds(self) -> float:
        return float(self.train.gpu_seconds + self.eval.gpu_seconds)

    def normalized_score(self) -> float:
        if self.status != STATUS_OK or not self.eval.episodes:
            return 0.0
        return min(1.0, self.eval.mean_score / max(self.eval.max_score, 1.0))

    def scores_by_seed(self) -> dict[int, float]:
        return self.eval.scores_by_seed()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "status": self.status,
            "error": self.error,
            "train": self.train.to_dict(),
            "eval": self.eval.to_dict(),
            "cost_wall_seconds": round(self.cost_wall_seconds, 3),
            "cost_gpu_seconds": round(self.cost_gpu_seconds, 3),
            "normalized_score": round(self.normalized_score(), 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentOutcome":
        train = TrainResult(**{**data.get("train", {}), "metrics": data.get("train", {}).get("metrics", {})})
        eval_data = data.get("eval", {})
        episodes = [EpisodeRow(**row) for row in eval_data.get("episodes", [])]
        eval_result = EvalResult(
            status=eval_data.get("status", STATUS_OK),
            episodes=episodes,
            max_score=float(eval_data.get("max_score", MAX_SCORE_2X5)),
            wall_seconds=float(eval_data.get("wall_seconds", 0.0)),
            gpu_seconds=float(eval_data.get("gpu_seconds", 0.0)),
            error=eval_data.get("error", ""),
        )
        return cls(
            candidate=CandidateSpec.from_dict(data["candidate"]),
            train=train,
            eval=eval_result,
            status=data.get("status", STATUS_OK),
            error=data.get("error", ""),
        )


@runtime_checkable
class ExperimentBackend(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...

    def run(
        self,
        candidate: CandidateSpec,
        train_budget: ExperimentBudget,
        eval_seeds: Sequence[int],
        max_eval_steps: int,
    ) -> ExperimentOutcome: ...


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path

    @property
    def outcomes_dir(self) -> Path:
        return self.run_dir / "outcomes"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def memory_path(self) -> Path:
        return self.run_dir / "memory.jsonl"

    @property
    def tree_path(self) -> Path:
        return self.run_dir / "tree.json"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.md"

    def ensure(self) -> "RunLayout":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class OptimizationSummary:
    run_id: str
    backend: str
    space_name: str
    root_candidate_id: str
    best_candidate_id: str = ""
    best_mean_score: float = 0.0
    experiments: int = 0
    accepted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "backend": self.backend,
            "space_name": self.space_name,
            "root_candidate_id": self.root_candidate_id,
            "best_candidate_id": self.best_candidate_id,
            "best_mean_score": self.best_mean_score,
            "experiments": self.experiments,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "budget": self.budget,
            "created_at": self.created_at,
            "config": self.config,
        }
