from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from a3opt.contracts import STATUS_OK, ExperimentOutcome, now_iso


@dataclass
class MemoryRecord:
    node_id: str
    candidate_id: str
    status: str
    mean_score: float
    success_rate: float
    normalized_score: float
    train_steps: int
    wall_seconds: float
    gpu_seconds: float
    artifact: str = ""
    error: str = ""
    candidate: dict[str, Any] = field(default_factory=dict)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "candidate_id": self.candidate_id,
            "candidate": dict(self.candidate),
            "status": self.status,
            "mean_score": float(self.mean_score),
            "success_rate": float(self.success_rate),
            "normalized_score": float(self.normalized_score),
            "train_steps": int(self.train_steps),
            "wall_seconds": float(self.wall_seconds),
            "gpu_seconds": float(self.gpu_seconds),
            "artifact": self.artifact,
            "error": self.error,
            "episodes": [dict(row) for row in self.episodes],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        return cls(
            node_id=str(data.get("node_id", "")),
            candidate_id=str(data.get("candidate_id", "")),
            status=str(data.get("status", "")),
            mean_score=float(data.get("mean_score", 0.0)),
            success_rate=float(data.get("success_rate", 0.0)),
            normalized_score=float(data.get("normalized_score", 0.0)),
            train_steps=int(data.get("train_steps", 0)),
            wall_seconds=float(data.get("wall_seconds", 0.0)),
            gpu_seconds=float(data.get("gpu_seconds", 0.0)),
            artifact=str(data.get("artifact", "")),
            error=str(data.get("error", "")),
            candidate=dict(data.get("candidate", {})),
            episodes=[dict(row) for row in data.get("episodes", [])],
            created_at=str(data.get("created_at", now_iso())),
        )


class ExperimentMemory:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._records: list[MemoryRecord] = []
        if self.path is not None and self.path.exists():
            self.reload()

    def __len__(self) -> int:
        return len(self.records())

    def reload(self) -> list[MemoryRecord]:
        self._records = []
        if self.path is not None and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    self._records.append(MemoryRecord.from_dict(json.loads(line)))
        return list(self._records)

    def append(self, record: MemoryRecord) -> MemoryRecord:
        self._records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def record_from_outcome(self, outcome: ExperimentOutcome, node_id: str) -> MemoryRecord:
        record = MemoryRecord(
            node_id=str(node_id),
            candidate_id=outcome.candidate.candidate_id,
            status=outcome.status,
            mean_score=float(outcome.eval.mean_score),
            success_rate=float(outcome.eval.success_rate),
            normalized_score=float(outcome.normalized_score()),
            train_steps=int(outcome.train.train_steps),
            wall_seconds=float(outcome.cost_wall_seconds),
            gpu_seconds=float(outcome.cost_gpu_seconds),
            artifact=outcome.train.artifact,
            error=outcome.error or outcome.train.error or outcome.eval.error,
            candidate=outcome.candidate.to_dict(),
            episodes=[row.to_dict() for row in outcome.eval.episodes],
        )
        return self.append(record)

    def records(self) -> list[MemoryRecord]:
        if self.path is not None:
            self.reload()
        return list(self._records)

    @staticmethod
    def _rank_key(record: MemoryRecord) -> tuple[float, float, str]:
        return (
            float(record.normalized_score),
            float(record.mean_score),
            record.created_at,
        )

    def best(self, k: int = 1) -> list[MemoryRecord]:
        if int(k) <= 0:
            return []
        ordered = sorted(self.records(), key=self._rank_key, reverse=True)
        return ordered[: int(k)]

    def by_candidate(self, candidate_id: str) -> list[MemoryRecord]:
        return [
            record
            for record in self.records()
            if record.candidate_id == str(candidate_id)
        ]

    def summary(self) -> dict[str, Any]:
        records = self.records()
        ok_records = [record for record in records if record.status == STATUS_OK]
        best_records = sorted(records, key=self._rank_key, reverse=True)[:1]
        best = best_records[0] if best_records else None
        mean_score = (
            sum(record.mean_score for record in ok_records) / len(ok_records)
            if ok_records
            else 0.0
        )
        return {
            "count": len(records),
            "records": len(records),
            "ok": len(ok_records),
            "failed": sum(1 for record in records if record.status != STATUS_OK),
            "best_candidate_id": best.candidate_id if best is not None else "",
            "best_node_id": best.node_id if best is not None else "",
            "best_mean_score": float(best.mean_score) if best is not None else 0.0,
            "mean_score": round(float(mean_score), 4),
            "total_train_steps": sum(int(record.train_steps) for record in records),
            "total_wall_seconds": round(sum(float(record.wall_seconds) for record in records), 3),
            "total_gpu_seconds": round(sum(float(record.gpu_seconds) for record in records), 3),
        }

    def condense(self) -> str:
        summary = self.summary()
        records = self.records()
        best_id = summary["best_candidate_id"] or "none"
        lines = [
            "# Experiment memory",
            "",
            f"- records: {summary['count']}",
            f"- best candidate: {best_id}",
            f"- best mean score: {summary['best_mean_score']:.4f}",
            "",
            "| node | candidate | status | mean score | success | normalized | steps | wall s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for record in records:
            lines.append(
                f"| {record.node_id} | {record.candidate_id} | {record.status} "
                f"| {record.mean_score:.4f} | {record.success_rate:.4f} "
                f"| {record.normalized_score:.4f} | {record.train_steps} "
                f"| {record.wall_seconds:.3f} |"
            )
        return "\n".join(lines) + "\n"
