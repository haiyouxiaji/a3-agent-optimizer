from __future__ import annotations

import json

import pytest

from a3opt.contracts import (
    CandidateSpec,
    EpisodeRow,
    EvalResult,
    ExperimentOutcome,
    TrainResult,
)
from a3opt.memory import ExperimentMemory, MemoryRecord


def make_outcome(
    candidate_id: str,
    scores: list[float],
    successes: list[bool] | None = None,
    status: str = "ok",
    train_steps: int = 40,
    wall_seconds: float = 10.0,
) -> ExperimentOutcome:
    outcome_successes = successes if successes is not None else [True] * len(scores)
    candidate = CandidateSpec(candidate_id=candidate_id)
    train = TrainResult(
        status=status,
        train_steps=train_steps,
        metrics={"final_loss": 0.25},
        artifact=f"artifacts/{candidate_id}",
        wall_seconds=wall_seconds,
    )
    episodes = [
        EpisodeRow(
            seed=1000 + index,
            score=float(score),
            success=bool(outcome_successes[index]),
            steps=25,
        )
        for index, score in enumerate(scores)
    ]
    evaluation = EvalResult(
        status=status,
        episodes=episodes,
        max_score=10.0,
        wall_seconds=1.5,
    )
    return ExperimentOutcome(
        candidate=candidate,
        train=train,
        eval=evaluation,
        status=status,
    )


def test_record_from_outcome_maps_contract_fields():
    outcome = make_outcome(
        "c1",
        [4.0, 6.0],
        successes=[True, False],
        train_steps=75,
        wall_seconds=12.5,
    )
    memory = ExperimentMemory()
    record = memory.record_from_outcome(outcome, "n001")
    assert record.node_id == "n001"
    assert record.candidate_id == "c1"
    assert record.status == "ok"
    assert record.mean_score == pytest.approx(5.0)
    assert record.success_rate == pytest.approx(0.5)
    assert record.normalized_score == pytest.approx(0.5)
    assert record.train_steps == 75
    assert record.wall_seconds == pytest.approx(14.0)
    assert record.gpu_seconds == 0.0
    assert record.artifact == "artifacts/c1"
    assert record.candidate["candidate_id"] == "c1"
    assert len(record.episodes) == 2
    assert len(memory) == 1
    assert memory.records() == [record]


def test_jsonl_append_and_reload(tmp_path):
    path = tmp_path / "memory.jsonl"
    memory = ExperimentMemory(path)
    memory.record_from_outcome(make_outcome("c1", [3.0, 4.0]), "n1")
    memory.record_from_outcome(make_outcome("c2", [5.0, 6.0]), "n2")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["node_id"] == "n1"
    assert parsed["candidate_id"] == "c1"
    reloaded = ExperimentMemory(path)
    records = reloaded.records()
    assert [record.node_id for record in records] == ["n1", "n2"]
    assert records[1].candidate_id == "c2"
    assert records[1].mean_score == pytest.approx(5.5)


def test_records_reads_external_appends(tmp_path):
    path = tmp_path / "memory.jsonl"
    memory = ExperimentMemory(path)
    memory.record_from_outcome(make_outcome("c1", [3.0]), "n1")
    extra = MemoryRecord(
        node_id="n9",
        candidate_id="c9",
        status="ok",
        mean_score=9.0,
        success_rate=1.0,
        normalized_score=0.9,
        train_steps=5,
        wall_seconds=1.0,
        gpu_seconds=0.0,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(extra.to_dict()) + "\n")
    assert "n9" in [record.node_id for record in memory.records()]


def test_best_returns_top_k_normalized_records(tmp_path):
    memory = ExperimentMemory(tmp_path / "memory.jsonl")
    memory.record_from_outcome(make_outcome("c1", [5.0, 5.0]), "n1")
    memory.record_from_outcome(make_outcome("c2", [1.0]), "n2")
    memory.record_from_outcome(make_outcome("c3", [8.0]), "n3")
    assert [record.node_id for record in memory.best(2)] == ["n3", "n1"]
    assert memory.best(1)[0].candidate_id == "c3"
    assert memory.best(0) == []
    assert len(memory.best(10)) == 3


def test_by_candidate_and_summary(tmp_path):
    memory = ExperimentMemory(tmp_path / "memory.jsonl")
    memory.record_from_outcome(make_outcome("c1", [2.0]), "n1")
    memory.record_from_outcome(make_outcome("c1", [4.0]), "n2")
    memory.record_from_outcome(make_outcome("c2", [6.0], status="failed"), "n3")
    assert [record.node_id for record in memory.by_candidate("c1")] == ["n1", "n2"]
    assert memory.by_candidate("missing") == []
    summary = memory.summary()
    assert summary["count"] == 3
    assert summary["records"] == 3
    assert summary["ok"] == 2
    assert summary["failed"] == 1
    assert summary["best_candidate_id"] == "c1"
    assert summary["best_node_id"] == "n2"
    assert summary["best_mean_score"] == pytest.approx(4.0)
    assert summary["mean_score"] == pytest.approx(3.0)
    assert summary["total_train_steps"] == 120


def test_condense_markdown_contains_records_and_best(tmp_path):
    memory = ExperimentMemory(tmp_path / "memory.jsonl")
    memory.record_from_outcome(make_outcome("c1", [7.0]), "n1")
    memory.record_from_outcome(make_outcome("c2", [2.0]), "n2")
    text = memory.condense()
    assert text.startswith("# Experiment memory")
    assert "| node | candidate |" in text
    assert "n1" in text
    assert "c1" in text
    assert "- best candidate: c1" in text
    assert "- records: 2" in text


def test_empty_memory_condense():
    text = ExperimentMemory().condense()
    assert "# Experiment memory" in text
    assert "- records: 0" in text
    assert "best candidate: none" in text


def test_memory_record_dict_roundtrip():
    record = MemoryRecord(
        node_id="n1",
        candidate_id="c1",
        status="ok",
        mean_score=1.0,
        success_rate=0.5,
        normalized_score=0.1,
        train_steps=4,
        wall_seconds=2.0,
        gpu_seconds=0.0,
        artifact="a",
        candidate={"candidate_id": "c1"},
        episodes=[{"seed": 1, "score": 1.0, "success": True, "steps": 2}],
    )
    restored = MemoryRecord.from_dict(record.to_dict())
    assert restored == record


def test_failed_outcome_records_zero_normalized_score(tmp_path):
    memory = ExperimentMemory(tmp_path / "memory.jsonl")
    record = memory.record_from_outcome(
        make_outcome("c1", [9.0], status="failed"), "n1"
    )
    assert record.status == "failed"
    assert record.normalized_score == 0.0
