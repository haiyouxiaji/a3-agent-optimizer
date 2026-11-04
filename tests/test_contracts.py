from __future__ import annotations

import json

import pytest

from a3opt.contracts import (
    CandidateSpec,
    EpisodeRow,
    EvalResult,
    ExperimentBudget,
    ExperimentOutcome,
    RunBudget,
    RunLayout,
    TrainResult,
)


def make_outcome(candidate_id: str = "c1", scores: tuple[float, ...] = (4.0, 6.0), steps: int = 100) -> ExperimentOutcome:
    candidate = CandidateSpec(candidate_id=candidate_id)
    episodes = [EpisodeRow(seed=2000 + i, score=score, success=False, steps=steps) for i, score in enumerate(scores)]
    return ExperimentOutcome(
        candidate=candidate,
        train=TrainResult(train_steps=steps, metrics={"final_loss": 0.5}, wall_seconds=1.5),
        eval=EvalResult(episodes=episodes, wall_seconds=2.5),
    )


def test_candidate_roundtrip() -> None:
    candidate = CandidateSpec(
        candidate_id="tiny_cnn-concat-chunk-k20-lr0.001",
        encoder="tiny_cnn",
        fusion="film",
        action_head="chunk",
        chunk_size=20,
        lr=1e-3,
        seed=7,
    )
    restored = CandidateSpec.from_dict(candidate.to_dict())
    assert restored == candidate


def test_eval_aggregates() -> None:
    result = EvalResult(
        episodes=[
            EpisodeRow(seed=1, score=10.0, success=True, steps=100),
            EpisodeRow(seed=2, score=0.0, success=False, steps=120),
        ]
    )
    assert result.mean_score == 5.0
    assert result.success_rate == 0.5
    assert result.scores_by_seed() == {1: 10.0, 2: 0.0}


def test_outcome_roundtrip_and_cost() -> None:
    outcome = make_outcome()
    payload = outcome.to_dict()
    restored = ExperimentOutcome.from_dict(payload)
    assert restored.candidate == outcome.candidate
    assert restored.eval.scores_by_seed() == outcome.eval.scores_by_seed()
    assert outcome.cost_wall_seconds == pytest.approx(4.0)
    assert outcome.normalized_score() == pytest.approx(0.5)


def test_run_budget_blocks_overspend() -> None:
    budget = RunBudget(max_experiments=1, max_train_steps_total=500)
    experiment = ExperimentBudget(max_train_steps=400)
    assert budget.can_afford(experiment)
    budget.record(make_outcome(steps=400))
    assert not budget.can_afford(experiment)
    summary = budget.summary()
    assert summary["spent_experiments"] == 1
    assert summary["spent_train_steps"] == 400
    assert summary["exhausted"] is True


def test_run_layout_creates_directories(tmp_path) -> None:
    layout = RunLayout(run_dir=tmp_path / "run").ensure()
    assert layout.outcomes_dir.is_dir()
    assert layout.logs_dir.is_dir()
    layout.memory_path.write_text("", encoding="utf-8")
    assert json.loads(json.dumps({"report": str(layout.report_path)}))["report"].endswith("report.md")
