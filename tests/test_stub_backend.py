from __future__ import annotations

from a3opt.backends.stub import StubBackend
from a3opt.contracts import CandidateSpec, ExperimentBudget


def candidate(candidate_id: str, **kwargs) -> CandidateSpec:
    return CandidateSpec(candidate_id=candidate_id, **kwargs)


def test_stub_is_deterministic() -> None:
    backend = StubBackend()
    budget = ExperimentBudget(max_train_steps=100)
    first = backend.run(candidate("c1", chunk_size=20), budget, (2000, 2001), 120)
    second = backend.run(candidate("c1", chunk_size=20), budget, (2000, 2001), 120)
    assert first.eval.scores_by_seed() == second.eval.scores_by_seed()
    assert first.train.metrics == second.train.metrics


def test_stub_scores_within_range_and_step_budget() -> None:
    backend = StubBackend()
    outcome = backend.run(candidate("c1"), ExperimentBudget(max_train_steps=50), (2000, 2001, 2002), 120)
    scores = list(outcome.eval.scores_by_seed().values())
    assert len(scores) == 3
    assert all(0 <= score <= 10 for score in scores)
    assert all(0 < row.steps <= 120 for row in outcome.eval.episodes)
    assert outcome.train.train_steps == 50
    assert outcome.status == "ok"


def test_stub_orders_quality() -> None:
    backend = StubBackend(seed_noise=0.0)
    budget = ExperimentBudget(max_train_steps=10)
    strong = backend.run(candidate("strong", encoder="tiny_cnn", fusion="film", chunk_size=50, lr=3e-3), budget, (2000,), 120)
    weak = backend.run(candidate("weak", encoder="state_mlp", fusion="concat", action_head="mlp", chunk_size=1, lr=1e-3), budget, (2000,), 120)
    assert strong.eval.mean_score > weak.eval.mean_score


def test_available_reports_capability() -> None:
    ok, reason = StubBackend().available()
    assert ok is True
    assert reason
