from __future__ import annotations

import hashlib
import time
from typing import Sequence

from a3opt.contracts import (
    MAX_SCORE_2X5,
    STATUS_OK,
    CandidateSpec,
    EpisodeRow,
    EvalResult,
    ExperimentBudget,
    ExperimentOutcome,
    TrainResult,
)


def _unit(*parts: object) -> float:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _quality(candidate: CandidateSpec) -> float:
    score = 1.0
    if candidate.encoder == "tiny_cnn":
        score += 3.0
    if candidate.fusion == "film" and candidate.encoder == "tiny_cnn":
        score += 1.0
    if candidate.action_head == "chunk" and candidate.chunk_size >= 20:
        score += 2.0
    if candidate.chunk_size >= 50:
        score += 1.0
    if candidate.lr >= 3e-3:
        score += 1.0
    return score


class StubBackend:
    name = "stub"

    def __init__(self, latency: float = 0.0, seed_noise: float = 1.0, max_score: float = MAX_SCORE_2X5):
        self.latency = float(latency)
        self.seed_noise = float(seed_noise)
        self.max_score = float(max_score)

    def available(self) -> tuple[bool, str]:
        return True, "synthetic backend; scores are deterministic functions of the candidate"

    def run(
        self,
        candidate: CandidateSpec,
        train_budget: ExperimentBudget,
        eval_seeds: Sequence[int],
        max_eval_steps: int,
    ) -> ExperimentOutcome:
        train_start = time.perf_counter()
        if self.latency:
            time.sleep(self.latency)
        quality = _quality(candidate)
        final_loss = max(0.01, 1.2 - quality * 0.12)
        train = TrainResult(
            status=STATUS_OK,
            train_steps=int(train_budget.max_train_steps),
            metrics={
                "first_loss": round(final_loss * (1.6 + 0.3 * _unit(candidate.candidate_id, "first")), 4),
                "final_loss": round(final_loss, 4),
            },
            artifact=f"stub://{candidate.candidate_id}",
            wall_seconds=time.perf_counter() - train_start,
        )
        eval_start = time.perf_counter()
        if self.latency:
            time.sleep(self.latency)
        rows: list[EpisodeRow] = []
        for seed in eval_seeds:
            value = quality + (2.0 * _unit(candidate.candidate_id, seed) - 1.0) * self.seed_noise
            value = max(0.0, min(self.max_score, value))
            score = int(round(value))
            rows.append(
                EpisodeRow(
                    seed=int(seed),
                    score=score,
                    success=score >= int(self.max_score),
                    steps=int(min(max_eval_steps, 40 + 5 * score)),
                )
            )
        evaluation = EvalResult(
            status=STATUS_OK,
            episodes=rows,
            max_score=self.max_score,
            wall_seconds=time.perf_counter() - eval_start,
        )
        return ExperimentOutcome(candidate=candidate, train=train, eval=evaluation, status=STATUS_OK)
