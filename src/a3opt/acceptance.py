from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

VERDICT_ACCEPT = "accept"
VERDICT_REJECT = "reject"
VERDICT_INSUFFICIENT = "insufficient"

NON_WORSE_TOLERANCE = 1e-9
THRESHOLD_TOLERANCE = 1e-9


@dataclass(frozen=True)
class AcceptanceCriteria:
    min_mean_gain: float = 1.0
    min_non_worse_fraction: float = 0.7
    max_cost_ratio: float = 3.0


@dataclass
class AcceptanceReport:
    verdict: str
    mean_gain: float = 0.0
    non_worse_fraction: float = 0.0
    cost_ratio: float = 0.0
    paired_seeds: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.verdict == VERDICT_ACCEPT

    def to_dict(self) -> dict[str, Any]:
        cost_ratio = float(self.cost_ratio)
        return {
            "verdict": self.verdict,
            "mean_gain": round(float(self.mean_gain), 6),
            "non_worse_fraction": round(float(self.non_worse_fraction), 6),
            "cost_ratio": round(cost_ratio, 6) if math.isfinite(cost_ratio) else float("inf"),
            "paired_seeds": [int(seed) for seed in self.paired_seeds],
            "reasons": list(self.reasons),
        }


def _as_score_map(scores: Mapping[int, float] | None) -> dict[int, float]:
    if not scores:
        return {}
    return {int(seed): float(score) for seed, score in dict(scores).items()}


def _cost_ratio(baseline_cost: float, candidate_cost: float) -> float:
    baseline = float(baseline_cost)
    candidate = float(candidate_cost)
    if baseline > 0.0:
        return candidate / baseline
    if candidate > 0.0:
        return float("inf")
    return 1.0


def compare_paired(
    baseline_scores: Mapping[int, float] | None,
    candidate_scores: Mapping[int, float] | None,
    baseline_cost: float = 0.0,
    candidate_cost: float = 0.0,
    criteria: AcceptanceCriteria | None = None,
) -> AcceptanceReport:
    effective = criteria if criteria is not None else AcceptanceCriteria()
    baseline = _as_score_map(baseline_scores)
    candidate = _as_score_map(candidate_scores)
    cost_ratio = _cost_ratio(baseline_cost, candidate_cost)
    paired = sorted(set(baseline) & set(candidate))
    if not paired:
        reasons = ["insufficient data: no paired evaluation seeds"]
        if not baseline:
            reasons.append("baseline scores are empty")
        if not candidate:
            reasons.append("candidate scores are empty")
        return AcceptanceReport(
            verdict=VERDICT_INSUFFICIENT,
            mean_gain=0.0,
            non_worse_fraction=0.0,
            cost_ratio=cost_ratio,
            paired_seeds=[],
            reasons=reasons,
        )
    gains = np.asarray(
        [candidate[seed] - baseline[seed] for seed in paired],
        dtype=float,
    )
    mean_gain = float(gains.mean())
    non_worse_fraction = float((gains >= -NON_WORSE_TOLERANCE).mean())
    reasons: list[str] = []
    failed = False
    if mean_gain + THRESHOLD_TOLERANCE < float(effective.min_mean_gain):
        failed = True
        reasons.append(
            f"mean_gain {mean_gain:.4f} below min_mean_gain {float(effective.min_mean_gain):.4f}"
        )
    if non_worse_fraction + THRESHOLD_TOLERANCE < float(effective.min_non_worse_fraction):
        failed = True
        reasons.append(
            f"non_worse_fraction {non_worse_fraction:.4f} below min_non_worse_fraction "
            f"{float(effective.min_non_worse_fraction):.4f}"
        )
    if cost_ratio > float(effective.max_cost_ratio) + THRESHOLD_TOLERANCE:
        failed = True
        reasons.append(
            f"cost_ratio {cost_ratio:.4f} above max_cost_ratio {float(effective.max_cost_ratio):.4f}"
        )
    if failed:
        verdict = VERDICT_REJECT
    else:
        verdict = VERDICT_ACCEPT
        reasons.append(
            f"all acceptance criteria satisfied: mean_gain {mean_gain:.4f}, "
            f"non_worse_fraction {non_worse_fraction:.4f}, cost_ratio {cost_ratio:.4f}"
        )
    return AcceptanceReport(
        verdict=verdict,
        mean_gain=mean_gain,
        non_worse_fraction=non_worse_fraction,
        cost_ratio=cost_ratio,
        paired_seeds=paired,
        reasons=reasons,
    )
