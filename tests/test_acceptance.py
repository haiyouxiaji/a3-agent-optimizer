from __future__ import annotations

import math

import pytest

from a3opt.acceptance import (
    VERDICT_ACCEPT,
    VERDICT_INSUFFICIENT,
    VERDICT_REJECT,
    AcceptanceCriteria,
    AcceptanceReport,
    compare_paired,
)

BASELINE = {100: 2.0, 101: 3.0, 102: 4.0}
CANDIDATE = {100: 3.5, 101: 4.5, 102: 5.0}


def test_default_criteria_values():
    criteria = AcceptanceCriteria()
    assert criteria.min_mean_gain == 1.0
    assert criteria.min_non_worse_fraction == 0.7
    assert criteria.max_cost_ratio == 3.0


def test_accept_when_all_thresholds_met():
    report = compare_paired(BASELINE, CANDIDATE, 100.0, 150.0, AcceptanceCriteria())
    assert report.verdict == VERDICT_ACCEPT
    assert report.accepted is True
    assert report.mean_gain == pytest.approx(4.0 / 3.0)
    assert report.non_worse_fraction == pytest.approx(1.0)
    assert report.cost_ratio == pytest.approx(1.5)
    assert report.paired_seeds == [100, 101, 102]
    assert report.reasons


def test_accept_at_exact_thresholds():
    baseline = {seed: 0.0 for seed in range(10)}
    candidate = {seed: 1.0 for seed in range(10)}
    report = compare_paired(baseline, candidate, 10.0, 10.0)
    assert report.mean_gain == pytest.approx(1.0)
    assert report.verdict == VERDICT_ACCEPT
    baseline = {seed: 0.0 for seed in range(10)}
    candidate = {seed: 2.0 for seed in range(7)}
    candidate.update({seed: -1.0 for seed in range(7, 10)})
    report = compare_paired(baseline, candidate, 10.0, 10.0)
    assert report.non_worse_fraction == pytest.approx(0.7)
    assert report.verdict == VERDICT_ACCEPT


def test_reject_when_mean_gain_below_threshold():
    candidate = {seed: score + 0.5 for seed, score in BASELINE.items()}
    report = compare_paired(BASELINE, candidate, 10.0, 10.0)
    assert report.verdict == VERDICT_REJECT
    assert report.accepted is False
    assert report.mean_gain == pytest.approx(0.5)
    assert any("mean_gain" in reason for reason in report.reasons)


def test_reject_when_non_worse_fraction_below_threshold():
    baseline = {seed: 0.0 for seed in range(10)}
    candidate = {seed: 5.0 for seed in range(5)}
    candidate.update({seed: -0.5 for seed in range(5, 10)})
    report = compare_paired(baseline, candidate, 10.0, 10.0)
    assert report.non_worse_fraction == pytest.approx(0.5)
    assert report.verdict == VERDICT_REJECT
    assert any("non_worse_fraction" in reason for reason in report.reasons)


def test_reject_when_cost_ratio_too_high():
    baseline = {seed: 0.0 for seed in range(3)}
    candidate = {seed: 2.0 for seed in range(3)}
    report = compare_paired(baseline, candidate, 10.0, 31.0)
    assert report.cost_ratio == pytest.approx(3.1)
    assert report.verdict == VERDICT_REJECT
    assert any("cost_ratio" in reason for reason in report.reasons)


def test_insufficient_when_no_paired_seeds():
    report = compare_paired({100: 1.0}, {200: 2.0}, 5.0, 5.0)
    assert report.verdict == VERDICT_INSUFFICIENT
    assert report.paired_seeds == []
    assert report.mean_gain == 0.0
    assert report.non_worse_fraction == 0.0
    assert report.reasons


def test_insufficient_when_one_side_empty():
    assert compare_paired({}, {100: 1.0}).verdict == VERDICT_INSUFFICIENT
    assert compare_paired({100: 1.0}, {}).verdict == VERDICT_INSUFFICIENT
    assert compare_paired(None, None).verdict == VERDICT_INSUFFICIENT


def test_incomplete_seed_lists_are_insufficient():
    baseline = {100: 1.0, 101: 2.0}
    candidate = {101: 4.0, 200: 99.0}
    report = compare_paired(baseline, candidate)
    assert report.paired_seeds == []
    assert report.verdict == VERDICT_INSUFFICIENT


def test_non_worse_counts_equal_scores_as_non_worse():
    baseline = {100: 1.0, 101: 2.0}
    candidate = {100: 3.0, 101: 2.0}
    report = compare_paired(baseline, candidate)
    assert report.non_worse_fraction == pytest.approx(1.0)
    assert report.mean_gain == pytest.approx(1.0)
    assert report.verdict == VERDICT_ACCEPT


def test_cost_ratio_edge_cases():
    report = compare_paired({1: 0.0}, {1: 2.0}, 0.0, 5.0)
    assert math.isinf(report.cost_ratio)
    assert report.verdict == VERDICT_REJECT
    assert any("cost_ratio" in reason for reason in report.reasons)
    report = compare_paired({1: 0.0}, {1: 2.0}, 0.0, 0.0)
    assert report.cost_ratio == 1.0
    assert report.verdict == VERDICT_ACCEPT


def test_custom_criteria_relaxes_thresholds():
    baseline = {100: 1.0}
    candidate = {100: 1.5}
    assert compare_paired(baseline, candidate, 1.0, 4.0).verdict == VERDICT_REJECT
    relaxed = AcceptanceCriteria(
        min_mean_gain=0.4,
        min_non_worse_fraction=0.5,
        max_cost_ratio=10.0,
    )
    assert (
        compare_paired(baseline, candidate, 1.0, 4.0, relaxed).verdict
        == VERDICT_ACCEPT
    )


def test_report_to_dict_exposes_fields():
    report = compare_paired(BASELINE, CANDIDATE, 100.0, 150.0)
    data = report.to_dict()
    assert data["verdict"] == VERDICT_ACCEPT
    assert data["mean_gain"] == pytest.approx(4.0 / 3.0)
    assert data["non_worse_fraction"] == pytest.approx(1.0)
    assert data["cost_ratio"] == pytest.approx(1.5)
    assert data["paired_seeds"] == [100, 101, 102]
    assert isinstance(data["reasons"], list)
    default_report = AcceptanceReport(verdict=VERDICT_INSUFFICIENT)
    assert default_report.accepted is False
    assert default_report.paired_seeds == []
    assert default_report.reasons == []
