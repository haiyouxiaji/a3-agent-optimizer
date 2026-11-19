from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from a3opt.contracts import RunLayout


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def _node_rows(tree: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for node in tree.get("nodes", []):
        outcome = node.get("outcome") or {}
        train = outcome.get("train", {})
        evaluation = outcome.get("eval", {})
        metrics = train.get("metrics", {})
        acceptance = node.get("acceptance") or {}
        candidate = node.get("candidate") or {}
        rows.append(
            [
                node.get("node_id", ""),
                candidate.get("candidate_id", ""),
                node.get("parent_id") or "-",
                node.get("stage") or "",
                outcome.get("status", node.get("status", "pending")) or "pending",
                metrics.get("final_loss", ""),
                evaluation.get("mean_score", ""),
                evaluation.get("success_rate", ""),
                outcome.get("cost_wall_seconds", ""),
                acceptance.get("verdict", ""),
            ]
        )
    return rows


def _acceptance_rows(tree: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for node in tree.get("nodes", []):
        acceptance = node.get("acceptance")
        if not acceptance:
            continue
        candidate = node.get("candidate") or {}
        rows.append(
            [
                candidate.get("candidate_id", ""),
                acceptance.get("verdict", ""),
                acceptance.get("mean_gain", ""),
                acceptance.get("non_worse_fraction", ""),
                acceptance.get("cost_ratio", ""),
                "; ".join(acceptance.get("reasons", [])),
            ]
        )
    return rows


def _memory_rows(records: list[Mapping[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for record in records[-10:]:
        train = record.get("train", {}) if isinstance(record.get("train"), dict) else {}
        metrics = train.get("metrics", {})
        rows.append(
            [
                record.get("node_id", ""),
                record.get("candidate_id", ""),
                record.get("status", ""),
                record.get("mean_score", ""),
                metrics.get("final_loss", ""),
                record.get("normalized_score", ""),
                record.get("wall_seconds", ""),
                record.get("created_at", ""),
            ]
        )
    return rows


def render_report(
    summary: Mapping[str, Any],
    tree: Mapping[str, Any],
    memory_records: Iterable[Mapping[str, Any]],
    criteria: Mapping[str, Any] | None = None,
) -> str:
    node_rows = _node_rows(tree)
    acceptance_rows = _acceptance_rows(tree)
    memory_list = list(memory_records)
    header = [
        f"# Optimization run: {summary.get('run_id', 'unknown')}",
        "",
        f"- Backend: `{summary.get('backend', '')}`",
        f"- Candidate space: `{summary.get('space_name', '')}`",
        f"- Root candidate: `{summary.get('root_candidate_id', '')}`",
        f"- Best candidate: `{summary.get('best_candidate_id', '')}` "
        f"(mean normalized reward {summary.get('best_mean_score', 0)})",
        f"- Experiments executed: {summary.get('experiments', 0)}",
        f"- Accepted candidates: {', '.join(summary.get('accepted', [])) or 'none'}",
        f"- Created: {summary.get('created_at', '')}",
        "",
        "## Budget",
        "",
        "```json",
        json.dumps(summary.get("budget", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Experiment tree",
        "",
        _table(
            [
                "Node",
                "Candidate",
                "Parent",
                "Stage",
                "Status",
                "Final loss",
                "Mean score",
                "Success rate",
                "Wall s",
                "Verdict",
            ],
            node_rows,
        )
        if node_rows
        else "_No experiments recorded._",
        "",
        "## Paired-seed acceptance",
        "",
        _table(
            ["Candidate", "Verdict", "Mean gain", "Non-worse frac.", "Cost ratio", "Reasons"],
            acceptance_rows,
        )
        if acceptance_rows
        else "_No paired-seed acceptance comparisons were recorded._",
        "",
        "## Acceptance criteria",
        "",
        "```json",
        json.dumps(criteria or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Experiment memory",
        "",
        f"- Records: {len(memory_list)}",
        "",
        _table(
            ["Node", "Candidate", "Status", "Mean score", "Final loss", "Normalized", "Wall s", "Created"],
            _memory_rows(memory_list),
        )
        if memory_list
        else "_No memory records._",
        "",
        "## Protocol notes",
        "",
        "- Paired acceptance in this report uses the development seeds. The confirmation command runs the reserved seed block after policy selection.",
        "- A candidate is retained only when the paired comparison passes the gain, non-inferiority, and cost-ratio thresholds.",
        "- Every record stores the candidate specification, training metrics, per-seed evaluation scores, and measured cost.",
        "",
        "_Generated by a3opt._",
        "",
    ]
    return "\n".join(header)


def write_report(layout: RunLayout, summary: Mapping[str, Any], tree: Mapping[str, Any], memory_records: Iterable[Mapping[str, Any]], criteria: Mapping[str, Any] | None = None) -> Path:
    layout.ensure()
    text = render_report(summary, tree, memory_records, criteria)
    layout.report_path.write_text(text, encoding="utf-8")
    return layout.report_path
