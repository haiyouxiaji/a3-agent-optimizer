from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from a3opt.acceptance import AcceptanceCriteria, compare_paired
from a3opt.contracts import (
    STATUS_FAILED,
    STATUS_OK,
    CandidateSpec,
    EvalResult,
    ExperimentBackend,
    ExperimentBudget,
    ExperimentOutcome,
    OptimizationSummary,
    RunBudget,
    RunLayout,
    TrainResult,
)
from a3opt.memory import ExperimentMemory
from a3opt.planner import StagedPlanner
from a3opt.report import write_report
from a3opt.space import CandidateSpace
from a3opt.tree import ExperimentTree, TreeNode


@dataclass
class LoopConfig:
    run_dir: Path
    backend: ExperimentBackend
    space: CandidateSpace = field(default_factory=CandidateSpace)
    experiments: int = 6
    eval_seeds: tuple[int, ...] = (2000, 2001)
    max_eval_steps: int = 120
    experiment_budget: ExperimentBudget = field(default_factory=ExperimentBudget)
    run_budget: RunBudget = field(default_factory=RunBudget)
    criteria: AcceptanceCriteria = field(default_factory=AcceptanceCriteria)
    exploration_constant: float = 0.7
    max_children: int = 3
    run_id: str = ""
    seed: int = 0


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _failed_outcome(candidate: CandidateSpec, error: str) -> ExperimentOutcome:
    return ExperimentOutcome(
        candidate=candidate,
        train=TrainResult(status=STATUS_FAILED, error=error),
        eval=EvalResult(status=STATUS_FAILED, error=error),
        status=STATUS_FAILED,
        error=error,
    )


def run_optimization(config: LoopConfig) -> dict[str, Any]:
    expected_seeds = set(config.eval_seeds)
    if not expected_seeds or len(expected_seeds) != len(config.eval_seeds):
        raise ValueError("development evaluation seeds must be nonempty and unique")
    if config.max_eval_steps < 1:
        raise ValueError("the evaluation step budget must be positive")
    layout = RunLayout(run_dir=Path(config.run_dir)).ensure()
    run_id = config.run_id or layout.run_dir.name
    space = config.space
    candidates = space.candidates(seed=config.seed)
    root_candidate = candidates[0]
    tree = ExperimentTree(
        root_candidate,
        c=config.exploration_constant,
        max_children=config.max_children,
    )
    planner = StagedPlanner(space)
    memory = ExperimentMemory(layout.memory_path)
    outcomes: dict[str, ExperimentOutcome] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    used: set[str] = {root_candidate.candidate_id}
    state = {"baseline": None, "executed": 0}

    def run_node(node: TreeNode, candidate: CandidateSpec) -> ExperimentOutcome:
        started = time.perf_counter()
        try:
            outcome = config.backend.run(
                candidate,
                config.experiment_budget,
                config.eval_seeds,
                config.max_eval_steps,
            )
        except Exception as exc:
            outcome = _failed_outcome(candidate, f"{type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - started
        if outcome.train.wall_seconds <= 0.0:
            outcome.train.wall_seconds = elapsed
        if outcome.status == STATUS_OK:
            rows = outcome.eval.episodes
            valid = (
                outcome.eval.status == STATUS_OK and len(rows) == len(expected_seeds)
                and {row.seed for row in rows} == expected_seeds
                and math.isfinite(outcome.eval.max_score) and outcome.eval.max_score > 0
                and all(0 <= row.score <= outcome.eval.max_score and 1 <= row.steps <= config.max_eval_steps
                        for row in rows)
            )
            if not valid:
                outcome.status = outcome.eval.status = STATUS_FAILED
                outcome.error = outcome.eval.error = "evaluation did not satisfy the complete development protocol"
        node.status = outcome.status
        node.error = outcome.error
        tree.backpropagate(node, outcome.normalized_score())
        outcomes[node.node_id] = outcome
        _write_json(layout.outcomes_dir / f"{candidate.candidate_id}.json", outcome.to_dict())
        config.run_budget.record(outcome)
        memory.record_from_outcome(outcome, node.node_id)
        state["executed"] += 1
        if node.node_id == tree.root_id:
            state["baseline"] = outcome
        else:
            baseline = state["baseline"]
            report = compare_paired(
                baseline.scores_by_seed() if baseline is not None and baseline.status == STATUS_OK else None,
                outcome.scores_by_seed() if outcome.status == STATUS_OK else None,
                baseline.cost_wall_seconds if baseline is not None else 0.0,
                outcome.cost_wall_seconds,
                config.criteria,
            )
            verdicts[candidate.candidate_id] = report.to_dict()
        return outcome

    root = tree.root
    if config.run_budget.can_afford(config.experiment_budget):
        run_node(root, root_candidate)
    while state["executed"] < config.experiments and config.run_budget.can_afford(config.experiment_budget):
        parent = tree.select()
        candidate = planner.propose(parent, used, space)
        if candidate is None:
            break
        used.add(candidate.candidate_id)
        child = tree.add_child(parent, candidate, stage=space.stage_of(candidate))
        run_node(child, candidate)

    best_candidate_id = ""
    best_normalized = -1.0
    for node_id, outcome in outcomes.items():
        node = tree.get(node_id)
        if node.candidate is None or outcome.status != STATUS_OK:
            continue
        if node_id != tree.root_id and verdicts[node.candidate.candidate_id]["verdict"] != "accept":
            continue
        score = outcome.normalized_score()
        if score > best_normalized:
            best_normalized = score
            best_candidate_id = node.candidate.candidate_id
    accepted = [cid for cid, report in verdicts.items() if report.get("verdict") == "accept"]
    rejected = [cid for cid, report in verdicts.items() if report.get("verdict") == "reject"]
    summary = OptimizationSummary(
        run_id=run_id,
        backend=config.backend.name,
        space_name=space.name,
        root_candidate_id=root_candidate.candidate_id,
        best_candidate_id=best_candidate_id,
        best_mean_score=float(max(best_normalized, 0.0)),
        experiments=int(state["executed"]),
        accepted=accepted,
        rejected=rejected,
        budget=config.run_budget.summary(),
        config={
            "eval_seeds": [int(seed) for seed in config.eval_seeds],
            "max_eval_steps": int(config.max_eval_steps),
            "experiments_requested": int(config.experiments),
            "exploration_constant": float(config.exploration_constant),
            "max_children": int(config.max_children),
            "experiment_budget": config.experiment_budget.to_dict(),
            "criteria": asdict(config.criteria),
            "space_grid_size": int(space.grid_size()),
            "verdicts": verdicts,
        },
    )
    summary_payload = summary.to_dict()
    _write_json(layout.summary_path, summary_payload)

    tree_payload = tree.to_dict()
    for node in tree_payload.get("nodes", []):
        node_id = node.get("node_id")
        outcome = outcomes.get(node_id)
        node["outcome"] = outcome.to_dict() if outcome is not None else None
        candidate_id = (node.get("candidate") or {}).get("candidate_id", "")
        node["acceptance"] = verdicts.get(candidate_id)
    _write_json(layout.tree_path, tree_payload)

    records = [record.to_dict() for record in memory.records()]
    write_report(layout, summary_payload, tree_payload, records, criteria=asdict(config.criteria))
    return {
        "summary": summary_payload,
        "tree": tree_payload,
        "memory": records,
        "run_dir": str(layout.run_dir),
        "report": str(layout.report_path),
    }
