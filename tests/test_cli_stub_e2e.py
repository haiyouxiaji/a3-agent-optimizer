from __future__ import annotations

import json

from a3opt.cli import main
from a3opt.contracts import EpisodeRow, EvalResult, ExperimentOutcome, TrainResult
from a3opt.loop import LoopConfig, run_optimization


def test_stub_run_end_to_end(tmp_path) -> None:
    out = tmp_path / "runs"
    code = main(
        [
            "run",
            "--backend",
            "stub",
            "--experiments",
            "4",
            "--out",
            str(out),
            "--run-id",
            "e2e",
            "--train-steps",
            "50",
            "--eval-seeds",
            "2000,2001",
        ]
    )
    assert code == 0
    run_dir = out / "e2e"
    for name in ("run.json", "tree.json", "memory.jsonl", "report.md"):
        assert (run_dir / name).is_file(), name
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert summary["experiments"] == 4
    assert summary["backend"] == "stub"
    assert summary["best_candidate_id"]
    assert (run_dir / "outcomes").is_dir()
    lines = [line for line in (run_dir / "memory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 4
    tree = json.loads((run_dir / "tree.json").read_text(encoding="utf-8"))
    assert len(tree["nodes"]) == 4
    root = tree["nodes"][0]
    assert root["parent_id"] is None
    assert root["outcome"]["status"] == "ok"
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Paired-seed acceptance" in report
    assert "Experiment tree" in report


def test_report_regeneration(tmp_path) -> None:
    out = tmp_path / "runs"
    main(["run", "--backend", "stub", "--experiments", "2", "--out", str(out), "--run-id", "regen", "--train-steps", "20"])
    run_dir = out / "regen"
    (run_dir / "report.md").unlink()
    code = main(["report", "--run", str(run_dir)])
    assert code == 0
    assert (run_dir / "report.md").is_file()


def test_plan_outputs_staged_candidates(capsys) -> None:
    code = main(["plan"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["space"] == "a3_policy_space"
    assert payload["staged"]
    assert all(not item["errors"] for item in payload["staged"])
    assert payload["staged"][0]["stage"] == "baseline"


def test_run_reports_unavailable_backend(tmp_path, capsys, monkeypatch) -> None:
    import a3opt.cli as cli_module
    from a3opt.backends import stub

    class Missing:
        name = "missing"

        def available(self):
            return False, "not installed"

    monkeypatch.setattr(cli_module, "make_backend", lambda name: Missing())
    code = main(["run", "--backend", "stub", "--out", str(tmp_path), "--experiments", "1"])
    assert code == 2
    assert "not installed" in capsys.readouterr().err


def test_summary_does_not_select_an_incomplete_or_rejected_candidate(tmp_path) -> None:
    class Backend:
        name = "test"
        calls = 0

        def run(self, candidate, budget, eval_seeds, max_eval_steps):
            self.calls += 1
            score = 4 if self.calls == 1 else 10
            seeds = eval_seeds[:1] if self.calls == 2 else eval_seeds
            cost = 10.0 if self.calls == 3 else 1.0
            return ExperimentOutcome(candidate, TrainResult(wall_seconds=cost), EvalResult(
                episodes=[EpisodeRow(seed, score, score == 10, 100) for seed in seeds],
            ))

    summary = run_optimization(LoopConfig(tmp_path, Backend(), experiments=3))["summary"]
    assert summary["best_candidate_id"] == summary["root_candidate_id"]
    assert summary["best_mean_score"] == 0.4
    verdicts = {report["verdict"] for report in summary["config"]["verdicts"].values()}
    assert verdicts == {"insufficient", "reject"}
