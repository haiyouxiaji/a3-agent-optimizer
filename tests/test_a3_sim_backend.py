from __future__ import annotations

from pathlib import Path

from a3opt.backends.a3_sim import (
    A3SimCliBackend,
    A3SimCliConfig,
    build_eval_command,
    build_train_command,
    parse_eval_episodes,
)
from a3opt.contracts import CandidateSpec, ExperimentBudget


def make_candidate(**kwargs) -> CandidateSpec:
    return CandidateSpec(candidate_id="c1", **kwargs)


def test_train_command_mapping() -> None:
    config = A3SimCliConfig(sim_root=Path("."), dataset_root=Path("datasets/demo"), device="cpu")
    command = build_train_command(config, make_candidate(chunk_size=50, lr=3e-3, seed=1000), ExperimentBudget(max_train_steps=200, batch_size=8), Path("out/model"))
    assert command[:2] == ["a3-sim", "train-act"]
    assert "--steps" in command and command[command.index("--steps") + 1] == "200"
    assert "--chunk-size" in command and command[command.index("--chunk-size") + 1] == "50"
    assert "--n-action-steps" in command and command[command.index("--n-action-steps") + 1] == "50"
    assert "--lr" in command and command[command.index("--lr") + 1] == "0.003"
    assert command[-3:] == ["--num-workers", "0", "cpu"] or command[-2:] == ["--device", "cpu"]


def test_eval_command_mapping() -> None:
    config = A3SimCliConfig(sim_root=Path("."), dataset_root=Path("datasets/demo"))
    command = build_eval_command(config, Path("out/model"), (2100, 2101), 120)
    assert "--episodes" in command and command[command.index("--episodes") + 1] == "2"
    assert "--seed-start" in command and command[command.index("--seed-start") + 1] == "2100"
    assert any(token.startswith("act:") for token in command)


def test_dry_run_writes_commands(tmp_path) -> None:
    config = A3SimCliConfig(
        sim_root=tmp_path / "sim",
        output_root=tmp_path / "out",
        dry_run=True,
    )
    config.sim_root.mkdir()
    backend = A3SimCliBackend(config)
    ok, _ = backend.available()
    assert ok is True
    outcome = backend.run(make_candidate(), ExperimentBudget(max_train_steps=20), (2100, 2101), 120)
    assert outcome.status == "ok"
    assert outcome.train.error == "dry_run"
    assert (tmp_path / "out" / "c1" / "commands.json").is_file()


def test_unavailable_without_repository(tmp_path) -> None:
    backend = A3SimCliBackend(A3SimCliConfig(sim_root=tmp_path / "missing"))
    ok, reason = backend.available()
    assert ok is False
    assert "not found" in reason


def test_parse_episode_lines() -> None:
    output = (
        "[01/02] Seed 2100 | Score:  6/10 | Steps:  120 | Time:  32.10s | Status: FAIL\n"
        "[02/02] Seed 2101 | Score: 10/10 | Steps:  118 | Time:  31.40s | Status: SUCCESS\n"
    )
    rows = parse_eval_episodes(output)
    assert [row.seed for row in rows] == [2100, 2101]
    assert [row.score for row in rows] == [6.0, 10.0]
    assert rows[1].success is True
    assert rows[0].steps == 120
