# A3 Agent Optimizer

Agent-driven experiment search for modular robot policy optimization on the A3 dual-arm
MuJoCo platform. The repository turns the optimization design of the A3 midterm report into a
runnable loop: the agent selects a policy candidate, trains it, evaluates it on paired seeds,
compares it with the baseline, writes the evidence into memory, and decides what to try next.

The layer never re-implements the simulator, the expert controller, or the task success rules.
It drives an existing A3 simulation checkout and keeps the runtime chain
`Observation -> Policy -> Action -> Environment -> Evaluation` unchanged.

- Design document: [docs/architecture.md](docs/architecture.md)
- Upstream simulation project: `a3_dual_arm_sim` (course-provided checkout, not vendored here)

## What it implements

| Component | Module | Description |
| --- | --- | --- |
| Candidate space | `src/a3opt/space.py` | legal module combinations and a staged screening list |
| Experiment tree | `src/a3opt/tree.py` | UCT selection, expansion, backpropagation |
| Acceptance gate | `src/a3opt/acceptance.py` | paired-seed gain, non-inferiority, cost ratio |
| Experiment memory | `src/a3opt/memory.py` | append-only JSONL evidence and best-k retrieval |
| Budget | `src/a3opt/contracts.py` | per-experiment and per-run budgets |
| Planner | `src/a3opt/planner.py` | staged hypothesis ordering |
| Loop | `src/a3opt/loop.py` | Analyze -> Hypothesize -> Train -> Evaluate -> Reflect -> Iterate |
| Report | `src/a3opt/report.py` | generated markdown run report |
| Backends | `src/a3opt/backends/` | synthetic, local MuJoCo + torch, upstream CLI adapter |

## Quick start

The loop needs Python 3.10+. The real simulation backend additionally needs the A3 simulation
checkout with its own dependencies (`mujoco`, `gymnasium`, `opencv-python`, `scipy`) and `torch`.

```bash
git clone <this-repository> a3_agent_optimizer
cd a3_agent_optimizer
python -m venv .venv
.venv/Scripts/activate            # Windows; use .venv/bin/activate on Linux
pip install -e ".[dev]"
```

Synthetic self-test, no simulator or GPU required:

```bash
a3opt plan
a3opt selftest --experiments 6
```

A self-test run writes `runs/<run-id>/` with `run.json`, `tree.json`, `memory.jsonl`,
`outcomes/`, and `report.md`.

### Real closed-loop experiments

1. Install the upstream simulation repository and its dependencies into the same environment
   (see its `README.md` and `docs/workflow_guide.md`).
2. Record expert demonstrations from the real simulator, train a behavior-cloning candidate, and
   evaluate it on the randomized 2x5 benchmark:

```bash
python scripts/local_backend_smoke.py --stage record     # 6 expert episodes, seeds 1000-1005
python scripts/local_backend_smoke.py --stage train      # 3 candidates, 400 steps each
python scripts/local_backend_smoke.py --stage eval       # closed-loop scores on seeds 2100-2101
```

Real artifacts land in `artifacts/real_runs/summary.json`, `artifacts/datasets/`, and
`artifacts/candidates/`. Datasets and checkpoints are intentionally not committed; the scripts
regenerate them.

3. Run the agent loop against the real backend. The dataset recorded above is reused:

```bash
a3opt run --backend local_sim --experiments 3 --eval-seeds 2000,2001 --max-eval-steps 120 --train-steps 300
```

## CLI

| Command | Purpose |
| --- | --- |
| `a3opt plan` | print the staged candidate space and validation results |
| `a3opt run --backend stub` | deterministic synthetic loop for CI and demos |
| `a3opt run --backend local_sim` | real loop: record, train, evaluate in MuJoCo |
| `a3opt report --run runs/<id>` | regenerate the markdown report of a run |
| `a3opt selftest` | one-command end-to-end synthetic check |

## Candidate space

| Dimension | Values |
| --- | --- |
| encoder | `tiny_cnn`, `state_mlp` |
| fusion | `concat`, `film` |
| action head | `mlp`, `chunk` |
| chunk size | `1`, `20`, `50` |
| image size | `0`, `64` |
| learning rate | `1e-3`, `3e-3` |

Compatibility rules reject illegal combinations before training (for example, `chunk` requires
`chunk_size > 1`, `film` requires an image encoder). The staged list evaluates one design question
at a time; the Cartesian grid (96 combinations) is never enumerated blindly.

## Acceptance gate

A candidate is retained only if, on identical seeds:

1. mean paired gain over the baseline is at least 1.0 cookie;
2. at least 70% of paired seeds are not worse;
3. measured wall-clock cost is at most 3x the baseline cost.

The verdict is one of `accept`, `reject`, or `insufficient` (fewer than two paired seeds).

## Backends

| Backend | Real training | Requirements |
| --- | --- | --- |
| `stub` | synthetic scores only | nothing |
| `local_sim` | single-camera torch behavior cloning on recorded MuJoCo demonstrations, closed-loop MuJoCo evaluation | `torch` + A3 simulation checkout |
| `local_sim_v2` | multi-camera torch behavior cloning with action normalization and previous-action conditioning; current best real policy (3.5/10 mean) | `torch` + A3 simulation checkout |
| `a3_sim_cli` | upstream ACT training and benchmark commands (dry-run by default) | `lerobot` + A3 simulation checkout |

## Real experiments

See [docs/results.md](docs/results.md) for full logs, commands, and interpretation. Executed on one
Windows workstation (Python 3.13.12, torch 2.15.0+cpu, MuJoCo 3.16.0):

| Stage | Result |
| --- | --- |
| Expert demonstration collection | 6 episodes (seeds 1000-1005), 2450 frames, all 10/10 success, 390.3 s |
| Behavior-cloning training | 3 candidates, 400 steps each, loss 1.01 -> 0.010 / 0.010 / 0.004, 0.7-4.8 s each |
| Closed-loop evaluation (real MuJoCo) | seeds 2100/2101: 0/10 for all three candidates at 120 and 500 steps |
| Expert reference on the same seeds | 0/10 at 120 steps, 5/10 at 200 steps, 10/10 at 500 steps |
| Agent loop against the real backend | 3 experiments (baseline + 2 staged candidates), 1200 training steps, 268.98 s; all scored 0/10, both candidates rejected by the paired-seed gate |
| Second-stage: multi-camera BC2 | 20 expert episodes (8178 frames, 128x128, three cameras), 6000-step behavior cloning; real closed-loop **2/10 and 5/10** on seeds 2000/2001 (mean 3.5) |
| Agent loop with the BC2 backend | 3 experiments, 12000 training steps, 693.04 s; best = multi-camera concat candidate, real closed-loop **5/10 and 2/10 (mean 3.5/10)**; film and state-only variants rejected |
| Third stage: 40-episode data + inference tuning | 40 expert episodes (16379 frames), 10000-step training, execution horizon 12; dev seeds 2000-2005: 5, 10 (success), 5, 5, 5, 5 -> **mean 5.83/10**; reserved confirmation seeds 3000-3001: 5, 0 |
| Backend / plugin verification | `backend.run` status ok and round-trip; upstream CLI via policy plugin returncode 0 |

The zero closed-loop scores are reported honestly: the compact behavior-cloning baseline learns the
mean pose but not yet the task, and a 120-step budget cannot score even for the scripted expert.
The framework, the data collection, the training, and the evaluation are real; policy competence is
the documented limitation, not a hidden failure.

## Handover

A detailed handover document for a new AI or human maintainer is at
[docs/AI_HANDOVER.md](docs/AI_HANDOVER.md) with a PDF version at
[docs/AI_HANDOVER.pdf](docs/AI_HANDOVER.pdf). It covers the current status, file map, environment
matrix (three Python environments), datasets and checkpoints, the architecture and experiment-tree
format, verified results, exact reproduction commands, known blockers (official ACT/SmolVLA,
Windows SSH, network speed), open tasks, and GitHub instructions.

## Reports

- Experiment log: [docs/results.md](docs/results.md)
- Architecture and experiment-tree report: [docs/A3_Agent_优化框架_架构与实验树报告.pdf](docs/A3_Agent_优化框架_架构与实验树报告.pdf) (generated by `scripts/build_tree_report.py` from the real run artifacts)
- Example tree run: `runs/final-tree-60ep` (3 nodes, 24000 training steps, 1201.8 s)

## Tests

```bash
pytest tests -q
```

The suite covers the candidate space, UCT selection, acceptance gate, memory, contract round-trips,
backend command construction, the synthetic end-to-end loop, and a torch training smoke test.
Simulation tests are skipped unless `A3OPT_RUN_SIM_TESTS=1` is set.

## Repository layout

```text
src/a3opt/            framework package
  contracts.py        data contracts (candidates, budgets, results, layout)
  space.py            candidate space and compatibility rules
  tree.py             experiment tree with UCT selection
  acceptance.py       paired-seed acceptance gate
  memory.py           append-only experiment memory
  planner.py          staged hypothesis ordering
  loop.py             the agent loop
  report.py           markdown report generation
  cli.py              command line interface
  backends/           synthetic, local simulation, upstream CLI
scripts/              real-data and real-experiment entry points
tests/                unit and end-to-end tests
docs/                 architecture, results, protocol
```

## Evaluation protocol

- Task: 2x5 cookie transfer, `basic` randomization profile, source column 1.
- Expert demonstration seeds: 1000-1005.
- Screening seeds: 2000-2019, `max_eval_steps = 120`.
- Confirmation seeds: 3000-3019 are reserved and never used for tuning.
- Success conditions follow the upstream repository defaults: ten cookies in the target slots,
  seventy in the source box, five-step hold, slot tolerance 0.015/0.005 m, wall tolerance 0.03 m.

## Limitations

- The `local_sim` policy is a compact behavior-cloning baseline used to make the loop runnable and
  reproducible on one workstation; it is not a SmolVLA or ACT reproduction.
- The `a3_sim_cli` backend is command-tested, not executed end-to-end here, because the upstream
  training stack (`lerobot`) is not vendored.
- ACT and SmolVLA were attempted but the official pipeline cannot run on this workstation:
  `lerobot==0.5.1` needs Python 3.12+ and torch 2.7+, the only GPU environment here is Python 3.10
  with torch 2.5.1, and the 4.1 GB A3 dataset plus multi-GB SmolVLA weights are not reachable at the
  measured 0.5 MB/s HuggingFace throughput. An ACT-style reimplementation (ResNet18 backbones,
  transformer encoder/decoder, CVAE latent, 50-step chunks) was trained on the local A3 data and
  scored 1.5/10 on seeds 2000/2001, well below the compact BC model (5.83/10).
- Six expert episodes validate the pipeline and the acceptance gate, not a general performance
  claim about the search itself.

## License

MIT. See [LICENSE](LICENSE).
