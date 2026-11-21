# A3 Agent Optimizer

Agent-driven experiment search for modular robot policy optimization on the A3 dual-arm
MuJoCo platform. The repository turns the optimization design of the A3 midterm report into a
runnable loop: the agent selects a policy candidate, trains it, evaluates it on paired seeds,
compares it with the baseline, writes the evidence into memory, and decides what to try next.

The layer never re-implements the simulator, the expert controller, or the task success rules.
It drives an existing A3 simulation checkout and keeps the runtime chain
`Observation -> Policy -> Action -> Environment -> Evaluation` unchanged.

## Current architecture and result

The frozen expert-release and source-separation policy scored **9.323/10** on 192 new
`2×5 / advanced` scenes, with a stratified 95% interval of **[9.036, 9.583]** and
strict completion in **158/192** scenes. The earlier paired comparison recorded
one five-cookie regression and resulted in a rollback. Across both panels the
policy scored **9.25/10**, with a 95% interval of **[8.992, 9.484]**.

The frozen `branch-joint` policy with targeted first-batch descent scored **9.14/10**
on 100 independent, paired `2×5 / advanced` scenes balanced across four source columns.
The timed reference scored 8.79/10. The candidate's 95% balanced bootstrap interval is
[8.77, 9.49]. The course-budget `2×10` experiment is tracked separately.

The expert-release and source-separation candidate scored **9.03125/10** on a new
64-pair advanced panel, with a 95% interval of **[8.515625, 9.5]**. Strict completion
improved from 42 to 49 episodes. A five-cookie regression and an inconclusive paired
gain kept the Agent's reference policy in place.

- [当前 Agent 与模型架构](docs/A3_CURRENT_ARCHITECTURE.md)
- [实验树、分支决策与运行统计](docs/A3_EXPERIMENT_TREE.md)
- [冻结策略、逐场景评测证据与复现命令](evidence/advanced-20261120/README.md)
- [专家开爪位置的 64 对高难确认与错列审计](evidence/expert-landing-20261121/README.md)
- [专家开爪与来源避让组合的独立确认](evidence/source-guard-refine-20261121/README.md)
- [192 局高难计分复核、逐列指标与原生复跑](evidence/source-score-20261121/README.md)

## Earlier designs and experiments

- Design document: [docs/architecture.md](docs/architecture.md)
- Refinement and measured results: [docs/A3_REFINEMENT_REPORT.md](docs/A3_REFINEMENT_REPORT.md)
- Advanced randomization, autonomous follow-up, and replay verification: [docs/A3_ADVANCED_AGENT.md](docs/A3_ADVANCED_AGENT.md)
- Phase-aware grasp and placement search with paired confirmation: [docs/A3_ADAPTIVE_AGENT.md](docs/A3_ADAPTIVE_AGENT.md)
- Second-batch release diagnosis and autonomous candidate evaluation: [docs/A3_SECOND_BATCH_OPTIMIZATION.md](docs/A3_SECOND_BATCH_OPTIMIZATION.md)
- Upstream simulation project: [a3_dual_arm_sim](https://github.com/Eter0109/a3_dual_arm_sim) (separate checkout)

Historical reports refer to generated `artifacts/` and `runs/` files and to course handouts.
The public evidence linked above is available directly in this checkout.

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
git clone https://github.com/haiyouxiaji/a3-agent-optimizer.git a3_agent_optimizer
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

### Feedback-guided data collection and confirmation

With the simulator checkout at `5fa2497`, run the following from the optimizer root with
the real simulation environment active. The dataset must have its `.meta.json` sidecar.

```bash
a3opt refine --dataset artifacts/datasets/expert_2x5_multicam_60ep.npz --reference artifacts/local_sim_v2_40ep/bc2-40ep-64/checkpoint --out artifacts/refinement
a3opt freeze-confirmation --selection artifacts/refinement/development-summary.json --out artifacts/refinement
a3opt confirm --lock artifacts/refinement/confirmation-lock.json --eager-replay
```

The refinement loop scouts training layouts, collects matched-budget expert demonstrations for
feedback and uniform branches, checks the shards, and fits a layout-conditioned trajectory model.
Use `--initial-episodes 8` for the limited-data ablation and a separate output directory for
each run configuration. The confirmation command binds the simulator, checkpoints, seed block,
and evaluator to the development selection. `--eager-replay` repeats the candidate with the
upstream per-step camera rendering and checks action digests.

### Native 2x10 experiment

`python -m a3opt.two_by_ten_agent --out artifacts/two-by-ten --plan` previews the sampling budget and frozen seed splits.
Run without `--plan` to collect demonstrations, fit mean and vision-conditioned policies, and evaluate frozen candidates
through the upstream simulator. The run reports the 500-step snapshot and the native 3,200-step result for each episode.
`python -m a3opt.two_by_ten_course_agent --source artifacts/two-by-ten --out artifacts/two-by-ten-course --workers 4`
compares the frozen visual average and nearest-demonstration policies on fresh 500-step episodes. The native run's
initial fit and dataset manifests supply both policies. The Agent opens a separate confirmation split only when the
candidate meets the development gate.
See [course alignment and evidence](docs/A3_COURSE_ALIGNMENT.md) for the benchmark rules and sampling controls.

### First-release diagnostics

Run `a3opt.release_feedback_agent` to fit and select a bounded correction using a frozen `branch-joint` run and the
second-batch diagnosis. Once its development receipts are complete, run `a3opt.release_outcome_agent` with the
same `--reference` and the release run as `--source`. It uses the paired receipts to test whether runtime features can
choose among the three fixed controllers and opens a fresh confirmation split only after leave-one-seed-out validation.
`a3opt.release_focus_eval` separately tests the locked release candidate on 20 fresh scenes from the active source column.
`a3opt.release_completion_eval` verifies both action digests at the archived endpoint and then measures released
cookies and sustained full-slot stability at step 500. Its separate audit receipt records the original score, end step,
and gripper contacts alongside the stricter measurement.
Use separate output directories per experiment. On Windows, begin with a small worker count. Runs validate and reuse
completed receipts when resumed with the same protocol and a lower `--workers` value.

### Release timing and balanced verification

`python -m a3opt.release_timing_agent --reference <branch-run> --diagnosis <release-audit> --out <timing-run>`
uses the frozen `branch-joint` checkpoint and its original confirmation replays to select a timing intervention.
The run locks the screened speed before evaluating the fixed-column confirmation split.
`python -m a3opt.release_timing_random_eval --reference <branch-run> --timing <timing-run> --out <random-audit>`
evaluates 20 new balanced random-column scenarios. Unaffected scenes share a simulation only after both policies match
every action. `python scripts/audit_timing_course.py --timing <timing-run> --random <random-audit> --out <course-audit>`
re-evaluates the active column on the upstream default scoring contract.

### Two-batch placement search

The placement agents compare both course endpoint counts and released counts at step 500 on paired advanced scenes.
Training can observe simulator geometry to label rollouts; the policy uses task instructions and public robot observations.
`a3opt.placement_visual_feedback` captures front-camera image pairs on locked training scenes, and
`a3opt.placement_visual_geometry` learns a second-batch gate from their spatial features. See
[placement experiments and receipts](docs/A3_PLACEMENT_SEARCH.md) for the exact splits and rollback decisions.

`python -m a3opt.placement_hover_agent --out artifacts/placement-hover --workers 6` starts a follow-up from the
published frozen policy. The Agent scouts fresh column-2 scenes, branches from the end of the second transfer, and
selects a hover release height using matched failures and full-score controls. A candidate that passes development is
frozen for four-column confirmation under the same two scoring rules, with native replay checks.
`python -m a3opt.placement_slot_refine --acquisition <hover-run> --out <row-run>` uses the hover run's scout receipts
to fit a bounded first-batch alignment correction and evaluates it on fresh scenes. The row candidate must increase
strict full-slot success while preserving both endpoint counts.
`python -m a3opt.placement_compose --hover <hover-run> --row <row-run> --out <combined-run>` checks the two
fitted corrections together on the row branch's development scenes. The final confirmation uses a new seed block.
`python -m a3opt.placement_first_hover --source <hover-run> --source <combined-run> --out <first-hover-run>`
uses historical reference rollouts to select first-batch failures. It learns from matched simulation forks and
checks both batches on fresh scenes before opening confirmation.
`python -m a3opt.placement_hover_guard --source <hover-run> --source <combined-run> --first-hover <first-hover-run> --out <guard-run>`
fits an activation threshold from full-episode gains and regressions before opening fresh development and confirmation scenes.
`python -m a3opt.placement_hover_guard_refine --previous <guard-run> --out <refined-run>` uses the guard's
development feedback to fit a threshold that also checks the planned descent endpoint.
`python -m a3opt.placement_routed_agent --late <hover-run> --early <refined-run> --out <routed-run>` audits
the source pairs, selects complementary residuals on distinct source columns, and runs a fresh four-column confirmation.
`--fit-only` saves the dataset and freezes the recipe and seed list. Rerun the same command without the flag to evaluate it.
The routed controller and frozen follow-up have [paired receipts and a reproducible summary](evidence/routed-20261120/README.md).
`python -m a3opt.placement_wall_clearance_agent --source <routed-replication> --out <clearance-run>` uses the
reference rollouts to learn a placement gate near the box walls, then compares bounded corrections over full episodes.
`python -m a3opt.placement_clearance_feedback --out <sensor-run>` learns the tray-wall clearance from public observations
and tests it against the planned-position estimate on withheld scenes before evaluating any action changes.
`python -m a3opt.placement_contact_feedback --out <contact-run>` runs matched simulator branches during descent and
learns whether wrist force, torque, gripper load, or height tracking can trigger a bounded recovery.
`python -m a3opt.placement_late_hover_gate --late <hover-run> --out <late-hover-run>` tests planned-position gates for
the second batch on multiple archived panels, withholding an entire panel at a time.
`python -m a3opt.placement_staged_descent --out <staged-run>` tests paths that finish lateral alignment before lowering.
`python -m a3opt.placement_staged_gate --source <staged-run> --out <gated-run>` uses paired gains and regressions to learn
when to apply the first-batch staged descent.
`python -m a3opt.placement_release_hold --out <release-hold-run>` resamples historical placement failures and successful
controls, evaluates fixed-pose opening trajectories, and validates the selected recipe on new scenes.
`python -m a3opt.placement_release_recovery --out <release-recovery-run>` collects matched branches late in descent
and uses held-out historical groups to test release recovery rules based on wrist force and position tracking.
`python -m a3opt.placement_release_router --out <release-router-run>` joins the two audited training archives and
resamples missing stationary-release branches. The Agent learns when to choose stationary opening at the start of
descent and when to keep contact recovery available. The development test compares a selected route with both the
published policy and contact recovery alone.
`python -m a3opt.placement_first_recovery --out <first-recovery-run>` targets the source column with the most first-batch
descent losses across the historical groups. The Agent evaluates each branch through the end of both batches, using
the confirmed second-batch recovery as its reference.
`python -m a3opt.placement_early_clearance --out <early-clearance-run>` resamples first-batch placement losses and
tries a higher stationary release before late contact. The Agent fits and checks the gate using separate historical
groups, evaluates the entire two-batch episode, and freezes before opening the four-column confirmation set.
`python -m a3opt.placement_clearance_refine --source <completed-early-clearance-run> --out <guard-run>` revisits the
outcomes of a rejected unconditional correction. It replays the public observations up to the decision and fits an
activation rule from gains and regressions before reserving it for fresh closed-loop evaluation.
`python -m a3opt.placement_interbatch --source <completed-early-clearance-run> --out <coordinated-run>` resamples
second placements after the first-batch correction. The Agent verifies each unchanged branch against the archived
full episode, then fits a coordinated policy using both scoring endpoints and full two-batch outcomes.
`python scripts/replicate_release_hold.py <completed-confirmation-run> --out <replication-run>` repeats a frozen release
policy on 128 fresh scenes, with 32 per source column. Pass `--policy-kind release-recovery` and a fresh seed block
for the recovery policy. The evaluator checks shared-prefix rollouts against archived full episodes before starting.
The clearance feedback agent accepts `--rgb-archive <evidence-dir>` to fit
explicit tray-edge features from its archived wrist frames and collect a separate sensor holdout.
The clearance, contact, and staged descent agents accept `--source <routed-replication>` to add audited historical scenarios.
The [feedback evidence bundle](evidence/feedback-20261120/README.md) contains the saved protocols, datasets, and outcomes.
`python scripts/audit_placement_confirmation.py <run> --policy-kind hover` verifies a completed hover run from its
freeze, individual rollouts, and native replay receipts. The other controller audit kinds are `row`, `compose`, `first-hover`,
`hover-guard`, `hover-guard-refine`, `routed`, `wall-clearance`, `clearance-feedback`, `contact-feedback`, `late-hover-gate`,
`staged-descent`, `staged-gate`, `release-hold`, `release-recovery`, `release-router`, `first-recovery`, `early-clearance`,
`clearance-refine`, and `interbatch`.

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
`artifacts/candidates/`. The scripts regenerate the training datasets and exploratory checkpoints.
The frozen `branch-joint` checkpoint used for the published paired evaluation is under `evidence/`.

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
- The refinement workflow uses expert seed IDs 1000–1059, a separate sampling pool starting at
  1200, and development seeds 2000–2005. The confirmation block is 3100–3119.
- Real CLI runs default to 600 evaluation steps. The 120-step budget is for synthetic or smoke runs.
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
