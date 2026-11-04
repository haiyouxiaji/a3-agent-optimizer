# Executed experiments and measured results

All numbers in this document were produced on one Windows workstation by the commands shown below.
Every run leaves artifacts under `artifacts/` (git-ignored) and its own log. No synthetic numbers are
mixed into the real sections; the synthetic self-test numbers appear only in the last section and are
labelled as such.

## Environment

| Item | Value |
| --- | --- |
| OS | Windows 11 10.0.26200 |
| Python | 3.13.12 |
| torch | 2.15.0+cpu |
| MuJoCo | 3.16.0 |
| numpy | 2.5.4 |
| A3 simulation checkout | `a3_dual_arm_sim` commit `5fa2497` |
| Task | 2x5 cookie transfer, `basic` randomization profile, source column 1 |

## 1. Expert demonstration collection (real MuJoCo)

Command:

```bash
python scripts/local_backend_smoke.py --stage record
```

| Item | Value |
| --- | --- |
| Seeds | 1000-1005 |
| Episodes | 6 |
| Episode steps | 413, 402, 410, 406, 408, 411 |
| Episode outcomes | 10/10 cookies, success, all six |
| Frames recorded | 2450 |
| Observation | front camera downsampled to 64x64 |
| State / action | 16D joint state, 16D applied joint target |
| Wall clock | 390.3 s |
| Dataset | `artifacts/datasets/expert_2x5_basic_6ep.npz` (2.59 MB) |

## 2. Real training runs (torch behavior cloning, CPU)

Command:

```bash
python scripts/local_backend_smoke.py --stage train
```

Each candidate trained 400 steps with batch 64 on the 2450 recorded frames.

| Candidate | First loss | Final loss | Last-10 mean | Train steps | Train wall |
| --- | --- | --- | --- | --- | --- |
| tiny_cnn / concat / chunk-k20 / lr 1e-3 | 1.012571 | 0.010029 | 0.010409 | 400 | 4.76 s |
| tiny_cnn / film / chunk-k20 / lr 1e-3 | 1.005398 | 0.010167 | 0.011112 | 400 | 3.90 s |
| state_mlp / concat / chunk-k1 / lr 1e-3 | 1.016318 | 0.003840 | 0.003279 | 400 | 0.67 s |

Training loss drops by two orders of magnitude, so the gradient pipeline, the chunk targets, and the
checkpoint round-trip all work. The open question is closed-loop competence, measured next.

Checkpoints: `artifacts/real_runs/candidates/<candidate>/checkpoint/{model.pt,spec.json,train_log.json}`.

## 3. Closed-loop evaluation (real MuJoCo)

Command:

```bash
python scripts/local_backend_smoke.py --stage eval
python scripts/local_backend_smoke.py --stage supplement
```

Evaluation protocol: seeds 2100 and 2101, policy executed in the randomized 2x5 benchmark until the
step budget is exhausted; score is the number of cookies validly placed in the target box.

| Candidate | Budget 120 steps | Budget 500 steps | Steps used at 500 |
| --- | --- | --- | --- |
| tiny_cnn / concat / chunk-k20 | 0/10, 0/10 | 0/10, 0/10 | 500, 500 |
| tiny_cnn / film / chunk-k20 | 0/10, 0/10 | 0/10, 0/10 | 500, 500 |
| state_mlp / concat / chunk-k1 | 0/10, 0/10 | 0/10, 0/10 | 500, 500 |

Offline action error (mean absolute error on held-out demonstration frames): 0.066, 0.067, 0.031 rad.

### Why every score is zero

The step budget alone explains the 120-step result: the scripted expert itself scores 0/10 at 120
steps on the same seeds. The 500-step result is not a budget artifact; the compact behavior-cloning
baseline is simply not competent at the task yet.

Expert reference on the same seeds (scripted controller, not a learned policy):

| Expert budget | Score | Steps used |
| --- | --- | --- |
| 120 steps | 0/10 | 120 (5 cookies grasped, none released in time) |
| 200 steps | 5/10 | 200 |
| 500 steps | 10/10, success | 412 and 409 |

The learned policies reach a fixed pose after roughly 20 closed-loop steps and then stay there.
The recorded demonstrations contain roughly 2450 frames from six episodes; a 400-step behavior
cloning model on 64x64 images learns the mean pose but not the phase-conditioned control sequence.
This is the limitation stated in `docs/architecture.md`: the local backend exists to run the whole
loop cheaply, not to claim task competence. Making the search discriminative requires longer
training and more demonstrations, which is future work rather than a property of the framework.

## 4. Agent loop against the real backend

Command:

```bash
python scripts/run_local_loop.py --experiments 3 --train-steps 400 --eval-seeds 2000,2001 --max-eval-steps 500
```

The loop runs the baseline candidate first, then asks the experiment tree which node to expand,
proposes the next staged candidate, trains it, evaluates it on the same paired seeds, compares it
with the baseline, records the evidence, and regenerates the report.

Protocol: seeds 2000 and 2001, `max_eval_steps = 500`, 400 training steps per candidate, batch 64,
acceptance thresholds at their defaults (gain >= 1.0 cookie, non-worse >= 70%, cost ratio <= 3x).

| Node | Candidate | Parent | Final loss | Mean score | Verdict |
| --- | --- | --- | --- | --- | --- |
| root | tiny_cnn / concat / chunk-k20 / lr 1e-3 | - | 0.010029 | 0/10 | baseline |
| n001 | tiny_cnn / film / chunk-k20 / lr 1e-3 | root | 0.010167 | 0/10 | reject |
| n002 | state_mlp / concat / chunk-k20 / lr 1e-3 | n001 | 0.010907 | 0/10 | reject |

Budget actually spent: 3 experiments, 1200 training steps, 268.98 s wall clock. The run artifacts are
`runs/real-loop-500/{run.json,tree.json,memory.jsonl,report.md,outcomes/}`.

Because all three policies score zero, the paired comparisons have zero gain and the gate rejects
both candidates; the baseline is retained and the report says so explicitly. This is the correct
behaviour of the gate, and it is the honest end state of the current baseline: the loop is fully
operational on real training and real evaluation, while the compact behavior-cloning policy is not
yet competent enough for the search to be discriminative. Making it discriminative requires longer
training on more demonstrations (the future work described in the report), not a change to the
framework.

## 5. Backend contract verification

`LocalSimBackend.available()` returns `(True, "torch 2.15.0+cpu, mujoco and a3_dual_arm_sim importable")`,
`make_backend("local_sim")` satisfies the `ExperimentBackend` protocol, `backend.run(...)` returns
`status=ok` and round-trips through `ExperimentOutcome.to_dict/from_dict`.

The policy plugin also works through the upstream benchmark CLI:

```bash
A3OPT_CHECKPOINT=artifacts/real_runs/candidates/tiny_cnn-concat-chunk-k20-lr0.001/checkpoint \
python -m a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli \
  --policy a3opt.backends.policy_plugin:create_policy \
  --episodes 1 --seed-start 2200 --max-steps 40 --workers 1 --task 2x5 --profile basic --source-column 1
```

Result: returncode 0, one episode, score 0/10, 40 steps, 2.73 s wall clock. The checkpoint loads and
the upstream benchmark drives it end to end.

## 6. Synthetic self-test (not a real experiment)

`a3opt selftest --experiments 6` runs the same loop with the deterministic synthetic backend. It is
used by the test suite and by anyone who wants to check the orchestration without a simulator.

Result: 6 experiments, best candidate `tiny_cnn / concat / chunk-k50`, mean normalized reward 0.7,
two candidates accepted (`chunk-k50`, `lr 3e-3`) and three rejected. These numbers are synthetic and
must not be read as robot results; they demonstrate that the UCT selection, the paired-seed gate, and
the report generation are wired together correctly.
