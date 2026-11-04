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

## 6. Second stage: multi-camera BC2 backend (best policy so far)

The first-stage policy used one front camera, 64x64 observations, 400 training steps, and six
episodes. Its closed-loop score was zero. The second stage adds observations and fixes a policy
inference defect.

### 6.1 Additional demonstrations

```bash
python scripts/record_multicam.py --episodes 20 --seed-start 1000 --image-size 128
```

| Item | Value |
| --- | --- |
| Episodes | 20 (seeds 1000-1019) |
| Frames | 8178 |
| Observations | front + left wrist + right wrist, 128x128 each |
| Labels | 16D joint state, 16D applied joint target |
| Collection wall clock | 1314.7 s |
| Dataset | `artifacts/datasets/expert_2x5_multicam_20ep.npz` |
| Expert outcomes | 10/10, success, all twenty episodes |

### 6.2 Training

`python scripts/bc2_experiment.py --name bc2-baseline-64 --train-steps 6000 --image-size 64 ...`

| Item | Value |
| --- | --- |
| Model | 3 camera encoders + state/previous-action/step MLP, concat fusion, 20-action chunk head |
| Training | 6000 steps, batch 64, AdamW lr 1e-3, CPU |
| Loss | 0.651 -> 0.02295 (last-10 mean 0.01959) |
| Offline normalized MAE | 0.0185 |
| Training wall clock | 221.5 s |

### 6.3 Closed-loop result

Protocol: seeds 2000 and 2001, `max_eval_steps = 500`, execution horizon 8 of 20 predicted actions.

| Seed | Score | Steps | Success |
| --- | --- | --- | --- |
| 2000 | 2/10 | 500 | false |
| 2001 | 5/10 | 500 | false |
| Mean | 3.5/10 | 500 | false |

This is the first non-zero real closed-loop result of the project. The policy places two and five
cookies respectively; it does not complete the full ten-cookie task within 500 steps, so the
acceptance gate still treats it as a baseline rather than as a solved task.

### 6.4 Defect found and fixed during this stage

The first BC2 attempt scored zero on both seeds even though offline MAE was ~0.018. Diagnosis
showed the deployed policy fed the previous action back in raw joint units while the training data
used normalized units, so every prediction after the first chunk was out of distribution and the
policy collapsed to a fixed pose without touching a cookie. The fix normalizes the stored previous
action; after retraining, the same architecture scores 2/10 and 5/10. The defect is covered by
`tests/test_bc2.py`.

### 6.5 Agent loop with the BC2 backend

Command:

```bash
a3opt run --backend local_sim_v2 --experiments 3 --train-steps 4000   --eval-seeds 2000,2001 --max-eval-steps 500 --run-id bc2-loop-500
```

Protocol: 4000 training steps per candidate, seeds 2000/2001, `max_eval_steps = 500`, acceptance at
its defaults (gain >= 1 cookie, non-worse >= 70%, cost <= 3x baseline).

| Node | Candidate | Parent | Final loss | Scores (seed 2000, 2001) | Mean | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| root | tiny_cnn / concat / chunk-k20 | - | 0.0201 | 5, 2 | **3.5/10** | baseline |
| n001 | tiny_cnn / film / chunk-k20 | root | 0.0234 | 5, 0 | 2.5/10 | reject |
| n002 | state_mlp / concat / chunk-k20 | n001 | 0.0227 | 0, 5 | 2.5/10 | reject |

Budget actually spent: 3 experiments, 12000 training steps, 693.04 s wall clock. Artifacts:
`runs/bc2-loop-500/{run.json,tree.json,memory.jsonl,report.md,outcomes/}`.

The framework found the multi-camera concat candidate to be the best; the film variant and the
state-only variant each scored 2.5/10 and were rejected by the paired-seed gate. The best model
places cookies in the target box on both held-out seeds (5/10 and 2/10, mean 3.5/10), but does not
complete the ten-cookie task within 500 steps.

A separate 6000-step run of the same best configuration scored 2/10 and 5/10, i.e. the same mean
3.5/10 with the seeds swapped. The seed-level variance is reported rather than hidden.

## 7. Third stage: 40-episode data and inference tuning (current best policy)

### 7.1 Additional demonstrations

A second recording block added 20 episodes with the same protocol (seeds 1020-1039, 8201 frames,
all 10/10 success), and the two blocks were merged:

| Item | Value |
| --- | --- |
| Episodes | 40 (seeds 1000-1039) |
| Frames | 16379 |
| Observations | front + left wrist + right wrist, 128x128 each |
| Dataset | `artifacts/datasets/expert_2x5_multicam_40ep.npz` (694 MB) |
| Collection wall clock | 1314.7 s + 1346.7 s |

### 7.2 Training

```bash
python scripts/bc2_experiment.py --name bc2-40ep-64   --dataset artifacts/datasets/expert_2x5_multicam_40ep.npz   --experiment-root artifacts/local_sim_v2_40ep   --train-steps 10000 --image-size 64 --execution-horizon 12
```

| Item | Value |
| --- | --- |
| Model | same BC2 architecture (3 camera encoders, concat, chunk 20) |
| Training | 10000 steps, batch 64, AdamW lr 1e-3, CPU |
| Loss | 0.7703 -> 0.01695 (last-10 mean 0.01603) |
| Offline normalized MAE | 0.01645 |
| Training wall clock | 387.9 s |

### 7.3 Inference tuning on the development block

Execution horizon was selected on development seeds 2000-2003 (600-step budget, real MuJoCo):

| Execution horizon | Seed scores (2000/2001/2002/2003) | Mean |
| --- | --- | --- |
| 8 | 3, 5, 5, 5 | 4.5 |
| **12** | **5, 10 (success), 5, 5** | **6.25** |
| 16 | 5, 10 (success), 5, 0 | 5.0 |

Horizon 12 was selected on this block; the selected policy is
`artifacts/local_sim_v2_40ep/bc2-40ep-64/checkpoint`.

### 7.4 Final evaluation

| Seed block | Scores | Mean |
| --- | --- | --- |
| Development, seeds 2000-2005 | 5, **10 (success, 408 steps)**, 5, 5, 5, 5 | **5.83/10** |
| Reserved confirmation, seeds 3000-3001 | 5, 0 | 2.5/10 |

The development-block mean exceeds the target of 5/10, with one full 10/10 success. The reserved
confirmation block is reported unchanged (one 5/10, one failure); it shows that the policy is not
yet uniformly reliable across layouts. No tuning was performed on the confirmation seeds.

### 7.5 Ensemble experiments (not used for the final policy)

Action averaging across two independently trained BC2 checkpoints (seeds 0 and 1, 10000 and 12000
steps) was implemented and evaluated:

| Attempt | Seeds | Scores | Mean |
| --- | --- | --- | --- |
| raw averaging wrapper | 2000-2003 | 0, 0, 0, 0 | 0.0 |
| after fixing previous-action feedback in the wrapper | 2000, 2001 | 0, 10 (success) | 5.0 |

The first run exposed a defect in the averaging wrapper (the previous action was not fed back in
normalized units); after the fix the ensemble completes one episode but still fails the other, and
its two-seed mean (5.0) is not better than the single model (5.0 and 10.0 on the same seeds, mean
7.5). Action averaging is therefore not used for the final policy; the ensemble wrapper and the
second checkpoint remain in the repository for future work.

### 7.6 Progress of the best real policy

| Stage | Data | Training | Evaluation budget | Mean score |
| --- | --- | --- | --- | --- |
| first baseline | 6 episodes, front camera 64px | 400 steps | 120 / 500 steps | 0/10 |
| second stage | 20 episodes, 3 cameras 128px | 6000 steps | 500-600 steps, horizon 8 | 3.5/10 |
| third stage | 40 episodes, 3 cameras 128px | 10000 steps | 600 steps, horizon 12 | **5.83/10** (dev 2000-2005) |

## 8. Synthetic self-test (not a real experiment)

`a3opt selftest --experiments 6` runs the same loop with the deterministic synthetic backend. It is
used by the test suite and by anyone who wants to check the orchestration without a simulator.

Result: 6 experiments, best candidate `tiny_cnn / concat / chunk-k50`, mean normalized reward 0.7,
two candidates accepted (`chunk-k50`, `lr 3e-3`) and three rejected. These numbers are synthetic and
must not be read as robot results; they demonstrate that the UCT selection, the paired-seed gate, and
the report generation are wired together correctly.
