# A3 Agent Optimizer —— AI 对接文档（Handover）

> 本文档写给**接手本项目的另一个 AI Agent**。目标：让你在不重新探索的情况下，5 分钟内知道
> "东西在哪、现在是什么状态、怎么跑、做到哪了、下一步做什么、哪里会踩坑"。
>
> 生成时间：2026-11-19（北京时间）。项目根目录：`D:\AI305-Fall2026\project\a3_agent_optimizer`。
> 最后提交：`eb0ba66`（main 分支，工作区干净）。

---

## 0. TL;DR（先读这 10 行）

- 项目是**课程中期项目**：在 A3 双臂 MuJoCo 仿真平台上，实现"**Agent 驱动机器人策略实验搜索**"的框架 `a3opt`，并用真实仿真数据训练/评估策略。
- 代码仓库：`D:\AI305-Fall2026\project\a3_agent_optimizer`（本地 git，5 个提交，已推送到私有 GitHub 仓库）。
- **当前最优策略分数：5.83/10**（6 个开发种子平均，含一次完整 10/10 成功；seeds 2000-2005，600 步预算）。
- 最优配置：`tiny_cnn / concat 融合 / chunk-20 / lr=1e-3 / img64` + 三路相机数据（128px 原始，模型输入 64），推理执行 horizon=12，训练 10000 步。
- 最优 checkpoint：`artifacts/local_sim_v2_40ep/bc2-40ep-64/checkpoint`。
- 实验树运行：`runs/final-tree-60ep`（3 节点：root 5.0 / n001 4.5 / n002 2.5，后两个被验收门拒绝）。
- 官方 ACT / SmolVLA 管线**未能跑通**（lerobot 版本/网络/环境限制，详见 §10），但已实现 ACT 结构重实现，分数 1.5/10，低于 BC 基线。
- 测试：`82 passed, 1 skipped`。核心入口：`src/a3opt/cli.py`（`a3opt` 命令）。
- **不要**把 `artifacts/`（2.8 GB 数据集/权重）提交进 git；**不要**把 GitHub token 写进任何文件。
- 中期报告/PPT 有一部分在 `D:\AI305-Fall2026\project` 与 `期中汇报最终版`，Desktop 上的 docx 已丢失（§13）。

---

## 1. 项目背景（为什么做这个）

课程要求（中期）：
- 熟悉 A3 双臂仿真项目；
- 调研 ≥3 个不同方向的机器人策略模型（报告里选了 ACT / Diffusion Policy / SmolVLA）；
- 调研 ≥2 种 Agent 使用方式（ReAct / SWE-agent / MLAgentBench / AIDE / RoboPhD / TREX）；
- 给出 **Agent 驱动的优化方案**（搜索空间、工具、失败分析、实验记忆、预算、人机分工）；
- Bonus：After-feedback 继续优化、多维度评估。

我的角色分工（用户设定）：**只做 Agent 层**，仿真/评估平台由老师提供且保持冻结。Agent 的"一次实验"= 一次训练 + 一次评估（一个 checkpoint）。

因此本仓库实现的是**实验搜索层**，而不是重新实现仿真平台。上游仿真仓库：
`D:\AI305-Fall2026\project\a3_dual_arm_sim`（commit `5fa2497`，勿修改，只读使用）。

---

## 2. 路径地图（所有关键位置）

| 类别 | 路径 |
| --- | --- |
| 主仓库 | `D:\AI305-Fall2026\project\a3_agent_optimizer` |
| 上游仿真仓库 | `D:\AI305-Fall2026\project\a3_dual_arm_sim` |
| 上游 venv（CPU torch，仿真/评估/测试用） | `D:\AI305-Fall2026\project\a3_dual_arm_sim\.venv\Scripts\python.exe` |
| GPU 环境（conda，torch2.5.1+cu124） | `D:\conda_envs\recommender\python.exe` |
| 文档/PDF 工具链 Python | `D:\Python313\python.exe` |
| md→PDF 脚本 | `D:\AI305-Fall2026\project\_build\md2pdf.py` |
| 数据集/权重（不入 git） | `D:\AI305-Fall2026\project\a3_agent_optimizer\artifacts` |
| 实验树运行记录 | `D:\AI305-Fall2026\project\a3_agent_optimizer\runs` |
| 交付 zip | `D:\AI305-Fall2026\project\a3_agent_optimizer_delivery.zip`（82.9 MB） |
| git bundle | `D:\AI305-Fall2026\project\a3_agent_optimizer.bundle` |
| GitHub | `https://github.com/haiyouxiaji/a3-agent-optimizer`（private） |
| 中期报告 PDF（合并版） | `D:\AI305-Fall2026\project\期中汇报最终版\midterm_report_merged.pdf` |
| 最终 PPT | `D:\AI305-Fall2026\project\A3机器人项目中期汇报_南科大模板_最终版.pptx` |
| PPT 预览 PDF | `D:\AI305-Fall2026\project\A3机器人项目中期汇报_南科大模板_最终版_公式版预览.pdf` |
| 架构/实验树报告 PDF | `D:\AI305-Fall2026\project\a3_agent_optimizer\docs\A3_Agent_优化框架_架构与实验树报告.pdf` |
| 实验结果详细记录 | `D:\AI305-Fall2026\project\a3_agent_optimizer\docs\results.md` |

---

## 3. 当前完成度一览

| 事项 | 状态 | 证据/位置 |
| --- | --- | --- |
| Agent 优化框架 `a3opt` | ✅ 完成 | `src/a3opt/`，37+ 文件 |
| 单元/端到端测试 | ✅ 82 passed, 1 skipped | `pytest tests -q` |
| 真实专家数据采集 | ✅ 60 episodes / 24553 帧 | `artifacts/datasets/expert_2x5_multicam_60ep.npz`（1.0 GB） |
| 真实训练（BC2 多相机） | ✅ 多个 checkpoint | `artifacts/local_sim_v2*` |
| 真实闭环评估 | ✅ 最优 5.83/10（6 种子） | `artifacts/local_sim_v2_40ep/final_6seeds.json` |
| Agent 实验树（UCT+验收+记忆） | ✅ 3 节点真实运行 | `runs/final-tree-60ep/` |
| 架构/树报告 PDF | ✅ 9 页 | `docs/A3_Agent_优化框架_架构与实验树报告.pdf` |
| ACT 结构重实现 + GPU 训练 | ✅ 1.5/10（低于 BC） | `src/a3opt/backends/act_style.py`，`artifacts/act_style` |
| 官方 ACT / SmolVLA | ❌ 环境/网络阻塞 | `docs/results.md` §9 |
| 中期报告 docx | ⚠️ Desktop 已丢失，仅 PDF 尚存 | 见 §13 |
| GitHub 上传 | ✅ 私有仓库，5 commits | 见 §12 |

git 提交历史（main）：

```
eb0ba66 Add ACT-style policy backend and document ACT/SmolVLA feasibility
8ac3290 Add search-tree run, report generator, and architecture/tree PDF
1ea4317 Tune BC2 to a 5.83/10 real closed-loop policy
4ffa547 Add multi-camera BC2 backend and real agent-loop results
d105ab5 Implement agent-driven modular policy optimization loop
```

---

## 4. 代码架构

### 4.1 三层决策（设计核心）

| 层 | 决定什么 | 实现 |
| --- | --- | --- |
| 部署层 | 每个控制周期输出 16 维关节目标 | 训练出的策略（BC2 / ACT-style） |
| 训练层 | 如何从示范数据学习权重 | `local_sim_v2` 后端（torch） |
| 实验搜索层 | 训练哪个候选、是否保留、下一步试什么 | `a3opt`（树+UCT+验收+记忆+预算） |

### 4.2 包结构（`src/a3opt/`）

| 文件 | 作用 |
| --- | --- |
| `contracts.py` | 数据契约：CandidateSpec、ExperimentBudget、RunBudget、TrainResult、EvalResult、ExperimentOutcome、RunLayout、OptimizationSummary |
| `space.py` | 候选空间：encoder × fusion × action_head × chunk_size × image_size × lr；`validate()` 兼容性规则；`staged_candidates()` 6 个分阶段候选 |
| `tree.py` | 实验树：`ExperimentTree.select()`（UCT）、`add_child()`、`backpropagate()`、`best()`；节点字段见 §9 |
| `acceptance.py` | 配对种子验收门：平均增益 ≥1 块、≥70% 种子不退化、成本 ≤3×，三态 accept/reject/insufficient |
| `memory.py` | 只追加 JSONL 实验记忆：`ExperimentMemory.record_from_outcome()`、best(k)、summary、condense |
| `budget.py`（无，逻辑在 contracts） | RunBudget 在 `contracts.py` |
| `planner.py` | `StagedPlanner.propose()` 按阶段返回下一个候选 |
| `loop.py` | 主循环：root → select → propose → train → eval → compare → backprop → memory → report |
| `report.py` | 生成 run 的 markdown 报告（树表/验收表/记忆表/预算） |
| `cli.py` | `a3opt plan / run / report / selftest`（argparse） |
| `backends/stub.py` | 合成后端：确定性分数，给测试/演示用 |
| `backends/local_sim.py` | 第一代仿真后端：单相机 BC（front 64），已被 v2 取代但保留 |
| `backends/bc2.py` | **主力后端**：三相机 BC（128px 数据→64 输入）+ 动作/状态归一化 + 上一动作/步数条件；`LocalSimV2Backend` |
| `backends/act_style.py` | ACT 结构重实现：3×ResNet18 + Transformer encoder/decoder + CVAE + 50-step chunk；`ACTStylePolicy` |
| `backends/a3_sim.py` | 上游 ACT/SmolVLA CLI 适配（默认 dry-run，未真跑） |
| `backends/policy_plugin.py` | 上游 benchmark CLI 的插件入口（`A3OPT_CHECKPOINT` 环境变量） |

### 4.3 实验循环（`loop.py`）

```
root(baseline) 训练+评估
  └→ loop:
       parent = tree.select()                    # UCT
       candidate = planner.propose(parent, used) # 分阶段候选
       child = tree.add_child(parent, candidate)
       outcome = backend.run(candidate, 预算, eval_seeds, max_steps)
       tree.backpropagate(child, outcome.normalized_score())
       compare_paired(baseline, candidate)       # 验收门
       memory.record_from_outcome(...)
       → 循环直到预算耗尽 / 候选用完
  → 写 run.json / tree.json / memory.jsonl / report.md
```

运行目录格式：

```
runs/<run_id>/
  run.json          # 摘要、预算、config（eval_seeds、criteria、verdicts）
  tree.json         # 树快照（扁平节点表）
  memory.jsonl      # 只追加证据
  report.md         # 自动报告
  outcomes/<candidate_id>.json   # 每个候选完整结局
  logs/
```

### 4.4 CLI 命令

```bash
a3opt plan                                   # 打印 6 个候选与合法性
a3opt selftest --experiments 6               # 合成后端跑通全流程（无需仿真）
a3opt run --backend stub --experiments 6 --out runs --run-id demo
a3opt run --backend local_sim_v2 --dataset artifacts/datasets/expert_2x5_multicam_60ep.npz \
  --experiment-root artifacts/local_sim_v2_60ep --experiments 3 --train-steps 8000 \
  --batch-size 64 --eval-seeds 2000,2001 --max-eval-steps 600 --out runs --run-id final-tree-60ep
a3opt report --run runs/final-tree-60ep      # 重新生成 report.md
```

---

## 5. 环境与依赖（重要：三个 Python 分工不同）

| 环境 | 路径 | 版本/能力 | 用途 |
| --- | --- | --- | --- |
| 上游 venv（主力） | `D:\AI305-Fall2026\project\a3_dual_arm_sim\.venv\Scripts\python.exe` | Python 3.13.12，torch **2.15.0+cpu**，numpy 2.5.4，mujoco 3.16.0，pytest 9.2.0 | 仿真、BC2 训练（CPU）、闭环评估、pytest、CLI |
| GPU conda 环境 | `D:\conda_envs\recommender\python.exe` | Python 3.10.21，torch **2.5.1+cu124**，CUDA 可用（RTX 4090 Laptop 16 GB，driver 610.88） | ACT-style GPU 训练（本机唯一 GPU 环境） |
| 文档工具 | `D:\Python313\python.exe` | Python 3.13，含 markdown + pymupdf | 生成 md→PDF（`_build/md2pdf.py`） |

注意：
- 主力 venv 是 **CPU-only torch**，跑 4090 不行；GPU 训练必须用 conda 环境。
- 不要把 pip 装到 `D:\AI305-Fall2026\project\a3_dual_arm_sim\.venv` 里换 CUDA torch——会破坏仿真和现有评估环境。
- 安装包建议用清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`（实测 4 MB/s；官方 PyPI/HuggingFace 只有 0.5–1.2 MB/s）。

### 5.1 环境变量 / PYTHONPATH

在仓库根目录运行任何脚本前：

```powershell
$env:PYTHONPATH = "src;D:\AI305-Fall2026\project\a3_dual_arm_sim\src"
```

大多数脚本（`scripts/*.py`）已经自己把这两个路径插进 `sys.path`，但 `a3opt` 命令行或 pytest 建议显式设置。

---

## 6. 数据与产物清单

### 6.1 数据集（`artifacts/datasets/`，**不入 git**）

| 文件 | 内容 | 大小 |
| --- | --- | --- |
| `expert_2x5_basic_6ep.npz` | 第一代单相机 64px，6 episodes（seeds 1000-1005），2450 帧 | 2.6 MB |
| `expert_2x5_multicam_20ep.npz` | 多相机 128px，20 episodes（1000-1019），8178 帧 | 347 MB |
| `expert_2x5_multicam_20b.npz` | 多相机，20 episodes（1020-1039），8201 帧 | 348 MB |
| `expert_2x5_multicam_20c.npz` | 多相机，20 episodes（1040-1059），8174 帧 | 347 MB |
| `expert_2x5_multicam_40ep.npz` | 40 episodes（1000-1039），16379 帧 | 694 MB |
| `expert_2x5_multicam_60ep.npz` | **60 episodes（1000-1059），24553 帧** | 1.04 GB |

npz 键：`front / left_wrist / right_wrist`（uint8 N×128×128×3）、`state`（float32 N×16）、
`action`（float32 N×16，env 实际应用的关节目标）、`seeds`（int32 N）。
元数据在同名 `.meta.json`（格式 `a3opt-expert-multicam-v1`，含 episode_reports）。

### 6.2 主要 checkpoint

| checkpoint 目录 | 训练 | 用途 |
| --- | --- | --- |
| `artifacts/local_sim_v2_40ep/bc2-40ep-64/checkpoint` | 40ep，10000 步，img64 | **当前最优 5.83/10** |
| `artifacts/local_sim_v2_40ep/bc2-40ep-64-20k/checkpoint` | 40ep，20000 步 | 对照（未提升） |
| `artifacts/local_sim_v2_60ep/bc2-60ep-64-16k/checkpoint` | 60ep，16000 步 | 对照（3.33/10，回归） |
| `artifacts/local_sim_v2_60ep/bc_policy-*`（3 个） | 60ep，8000 步 | 实验树 3 节点 |
| `artifacts/act_style/act_60ep/model.pt` | ACT-style，8000 步 GPU | ACT 尝试（1.5/10） |

checkpoint 结构（BC2）：`model.pt`（torch state_dict）+ `spec.json`（候选、归一化 stats、架构、训练指标）。

### 6.3 运行记录（`runs/`）

| 目录 | 内容 | 结果 |
| --- | --- | --- |
| `runs/real-loop-500` | 第一代 local_sim 循环（20ep，4000 步） | 3 节点全 0，2 reject |
| `runs/bc2-loop-500` | BC2 循环（20ep，4000 步） | baseline 3.5，film/state 2.5 reject |
| `runs/final-tree-60ep` | **最终树（60ep，8000 步）** | root 5.0 / n001 4.5 / n002 2.5，2 reject |

### 6.4 其他产物

- `artifacts/local_sim_v2_40ep/final_6seeds.json` —— 最优模型 6 种子评估（5.83）
- `artifacts/local_sim_v2_40ep/horizon_sweep_4seeds.json` —— horizon 8/12/16 选型
- `artifacts/local_sim_v2_60ep/final_16k_6seeds.json` —— 16k 回归结果（3.33）
- `artifacts/act_style/eval_60ep_2seeds.json` —— ACT-style 评估（3/0）
- `artifacts/real_runs/summary.json` —— 第一轮真实训练/评估总表（含环境、协议）

---

## 7. 真实实验结果（可直接引用）

评估协议：2×5 饼干装盒任务，`basic` 随机化，source column 1，真实 MuJoCo 闭环；
种子 2000-2005（开发块），每 episode 最多 600 步；分数 = 目标盒中饼干数（0-10，10 需完全成功条件）。

### 7.1 三个阶段的分数演进

| 阶段 | 数据/训练 | 评估 | 平均分 |
| --- | --- | --- | --- |
| 第一版（local_sim，单相机 64px） | 6 eps / 400 步 | seeds 2000-2001，120/500 步 | **0/10** |
| 第二版（BC2 多相机 20ep） | 20 eps / 6000 步 | seed 2000/2001，500-600 步 | **3.5/10** |
| 第三版（BC2 多相机 40ep） | 40 eps / 10000 步，horizon 12 | seeds 2000-2005，600 步 | **5.83/10** |

### 7.2 最终模型逐种子

| seed | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 均值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| score | 5 | **10（成功，408 步）** | 5 | 5 | 5 | 5 | **5.83** |

确认种子块 3000/3001（未参与调参）：5、0（均值 2.5，说明鲁棒性不足，需如实报告）。

### 7.3 实验树（`runs/final-tree-60ep`）

| 节点 | 候选 | 父 | 训练步 | 种子 2000/2001 | 均分 | 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| root | tiny_cnn/concat/chunk-k20/lr1e-3/img64 | - | 8000 | 5, 5 | **5.0** | baseline |
| n001 | tiny_cnn/film/chunk-k20/lr1e-3/img64 | root | 8000 | 5, 4 | 4.5 | reject |
| n002 | state_mlp/concat/chunk-k20/img0 | n001 | 8000 | 0, 5 | 2.5 | reject |

预算：3 实验 / 24000 训练步 / 1201.8 s；验收门 accepted=[]，rejected=2。

### 7.4 其他尝试（全部为真实运行）

| 尝试 | 结果 |
| --- | --- |
| horizon 8 / 12 / 16（40ep 模型，seeds 2000-2003） | 4.5 / **6.25** / 5.0 → 选 12 |
| 60ep 16000 步 | 3.33/10（比 40ep 10k 更差） |
| 两模型 action 平均 ensemble | 两种子 5.0/10，不超过单模型 |
| 时序 ensemble（coefficient 0.01） | 2.5/10，更差 |
| ACT-style（ResNet18+Transformer+CVAE，8000 步 GPU） | 3/0 → **1.5/10** |
| 官方 ACT dry-run | 因缺 4.1 GB 数据集，在 audit 处 FileNotFoundError |

---

## 8. 复现/继续实验的命令手册

### 8.1 自检（最先跑）

```powershell
cd D:\AI305-Fall2026\project\a3_agent_optimizer
$env:PYTHONPATH = "src;D:\AI305-Fall2026\project\a3_dual_arm_sim\src"
& "D:\AI305-Fall2026\project\a3_dual_arm_sim\.venv\Scripts\python.exe" -m pytest tests -q
# 期望：82 passed, 1 skipped
& "D:\AI305-Fall2026\project\a3_dual_arm_sim\.venv\Scripts\python.exe" -c "from a3opt.cli import main; import sys; sys.exit(main(['selftest','--experiments','6']))"
```

### 8.2 复现最优模型训练（CPU，约 6.5 分钟）

```powershell
cd D:\AI305-Fall2026\project\a3_agent_optimizer
$env:PYTHONPATH = "src;D:\AI305-Fall2026\project\a3_dual_arm_sim\src"
& "D:\AI305-Fall2026\project\a3_dual_arm_sim\.venv\Scripts\python.exe" scripts\bc2_experiment.py `
  --name bc2-40ep-64 --dataset artifacts\datasets\expert_2x5_multicam_40ep.npz `
  --experiment-root artifacts\local_sim_v2_40ep --seed 0 --train-steps 10000 `
  --image-size 64 --eval-seeds 2000,2001 --max-eval-steps 600 --execution-horizon 12
```

注意：若只想评估、不要重训，不要用 1 步训练覆盖 checkpoint（会覆盖原模型！）；直接用 `scripts/sweep_horizon.py`。

### 8.3 复现最优评估（6 种子，约 8-10 分钟）

```powershell
& "D:\AI305-Fall2026\project\a3_dual_arm_sim\.venv\Scripts\python.exe" scripts\sweep_horizon.py `
  --checkpoint artifacts\local_sim_v2_40ep\bc2-40ep-64\checkpoint `
  --horizons 12 --seeds 2000,2001,2002,2003,2004,2005 --max-steps 600 `
  --out artifacts\local_sim_v2_40ep\final_6seeds.json
```

### 8.4 跑实验搜索树

```powershell
& "D:\AI305-Fall2026\project\a3_dual_arm_sim\.venv\Scripts\a3opt.exe" run `
  --backend local_sim_v2 --dataset artifacts\datasets\expert_2x5_multicam_60ep.npz `
  --experiment-root artifacts\local_sim_v2_60ep --experiments 3 --train-steps 8000 `
  --batch-size 64 --eval-seeds 2000,2001 --max-eval-steps 600 --out runs --run-id new-tree
```

### 8.5 采集更多专家数据（约 1.3 分钟/episode）

```powershell
& "...\.venv\Scripts\python.exe" scripts\record_multicam.py `
  --dataset artifacts\datasets\expert_2x5_multicam_20d.npz --episodes 20 --seed-start 1060 --image-size 128
& "...\.venv\Scripts\python.exe" scripts\merge_datasets.py `
  --inputs artifacts\datasets\expert_2x5_multicam_60ep.npz artifacts\datasets\expert_2x5_multicam_20d.npz `
  --output artifacts\datasets\expert_2x5_multicam_80ep.npz
```

### 8.6 训练 ACT-style（GPU，需 conda 环境）

```powershell
cd D:\AI305-Fall2026\project\a3_agent_optimizer
D:\conda_envs\recommender\python.exe scripts\train_act_style.py `
  --dataset artifacts\datasets\expert_2x5_multicam_60ep.npz --output artifacts\act_style\act_60ep_20k `
  --steps 20000 --batch-size 16 --device cuda
D:\AI305-Fall2026\project\a3_dual_arm_sim\.venv\Scripts\python.exe scripts\eval_act_style.py `
  --checkpoint artifacts\act_style\act_60ep_20k\model.pt --seeds 2000,2001 --max-steps 600
```

### 8.7 生成"架构 + 实验树"PDF 报告

```powershell
cd D:\AI305-Fall2026\project\a3_agent_optimizer
& "D:\Python313\python.exe" scripts\build_tree_report.py --run runs\final-tree-60ep `
  --out "docs\A3_Agent_优化框架_架构与实验树报告.md" `
  --final-eval artifacts\local_sim_v2_40ep\final_6seeds.json
& "D:\Python313\python.exe" "D:\AI305-Fall2026\project\_build\md2pdf.py" `
  "docs\A3_Agent_优化框架_架构与实验树报告.md" "docs\A3_Agent_优化框架_架构与实验树报告.pdf" `
  "A3 Agent 优化框架 · 架构与实验树报告"
```

---

## 9. 实验树/tree.json 结构（扩展时必须知道）

`runs/<run_id>/tree.json` 是**扁平节点表**（不是嵌套 JSON）：

```json
{
  "root_id": "root",
  "c": 0.7,
  "max_children": 3,
  "nodes": [
    {
      "node_id": "root",
      "parent_id": null,
      "candidate": {"candidate_id": "...", "encoder": "tiny_cnn", "fusion": "concat",
                     "action_head": "chunk", "chunk_size": 20, "lr": 0.001,
                     "image_size": 64, "state_dim": 16, "seed": 0},
      "children": ["n001"],
      "visits": 3,
      "total_value": 1.2,
      "stage": "",
      "status": "ok",
      "outcome": {"train": {...}, "eval": {"mean_score": 5.0, "episodes": [...]}, "cost_wall_seconds": 512.1},
      "acceptance": null
    }
  ]
}
```

注意：
- `root` 的 `visits`/`total_value` 会被 UCT 回传累计，**不代表 root 自身分数**；报告展示用 `node.outcome.eval.mean_score`（loop 已修正 best 选取）。
- `acceptance` 仅出现在非 root 节点（配对验收结果）。
- 扩展候选空间时改 `space.py`；扩展后端时实现 `ExperimentBackend` 协议（`name`、`available()`、`run()`）。

---

## 10. 已知问题 / 坑（务必先读）

### 10.1 官方 ACT / SmolVLA 在本机跑不通（实测）

| 阻塞 | 具体 |
| --- | --- |
| lerobot 版本 | `lerobot==0.5.1` 要求 Python ≥3.12 + torch ≥2.7；本机唯一 GPU 环境是 conda Python 3.10 + torch 2.5.1（import 报 PEP 695 `class Backtrackable[T]` SyntaxError） |
| Python 3.13 环境 | 只有 CPU torch 2.15（`a3-sim train-act` 会退化为 CPU 训练） |
| 数据集 | `Eter0109/a3-front-close-left-100` 共 4.1 GB；HF 实测 0.5 MB/s → 约 2.3 小时 |
| CUDA torch（py3.13） | cu128 wheel 2.7 GB；aliyun 镜像 1.2 MB/s → 约 38 分钟 |
| ResNet18 ImageNet 权重 | torchvision 下载 44.7 MB 实测 24 kB/s，被截断导致 hash 校验失败；ACT-style 20k 预训练版因此中止 |
| SmolVLA | 还需要 `smolvla_base` + VLM 权重（数 GB），450M 参数模型 CPU 训练不现实 |

官方 CLI 探针结果：`a3-sim train-act --dry-run` 报 `FileNotFoundError: Missing LeRobot metadata`。
在具备"Python 3.12 + CUDA torch ≥2.7 + 快速网络"的机器上，可继续：
`a3-sim download-smolvla` → `a3-sim train-act` / `a3-sim train-smolvla` → 用 `benchmark_cookie_batch_cli --policy act:<ckpt>` 评估。

### 10.2 Windows 环境

- 仓库路径含中文用户名：`C:\Users\ZhuanZ（无密码）`，部分工具（git 自带 ssh）会因路径编码失败。
- 本机 `git push` 走 SSH 会报 `Host key verification failed`（git 调用的 Windows OpenSSH 拿到坏 HOME 路径）；
  直接用 `ssh -T git@github.com` 是通的，但 git 不走它。上传时用 **HTTPS + token** 一次性推送（见 §12）。
- 数据采集/训练使用 CPU 多线程：两个训练并行会互相抢 CPU；评估时的 MuJoCo 渲染约 0.14 s/step。

### 10.3 实验本身

- 分数对种子敏感（2001 可 10/10，2004/2005 只 5/10；确认种子 3000/3001 = 5/0）；不要用单种子下结论。
- 600 步预算才合理：专家约 410 步完成；120 步预算下连专家都是 0 分。
- `artifacts/` 共 ~2.8 GB，**已被 .gitignore 排除**；clone 仓库后需重新运行采集/训练，或从交付 zip 解压。
- 训练 checkpoint 路径覆盖风险：`scripts/bc2_experiment.py` 会用同名目录重训并覆盖；评估旧模型时用 `sweep_horizon.py` 只读 checkpoint。
- `--eval-seeds` 必须是连续种子（benchmark 用 seed_start + episodes）。

---

## 11. 未完成任务 / 下一步建议（按优先级）

1. **官方 ACT（更靠谱的下一步）**
   - 需要：Python 3.12 环境 + CUDA torch ≥2.7（2.7 GB）+ 4.1 GB 数据集（最好用快网/预下载）。
   - 命令：`a3-sim download-smolvla`、`a3-sim train-act --root datasets/a3_front_close_left_100 --steps 20000 --device cuda`。
   - 评估：`benchmark_cookie_batch_cli --policy act:<checkpoint> --dataset-root ... --episodes 2 --seed-start 2000 --max-steps 600`。
   - 预期：官方 ACT 20k 步训练若成功，分数大概率超过 BC 的 5.83。
2. **SmolVLA**：需要额外 base/VLM 权重（数 GB）与 GPU 内存；本机 4090 16GB 可尝试 batch 1 冻结 VLM 微调，但下载是主要瓶颈。
3. **提升 BC2 鲁棒性**：更多数据（80-120 eps）、更长训练、DAgger（用专家纠正策略状态）、种子集成，目标把 6 种子均值稳定到 6-7+ 且确认种子 >5。
4. **实验搜索扩展**：把"数据规模、训练步数、执行 horizon"纳入候选空间（目前只搜索模块组合）；把 `local_sim_v2` 的评估种子扩到 4-6 个以降低验收噪声（成本会上升）。
5. **报告/PPT 收尾**：见 §13；把仓库更新（ACT 尝试、最优分数）同步进报告。
6. **GitHub**：本地 commit 已推送；如需继续推送，用 HTTPS + token（见 §12）。

---

## 12. GitHub 仓库状态与上传方式

- 仓库：`https://github.com/haiyouxiaji/a3-agent-optimizer`（**private**，8 个仓库现已全部私有）
- 分支：`main`；已推送 5 个提交（截至 `eb0ba66`）
- 本地 remote：`https://github.com/haiyouxiaji/a3-agent-optimizer.git`（不含 token）
- 上传方式：该机 SSH push 不可用（§10.2），用 HTTPS + Personal Access Token：

```powershell
cd D:\AI305-Fall2026\project\a3_agent_optimizer
$token = "ghp_xxx"   # 不要提交/写入文件
git push "https://haiyouxiaji:$token@github.com/haiyouxiaji/a3-agent-optimizer.git" main
```

- token 权限：classic token 勾 `repo` 即可；用完请立刻在 GitHub Settings → Developer settings → Tokens 删除。
- **绝对不要**把 token 写进 README、脚本、git config 或提交记录。
- 若想公开仓库：repo → Settings → General → Danger Zone → Change visibility。
- 若想改 SSH：把 `core.sshCommand` 指向 `C:\Program Files\Git\usr\bin\ssh.exe`，并检查 known_hosts/密钥路径（当前 HOME 路径含中文，容易出现编码问题）。

---

## 13. 课程交付物状态（中期报告 / PPT）

| 文件 | 状态 |
| --- | --- |
| `D:\AI305-Fall2026\project\期中汇报最终版\midterm_report_merged.pdf` | ✅ 存在（250 KB，12 页，含封面/两作者） |
| `D:\AI305-Fall2026\project\A3机器人项目中期汇报_南科大模板_最终版.pptx` | ✅ 存在（1.8 MB，47 页，公式已修正） |
| `D:\AI305-Fall2026\project\A3机器人项目中期汇报_南科大模板_最终版_公式版预览.pdf` | ✅ 存在 |
| `Midterm_report_merged.docx`（可编辑 Word） | ⚠️ **桌面已丢失**（2026-11 后桌面只剩 2 个文件）；如需修改报告，可从 PDF 重建或由用户找回 |
| `docs/A3_Agent_优化框架_架构与实验树报告.pdf` | ✅ 存在（9 页，架构+树+节点分数） |

报告要点（若需要改报告）：
- 中期报告章节：1 项目理解与初步执行（含 1.5 随机 benchmark 证据，SHA-256 已核验）；2 机器人策略综述（ACT/Diffusion Policy/SmolVLA + Table 2）；3 Agent 方法与优化方案（6 篇文献 + 搜索空间/工具/失败分析/记忆/预算/人机分工/多维评估 + Table 3/4 + 参考文献 [1]-[14]）。
- 最终 PPT 47 页，Agent 部分在 P29 之后；公式已统一 Cambria Math。
- 详细修改历史在对话记录与备份文件（如 `_公式修改前备份.pptx`）。

---

## 14. 给接手 AI 的约定（重要）

1. **语言**：与用户交流用中文；代码注释/文档字符串**不要写**（本会话有"无注释"输出约束，仓库代码也遵循此风格）。
2. **先验证再改**：动手前跑 §8.1 自检；改完跑 `pytest tests -q`，并至少跑一次 `a3opt selftest`。
3. **不要动上游**：`D:\AI305-Fall2026\project\a3_dual_arm_sim` 是老师给的平台，保持只读；需要新功能写在本仓库。
4. **不要提交大文件**：`artifacts/`、`runs/` 已 gitignore；数据集 2.8 GB，权重几十 MB 各自保留在本地/交付 zip。
5. **实验必须真实**：本项目的核心要求是"真实训练 + 真实闭环评估"；禁止把合成分数写成真实结果。`stub` 后端只能用于测试/演示，报告中要标注。
6. **报告分数要带种子明细**：不能只写平均值；确认种子 3000-3019 保留，不参与调参。
7. **安全**：不写入/泄露 GitHub token；不要改动用户其他私有仓库。
8. **交付物优先**：用户最终需要的是「中期报告 + PPT + 真实分数 + 可复现代码」，而不是无止境调参。若调参收益 <1 分，优先完善文档/复现性。

---

## 15. 关键文件索引

```
a3_agent_optimizer/
├── README.md                       仓库说明（安装/CLI/结果/限制）
├── pyproject.toml                  a3-agent-optimizer 包定义（console script: a3opt）
├── docs/
│   ├── architecture.md             架构（三层/循环/候选空间/UCB/验收/预算/记忆/后端/协议）
│   ├── results.md                  全部真实实验记录（含 ACT/SmolVLA 阻塞证据）
│   ├── A3_Agent_优化框架_架构与实验树报告.md/.pdf   9 页报告
│   └── example_*_report.md         示例 run 报告
├── src/a3opt/                      框架源码（见 §4.2）
├── scripts/
│   ├── record_multicam.py          三相机专家数据采集
│   ├── merge_datasets.py           合并 npz 数据集
│   ├── bc2_experiment.py           单候选训练+评估
│   ├── sweep_horizon.py            执行 horizon 扫描/单模型多种子评估
│   ├── eval_ensemble.py            多 checkpoint action 平均集成
│   ├── run_local_loop.py           真实 Agent 循环入口
│   ├── local_backend_smoke.py      第一代后端全流程（record/train/eval/backend/cli）
│   ├── train_act_style.py          ACT-style 训练（GPU）
│   ├── eval_act_style.py           ACT-style 评估
│   └── build_tree_report.py        从 runs 生成架构+树报告 md（配合 _build/md2pdf.py 出 PDF）
├── tests/                          82 passed / 1 skipped
├── artifacts/                      （gitignored）数据集/checkpoint/评估 JSON
└── runs/                           （gitignored）实验树运行记录
```

---

## 附录 A：一分钟健康检查

```powershell
cd D:\AI305-Fall2026\project\a3_agent_optimizer
git log --oneline -1                                     # 应为 eb0ba66
git status --short                                       # 应为空
Test-Path artifacts\datasets\expert_2x5_multicam_40ep.npz # True
Test-Path artifacts\local_sim_v2_40ep\bc2-40ep-64\checkpoint\model.pt # True
$env:PYTHONPATH = "src;D:\AI305-Fall2026\project\a3_dual_arm_sim\src"
& "D:\AI305-Fall2026\project\a3_dual_arm_sim\.venv\Scripts\python.exe" -m pytest tests -q  # 82 passed, 1 skipped
```

## 附录 B：关键数字速查

| 项 | 数值 |
| --- | --- |
| 任务 | 2×5 饼干装盒，basic，source column 1，16D 关节目标，20Hz |
| 专家完成 | ≈410 步（10/10 success） |
| 当前最优 | BC2 多相机，5.83/10（6 开发种子，600 步） |
| 最优 checkpoint | `artifacts/local_sim_v2_40ep/bc2-40ep-64/checkpoint` |
| 最优配置 | tiny_cnn / concat / chunk-20 / lr 1e-3 / img64 / horizon 12 / 10000 步 |
| 训练数据 | 60ep 可用；最优模型用 40ep（16379 帧） |
| 树运行 | 3 节点，24000 步，1201.8 s |
| ACT-style | 1.5/10（8k 步 GPU） |
| 测试 | 82 passed, 1 skipped |
| 仓库 | private，5 commits，`eb0ba66` |

## 附录 C：种子协议

- 专家数据采集：seeds **1000-1059**（60 个 episode）。
- 开发/筛选：seeds **2000-2019**（本报告用 2000-2005）。
- 诊断/对照：seeds **2100-2101**（早期后端验证）。
- 确认块：seeds **3000-3019**（保留，不参与调参；目前只测了 3000/3001）。
- 每次评估必须使用相同的种子集合做配对比较；`acceptance.compare_paired` 依赖逐种子配对。
