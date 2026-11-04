# A3 Agent 优化框架：架构与实验树报告

生成时间：2026-11-05 01:15　运行编号：`final-tree-60ep`

本文档由 `scripts/build_tree_report.py` 从真实运行产物自动生成：运行摘要 `run.json`、实验树 `tree.json`、评估明细 `outcomes/*.json`。报告中所有分数均为 MuJoCo 闭环评估的真实得分（0-10 分制，10 分为把 10 块饼干全部放入目标盒）。

## 一、结论摘要

| 项目 | 结果 |
| --- | --- |
| 最优候选组合 | tiny_cnn / concat / chunk-k20 / lr=0.001 / img64 |
| 最优平均分 | 5.0 / 10 |
| 候选实验节点数 | 3 |
| 训练预算消耗 | 24000 步 |
| 墙钟预算消耗 | 1201.834 s |
| 通过验收的候选 | 无（最优为基础候选） |
| 被拒绝的候选 | 2 个 |
| 调优后最佳模型（6 个开发种子复核） | 5.83 / 10（40ep 数据集，10000 步，horizon 12） |

## 二、当前架构

### 2.1 三层决策

| 层 | 决定什么 | 实现位置 |
| --- | --- | --- |
| 部署层 | 每个控制周期输出 16 维关节目标 | 训练得到的机器人策略 |
| 训练层 | 如何从示范数据学习权重 | `local_sim_v2` 后端（torch 行为克隆） |
| 实验搜索层 | 训练哪个候选、是否保留、下一步试什么 | 本仓库 `a3opt`（实验树 + UCT + 验收门 + 记忆 + 预算） |

### 2.2 组件与职责

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| 数据契约 | `src/a3opt/contracts.py` | 候选规格、预算、训练/评估结果、实验结局、运行目录 |
| 候选空间 | `src/a3opt/space.py` | 模块组合的合法性与分阶段候选列表 |
| 实验树 | `src/a3opt/tree.py` | UCT 选择、节点扩展、奖励回传、最优节点 |
| 验收门 | `src/a3opt/acceptance.py` | 配对种子的增益 / 不屈化 / 成本三条件 |
| 实验记忆 | `src/a3opt/memory.py` | 只追加 JSONL 证据 + best-k + 摘要 |
| 预算 | `src/a3opt/contracts.py` | 单实验与整轮实验的预算与停止条件 |
| 规划器 | `src/a3opt/planner.py` | 按阶段给出下一个候选 |
| 主循环 | `src/a3opt/loop.py` | 分析→假设→训练→评估→反思→记录→迭代 |
| 报告 | `src/a3opt/report.py` | 生成树表、验收表、记忆表的 Markdown 报告 |
| 后端 | `src/a3opt/backends/` | stub / local_sim / local_sim_v2 / a3_sim_cli |
| 命令行 | `src/a3opt/cli.py` | `a3opt plan / run / report / selftest` |

### 2.3 实验循环

```text
Analyze → Hypothesize → Implement → Train → Evaluate → Reflect → Iterate
```

每一轮 = 一个候选实验：指定模块组合与训练预算 → 训练 → 在固定种子块上闭环评估 → 与基线做配对比较 → 写入实验记忆 → 回到实验树选择下一个节点。

### 2.4 候选空间

| 维度 | 取值 |
| --- | --- |
| encoder | `tiny_cnn`（视觉编码器）、`state_mlp`（仅状态） |
| fusion | `concat`、`film` |
| action_head | `mlp`、`chunk` |
| chunk_size | 1、20、50 |
| image_size | 0、64 |
| learning_rate | 1e-3、3e-3 |

兼容性规则在训练前拦截非法组合：`state_mlp` 要求 `image_size=0`；`film` 要求图像输入；`chunk` 要求 `chunk_size>1`；`mlp` 要求 `chunk_size=1`。分阶段候选顺序为：baseline → 编码器/融合变体 → 动作头/chunk 变体 → 学习率变体。

### 2.5 选择、验收、预算与记忆

```text
UCT(v) = Q_v / N_v + c * sqrt( ln N_parent(v) / N_v ),  c = 0.7
```

- **验收门**：同一批种子上平均增益 ≥ 1 块饼干、≥ 70% 种子不退化、成本 ≤ 基线 3 倍，三者同时满足才 `accept`；配对种子不足两个判定为 `insufficient`。
- **预算**：整轮受实验数、训练总步数、墙钟秒数、GPU 秒数约束；单实验另有训练步数、batch、墙钟上限。
- **记忆**：每次实验结束追加一条 JSONL 记录（候选、训练指标、逐种子分数、成本、节点号），支持 best-k 查询与压缩摘要。

### 2.6 后端

| 后端 | 真实训练 | 依赖 |
| --- | --- | --- |
| `stub` | 合成分数，仅用于测试与演示 | 无 |
| `local_sim` | 单相机行为克隆 + MuJoCo 闭环 | torch + A3 仿真仓库 |
| `local_sim_v2` | 三相机行为克隆 + MuJoCo 闭环（本次结果使用） | torch + A3 仿真仓库 |
| `a3_sim_cli` | 上游 ACT/SmolVLA 命令（默认 dry-run，未在本机执行） | lerobot + A3 仿真仓库 |

### 2.7 运行目录

```text
runs/<run_id>/
  run.json          运行摘要、预算、验收阈值、verdicts
  tree.json         实验树快照（节点、父子关系、访问计数、结果）
  memory.jsonl      只追加的实验记忆
  report.md         自动生成的人类可读报告
  outcomes/<candidate_id>.json   每个候选的完整结局
  logs/
```

## 三、实验树是怎么保存的

实验树保存在 `runs/<run_id>/tree.json`。它是一个扁平节点表加父子指针，**不是**嵌套 JSON，便于追加、回传和断点恢复。字段如下：

| 字段 | 含义 |
| --- | --- |
| `root_id` | 根节点编号（基础候选，固定为 `root`） |
| `c` / `max_children` | UCT 探索常数与单节点最大子节点数 |
| `nodes[].node_id` | 节点编号（`root`、`n001`、`n002`…） |
| `nodes[].parent_id` | 父节点编号，根为 `null` |
| `nodes[].candidate` | 该节点的候选组合（encoder/fusion/head/chunk/lr/image_size/seed） |
| `nodes[].children` | 子节点编号列表 |
| `nodes[].visits` / `total_value` | 回传的访问次数与累计归一化奖励 |
| `nodes[].stage` | 该候选所属的筛选阶段 |
| `nodes[].outcome` | 训练结果 + 逐种子评估结果 + 成本（由 loop 写入） |
| `nodes[].acceptance` | 相对基线的配对验收结论（accept / reject / insufficient） |

以本运行的最优节点为例，`tree.json` 中的一个节点形如：

```json
{
  "node_id": "root",
  "candidate": {
    "candidate_id": "bc_policy-tiny_cnn-concat-chunk-k20-lr0.001-img64-sd16-s0",
    "encoder": "tiny_cnn",
    "fusion": "concat",
    "action_head": "chunk",
    "chunk_size": 20,
    "lr": 0.001,
    "image_size": 64,
    "state_dim": 16,
    "seed": 0,
    "family": "bc_policy",
    "notes": ""
  },
  "parent_id": null,
  "visits": 3,
  "total_value": 1.2,
  "children": [
    "n001"
  ],
  "created_index": 0,
  "stage": "",
  "status": "ok",
  "error": "",
  "outcome": {
    "candidate": {
      "candidate_id": "bc_policy-tiny_cnn-concat-chunk-k20-lr0.001-img64-sd16-s0",
      "encoder": "tiny_cnn",
      "fusion": "concat",
      "action_head": "chunk",
      "chunk_size": 20,
      "lr": 0.001,
      "image_size": 64,
      "state_dim": 16,
      "seed": 0,
      "family": "bc_policy",
      "notes": ""
    },
    "status": "ok",
    "error": "",
    "train": {
      "status": "ok",
      "train_steps": 8000,
      "metrics": {
        "first_loss": 0.7718890309333801,
        "final_loss": 0.015755468979477882,
        "loss_last_10_mean": 0.019106738083064557,
        "offline_normalized_mae": 0.020554091781377792,
        "train_steps": 8000.0,
        "device": 0.0,
        "num_frames": 24553.0,
        "num_episodes": 60.0
      },
      "artifact": "D:\\AI305-Fall2026\\project\\a3_agent_optimizer\\artifacts\\local_sim_v2_60ep\\bc_policy-tiny_cnn-concat-chunk-k20-lr0.001-img64-sd16-s0\\checkpoint\\model.pt",
      "wall_seconds": 341.311,
      "gpu_seconds": 0.0,
      "error": ""
    },
    "eval": {
      "status": "ok"
  ... (截断)
```

树形结构如下（缩进表示父子关系，括号内为该节点平均分与验收结论）：

```text
root · tiny_cnn / concat / chunk-k20 / lr=0.001 / img64 · score=5.0
    └── n001 · tiny_cnn / film / chunk-k20 / lr=0.001 / img64 · score=4.5 · reject
        └── n002 · state_mlp / concat / chunk-k20 / lr=0.001 / img0 · score=2.5 · reject
```

## 四、每个节点的得分

所有分数为 10 分制闭环评估；训练步数与 loss 来自该节点的训练产物；验收结论相对根节点（baseline）的配对比较。

| 节点 | 候选组合 | 步数 | loss | 逐种子得分 | 均分 | 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| root | tiny_cnn/concat/chunk-k20/lr0.001/img64 | 8000 | 0.0158 | 2000:5, 2001:5 | 5.0 | - |
| n001 | tiny_cnn/film/chunk-k20/lr0.001/img64 | 8000 | 0.0158 | 2000:5, 2001:4 | 4.5 | 拒绝 |
| n002 | state_mlp/concat/chunk-k20/lr0.001/img0 | 8000 | 0.0242 | 2000:0, 2001:5 | 2.5 | 拒绝 |

### 4.1 预算与实际耗时

| 项目 | 数值 |
| --- | --- |
| 实验数 | 3 / 3 |
| 训练步数 | 24000 / 24000 |
| 墙钟秒数 | 1201.834 |
| GPU 秒数 | 0.0 |

## 五、最高分组合

| 维度 | 取值 |
| --- | --- |
| candidate_id | `bc_policy-tiny_cnn-concat-chunk-k20-lr0.001-img64-sd16-s0` |
| encoder | `tiny_cnn` |
| fusion | `concat` |
| action_head | `chunk` |
| chunk_size | `20` |
| lr | `0.001` |
| image_size | `64` |
| state_dim | `16` |
| seed | `0` |
| family | `bc_policy` |

| 指标 | 数值 |
| --- | --- |
| 平均分 | 5.0 / 10 |
| 逐种子得分 | 2000:5, 2001:5 |
| 成功次数 | 0 |
| 训练步数 | 8000 |
| 训练 final loss | 0.0158 |
| 训练耗时 | 341.311 s |
| 评估耗时 | 170.798 s |

最优 checkpoint：`D:\AI305-Fall2026\project\a3_agent_optimizer\artifacts\local_sim_v2_60ep\bc_policy-tiny_cnn-concat-chunk-k20-lr0.001-img64-sd16-s0\checkpoint\model.pt`

### 5.1 模型结构

| 组件 | 配置 |
| --- | --- |
| camera_count | 3 |
| encoder | tiny_cnn |
| fusion | concat |
| action_head | chunk |
| chunk_size | 20 |
| image_size | 64 |
| state_dim | 16 |
| action_dim | 16 |
| hidden | 128 |

### 5.2 复现命令

```bash
python scripts/bc2_experiment.py \
  --name <候选名> --dataset artifacts/datasets/expert_2x5_multicam_40ep.npz \
  --train-steps 10000 --image-size 64 --execution-horizon 12
```

## 五之二、最终调优模型（独立复核）

实验树之外，另用更大的训练预算训练了最终模型，并在开发块 6 个种子上复核：

| 项目 | 数值 |
| --- | --- |
| checkpoint | `artifacts/local_sim_v2_40ep/bc2-40ep-64/checkpoint` |
| 执行 horizon | 12 |
| 每种子分数 | 5.0, 10.0, 5.0, 5.0, 5.0, 5.0 |
| 平均分 | 5.83 / 10 |
| 成功次数 | 1 |

## 六、真实训练记录汇总

| 模型 | 数据 | 训练步数 | final loss | 离线 MAE | 训练耗时(s) | checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| bc2-baseline-64 | - | 6000 | 0.02295 | 0.01848 | 221.518 | D:\AI305-Fall2026\project\a3_agent_optimizer\artifacts\local_sim_v2\bc2-baseline-64\checkpoint |
| bc2-long-64 | - | 12000 | 0.01233 | 0.01247 | 449.588 | D:\AI305-Fall2026\project\a3_agent_optimizer\artifacts\local_sim_v2\bc2-long-64\checkpoint |
| bc_policy-state_mlp-concat-chunk-k20-lr0.001-img0-sd16-s0 | - | 4000 | 0.02267 | 0.0252 | 10.999 | D:\AI305-Fall2026\project\a3_agent_optimizer\artifacts\local_sim_v2\bc_policy-state_mlp-concat-chunk-k20-lr0.001-img0-sd16-s0\checkpoint |
| bc_policy-tiny_cnn-concat-chunk-k20-lr0.001-img64-sd16-s0 | - | 4000 | 0.02007 | 0.0226 | 153.543 | D:\AI305-Fall2026\project\a3_agent_optimizer\artifacts\local_sim_v2\bc_policy-tiny_cnn-concat-chunk-k20-lr0.001-img64-sd16-s0\checkpoint |
| bc_policy-tiny_cnn-film-chunk-k20-lr0.001-img64-sd16-s0 | - | 4000 | 0.02336 | 0.02077 | 120.949 | D:\AI305-Fall2026\project\a3_agent_optimizer\artifacts\local_sim_v2\bc_policy-tiny_cnn-film-chunk-k20-lr0.001-img64-sd16-s0\checkpoint |
| bc2-40ep-64 | - | 10000 | 0.01695 | 0.01645 | 387.846 | D:\AI305-Fall2026\project\a3_agent_optimizer\artifacts\local_sim_v2_40ep\bc2-40ep-64\checkpoint |
| bc2-40ep-64-20k | - | 20000 | 0.01056 | 0.01076 | 750.364 | D:\AI305-Fall2026\project\a3_agent_optimizer\artifacts\local_sim_v2_40ep\bc2-40ep-64-20k\checkpoint |
| bc2-40ep-64-s1 | - | 12000 | 0.01805 | 0.01655 | 542.685 | D:\AI305-Fall2026\project\a3_agent_optimizer\artifacts\local_sim_v2_40ep\bc2-40ep-64-s1\checkpoint |

外部参照（专家策略在相同任务上的水平）：脚本专家在同一 2x5 任务上约 410 步完成 10/10；当前学习策略在开发种子上平均分见第一节。

## 七、复现与测试

```bash
a3opt plan
a3opt selftest --experiments 6
pytest tests -q
```

测试结果：80 passed, 1 skipped（真实仿真测试默认由 `A3OPT_RUN_SIM_TESTS=1` 控制）。

## 八、局限与下一步

- 行为克隆策略仍会出现逐种子波动：部分布局能放 5-10 块，部分布局提前停滞；预留确认种子块分数明显低于开发块。
- 执行预算为 600 步（约 30 s 仿真时间）；专家需要约 410 步，学习策略偶尔更慢。
- 下一步：继续扩大示范数据、训练更长步数；有条件时在 GPU 上训练 ACT/SmolVLA 作为主力候选，并用专家做 DAgger 纠偏。

_本报告由 a3opt 运行产物自动生成。_