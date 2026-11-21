# A3 课程规则与评测证据核查

2026 年 11 月 21 日。核查以教师中期任务书《基于 Agent 的机器人策略模型优化》第 1–3 页、项目介绍《A3 双臂机器人仿真与策略学习平台》第 6、9–11 页的课件文字存档，以及锁定的模拟器提交 `5fa2497` 为依据。分数目标 7.5 和 9 属于项目验收目标。 [上游模拟器固定版本](https://github.com/Eter0109/a3_dual_arm_sim/tree/5fa24970648f53a24c0b5038d19bf6a14e1be5b9)

已核实 `2×5 / advanced` 的真实闭环仿真、500 步上限和原生终局计分。四列平衡的 192 局复核均分为 **9.323/10**。最终冻结策略的完整六格成绩仍缺统一验收，四列结果适用于本次补充协议。 [实验树](A3_EXPERIMENT_TREE.md#实验总览) · [逐局回执与复算入口](../evidence/source-score-20261121/README.md)

## 逐项核查

| 要求 | 核查结果 | 核查依据 |
| --- | --- | --- |
| 跑通 `Observation → Policy → Action → Environment → Evaluation` | 符合。冻结策略通过上游策略入口运行，按控制步输出 14 个关节目标和 2 个夹爪目标。闭环由 MuJoCo 执行并生成终局回执。 | [公开策略入口](../scripts/frozen_source_policy.py) · [上游观测与动作契约](https://github.com/Eter0109/a3_dual_arm_sim/blob/5fa24970648f53a24c0b5038d19bf6a14e1be5b9/docs/contracts.md) |
| 保持部署观测边界 | 冻结策略从任务文本读来源列，从首帧 RGB 估计盒位姿。接触恢复使用公开末端位姿和关节状态。专家采集与评测诊断可访问仿真真值，两者位于独立路径。 | [闭爪策略](../src/a3opt/placement_source_guard.py#L63) · [接触恢复](../src/a3opt/placement_release_recovery.py#L66) · [诊断与评测](../src/a3opt/placement_sim.py) |
| 采用原生 `advanced` 扰动 | 符合。192 局回执均标为 `advanced`，含四种饼干配色。该档源盒及目标盒 XY 每轴 ±20 mm、目标盒偏航 ±0.08 rad、固定相机位置 ±9 mm、腕相机位置 ±3 mm、灯光 ±20%，并有相机角度和视场扰动。三档饼干初始位置抖动均为 ±0.3 mm，所以高难场景也可能排列整齐。 | [上游随机化配置](https://github.com/Eter0109/a3_dual_arm_sim/blob/5fa24970648f53a24c0b5038d19bf6a14e1be5b9/configs/randomization.yaml) · [冻结协议](../evidence/source-score-20261121/protocol.json) |
| 限时、计分和成功判定 | `2×5` 上限 500 控制步，按 20 Hz 为 25 秒。上游原始分是终止时入盒数。回执分别记录指定来源入盒数、500 步已释放数和稳定满槽成功。该批 192 局的三种数量均分相同。 | [上游评测入口](https://github.com/Eter0109/a3_dual_arm_sim/blob/5fa24970648f53a24c0b5038d19bf6a14e1be5b9/src/a3_dual_arm_sim/workflows/episode_execution.py) · [计分采集](../src/a3opt/placement_benchmark.py) |
| 课件的六格评测 | 课件展示 `2×5 / 2×10` × `basic / medium / advanced`，每格 20 局。锁定版上游网格固定来源第 1 列，默认分别给 `2×5` 和 `2×10` 500 与 3,200 步。当前 192 局只覆盖 `2×5 / advanced`，四列各 48 局。旧策略的三档和迁移结果有记录，最终冻结策略组合尚缺统一六格报告。 | [上游六格参数](https://github.com/Eter0109/a3_dual_arm_sim/blob/5fa24970648f53a24c0b5038d19bf6a14e1be5b9/src/a3_dual_arm_sim/workflows/evaluate_smolvla_grid_cli.py#L94-L109) · [历史难度结果](A3_DIFFICULTY_EVALUATION.md) |
| 调研至少三种策略方法及一种 Agent | 已覆盖 ACT、Diffusion Policy、SmolVLA 的方法比较，以及 AIDE 和 MLAgentBench 的实验组织方式。已跑通的主要收益来自分阶段示范策略。三种方法的统一预算训练与同协议对比仍缺。 | [方法比较与实证范围](A3_COURSE_ALIGNMENT.md#课程所要求的方法比较) · [后端运行记录](results.md#9-act-and-smolvla-attempts) |
| Agent 自主采样、构建数据集、拟合、验证和回退 | 专项入口有完整阶段证据。以专家开爪为例，脚本从历史回执选失分及对照场景，调用专家录制，生成带拒收原因的数据集清单，拟合并冻结候选，再用独立场景验证和决定回退。候选模板、预算与闸门由框架定义。通用 UCT 与专项脚本各自留痕，跨全部分支的端到端自主调度尚无统一运行证据。 | [专家开爪实验与回执](../evidence/expert-landing-20261121/README.md) · [采样到确认的实现](../src/a3opt/placement_expert_landing.py#L232) · [Agent 架构与人机分工](A3_CURRENT_ARCHITECTURE.md) |

`2×10` 的 500 步补充实验中，加权平均和最近示范的 20 对确认分数分别为 **1.80/20** 和 **3.15/20**，配对差区间跨零，完整成功均为 0，Agent 保留加权平均。原生任务另有 3,200 步开发结果。 [2×10 协议与结果](A3_COURSE_ALIGNMENT.md#210-自主实验)

## 分数与晋级结论

| 冻结策略或样本 | 复核结果 | 可据此支持的结论 |
| --- | --- | --- |
| 发布策略首批 100 对 `2×5 / advanced` | 8.79 → **9.14/10**，严格完整成功 59 → 65/100 | 同场景配对确认通过。该发布策略在合并后续 128 个场景后的 228 局均分为 **8.9956/10**。 |
| 专家开爪与 0.75 mm 横移组合，64 对独立确认 | 8.844 → **9.031/10**，配对增益 95% 区间 `[−0.125, +0.547]`，一局少搬 5 块 | Agent 回退并保留参考策略。 |
| 同一冻结组合，192 局新场景 | **9.323/10**，分层 95% 区间 `[9.036, 9.583]`，严格完整成功 158/192 | 该批绝对计分超过 9 分，支持本地四列均衡协议下的绝对表现结论。 |
| 合看该组合的 64 局候选和 192 局复核 | **9.25/10**，分层 95% 区间 `[8.992, 9.484]`，严格完整成功 207/256 | 合计区间涵盖 9 分，配对晋级结论仍为回退。 |

[100 对原始评分](../evidence/advanced-20261120/README.md) · [228 局合计](A3_CURRENT_ARCHITECTURE.md) · [64 对回退](../evidence/source-guard-refine-20261121/README.md) · [192 局与 256 局计分](../evidence/source-score-20261121/README.md)

两批发布策略压缩回执通过 SHA-256 校验，两种计分口径均复算出 `8.79 → 9.14`。192 局的冻结检查点、协议、逐局文件和汇总均匹配，归档的 8 次原生复跑在终局、步数、成功标志和动作哈希等字段上匹配。可见的其他试验协议中，涉及这批种子的本地目录是同一份协议的副本。 [发布证据复算](../evidence/advanced-20261120/README.md#从回执核对发布分数) · [192 局复算命令](../evidence/source-score-20261121/README.md#复算回执)

本次另通过上游 CLI 重新执行两局。第 2 列种子 `179600` 在 500 步终局计 5 分，完整成功为假。第 4 列种子 `179601` 在 465 步完整成功，终局计 10 分。两局来源列、计分、步数和成功标志均与归档逐局回执一致。

```powershell
$sim = (Resolve-Path '..\a3_dual_arm_sim').Path
$env:PYTHONPATH = "$PWD\src;$sim\src;$PWD\scripts"
$env:A3OPT_PLACEMENT_LOCK = (Resolve-Path 'evidence/source-guard-refine-20261121/confirmation-lock.json').Path
$env:A3OPT_PLACEMENT_PARENT_LOCK = (Resolve-Path 'evidence/expert-landing-20261121/confirmation-lock.json').Path
$env:A3OPT_PLACEMENT_CHECKPOINT = (Resolve-Path 'evidence/advanced-20261120/policy.npz').Path
& "$sim\.venv\Scripts\python.exe" -m a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli `
  --task 2x5 --policy frozen_source_policy:upstream_policy --profile advanced `
  --source-column random --prompt-style task --episodes 2 --seed-start 179600 --max-steps 500 --workers 1 `
  --output artifacts/course-audit-spot-check.json
```
