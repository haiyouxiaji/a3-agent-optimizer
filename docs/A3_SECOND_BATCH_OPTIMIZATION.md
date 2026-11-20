# A3 最高档第二批放置优化

2026 年 11 月 19 日 · `2×5 / advanced / 500` 控制步 · 冻结参考 `branch-joint`

## 评测口径

使用教师原始任务书、项目介绍第 9–11 页和上游评测执行器。MuJoCo 模拟器固定在 `5fa24970648f53a24c0b5038d19bf6a14e1be5b9`。报告中的均分是终局正确来源饼干数，满分 10。完整成功沿用上游的槽位匹配、四壁接触、源盒余量和连续 5 步保持要求，另核对来源列。最高档保留盒位、目标盒偏航、相机、灯光和颜色扰动。 [教师任务书](../../midterm.pdf#page=1) · [原始项目介绍逐页存档](../../_build/project_intro.txt#L95) · [评测参数与扰动范围](A3_ADAPTIVE_AGENT.md#判分与实验范围)

## Agent 如何收集数据

Agent 在新种子上执行 16 局平衡随机列初查，每列 4 局。追踪器在第二批抬升结束、搬运结束、放置下降结束、开爪结束、退臂结束和终局记录饼干 ID、目标盒坐标、左右夹爪接触、目标盒几何范围以及上游计分与槽位。每份逐局回执还保存初始扰动信息与动作哈希。第一批饼干也在这些时刻按 ID 追踪，用于观察第二批放置对第一批的影响。 [追踪与回执](../src/a3opt/second_batch_sim.py) · [初查逐局记录](../artifacts/second-batch-20261119/scout/results.json)

| 来源列 | 完整成功 | 第二批放置前的失败 | 第二批终局少计 | 终局数量达标但槽位或保持未达标 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2/4 | 1 | 1 | 0 |
| 2 | 2/4 | 0 | 1 | 1 |
| 3 | 4/4 | 0 | 0 | 0 |
| 4 | 4/4 | 0 | 0 | 0 |

Agent 按第一批已计满 5 块、第二批搬运仍保有 5 块时的放置失败次数排优先级，得到第 2 列和第 1 列。每个选中的列再采 8 个场景。放置标注使用开爪前同一组 5 块饼干的目标盒坐标，要求双侧夹爪仍有接触。Agent 用有效标注的中位偏移拟合修正量，优先使用失败场景。拟合至少需要两个有效场景。仿真真值用于训练标注与评测。推理使用任务文本、首帧视觉定位和原有策略输入。 [诊断](../artifacts/second-batch-20261119/scout-diagnosis.json) · [标注和拟合](../src/a3opt/second_batch_fit.py)

## 候选与晋级

- 候选比较开爪后停留、增加 6 毫米释放间隙、拟合偏移与拟合偏移加停留。偏移经名义机械臂运动学转换为有限关节修正，作用于指定列的第二批下降、开爪和退臂。所有配对回执核验到第二批搬运结束为止的策略动作和仿真实施动作哈希。 [策略实现](../src/a3opt/backends/second_batch.py)
- 训练候选和冻结参考使用相同的 8 个追加场景。Agent 以完整成功数、正确来源均分排序，要求两项均不下降，至少一项上升，且候选没有安全停机或错列计分。每个优先列最多产生一个开发候选，在 12 个独立种子上应用同一准入条件。 [采样与晋级](../src/a3opt/second_batch_agent.py)
- 通过开发后，检查点、开发选择、协议与确认种子通过 SHA-256 绑定。确认阶段固定第 1 列与平衡随机列各 20 局，参考和候选分别执行。两组正确来源均分均须超过 7.50，两组均通过开发阶段的无回退条件，且至少一组有提升。区间使用按来源列分层的配对 bootstrap。 [预先写入的协议](../artifacts/second-batch-20261119/protocol.json)

## 第二批候选的实测结果

| 来源列 | 冻结参考均分 | 冻结参考完整成功 | 训练候选数 | 各候选配对均分变化 | 各候选完整成功数变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 10.00 | 4/8 | 4 | 0.00 | 0 |
| 1 | 9.13 | 7/8 | 2 | 0.00 | 0 |

两列共执行 64 局训练比较，候选与参考均无安全停机或错列计分。Agent 在训练筛选后保留 `branch-joint`。第 2 列的 4 局超时均有 10 块正确入盒，但只匹配 9 个槽位。缺槽在第二批下降前已经存在，终局未分配槽位的饼干都来自首批。以 `72102` 为例，未匹配的 20 号饼干的盒内 Y 为 −22.6 mm，最末端名义槽位 Y 为 −12.7 mm，Y 容差为 ±5 mm。 [候选结果](../artifacts/second-batch-20261119/result.json) · [逐局槽位证据](../artifacts/second-batch-20261119/column-2/training/reference/episodes/2/72102.json)

## 首批槽位校准

Agent 从初查和第二批训练区的冻结参考回合中统计首批缺槽。第 2 列有 5 局在第二批放置前即缺槽，且终局第二批 5 块全部匹配槽位。Agent 据此选择第 2 列，并使用 20 个全新场景采样。采样回合在首批退臂结束时记录上游计分、槽位和饼干坐标。保留的样本要求 5 块均已计分、同属指定列、落在目标列的 X 容差内，且一维刚性平移可以解释它们的 Y 偏差。 [采样协议](../artifacts/slot-alignment-20261119/protocol.json) · [标注数据与拒收原因](../artifacts/slot-alignment-20261119/slot-dataset.json)

Agent 保留 12 条标注，以排序后逐个槽位的 Y 残差拟合纵向位移。按场景留一交叉验证选中了根据首帧视觉和名义关节轨迹计算的两个末端 Y 特征。训练区几何预测中，5 块同时满足 Y 容差的样本由 0/12 增至 12/12，平均每组最大 Y 误差由 9.56 mm 降至 0.51 mm。该几何预测基于刚性平移。开发和确认阶段执行完整仿真，检查真实入盒和完整成功。 [训练与交叉验证](../src/a3opt/slot_alignment_fit.py)

候选通过名义运动学将目标盒坐标系的 Y 修正转换为关节命令，位移上限为 12 mm。修正作用于选中来源列的首批下降、开爪和退臂。首批开始下降前的策略动作及实际执行动作均逐局核对。开发集用 12 个独立种子，确认沿用两组各 20 局的协议。 [策略](../src/a3opt/backends/slot_alignment.py) · [Agent 入口](../src/a3opt/slot_alignment_agent.py)

| 最高档第 2 列开发集 | 局数 | 首批退臂时 5 槽均匹配 | 完整成功 | 终局正确来源均分 |
| --- | ---: | ---: | ---: | ---: |
| 冻结参考 | 12 | 3/12 | 3/12 | 8.25 |
| 槽位校准 | 12 | 9/12 | 5/12 | 7.50 |

配对均分变化为 −0.75，95% 区间为 −1.67 至 0.00。`73805`、`73806` 和 `73807` 的第二批都搬到下降前，终局分别少计 2、2 和 5 块。Agent 根据终局均分的晋级条件保留 `branch-joint`。 [配对结果](../artifacts/slot-alignment-20261119/development-selection.json) · [逐局证据](../artifacts/slot-alignment-20261119/development/results.json)

## 根据配对反馈限制修正幅度

Agent 将上一轮 12 个开发配对转为本轮反馈训练数据。训练标签记录校准幅度、正确来源分变化、完整成功的变化及安全和来源核验。Agent 在 1 mm 间隔的阈值上搜寻幅度区间，要求训练区间内没有掉分或违规，且至少两局从不完整变为完整成功，再按净成功增量和区间宽度排序。由此得到 6–9 mm 的启用区间。范围外执行参考命令。 [反馈数据与学习结果](../artifacts/slot-guard-20261119/feedback-dataset.json) · [Agent 实现](../src/a3opt/slot_guard_agent.py)

在新开发种子 `74800–74811` 上，两种策略均为 **7.42/10**。限幅候选把首批退臂时 5 槽匹配从 3/12 提升到 5/12，把完整成功从 2/12 提升到 4/12。12 个配对的计分逐局相同，无安全停机或错列计分。Agent 冻结候选后才开始确认评测。 [开发结果](../artifacts/slot-guard-20261119/development-selection.json) · [确认锁](../artifacts/slot-guard-20261119/confirmation-lock.json)

开发种子 `74801` 的冻结参考和候选均为 10 分。参考到第 500 步超时，候选在第 420 步完整成功。三相机完整录像的重跑逐局匹配策略动作哈希、实际动作哈希、终局饼干 ID、槽位状态和步数。录像含重置、完整控制序列及终局停留，解码分别得到 541 和 461 帧。上游评测 CLI 用相同冻结候选再次复现 10 分和第 420 步完整成功。 [参考录像](../artifacts/slot-guard-20261119/videos/74801-reference.mp4) · [候选录像](../artifacts/slot-guard-20261119/videos/74801-selected.mp4) · [录像核验](../artifacts/slot-guard-20261119/videos/74801-audit.json) · [上游 CLI 回执](../artifacts/slot-guard-20261119/upstream-74801.json)

## 最高档独立确认

确认种子为固定第 1 列 `74900–74919` 和平衡随机列 `75000–75019`。参考和候选分别在同一组场景运行，共 80 局。全部逐局回执与冻结协议、场景种子、检查点哈希、动作前缀和上游终局计分一致。 [冻结协议](../artifacts/slot-guard-20261119/protocol.json) · [逐局结果](../artifacts/slot-guard-20261119/confirmation/results.json) · [独立核验](../artifacts/slot-guard-20261119/confirmation-audit.json)

| `2×5 / advanced / 500` | 局数／策略 | 参考与候选均分（95% 区间） | 参考与候选完整成功 |
| --- | ---: | ---: | ---: |
| 固定第 1 列 | 20 | 8.50（7.50–9.50） | 12/20 |
| 平衡随机列 | 20 | 8.80（7.60–9.75） | 13/20 |

两种策略的确认点估计均为 **8.50/10**，达到 7.50 的项目目标。固定列区间下界为 7.50。随机列组的 5 局第 2 列均未触发限幅修正，两个策略在全部 40 个确认场景的动作哈希一致。两组的均分和完整成功数相同，Agent 保留 `branch-joint`。 [晋级判定](../artifacts/slot-guard-20261119/result.json)

## 修正触发场景复测

Agent 用两个冻结策略验证实际触发时的表现。在 64 个新的最高档种子 `76000–76063` 上，用重置首帧估计修正幅度。15 个场景符合启用区间，按种子顺序取前 12 个，写入锁后执行 24 次完整仿真。12 个配对均确认施加了不同动作。初筛只计算首帧特征与策略幅度，场景入选与终局结果无关。 [抽样协议](../artifacts/slot-guard-coverage-20261119/protocol.json) · [抽样锁](../artifacts/slot-guard-coverage-20261119/paired-lock.json) · [逐局结果](../artifacts/slot-guard-coverage-20261119/paired/results.json)

| 固定第 2 列，启用条件内 | 首批退臂时 5 槽均匹配 | 完整成功 | 终局正确来源均分 |
| --- | ---: | ---: | ---: |
| 冻结参考 | 0/12 | 0/12 | 8.83 |
| 限幅校准 | 10/12 | 9/12 | 8.75 |

启用条件内的配对均分差为 −0.08，95% 区间为 −1.25 至 1.00。`76008` 从 10 分降到 5 分，`76036` 从 6 分升到 10 分。两局的两批均抬起 5 块饼干。两种策略共 24 局无安全停机或错列计分。当前候选提升了启用条件内的完整成功率，终局数量仍有掉分，正式采用 `branch-joint`。 [覆盖度审计与汇总](../artifacts/slot-guard-coverage-20261119/result.json)

## 复现命令

```powershell
$sim = (Resolve-Path '..\a3_dual_arm_sim').Path
$env:PYTHONPATH = "$PWD\src;$sim\src"
& "$sim\.venv\Scripts\python.exe" -u -m a3opt.second_batch_agent `
    --reference artifacts/adaptive-branch-20261119 `
    --out artifacts/second-batch-20261119 --seed-block 72000 --workers 12
& "$sim\.venv\Scripts\python.exe" -u -m a3opt.slot_alignment_agent `
    --reference artifacts/adaptive-branch-20261119 `
    --diagnosis artifacts/second-batch-20261119 `
    --out artifacts/slot-alignment-20261119 --seed-block 73600 --workers 12
& "$sim\.venv\Scripts\python.exe" -u -m a3opt.slot_guard_agent `
    --reference artifacts/adaptive-branch-20261119 `
    --calibration artifacts/slot-alignment-20261119 `
    --out artifacts/slot-guard-20261119 --seed-block 74800 --workers 14
& "$sim\.venv\Scripts\python.exe" -u -m a3opt.slot_guard_coverage `
    --guard artifacts/slot-guard-20261119 `
    --out artifacts/slot-guard-coverage-20261119 --seed-block 76000 --workers 12
```

在优化器目录运行。相同协议与输出目录可复核已有逐局回执，并从未完成阶段续跑。选择新的输出目录可从仿真重跑。 [逐局校验及缓存入口](../src/a3opt/second_batch_sim.py)
