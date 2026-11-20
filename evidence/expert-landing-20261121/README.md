# 第 2 列首批专家示范开爪位置

Agent 按搬运成功后下降损失的饼干数，对两个历史场景组中的来源列和批次排序。它选择第 2 列首批，包括 6 局下降失分和 4 局满分对照，随后通过上游专家采集示范，10 次均完成两批。 [采集协议](protocol.json) · [Agent 入口](../../src/a3opt/placement_expert_landing.py)

Agent 将专家开爪时的末端位置表示为盒相对水平坐标和世界高度。留一验证逐轴选择中位数或岭回归，平均绝对误差由 **2.91、5.44、2.18 mm** 降至 **0.80、1.08、0.02 mm**。部署时用正面图像的盒位估计和实时末端位置，将下降尾段平滑连接至预测位置，并在开爪阶段保持位置。 [冻结拟合](fit.json)

| 16 对第 2 列独立开发场景 | 参考 | 专家开爪 |
| --- | ---: | ---: |
| 课程终局均分 | 8.0625 | 8.7500 |
| 第 500 步已释放均分 | 8.0625 | 8.7500 |
| 严格完整成功 | 4/16 | 11/16 |

4 局分别多计 4、5、1、1 块，配对增益的 95% 区间为 **[+0.0625, +1.5000]**。两组的错列、安全停机和严重退化计数均为 0。参考和候选均启用同一冻结的第二批接触恢复。 [开发汇总](development-selection.json) · [冻结确认锁](confirmation-lock.json)

## 四列平衡确认

64 对独立高难场景每列各 16 对。任务指定来源的终局及第 500 步已释放均分均为 **9.171875 → 9.53125/10**。候选的分层 95% 区间为 **[9.0625, 9.921875]**，配对增益区间为 **[+0.109375, +0.65625]**，严格完整成功由 **35/64 升至 48/64**。6 局增分且所有场景的指定来源计数均持平或上升。 [逐列复算](confirmation-audit.json) · [冻结决策](result.json)

| 来源列 | 课程终局与第 500 步已释放均分 | 严格完整成功 |
| --- | ---: | ---: |
| 1 | 9.6875 → 9.6875 | 10 → 10/16 |
| 2 | 8.25 → 9.6875 | 2 → 15/16 |
| 3 | 9.375 → 9.375 | 12 → 12/16 |
| 4 | 9.375 → 9.375 | 11 → 11/16 |

种子 `167624` 指定第 3 列。两个控制器均搬入该列 10 块和第 4 列的 1 块。模拟器原始计数在这一局为 11，原始面板均分为 **9.1875 → 9.546875**。晋级同时要求错列数为零，Agent 回退并保留参考，预登记的复验未触发。 [来源核对与原始计数](confirmation-audit.json) · [复验计划](replication-plan.json)

该局 500 步的原生复跑匹配动作哈希和全部终局字段。相邻列饼干 `67` 在第 **317** 步与指定列饼干 `46` 发生接触，在第二批抬升的第 **341** 步离开来源盒，第 **436** 步进入目标盒的计数范围。记录到的接触发生在饼干之间。 [错列复跑摘要](wrong-source-audit.json) · [逐步接触轨迹](wrong-source-167624.json.gz)

12 次完整开发运行和共享前缀运行的回执一致。确认面板每列选一局，以课程计分和第 500 步已释放计分分别原生复跑，8 次均匹配步数、计数、槽位、正确来源、成功标记和动作哈希。 [完整确认与原生复跑证据包](confirmation-receipts.zip)

## 开发证据复算

在仓库根目录执行。示范、采集回执、逐局开发回执和原始数据哈希均保存在[开发证据包](development-receipts.zip)。脚本从示范轨迹重建标签、重拟合模型、核对逐局记录并复算两套开发指标。

```powershell
$sim = (Resolve-Path '..\a3_dual_arm_sim').Path
$env:PYTHONPATH = "$PWD\src;$sim\src;$PWD\scripts"
$archive = 'evidence/expert-landing-20261121'
$review = 'artifacts/expert-landing-review'
New-Item -ItemType Directory -Path $review -Force | Out-Null
Copy-Item "$archive/*.json" $review
Expand-Archive "$archive/development-receipts.zip" $review -Force
& "$sim\.venv\Scripts\python.exe" scripts/audit_placement_confirmation.py $review `
  --policy-kind expert-landing --development-only
```

解开确认包后可从同一目录复算 64 对评测及 8 次原生复跑记录。

```powershell
Expand-Archive "$archive/confirmation-receipts.zip" $review -Force
& "$sim\.venv\Scripts\python.exe" scripts/audit_placement_confirmation.py $review `
  --policy-kind expert-landing --checkpoint evidence/advanced-20261120/policy.npz
```
