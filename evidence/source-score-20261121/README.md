# 高难冻结策略计分复核

复核对象在 `branch-joint` 分阶段示范策略上叠加第 1 列首批垂直下降、第 2 列首批专家开爪与第二批末端跟踪恢复、第 3 列第二批闭爪横移 0.75 mm，以及第 4 列第二批缩时。位置和来源列分别从首帧 RGB 与任务文本读取。控制器只使用部署侧公开观测。 [冻结控制器](../source-guard-refine-20261121/confirmation-lock.json) · [架构](../../docs/A3_CURRENT_ARCHITECTURE.md)

冻结协议指定 `2×5 / advanced`、模拟器提交 `5fa2497`、控制上限 500 步及种子 `179600–179791`。四列各 48 局。Agent 按列独立重采样并计算四列等权均值的 95% 区间。目标为课程终局指定来源均分的区间下界严格大于 9。另记模拟器原始入盒数、500 步已释放数、槽位、连续 20 步稳定、错列与安全停机。原生复跑在每列各执行课程及严格规则各一局。 [冻结协议](protocol.json) · [计分入口](../../scripts/validate_source_score.py)

## 192 局结果

课程终局原始计分、指定来源计分、第 500 步已释放均分均为 **9.3229167/10**，指定来源均分的分层 95% 区间为 **[9.0358073, 9.5833333]**。严格完整成功 **158/192**，第 500 步平均占用 **9.2552083 个槽位**，十槽全占 **160/192**，两套口径均无错列或安全停机。8 次原生复跑的计分、步数、槽位、成功标志和动作哈希均匹配。 [指标明细](result.json) · [逐局评测与原生复跑](validation-receipts.zip)

| 来源列 | 场景数 | 指定来源终局与已释放均分 | 十槽全占 | 严格完整成功 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 48 | 9.4792 | 38 | 38 |
| 2 | 48 | 8.7500 | 37 | 36 |
| 3 | 48 | 9.1667 | 41 | 40 |
| 4 | 48 | 9.8958 | 44 | 44 |

第 2 列失分最多。按阶段快照统计，第二批抬升结束时累计少起 25 块，首批有 30 块在下降时倾斜且终局未计分。 [逐局阶段快照](validation-receipts.zip)

192 局按同一冻结候选测绝对分数。此前 64 对比较中，候选在第 2 列一局少计 5 块，配对增益的区间包含零，Agent 判定回退。合看同一候选的 256 个场景，四列各 64 局，均分为 **9.25/10**，95% 区间 **[8.9921875, 9.484375]**，严格完整成功 **207/256**，错列和安全停机均为 0。 [此前 64 对确认](../source-guard-refine-20261121/README.md) · [192 局复核](result.json)

种子 `179601` 的满分回放和种子 `179600` 的五分回放均包含初始帧及完整 500 步。两个三视角录像的动作哈希、终局和随机化参数均与本轮确认回执一致。 [录像核对记录](video-examples-audit.json)

## 复算回执

在仓库根目录配置两个仓库的源码与 `scripts`，将协议、结果和回执包解到同一目录。复算入口校验每个种子的动作和终局回执，核对原生复跑字段，并重新计算两套指标。

```powershell
$sim = (Resolve-Path '..\a3_dual_arm_sim').Path
$env:PYTHONPATH = "$PWD\src;$sim\src;$PWD\scripts"
$archive = 'evidence/source-score-20261121'
$review = 'artifacts/source-score-review'
New-Item -ItemType Directory -Path $review -Force | Out-Null
Copy-Item "$archive/*.json" $review
Expand-Archive "$archive/validation-receipts.zip" $review -Force
& "$sim\.venv\Scripts\python.exe" scripts/audit_source_score.py $review `
  --source evidence/source-guard-refine-20261121 --parent evidence/expert-landing-20261121
```
