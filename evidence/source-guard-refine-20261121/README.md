# 第 3 列闭爪横移幅度细化

Agent 根据上轮夹带与抓取失败的取舍，比较 **0.75、1.5、2.25 mm** 的闭爪横移。前两项在训练阶段均消除已知错列，并保留四局正常对照的各项计分。2.25 mm 的候选使一局对照少搬 5 块。Agent 选择通过训练约束且横移最小的 **0.75 mm** 候选。 [父实验](../source-guard-20261121/README.md) · [采样协议](protocol.json) · [训练筛选](fit.json)

8 对独立第 3 列开发场景的指定来源计分持平，候选没有错列、安全停机或完整成功退化。Agent 将横移与第 2 列首批专家开爪合并冻结。旧错列种子 `167624` 的原生复跑达到 **10 块指定来源、零错列和 500 步严格成功**。该种子也参与了避让方案的训练筛选。 [开发汇总](development-selection.json) · [采样及开发回执包](development-receipts.zip) · [冻结锁](confirmation-lock.json) · [旧场景原生复跑](known-offcolumn-native.json)

## 四列确认

64 对 `2×5 / advanced` 新场景每列 16 对，种子为 `171600–171663`。课程终局的模拟器原始计分、指定来源计分以及第 500 步已释放计分均由 **8.84375 升至 9.03125/10**。候选绝对均分的分层 95% 区间为 **[8.515625, 9.5]**，配对增益区间为 **[−0.125, +0.546875]**。 [确认复算](confirmation-audit.json) · [逐局回执及原生复跑包](confirmation-receipts.zip)

| 来源列 | 指定来源终局与已释放均分 | 严格完整成功 |
| --- | ---: | ---: |
| 1 | 9.0625 → 9.0625 | 12 → 12/16 |
| 2 | 7.6875 → 8.4375 | 4 → 10/16 |
| 3 | 9.375 → 9.375 | 13 → 14/16 |
| 4 | 9.25 → 9.25 | 13 → 13/16 |
| 总计 | **8.84375 → 9.03125** | **42 → 49/64** |

第 500 步已占槽位均值从 **8.640625 升至 8.984375/10**，十槽全占由 **44 增至 51 局**。两组错列与安全停机均为 **0**。四列各用两套规则复跑一局，8 次原生复跑的步数、计分、槽位、成功标志及动作哈希均匹配。 [终局明细](confirmation-audit.json)

第 2 列的 `171656` 由 10 块降至 5 块。首批正常抬升并完成搬运，5 块在下降时倾斜，第二批仍计入 5 块。Agent 根据严重退化与跨零的配对区间回退。预登记的 128 对复验要求候选晋级，本轮回退结束该计划。后续 Agent 用两批历史回执采样，检查专家落点之前的下降路径。 [冻结决策](result.json) · [复验计划](replication-plan.json) · [后续调优入口](../../scripts/tune_landing_path.py)

## 回执复算

在仓库根目录配置两个仓库的源码与 `scripts` 目录后，将本目录 JSON 和确认包解到同一目录。复算入口读取冻结协议、检查点和逐局回执，并核对原生复跑字段。

```powershell
$sim = (Resolve-Path '..\a3_dual_arm_sim').Path
$env:PYTHONPATH = "$PWD\src;$sim\src;$PWD\scripts"
$archive = 'evidence/source-guard-refine-20261121'
$review = 'artifacts/source-guard-review'
New-Item -ItemType Directory -Path $review -Force | Out-Null
Copy-Item "$archive/*.json" $review
Expand-Archive "$archive/confirmation-receipts.zip" $review -Force
& "$sim\.venv\Scripts\python.exe" scripts/audit_source_guard.py $review `
  --parent evidence/expert-landing-20261121 --checkpoint evidence/advanced-20261120/policy.npz
```
