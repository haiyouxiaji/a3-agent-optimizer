# 第 2 列第二批接触恢复

Agent 从四组历史评测中复采 23 个场景，包含 11 局搬运后失分和 12 局成功对照。在第二批下降约 75% 时，它比较原轨迹、原地抬升后定点开爪、抬升后向外移再定点开爪。23 个原轨迹分支的末态、动作和入盒饼干与连续回合一致。 [协议](protocol.json) · [训练回执](training.json.gz)

| 训练筛选 | 净增饼干 | 增分 / 减分场景 | 严格完整成功数变化 |
| --- | ---: | ---: | ---: |
| 原地抬升常开 | +1 | 4 / 4 | 0 |
| 抬升外移常开 | +33 | 11 / 4 | −4 |
| 逐组留出的恢复条件 | +37 | 10 / 0 | 0 |

Agent 冻结抬升后外移配方，按公开观测的末端高度启用。测得末端高度比计划动作高至少 **1.032 mm** 时触发，开爪保持原夹爪动作。训练集解压后的 SHA-256 为 `8e4a66423cf912546c5fc50464d6492119e41adbc446b1194b6be6a74ce46dd2`。每一折用另外三组拟合启用条件，保留组的结果见回执。 [冻结拟合](fit.json) · [Agent 入口](../../src/a3opt/placement_release_recovery.py)

16 对第 2 列独立开发场景中，课程终局和第 500 步释放均由 **7.625 升至 8.4375/10**，配对增益 95% 区间为 **[0, +1.75]**。3 局分别多计 5、3、5 块，其余 13 局计分一致。严格完整成功均为 **1/16**，两组均无错列和安全停机。逐局回执解压后的 SHA-256 为 `269612d329f3728cbc5b3a26b07cd7f97bfc7d9d0ca8400881207088716a39b6`。Agent 在开发筛选通过后冻结策略，预留的四列平衡确认种子为 `159400–159463`。 [逐局回执](development.json.gz) · [开发汇总](development-selection.json)

## 四列平衡确认

64 对独立高难场景每列各 16 对。课程终局及第 500 步释放均分均由 **8.921875 升至 9.171875/10**。按列分层的配对增益 95% 区间为 **[+0.046875, +0.484375]**，候选绝对均分区间为 **[8.734375, 9.5625]**。严格完整成功均为 **35/64**。4 局多计 3、5、3、5 块，其余 60 局计分相同。两组均无错列和安全停机。Agent 按预设的点估计与配对晋级规则选中接触恢复策略，绝对均分区间仍涵盖 9 分。 [冻结锁](confirmation-lock.json) · [汇总](result.json)

| 来源列 | 课程终局与第 500 步释放均分 | 严格完整成功 |
| --- | ---: | ---: |
| 1 | 9.0625 → 9.0625 | 10 → 10/16 |
| 2 | 7.25 → 8.25 | 1 → 1/16 |
| 3 | 10 → 10 | 15 → 15/16 |
| 4 | 9.375 → 9.375 | 9 → 9/16 |

每列各抽一局，以课程口径和严格口径分别进行原生复跑。8 次均匹配步数、目标饼干身份、槽位、成功标记、命令与执行动作哈希。确认回执解压后的 SHA-256 为 `e874009b292e7574f87cf7fb9eb146a8a40886489c0ee64128b19a1261ace15f`。 [确认逐局回执](confirmation.json.gz) · [原生复跑](native-replays.json.gz) · [复算审计](confirmation-audit.json)

种子 `159428` 的三视角录像完整覆盖 500 步。该局从 0 块增至 5 块，两次完整成功标记均为假。录像与确认回执的终局和动作哈希一致。 [录像审计](video-159428-audit.json)

## 从公开检查点复跑

在本仓库根目录运行。复跑使用协议记录的模拟器提交与依赖环境。检查点路径可按本机检出位置设置，加载时仍核对冻结 SHA-256。

```powershell
$sim = (Resolve-Path '..\a3_dual_arm_sim').Path
$env:PYTHONPATH = "$PWD\src;$sim\src;$PWD\scripts"
$env:A3OPT_PLACEMENT_LOCK = (Resolve-Path 'evidence/release-recovery-20261121/confirmation-lock.json').Path
$env:A3OPT_PLACEMENT_CHECKPOINT = (Resolve-Path 'evidence/advanced-20261120/policy.npz').Path
$env:A3OPT_PLACEMENT_KIND = 'release-recovery'
& "$sim\.venv\Scripts\python.exe" -m a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli `
  --task 2x5 --policy replay_placement:upstream_policy --profile advanced `
  --source-column random --prompt-style task --episodes 1 --seed-start 159428 --max-steps 500 --workers 1 `
  --output artifacts/replay-release-recovery-159428.json
```
