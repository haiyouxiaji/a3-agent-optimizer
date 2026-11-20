# 最高档放置策略优化

发布版提供冻结策略、原始评测协议、逐局回执和复现代码。文中的 `artifacts/` 路径是本地运行产物。 [公开证据](../evidence/advanced-20261120/README.md)

本轮目标为 `2×5 / advanced` 的四列平衡随机评测均分超过 9/10。课程主指标沿用项目介绍第 9–11 页的回合终局入盒数和 500 控制步上限。早期放置候选使用每列 5 局的确认集，后续分支预留每列 10 局。每个种子配对比较原策略与候选策略，同时记录第 500 步的已释放数量和连续 20 步完整成功。报告正确来源计数，并核对上游终局总数和错列情况。 [教师任务书](../../midterm.pdf) · [项目介绍](../../_build/project_intro.txt#L95)

## 放置失分与策略改动

Agent 复现前次平衡评测中 6 个失分场景的结果和动作哈希，并在 16 个新场景上记录逐块位置、倾角、接触和计分状态。种子 `110002` 和 `110010` 的参考策略两批都搬运了 5 块，但放置后饼干沿盒内边缘倾斜，终局为 0 块。定向采样中，第一批增加 8 mm 高度和 3 mm 向内位移，使两个场景各恢复 5 块。种子 `110009` 的第一批搬运终点超出横向边界，五种分支的第一批都未计分。 [历史复跑](../artifacts/placement-scout-20261120/archive-diagnosis/results.json) · [新场景追踪](../artifacts/placement-scout-20261120/scout/results.json) · [定向样本](../artifacts/placement-active-20261120/training/dataset.json)

策略根据首帧前相机图像预测盒位，根据任务文本选择来源列，并用原有分阶段轨迹搬运两批饼干。训练选出的限幅位移用名义机械臂运动学转换成下降、开爪和退臂的关节修正。Agent 对搬运终点的模拟器状态进行分支采样。每组零修正分支必须与完整参考回合的同段命令、物理状态及计分一致。逐块仿真真值用于采样标注和评测，策略的修正只读取学习结果、来源列和首帧视觉预测。 [策略与逐块追踪](../src/a3opt/placement_sim.py#L35) · [分支采样](../src/a3opt/placement_train.py#L45)

## 复现选模与评测

Agent 首轮读取 8 局平衡训练样本及 3 局定向训练样本。按批次及按列批次的候选在开发集从 8.42 升至 9.75，严格完整成功均为 8/12。Agent 选择按列和批次修正的方案。独立确认中，终局均分从 9.00 降至 8.60，完整成功从 12/20 降至 11/20，有 3 局少计至少 4 块。种子 `110402` 的第一批均入盒 5 块，候选的第二批从 5 块降至 0 块。Agent 保留参考策略。 [首轮结果](../artifacts/placement-confirm-20261120/result.json) · [逐局回执](../artifacts/placement-confirm-20261120/confirmation/rollouts/results.json)

下一轮读取 16 局平衡训练样本及 3 局定向训练样本，要求一项修正在至少两局带来正向计数变化。Agent 选出首批向内 3 mm 和抬高 8 mm 的方案。开发集均分从 8.42 升至 9.75，独立确认的均分从 9.45 降至 8.45，配对差 −1.00 块（95% 区间 −1.85 至 −0.15）。种子 `110618` 的第一批早已入盒 5 块，第二批却从 5 块降至 0 块。第一批在目标盒中的位置变化会影响第二批下降。 [独立确认](../artifacts/placement-repeated-gains-20261120/result.json) · [逐局回执](../artifacts/placement-repeated-gains-20261120/confirmation/rollouts/results.json)

选择性修正将搬运终点可观测的夹爪位置投影到估计目标盒的横轴。Agent 按来源列学习启用阈值，在训练集保留两局成功修复，并将启用场景从 19 例缩至 7 例。候选开发集均分从 8.42 升至 9.33。批间距搜索在这些启用场景里重跑完整回合，检验首批动作改变后，第二批向盒外侧偏移 3、6 或 9 mm 对两批终局计数的影响。 [启用阈值和训练场景](../artifacts/placement-activation-20261120/fit.json) · [批间距搜索协议](../artifacts/placement-spacing-20261120/protocol.json)

联合回放使用上游默认判分和默认首次终止时刻，并继续记录第 500 步状态。释放口径调用上游单块释放判定。历史校验逐项匹配 10 份原生回执，正式确认额外使用锁定的每列一个种子，对候选分别原生运行两种口径。 [联合回放](../src/a3opt/placement_benchmark.py#L16) · [历史校验](../artifacts/placement-benchmark-parity-20261120/audit.json)

## 最高档独立复核

`advanced` 的盒位各轴在 ±20 mm 内变化，目标盒偏航在 ±0.08 rad 内变化，固定与腕部相机、视场、灯光和饼干颜色按上游配置随机化。饼干的初始平面扰动为 ±0.3 mm，偏航为 ±0.015 rad，因此仍能看到规则排布。每组 20 局使用 4 列各 5 局，策略从任务文本读取列号。 [上游随机化说明](../../a3_dual_arm_sim/docs/randomization.md) · [项目介绍第 9–11 页](../../_build/project_intro.txt#L95)

| 冻结候选 | 独立种子 | 默认终局均分 | 第 500 步释放均分 | 严格完整成功 | 配对差 95% 区间 |
| --- | --- | ---: | ---: | ---: | ---: |
| 选择性首批修正 | `110800–110819` | 8.45 → 8.30 | 8.45 → 8.30 | 12/20 → 10/20 | −0.80 至 +0.50 |
| 首批修正后第二批平移 3 mm | `111100–111119` | 8.80 → 9.00 | 8.80 → 9.00 | 14/20 → 13/20 | 0.00 至 +0.60 |

首批修正在 `110815` 救回 5 块，在 `110807` 损失 5 块。批间距候选的全部计数增益来自 `111107` 的 1→5 块。`111119` 仍有 10 块入盒，槽位匹配从 10 个降到 9 个，严格保持从 27 步降到 0 步。批间距候选均分的 95% 区间为 8.25–9.75。两轮的候选各有 8 次原生评测复跑吻合，安全停机和错列计分均为 0。Agent 根据完整成功门槛保留 `branch-joint` 加第 4 列第二批 `0.75×` 时序。 [首批确认](../artifacts/placement-activation-20261120/result.json) · [批间距确认](../artifacts/placement-spacing-20261120/result.json) · [原生复跑核验](../artifacts/placement-spacing-20261120/confirmation/audit.json)

较早的三组共 60 个确认种子上，冻结参考的第 1–4 列均分分别为 8.67、7.73、9.80 和 9.67，每列 15 局。第 2 列占这三组的 34/62 块失分，因此本轮按批次记录下降前末端位置，再用额外训练回执检查第二批的落点和首批已入盒饼干是否被带动。确认种子仅供审计和故障归因。

汇总四组放置候选的冻结参考回执，共 80 个不重复的最高档确认种子，每列 20 局。默认终局和第 500 步释放均分均为 **8.925/10**，按列分层 bootstrap 的 95% 区间为 **8.45–9.38**，严格完整成功 **49/80**。第 1–4 列均分为 **8.75 / 7.60 / 9.60 / 9.75**。安全停机与错列均为 0。四个批次参考使用同一个检查点及第 4 列二批时序。 [第一组](../artifacts/placement-confirm-20261120/confirmation/rollouts/results.json) · [第二组](../artifacts/placement-repeated-gains-20261120/confirmation/rollouts/results.json) · [第三组](../artifacts/placement-activation-20261120/confirmation/rollouts/results.json) · [第四组](../artifacts/placement-spacing-20261120/confirmation/rollouts/results.json)

## Agent 搜索分支

| 搜索分支 | 训练数据和选择方式 | 推理观测 |
| --- | --- | --- |
| 二批终局搜索 | 24 个新训练种子，第二批搬运结束后分支。首批门控实际触发时重跑状态改变的整段回合。零修正的分支须重现命令、物理和终局计数。 | 任务文本、首帧盒位估计、下降前末端位置 |
| 力觉触发释放 | Agent 从旧训练集挑选 9 个场景。有 3 个场景触发了最灵敏的二批分支，训练中各阈值与参考的计数增量均为 0。Agent 结束该分支。 | 下降后段连续两步的腕部力变化 |
| 横向落点校准 | 从 19 个训练场景提取 38 条搬运末端数据。逐种子留一验证夹爪与饼干中心偏差，拟合各列各批的中位修正和误差死区。 | 任务文本、首帧盒位估计、末端位置 |
| 源位与盒位联合拟合 | Agent 在训练槽位失分较多的列上采样，验证首批、第二批、两批使用联合姿态回归时的终局配对增益。 | 任务文本、首帧源盒和目标盒的视觉估计 |

项目 Agent 负责训练样本选择、MuJoCo 采集、拟合和选模。上游特权状态提供训练标注和评测回执。继续确认时，独立种子、规则、模型和训练回执通过锁文件绑定。 [二批终局搜索](../src/a3opt/placement_tail.py) · [力觉拟合](../artifacts/placement-contact-v2-20261120/fit.json) · [横向落点拟合](../artifacts/placement-centering-20261120/fit.json) · [联合拟合](../src/a3opt/placement_conditioning.py)

## 后续搜索与回退

| 分支 | Agent 观察到的变化 | 决策 |
| --- | --- | --- |
| 横向落点校准 | 38 条搬运末端测量。仅改二批和同时改两批在 12 个新开发种子上的两种口径均为 8.75→8.75，严格完整成功均为 6/12。 | 开发集回退。 [结果](../artifacts/placement-centering-20261120/result.json) |
| 源位和盒位联合拟合 | 第 2 列 5 个训练种子分别复跑首批、二批和两批联动，得分总变化 −1、−2、−1。 | 训练集回退。 [拟合](../artifacts/placement-conditioning-20261120/fit.json) |
| 开爪曲线 | Agent 从历史训练回执选出第 2 列两个失败局和两个批次对照。加快开爪、增大开度、抬高 6 mm 后加快开爪的终局计数和完整成功变化均为 0。 | 训练集回退。 [拟合](../artifacts/placement-release-20261120/fit.json) |
| 按列搜索二批落点 | 24 局平衡训练，9 种非零位移。各列未出现至少两次可重复的净改善。 | 训练集回退。 [拟合](../artifacts/placement-tail-20261120/fit.json) |
| 跨列共享二批外移 | 重用这 24 局中未经首批干预的分支。二批外移 3 mm 分别在第 2、3 列救回 5 块，24 局中减分和槽位退化均为 0。外移 6 mm 的终局总变化为 −5 块。 | 16 局开发集两种计数均为 9.375→9.375，严格完整成功持平于 12/16。 [结果](../artifacts/placement-pooled-20261120/result.json) |
| 首批与二批联合搜索 | 用同一二批动作比较两种实际到达的物理状态。训练为第 1、4 列启用首批门控。 | 开发集 9.375→9.0625，严格完整成功 12→11。第 4 列 `118010` 从 10 降到 5。 [结果](../artifacts/placement-hybrid-20261120/result.json) |
| 腕部姿态修正 | 在 13 个训练场景比较 ±0.06 rad 后，Agent 在第 2、3 列选择首批姿态修正，与共享二批外移联合。 | 开发集 9.375→9.0625，严格完整成功持平于 12/16。第 2 列 `118013` 从 10 降到 5。 [结果](../artifacts/placement-wrist-20261120/result.json) |
| 失败驱动的专家采样 | Agent 在二批外移后仍然失分的训练场景中，每列最多选择 2 局重采专家示范。6 局专家回合完整成功，保留 12 个批次轨迹。补充数据用于重拟合从下降到撤臂的动作。 | 两种口径开发均分均从 9.375 降到 9.0625；严格完整成功 12→13，但第 2 列一局 10→5，因此回退。 [训练](../artifacts/placement-demonstrations-v2-20261120/fit.json) · [开发](../artifacts/placement-demonstrations-v2-20261120/result.json) |
| 第二批抓取恢复 | 16 局平衡新训练场景，对 1 个漏抓和 4 个预定对照分支测试闭爪等待、增力与下探重夹。 | 下探重夹只使漏抓局 5→6；增力使另一局失去严格完整成功。尚不足以部署。 [回执](../artifacts/placement-grasp-20261120/result.json) |
| 放置阶段时序 | 12 个训练分支点分别比较原动作、平滑下降、提前卸力和下降后夹持停留。每个分支一直运行到终局，原动作分支必须精确复现原回合。 | 平滑下降在两个种子各多计 1 块，但分别来自不同批次。提前卸力和停留各净减 4 块；单批候选未晋级。 [配对数据](../artifacts/placement-phase-20261120/training/dataset.json) · [筛选](../artifacts/placement-phase-20261120/fit.json) |

### 按失分机制验证整条策略

训练回执中，`110002` 和 `110010` 的两批饼干在搬运结束时最大倾角都低于 1°，到下降结束时已约为 24–28°。新增时序模块保持放置路径的起点、终点和步数，仅改变下降段的速度分布。Agent 在第 2 列的五组整局训练配对中同时启用两批，两种口径均分从 7.2 升到 7.4，严格完整成功仍为 1/5。只有 `110107` 从 9 升到 10，另一局的单批收益未能在组合后复现，因此不进入开发或独立确认。 [时序模块](../src/a3opt/placement_phase_agent.py) · [组合结果](../artifacts/placement-phase-compose-20261120/result.json)

抓取分支另行处理源盒内堵塞。历史训练局 `123005` 的闭爪指令和实际开度之差约为 0.238，15 个正常抓取约为 0.026–0.028。Agent 据此在第 2 列重放该局并新采 20 局，比较下探重夹、抬起再插入以及两个有界的侧向重夹方案。21 局中有 2 局出现开度异常且未抓起五块：只有旧局从 5 升到 6，新局各种重夹均为 5 分，因而没有部署。另有一局 `125063` 未抓起任何饼干，但开度几乎追上指令，表明仅靠高开度差不能覆盖空抓。 [重夹模块](../src/a3opt/placement_regrasp_agent.py) · [训练筛选](../artifacts/placement-regrasp-20261120/fit.json)

视觉反馈分支在相同的 24 个历史训练种子上补采正面相机首帧和二批抬升前的帧，保存图像、公开视觉特征和第一批的训练标签。24 局的动作前缀、后续终局命令及物理状态均与原回执匹配。简单的颜色差总量在几乎未计分（0–1 块）与正常（至少 4 块）的场景中重叠，Agent 未选中候选。未计分的倾斜饼干也可能出现在盒内图像里。 [训练回执](../artifacts/placement-visual-feedback-20261120/training/dataset.json) · [结果](../artifacts/placement-visual-feedback-20261120/result.json)

Agent 在同一批图像上将目标盒区域透视校正，使用 4×4 网格的颜色变化及其相对分布，训练带类别均衡权重的岭回归门控。逐种子留一中，6 个 0–1 块的样本均触发、17 个至少 4 块的样本均未触发，1 个中间样本不参与二分类拟合。二批外移 3 mm 在门控触发的训练分支中只在第 2 和第 3 列各多计 5 块，另外 4 局持平。真值计数只用于训练标签，部署只读取两帧 RGB；进入 12 局全新四列平衡开发评测，留一分数不是课程分数。 [排布模型](../src/a3opt/placement_visual_geometry.py) · [训练与选模](../artifacts/placement-visual-geometry-20261120/fit.json)

### 两帧门控独立确认与双视角校准

两帧门控在 12 个开发种子上从 **8.33 提至 8.75**。计数变化来自第 2 列种子 `128010` 的 0→5。20 个新确认种子 `128100–128119` 上，默认终局和第 500 步释放均为 **8.80→8.80**，候选均分的 95% 分层区间为 **[8.05, 9.50]**。4 局动作发生变化，20 局计数全部持平，严格完整成功均为 **10/20**。每列一个种子分别按两种口径原生复跑，共 8 条记录吻合。Agent 驳回门控候选。 [独立确认](../artifacts/placement-visual-geometry-20261120/result.json) · [原生复跑](../artifacts/placement-visual-geometry-20261120/confirmation/audit.json)

加入这组确认的 20 局参考回执后，同一冻结基线在 100 个唯一种子上的两种计数均为 **8.90/10**，95% 分层区间为 **[8.47, 9.29]**，严格完整成功 **59/100**。第 1–4 列各 25 局，均分分别为 **8.40 / 7.92 / 9.68 / 9.60**。第 1、2 列共少计 92 块，占全部 110 块失分的约 84%。 [最新一组参考回执](../artifacts/placement-visual-geometry-20261120/confirmation/rollouts/results.json)

阶段核对显示，终局少计的 110 块中，100 块有搬运结束位置记录。81 块在搬运结束时位于盒底上方至少 35 mm 且倾角不超过 15°，下降结束后倾角超过 15°。其中第 1 列的全部 40 块失分均符合这个条件。这里用确认回执核验故障分布，Agent 的训练采样清单来自训练回执。

Agent 另采 360 个全新 `advanced` 重置场景，使用 240 局拟合、60 局选结构、60 局验收。候选结构比较正面 RGB 与正面加左腕 RGB，在源盒 XY 和目标盒 XY／偏航上分别学习对冻结估计的限幅修正。候选在验收集的源盒 XY 平均误差从 **0.648 mm** 升至 **0.727 mm**，第 90 百分位误差从 **1.264 mm** 升至 **1.353 mm**。目标盒修正的最佳权重为零。Agent 在视觉验收阶段结束该支路。 [校准模块](../src/a3opt/placement_calibration.py) · [选模和验收](../artifacts/placement-calibration-20261120/fit.json)

完整三视角录像各含初始帧、500 步画面和 40 帧终局停留。录像逐局校验终局计数、实际执行动作和原始命令哈希，并逐帧解码。

- 确认局 `128100` 为 5→5。 [参考 MP4](../artifacts/placement-visual-geometry-20261120/videos/strict-500/128100-reference.mp4) · [门控 MP4](../artifacts/placement-visual-geometry-20261120/videos/strict-500/128100-selected.mp4) · [录像审计](../artifacts/placement-visual-geometry-20261120/videos/strict-500/128100-audit.json)
- 开发局 `128010` 为 0→5。 [参考 MP4](../artifacts/placement-visual-geometry-20261120/videos/strict-500/development/128010-reference.mp4) · [门控 MP4](../artifacts/placement-visual-geometry-20261120/videos/strict-500/development/128010-selected.mp4) · [录像审计](../artifacts/placement-visual-geometry-20261120/videos/strict-500/development/128010-audit.json)

### 下降轨迹与腕部反馈

Agent 从已审计训练回执选取故障和成功场景，再补采新场景。下降控制比较现有轨迹、保持搬运终点 XY 的下降、提前对齐到计划落点、对齐到中点及提前对齐并抬高 4 mm。每种轨迹分别用于第一批、第二批和两批，分支一直运行到整局终点。末端位置取自公开关节和工具观测，横向命令修正限于每轴 8 mm，夹爪命令和各阶段步数沿用冻结计划。Agent 按列要求两次独立收益和至少两个成功计数对照。 [下降控制与选模](../src/a3opt/placement_descent_agent.py) · [协议](../artifacts/placement-descent-20261120/protocol.json)

腕部反馈记录每批搬运结束时的左腕 RGB、工具位姿和计划开爪位置。Agent 从同一批图像提取目标盒与彩色饼干的空间特征，用仿真几何标注夹持批次相对工具的中心偏移，再比较只用工具状态和加入腕部图像的回归误差。推理在下降前估计批次落点，以 `2×5` 实际槽位和留出误差计算有界修正。几何标注存放在训练回执中，控制器从 RGB、工具状态和冻结参数生成 16 维动作。 [腕部反馈](../src/a3opt/placement_wrist_rgb.py) · [协议](../artifacts/placement-wrist-geometry-20261120/protocol.json)

56 个训练回合留下 111 组可用的夹持末端记录。加入腕部 RGB 后，留出集横向平均误差从 **1.073 mm** 增至 **1.281 mm**，纵向从 **0.075 mm** 增至 **0.101 mm**。Agent 在视觉验收处结束该候选。随后，定向采样分支复用这些训练回合，从下降预采样结果中选出在多个种子受益的轨迹，再从腕部采集中自动选出两批均完成搬运的失分场景和成功对照，按首批、二批和两批组合重做整局训练比较。 [腕部验收](../artifacts/placement-wrist-geometry-20261120/fit.json) · [定向采样](../src/a3opt/placement_descent_refine.py)

共享外移和两批联合使用同一组 16 个开发种子，每列 4 局。两批联合的开发门槛同时比较冻结参考和共享外移。候选确认集各自预留 20 个新种子，每列 5 局。开跑前冻结检查点、动作参数、开发回执和教师评测口径。晋级要求默认终局均分超过 9，同时两种口径都提高终局均分、完整成功数持平或增加、没有单局少计至少 4 块、错列或安全停机。确认结果分别报告点估计及按列分层配对 bootstrap 区间。 [共享协议](../artifacts/placement-pooled-20261120/protocol.json) · [联合协议](../artifacts/placement-hybrid-20261120/protocol.json)

9 分目标指四列平衡 `2×5 / advanced` 的默认终局均分。训练集的增益集中在少数场景，且同一套训练分支供两种结构比较，因此确认回执与开发回执需要分开解读。

### 下降与释放的对照设计

固定预算为 500 个控制步，每个确认集使用 40 个唯一种子且每列 10 局。主要指标是教师默认条件下回合终止时的入盒数，另记第 500 步的已释放数和严格条件下连续 20 步完整成功。参考与候选每对共享场景初始化。候选训练配置、代码和种子在确认前冻结，并用每列一局、每局两种口径的原生回放核验动作和终局。 [课程主指标](../../_build/project_intro.txt#L126) · [独立复核入口](../scripts/audit_placement_confirmation.py)

| 候选 | Agent 的训练数据与选模信号 | 干预的位置 |
| --- | --- | --- |
| 约束下降 | 从失败与成功训练回执选场景，再补采随机场景。比较不同下降路径和适用批次；每列至少两局收益和两局满分对照。 | 在下降开始时读取末端 XY，各配置改变横向移动的时机。 |
| 定向约束下降 | 从腕部训练回执复采失分与成功场景，比较首批、第二批和两批组合。 | 与约束下降共用控制器，单独验证另一组训练数据。 |
| 末端位置反馈 | Agent 自动选出下降倾倒的训练场景与满分对照，补采新种子。候选与冻结参考、开环垂直下降配对。 | 下降与开爪每个控制步读取末端 XY，采用 0.5 mm 死区与 3 mm 残差幅度上限。 |
| 开爪高度与姿态 | 在训练回合中拟合饼干底面到末端的释放高度，并在同一物理状态分支比较 4 mm、7 mm 间隙和姿态保持。 | 下降开始时读取末端四元数，按配对收益决定何时保持姿态。 |
| 盒坐标落点 | 从专家执行动作拟合每列、每批在盒坐标中的开爪位置，用腕部训练回执确定偏差与复采对象。 | 根据首帧视觉估计的盒位，将纵向对齐提前到下降前 40% 轨迹，夹爪时序沿用原计划。 |

训练用仿真位姿和得分生成标签。部署控制使用首帧 RGB、任务文本、公开末端位姿、标称运动学和冻结参数。对照脚本分别查询改动前后的同一场景，提升时还要求两种计数的配对增益为正、完整成功数持平或增加，并保持零严重回退、零错列和零安全停机。 [训练示例](../artifacts/placement-servo-20261120/protocol.json) · [释放高度协议](../artifacts/placement-clearance-20261120/protocol.json) · [盒坐标协议](../artifacts/placement-retarget-20261120/protocol.json)

### 定向下降确认结果

Agent 复采第 1 列的三局失分场景和三局满分对照，从首批、第二批和两批方案中选定首批垂直下降。`branch-joint` 保留第 4 列第二批的 `0.75×` 接近与搬运时长。改动在第 1 列首批的下降、开爪和退臂生效。策略以搬运结束时实际观测的末端 XY 为下降锚点，并在退臂过程中平滑撤除修正。 [训练拟合](../artifacts/placement-descent-focused-v2-20261120/fit.json) · [冻结策略](../artifacts/placement-descent-focused-v2-20261120/confirmation-lock.json)

| 选模方案 | 开发集均分 | 40 局确认均分 | 严格完整成功 | 少计至少 4 块 |
| --- | ---: | ---: | ---: | ---: |
| 第 1 列两批和第 2 列首批垂直下降 | 8.25 → 8.75 | 9.175 → 8.15 | 26 → 23 | 5 局 |
| 第 1 列仅首批垂直下降 | 7.90 → 8.15 | 8.675 → **9.05** | 20 → **24** | 0 局 |

两个确认集的默认终局和第 500 步释放计数在各自组内相同。定向候选的均分 95% 分层区间为 **[8.475, 9.55]**，配对增益为 **+0.375 块**，95% 区间为 **[0, +0.75]**。三个第 1 列场景各从 5 增至 10 块，其余 37 局计数持平。第 1 列均分从 7.5 提到 9.0。两种计数口径中，错列和安全停机均为 0。两组各有 8 次原生复跑，动作及终局与冻结评测一致。 [全程结果](../artifacts/placement-descent-20261120/result.json) · [定向结果](../artifacts/placement-descent-focused-v2-20261120/result.json) · [定向候选原生复跑](../artifacts/placement-descent-focused-v2-20261120/confirmation/audit.json)

三个增分场景的收益都来自首批。下降结束时首批最大倾角从 20.6°–23.3° 降至 1.0°–1.4°，第二批在前后各计 5 块。第四个场景前后都计 10 块，满足严格完整条件的连续保持由 0 步升至 81 步。较宽的候选在第 2 列使四个满分场景掉到 0 或 1 块，说明首批改动可能影响随后两批是否入盒。 [定向逐局记录](../artifacts/placement-descent-focused-v2-20261120/confirmation/rollouts/results.json) · [较宽配置逐局记录](../artifacts/placement-descent-20261120/confirmation/rollouts/results.json)

### 冻结策略独立复验

沿用 `132100–132139` 确认时的策略和评测规则，在 60 个额外场景 `138000–138059` 比较冻结参考与候选，每列 15 局。审计脚本在运行前核验原始确认和种子冲突，写入策略、模拟器及协议哈希。复验新增 8 次原生回放均匹配动作哈希、饼干身份、终局和成功标记。 [复验协议](../artifacts/placement-descent-replication-20261120/protocol.json) · [复验结果](../artifacts/placement-descent-replication-20261120/result.json) · [原生回放](../artifacts/placement-descent-replication-20261120/confirmation/audit.json)

| 数据 | 默认终局均分 | 第 500 步释放均分 | 严格完整成功 | 均分 95% 区间 | 配对增益 95% 区间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 首次确认 40 局 | 8.675 → 9.05 | 8.675 → 9.05 | 20/40 → 24/40 | [8.475, 9.55] | [0, +0.75] |
| 独立复验 60 局 | 8.867 → 9.20 | 8.867 → 9.20 | 39/60 → 41/60 | [8.70, 9.62] | [+0.08, +0.67] |
| 合计 100 局 | 8.79 → **9.14** | 8.79 → **9.14** | 59/100 → **65/100** | [8.77, 9.49] | [+0.15, +0.60] |

两组共 100 个唯一种子，每列 25 局。候选的第 1–4 列均分为 **9.20 / 8.32 / 9.60 / 9.44**。第 1 列从 7.80 提升到 9.20，七个确认或复验场景各多计 5 块。100 局中计数下降、错列及安全停机均为 0。两种计数的配对均分提升为 +0.35；候选均分的区间下限为 8.77，9 分落在区间内。严格完整成功要求槽位和释放条件同时满足并持续 20 步。 [100 局分列结果](../artifacts/placement-descent-replication-20261120/result.json)

### 其他候选与复现

列级组合与第一种配置相同，Agent 在读取开发回执后跳过了重复运行。实时末端反馈和校准开爪高度在训练筛选处结束。盒坐标落点在第 1 列六个训练场景的两种计数仍为 7.50，完整成功从 1/6 增至 2/6。 [列级组合](../artifacts/placement-descent-compose-20261120/result.json) · [末端反馈](../artifacts/placement-servo-20261120/fit.json) · [开爪高度](../artifacts/placement-clearance-20261120/fit.json) · [盒坐标落点](../artifacts/placement-retarget-20261120/fit.json)

确认种子 `132114` 的释放计数从 5 提到 10。两份三视角录像各有 501 帧控制画面（含重置帧）和 40 帧终局停留，已逐帧解码并核对每局动作哈希与终局。 [参考 MP4](../artifacts/placement-descent-focused-v2-20261120/videos/strict-500/132114-reference.mp4) · [定向候选 MP4](../artifacts/placement-descent-focused-v2-20261120/videos/strict-500/132114-selected.mp4) · [录像审计](../artifacts/placement-descent-focused-v2-20261120/videos/strict-500/132114-audit.json)

在项目目录运行以下命令可使用上游公开评测入口复现该场景，默认条件下第 409 步判定完整成功并记 10 块。 [上游入口回执](../artifacts/placement-descent-focused-v2-20261120/upstream-cli-132114.json)

```powershell
$sim = (Resolve-Path '..\a3_dual_arm_sim').Path
$env:PYTHONPATH = "$PWD\src;$sim\src"
$env:A3OPT_DESCENT_LOCK = (Resolve-Path 'artifacts/placement-descent-focused-v2-20261120/confirmation-lock.json').Path
& "$sim\.venv\Scripts\python.exe" -m a3_dual_arm_sim.workflows.benchmark_cookie_batch_cli `
  --task 2x5 --policy a3opt.placement_descent_agent:upstream_policy `
  --profile advanced --source-column random --prompt-style task `
  --episodes 1 --seed-start 132114 --max-steps 500 --workers 1 `
  --output artifacts/placement-descent-focused-v2-20261120/upstream-cli-132114.json
```

```powershell
$sim = (Resolve-Path '..\a3_dual_arm_sim').Path
$env:PYTHONPATH = "$PWD\src;$sim\src"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
& "$sim\.venv\Scripts\python.exe" -u -m a3opt.placement_search `
  --out artifacts/placement-spacing-20261120 `
  --reference artifacts/adaptive-branch-20261119 `
  --timing artifacts/release-timing-20261120 `
  --reuse-acquisition artifacts/placement-search-20261120 `
  --active-training artifacts/placement-active-20261120 `
  --parity-audit artifacts/placement-benchmark-parity-20261120/audit.json `
  --training-scenes 16 --minimum-training-gains 2 --confirmation-offset 1000 `
  --gated-placement --learn-interbatch-spacing --workers 8
```
