# `2×5 / advanced` 放置反馈实验

各分支从同一个发布检查点出发。课程主指标为上游模拟器的回合终局入盒数，每回合最多 500 控制步。拟合和开发先使用特定来源列，确认使用四列平衡的独立种子。项目 Agent 自主选择训练场景、采集数据及拟合候选。 [放置实验记录](../../docs/A3_PLACEMENT_SEARCH.md) · [发布检查点](../advanced-20261120/README.md)

## 搬运末端净空

Agent 用 30 个场景拟合，留一验证选出末端位姿模型，冻结后采集 24 个新场景。物理净空不超过 2 mm 时触发。公开观测和离线净空标签按场景记录。

| 模型 | 漏判 | 误触发 | 标签不可用时触发 | 净空平均绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| 计划位置 | 6/15 | 3/30 | 2/3 | 1.648 mm |
| 末端位姿 | 6/15 | 3/30 | 2/3 | 1.663 mm |

两种模型的漏判和误触发数相同。Agent 在留出阶段回退候选。20 个开发场景和 64 个确认场景均保留。 [协议](clearance-protocol.json) · [拟合回执](clearance-fit.json) · [逐批预测与标签](clearance-sensor-holdout.json) · [判定](clearance-result.json)

| 文件 | 内容 |
| --- | --- |
| [`clearance-fit-dataset.json.gz`](clearance-fit-dataset.json.gz) | 30 个拟合场景的观测、净空标签、动作前缀和图像分片哈希。 |
| [`clearance-holdout-dataset.json.gz`](clearance-holdout-dataset.json.gz) | 24 个独立留出场景的同格式回执。 |
| [`clearance-wrist-frames.zip`](clearance-wrist-frames.zip) | 54 个场景在两批搬运末端的原始腕相机图像，按种子保存为 NumPy 分片。 |

解压数据集后可用 `clearance-fit.json` 中已冻结的两组参数复算预测。各图像分片可与数据集中的 `shard_sha256` 核对。驱动在留出完成后修正了首次推理时的相机请求，对已记录的采集与预测无影响。

## 下降中的接触反馈

Agent 复采 14 个历史场景，每例保存相同模型状态下原动作和三个候选动作的后续回执。原动作分支的物理末态与命令哈希均匹配实际连续回合。留一验证选择第 1 列首批的腕部力矩门控。12 对新场景的课程终局均分和第 500 步释放均分均为 **9.5833 → 9.5833/10**，严格完整成功为 **10 → 10/12**。其中 6 局命令发生改变，计数均持平。Agent 回退，保留 64 个确认种子。 [协议](contact-protocol.json) · [拟合回执](contact-fit.json) · [训练分支](contact-training.json.gz) · [逐局开发回执](contact-development.json.gz) · [结果](contact-result.json)

## 第 2 列第二批抬高开爪门控

Agent 从先前评测中重建 4 组共 68 对训练样本。两套计分下，逐组留出门控的净增均为 **13 块**，常开抬高开爪为 **46 块**。误用导致的减分局由 4 局减至 2 局，增分局也由 12 局减至 4 局。Agent 在训练筛选后回退门控。 [协议](late-hover-protocol.json) · [训练数据](late-hover-training.json.gz) · [逐组结果](late-hover-fit.json)
