# `2×5 / advanced` 放置策略组合评测

候选从已有的分阶段示范策略出发，按任务来源列选择放置修正。第 2 列仅在第二批抬高开爪，第 4 列仅在首批的计划搬运与下降终点同时达到阈值时启用修正。Agent 依据已校验的历史回执拟合并冻结组合。两组评测各有 64 个不同种子，每列 16 个，使用上游 MuJoCo 模拟器和 500 控制步上限。 [训练与实验记录](../../docs/A3_PLACEMENT_SEARCH.md) · [基础检查点](../advanced-20261120/README.md)

| 场景 | 课程终局均分 | 第 500 步释放均分 | 严格完整成功 |
| --- | ---: | ---: | ---: |
| 首次确认 | 9.375 → 9.59375 | 9.375 → 9.59375 | 44 → 44/64 |
| 独立复验 | 8.390625 → 8.53125 | 8.390625 → 8.53125 | 34 → 33/64 |
| 合并统计 | 8.8828125 → 9.0625 | 8.8828125 → 9.0625 | 78 → 77/128 |

Agent 在独立复验后回退组合。复验的课程配对增益 95% 区间为 [−0.078125, +0.421875]，合并的候选均分 95% 区间为 [8.640625, 9.4453125]。两组总计 16 次原生复跑匹配。 [独立复验结果](replication-result.json)

| 文件 | 内容 |
| --- | --- |
| [`manifest.json`](manifest.json) | 冻结候选配方、参考检查点及逐局回执的 SHA-256。 |
| [`confirmation-result.json`](confirmation-result.json) 与 [`replication-result.json`](replication-result.json) | 首次确认和独立复验的配对摘要与判定。合并结果只作描述，晋级依据独立复验。 |
| [`confirmation-rollouts.json.gz`](confirmation-rollouts.json.gz) 与 [`replication-rollouts.json.gz`](replication-rollouts.json.gz) | 原始逐局回执的无损压缩，两套计分均保存每对场景的动作及终局。 |
| [`confirmation-protocol.json`](confirmation-protocol.json) 与 [`replication-protocol.json`](replication-protocol.json) | 种子、评分规则、代码和仿真配置指纹。 |
| [`native-replays.json.gz`](native-replays.json.gz) | 每组按列抽取一局，每局以两套原生口径复跑。 |

## 重算配对分数

在仓库根目录运行以下片段。重算步骤只需要安装优化器。模拟器独立复跑使用协议中记录的上游提交和依赖环境。

```python
import gzip
import json
from hashlib import sha256
from pathlib import Path

from a3opt.placement_routed_agent import decode
from a3opt.placement_search import compare

folder = Path('evidence/routed-20261120')
manifest = json.loads((folder / 'manifest.json').read_bytes())
model = decode(manifest['corrections'])
for phase in ('confirmation', 'replication'):
    filename = f'{phase}-rollouts.json.gz'
    raw = gzip.decompress((folder / filename).read_bytes())
    assert sha256(raw).hexdigest() == manifest['rollout_sha256'][filename]
    paired = json.loads(raw)
    result = json.loads((folder / f'{phase}-result.json').read_bytes())[phase]
    for rule in ('default', 'strict-500'):
        measured = compare(*([episode[rule] for episode in paired[arm]] for arm in ('reference', 'selected')), model)
        assert all(result[rule][key] == value for key, value in measured.items())
        print(phase, rule, measured['reference_mean'], measured['mean'], measured['paired_count_ci95'])
```
