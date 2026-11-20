# `2×5 / advanced` 冻结评测证据

该目录记录 2026 年 11 月 20 日发布策略的本地 MuJoCo 配对评测。默认终局和第 500 步释放口径均为 **8.79 → 9.14/10**，严格完整成功为 **59 → 65/100**。每批评测均平衡四个来源列。 [架构](../../docs/A3_CURRENT_ARCHITECTURE.md) · [实验树](../../docs/A3_EXPERIMENT_TREE.md)

| 文件 | 内容 |
| --- | --- |
| [`controller.json`](controller.json)、[`policy.npz`](policy.npz) | 发布版冻结配方、检查点、代码和原始回执哈希。 |
| [`paired-endpoints.csv`](paired-endpoints.csv) | 每个种子两套口径的配对成绩、步数、成功标记、错列与安全事件，以及命令与实际执行的动作哈希。 |
| [`confirmation-rollouts.json.gz`](confirmation-rollouts.json.gz)、[`replication-rollouts.json.gz`](replication-rollouts.json.gz) | 原始逐局回执的无损压缩，解压后的内容可与原始 SHA-256 核对。 |
| [`native-replays.json.gz`](native-replays.json.gz) | 16 份原生复跑回执及两批复跑清单，均来自同一冻结候选。 |
| [`confirmation-protocol.json`](confirmation-protocol.json)、[`replication-protocol.json`](replication-protocol.json) | 首次确认和复验的原始协议，含种子、仿真版本、配置及代码指纹。 |
| [`confirmation-result.json`](confirmation-result.json)、[`replication-result.json`](replication-result.json) | 原始汇总及按列分层的区间、成功数和风险指标。 |
| [`acquisition-plan.json`](acquisition-plan.json)、[`broad-descent-result.json`](broad-descent-result.json)、[`two-by-ten-result.json`](two-by-ten-result.json) | 定向采集对象、较宽下降方案的回退回执、独立 `2×10` 分支的决策。 |
| [`run-index.csv`](run-index.csv) | 本地专项运行目录及根目录结果状态的快照。 |

## 从回执核对发布分数

安装本仓库的 Python 包后，在仓库根目录运行。该步骤可在未安装模拟器的环境中完成。

```python
import gzip
import json
from pathlib import Path

from hashlib import sha256
from a3opt.placement_descent_agent import Descent
from a3opt.placement_search import compare

folder = Path('evidence/advanced-20261120')
controller = json.loads((folder / 'controller.json').read_bytes())
expected = json.loads((folder / 'replication-result.json').read_bytes())['pooled']
model = tuple(Descent(*record) for record in controller['corrections'])
rollouts = {'reference': [], 'selected': []}
for name, digest in controller['rollout_sha256'].items():
    raw = gzip.decompress((folder / name).read_bytes())
    assert sha256(raw).hexdigest() == digest
    report = json.loads(raw)
    for arm in rollouts:
        rollouts[arm].extend(report[arm])
for rule in expected:
    measured = compare(*([row[rule] for row in rollouts[arm]] for arm in rollouts), model)
    assert all(expected[rule][key] == value for key, value in measured.items())
    print(rule, measured['reference_mean'], measured['mean'], measured['episodes'])
```

## 在上游模拟器复跑一对场景

在同一 Python 3.12+ 环境中安装本仓库的 `.[sim]` 与单独检出的上游模拟器。将上游切换到 `controller.json` 中的 `sim_revision`。历史评测的 MuJoCo、NumPy 与 Gymnasium 版本记录在确认协议中，跨环境可先核对终局及来源列；动作哈希精确匹配还取决于仿真与依赖版本。 [上游仓库](https://github.com/Eter0109/a3_dual_arm_sim/tree/5fa24970648f53a24c0b5038d19bf6a14e1be5b9)

在本仓库根目录用 Python 运行以下片段。种子 `132114` 的两种计分均为 5 → 10。

```python
import json
from hashlib import sha256
from pathlib import Path

from a3opt.placement_benchmark import episode
from a3opt.placement_descent_agent import Descent, DescentPolicy

folder = Path('evidence/advanced-20261120')
controller = json.loads((folder / 'controller.json').read_bytes())
checkpoint = folder / controller['checkpoint']
assert sha256(checkpoint.read_bytes()).hexdigest() == controller['checkpoint_sha256']
corrections = tuple(Descent(*record) for record in controller['corrections'])
for label, model in (('reference', ()), ('selected', corrections)):
    result = episode(checkpoint, *controller['timing'], 132114, model, policy_type=DescentPolicy)
    print(label, {rule: row['requested_score'] for rule, row in result.items()})
```
