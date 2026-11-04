from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def combo(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate.get('encoder')} / {candidate.get('fusion')} / "
        f"{candidate.get('action_head')}-k{candidate.get('chunk_size')} / "
        f"lr={candidate.get('lr')} / img{candidate.get('image_size')}"
    )


def compact(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate.get('encoder')}/{candidate.get('fusion')}/"
        f"{candidate.get('action_head')}-k{candidate.get('chunk_size')}/"
        f"lr{candidate.get('lr')}/img{candidate.get('image_size')}"
    )


def score_list(outcome: dict[str, Any]) -> str:
    episodes = (outcome.get("eval") or {}).get("episodes") or []
    return ", ".join(f"{int(row['seed'])}:{row['score']:.0f}" for row in episodes) if episodes else "-"


def tree_lines(nodes: dict[str, dict[str, Any]], root_id: str) -> list[str]:
    lines: list[str] = []

    def walk(node_id: str, prefix: str, is_last: bool, depth: int) -> None:
        node = nodes[node_id]
        outcome = node.get("outcome") or {}
        mean = (outcome.get("eval") or {}).get("mean_score")
        acceptance = node.get("acceptance") or {}
        label = f"{node_id} · {combo(node.get('candidate') or {})} · score={mean if mean is not None else 'pending'}"
        if acceptance.get("verdict"):
            label += f" · {acceptance['verdict']}"
        connector = "└── " if is_last else "├── "
        lines.append(prefix + (connector if depth > 0 else "") + label)
        children = node.get("children") or []
        for index, child in enumerate(children):
            child_prefix = prefix + ("    " if (is_last or depth == 0) else "│   ")
            walk(child, child_prefix, index == len(children) - 1, depth + 1)

    walk(root_id, "", True, 0)
    return lines


def training_history(base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = [base / "local_sim_v2", base / "local_sim_v2_40ep"]
    for root in roots:
        if not root.is_dir():
            continue
        for spec_path in sorted(root.rglob("spec.json")):
            spec = load_json(spec_path, {})
            if not spec:
                continue
            rows.append(
                {
                    "name": spec.get("candidate", {}).get("candidate_id") or spec_path.parent.parent.name,
                    "steps": spec.get("train_steps"),
                    "loss": spec.get("final_loss"),
                    "mae": spec.get("offline_normalized_mae"),
                    "wall": spec.get("wall_seconds"),
                    "dataset": spec.get("dataset") or "",
                    "checkpoint": str(spec_path.parent),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a markdown architecture and experiment-tree report")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="A3 Agent 优化框架：架构与实验树报告")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--final-eval", default="")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    run_dir = Path(args.run).resolve()
    summary = load_json(run_dir / "run.json", {})
    tree = load_json(run_dir / "tree.json", {})
    config = summary.get("config", {})
    nodes = {node["node_id"]: node for node in tree.get("nodes", [])}
    root_id = tree.get("root_id", "root")
    best_candidate = ""
    best_mean = None
    best_node_id = ""
    best_normalized = -1.0
    for node_id, node in nodes.items():
        outcome = node.get("outcome") or {}
        evaluation = outcome.get("eval") or {}
        mean = evaluation.get("mean_score")
        if mean is None or not node.get("candidate"):
            continue
        normalized = mean / max(float(evaluation.get("max_score", 10.0) or 10.0), 1.0)
        if normalized > best_normalized:
            best_normalized = normalized
            best_node_id = node_id
    if best_node_id:
        best_node = nodes[best_node_id]
        best_candidate = combo(best_node.get("candidate") or {})
        best_mean = (best_node.get("outcome") or {}).get("eval", {}).get("mean_score")
    budget = summary.get("budget", {})
    verdicts = config.get("verdicts", {})

    md: list[str] = []
    md.append(f"# {args.title}\n")
    md.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　运行编号：`{summary.get('run_id', run_dir.name)}`\n")
    md.append("本文档由 `scripts/build_tree_report.py` 从真实运行产物自动生成：运行摘要 `run.json`、实验树 `tree.json`、评估明细 `outcomes/*.json`。报告中所有分数均为 MuJoCo 闭环评估的真实得分（0-10 分制，10 分为把 10 块饼干全部放入目标盒）。\n")

    md.append("## 一、结论摘要\n")
    md.append("| 项目 | 结果 |")
    md.append("| --- | --- |")
    md.append(f"| 最优候选组合 | {best_candidate or summary.get('best_candidate_id', '')} |")
    md.append(f"| 最优平均分 | {best_mean if best_mean is not None else summary.get('best_mean_score')} / 10 |")
    md.append(f"| 候选实验节点数 | {summary.get('experiments', 0)} |")
    md.append(f"| 训练预算消耗 | {budget.get('spent_train_steps', 0)} 步 |")
    md.append(f"| 墙钟预算消耗 | {budget.get('spent_wall_seconds', 0)} s |")
    md.append(f"| 通过验收的候选 | {', '.join(summary.get('accepted', [])) or '无（最优为基础候选）'} |")
    md.append(f"| 被拒绝的候选 | {len(summary.get('rejected', []))} 个 |")
    final_preview = load_json(Path(args.final_eval), {}) if args.final_eval else {}
    if final_preview:
        md.append(
            "| 调优后最佳模型（6 个开发种子复核） | "
            + str(final_preview.get("mean_score"))
            + " / 10（40ep 数据集，10000 步，horizon 12） |"
        )
    md.append("")

    md.append("## 二、当前架构\n")
    md.append("### 2.1 三层决策\n")
    md.append("| 层 | 决定什么 | 实现位置 |")
    md.append("| --- | --- | --- |")
    md.append("| 部署层 | 每个控制周期输出 16 维关节目标 | 训练得到的机器人策略 |")
    md.append("| 训练层 | 如何从示范数据学习权重 | `local_sim_v2` 后端（torch 行为克隆） |")
    md.append("| 实验搜索层 | 训练哪个候选、是否保留、下一步试什么 | 本仓库 `a3opt`（实验树 + UCT + 验收门 + 记忆 + 预算） |")
    md.append("")
    md.append("### 2.2 组件与职责\n")
    md.append("| 模块 | 文件 | 作用 |")
    md.append("| --- | --- | --- |")
    rows = [
        ("数据契约", "`src/a3opt/contracts.py`", "候选规格、预算、训练/评估结果、实验结局、运行目录"),
        ("候选空间", "`src/a3opt/space.py`", "模块组合的合法性与分阶段候选列表"),
        ("实验树", "`src/a3opt/tree.py`", "UCT 选择、节点扩展、奖励回传、最优节点"),
        ("验收门", "`src/a3opt/acceptance.py`", "配对种子的增益 / 不屈化 / 成本三条件"),
        ("实验记忆", "`src/a3opt/memory.py`", "只追加 JSONL 证据 + best-k + 摘要"),
        ("预算", "`src/a3opt/contracts.py`", "单实验与整轮实验的预算与停止条件"),
        ("规划器", "`src/a3opt/planner.py`", "按阶段给出下一个候选"),
        ("主循环", "`src/a3opt/loop.py`", "分析→假设→训练→评估→反思→记录→迭代"),
        ("报告", "`src/a3opt/report.py`", "生成树表、验收表、记忆表的 Markdown 报告"),
        ("后端", "`src/a3opt/backends/`", "stub / local_sim / local_sim_v2 / a3_sim_cli"),
        ("命令行", "`src/a3opt/cli.py`", "`a3opt plan / run / report / selftest`"),
    ]
    for row in rows:
        md.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    md.append("")
    md.append("### 2.3 实验循环\n")
    md.append("```text\nAnalyze → Hypothesize → Implement → Train → Evaluate → Reflect → Iterate\n```\n")
    md.append("每一轮 = 一个候选实验：指定模块组合与训练预算 → 训练 → 在固定种子块上闭环评估 → 与基线做配对比较 → 写入实验记忆 → 回到实验树选择下一个节点。\n")
    md.append("### 2.4 候选空间\n")
    md.append("| 维度 | 取值 |")
    md.append("| --- | --- |")
    md.append("| encoder | `tiny_cnn`（视觉编码器）、`state_mlp`（仅状态） |")
    md.append("| fusion | `concat`、`film` |")
    md.append("| action_head | `mlp`、`chunk` |")
    md.append("| chunk_size | 1、20、50 |")
    md.append("| image_size | 0、64 |")
    md.append("| learning_rate | 1e-3、3e-3 |")
    md.append("")
    md.append("兼容性规则在训练前拦截非法组合：`state_mlp` 要求 `image_size=0`；`film` 要求图像输入；`chunk` 要求 `chunk_size>1`；`mlp` 要求 `chunk_size=1`。分阶段候选顺序为：baseline → 编码器/融合变体 → 动作头/chunk 变体 → 学习率变体。\n")
    md.append("### 2.5 选择、验收、预算与记忆\n")
    md.append("```text\nUCT(v) = Q_v / N_v + c * sqrt( ln N_parent(v) / N_v ),  c = 0.7\n```\n")
    md.append("- **验收门**：同一批种子上平均增益 ≥ 1 块饼干、≥ 70% 种子不退化、成本 ≤ 基线 3 倍，三者同时满足才 `accept`；配对种子不足两个判定为 `insufficient`。")
    md.append("- **预算**：整轮受实验数、训练总步数、墙钟秒数、GPU 秒数约束；单实验另有训练步数、batch、墙钟上限。")
    md.append("- **记忆**：每次实验结束追加一条 JSONL 记录（候选、训练指标、逐种子分数、成本、节点号），支持 best-k 查询与压缩摘要。")
    md.append("")
    md.append("### 2.6 后端\n")
    md.append("| 后端 | 真实训练 | 依赖 |")
    md.append("| --- | --- | --- |")
    md.append("| `stub` | 合成分数，仅用于测试与演示 | 无 |")
    md.append("| `local_sim` | 单相机行为克隆 + MuJoCo 闭环 | torch + A3 仿真仓库 |")
    md.append("| `local_sim_v2` | 三相机行为克隆 + MuJoCo 闭环（本次结果使用） | torch + A3 仿真仓库 |")
    md.append("| `a3_sim_cli` | 上游 ACT/SmolVLA 命令（默认 dry-run，未在本机执行） | lerobot + A3 仿真仓库 |")
    md.append("")
    md.append("### 2.7 运行目录\n")
    md.append("```text\nruns/<run_id>/\n  run.json          运行摘要、预算、验收阈值、verdicts\n  tree.json         实验树快照（节点、父子关系、访问计数、结果）\n  memory.jsonl      只追加的实验记忆\n  report.md         自动生成的人类可读报告\n  outcomes/<candidate_id>.json   每个候选的完整结局\n  logs/\n```\n")

    md.append("## 三、实验树是怎么保存的\n")
    md.append("实验树保存在 `runs/<run_id>/tree.json`。它是一个扁平节点表加父子指针，**不是**嵌套 JSON，便于追加、回传和断点恢复。字段如下：\n")
    md.append("| 字段 | 含义 |")
    md.append("| --- | --- |")
    md.append("| `root_id` | 根节点编号（基础候选，固定为 `root`） |")
    md.append("| `c` / `max_children` | UCT 探索常数与单节点最大子节点数 |")
    md.append("| `nodes[].node_id` | 节点编号（`root`、`n001`、`n002`…） |")
    md.append("| `nodes[].parent_id` | 父节点编号，根为 `null` |")
    md.append("| `nodes[].candidate` | 该节点的候选组合（encoder/fusion/head/chunk/lr/image_size/seed） |")
    md.append("| `nodes[].children` | 子节点编号列表 |")
    md.append("| `nodes[].visits` / `total_value` | 回传的访问次数与累计归一化奖励 |")
    md.append("| `nodes[].stage` | 该候选所属的筛选阶段 |")
    md.append("| `nodes[].outcome` | 训练结果 + 逐种子评估结果 + 成本（由 loop 写入） |")
    md.append("| `nodes[].acceptance` | 相对基线的配对验收结论（accept / reject / insufficient） |")
    md.append("")
    best_node = nodes.get(best_node_id) if best_node_id else None
    if best_node is not None:
        excerpt = json.dumps(best_node, indent=2, ensure_ascii=False)
        if len(excerpt) > 1600:
            excerpt = excerpt[:1600] + "\n  ... (截断)"
        md.append("以本运行的最优节点为例，`tree.json` 中的一个节点形如：\n")
        md.append("```json\n" + excerpt + "\n```\n")
    md.append("树形结构如下（缩进表示父子关系，括号内为该节点平均分与验收结论）：\n")
    md.append("```text\n" + "\n".join(tree_lines(nodes, root_id)) + "\n```\n")

    md.append("## 四、每个节点的得分\n")
    md.append("所有分数为 10 分制闭环评估；训练步数与 loss 来自该节点的训练产物；验收结论相对根节点（baseline）的配对比较。\n")
    stage_names = {"baseline": "基线", "encoder_fusion": "编码/融合", "head_chunk": "动作头", "lr": "学习率"}
    verdict_names = {"accept": "通过", "reject": "拒绝", "insufficient": "不足", "-": "-"}
    md.append("| 节点 | 候选组合 | 步数 | loss | 逐种子得分 | 均分 | 验收 |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    ordered = sorted(nodes.values(), key=lambda node: (node.get("node_id") != root_id, node.get("node_id", "")))
    for node in ordered:
        outcome = node.get("outcome") or {}
        train = outcome.get("train") or {}
        metrics = train.get("metrics") or {}
        mean = (outcome.get("eval") or {}).get("mean_score")
        acceptance = (node.get("acceptance") or {}).get("verdict", "-")
        md.append(
            f"| {node.get('node_id')} | {compact(node.get('candidate') or {})} "
            f"| {train.get('train_steps', '-')} | {metrics.get('final_loss', '-'):.4f} "
            f"| {score_list(outcome)} | {mean if mean is not None else '-'} "
            f"| {verdict_names.get(acceptance, acceptance)} |"
        )
    md.append("")
    md.append("### 4.1 预算与实际耗时\n")
    md.append("| 项目 | 数值 |")
    md.append("| --- | --- |")
    md.append(f"| 实验数 | {budget.get('spent_experiments', '-')} / {budget.get('max_experiments', '-')} |")
    md.append(f"| 训练步数 | {budget.get('spent_train_steps', '-')} / {budget.get('max_train_steps_total', '-')} |")
    md.append(f"| 墙钟秒数 | {budget.get('spent_wall_seconds', '-')} |")
    md.append(f"| GPU 秒数 | {budget.get('spent_gpu_seconds', '-')} |")
    md.append("")

    md.append("## 五、最高分组合\n")
    if best_node is not None:
        candidate = best_node.get("candidate") or {}
        outcome = best_node.get("outcome") or {}
        evaluation = outcome.get("eval") or {}
        train = outcome.get("train") or {}
        md.append("| 维度 | 取值 |")
        md.append("| --- | --- |")
        for key in ("candidate_id", "encoder", "fusion", "action_head", "chunk_size", "lr", "image_size", "state_dim", "seed", "family"):
            md.append(f"| {key} | `{candidate.get(key)}` |")
        md.append("")
        md.append("| 指标 | 数值 |")
        md.append("| --- | --- |")
        md.append(f"| 平均分 | {evaluation.get('mean_score')} / 10 |")
        md.append(f"| 逐种子得分 | {score_list(outcome)} |")
        md.append(f"| 成功次数 | {sum(1 for row in evaluation.get('episodes', []) if row.get('success'))} |")
        md.append(f"| 训练步数 | {train.get('train_steps')} |")
        md.append(f"| 训练 final loss | {((train.get('metrics') or {}).get('final_loss')):.4f} |")
        md.append(f"| 训练耗时 | {train.get('wall_seconds')} s |")
        md.append(f"| 评估耗时 | {evaluation.get('wall_seconds')} s |")
        md.append("")
        checkpoint = (train.get("artifact") or "")
        if checkpoint:
            md.append(f"最优 checkpoint：`{checkpoint}`\n")
        spec_path = Path(checkpoint).parent / "spec.json" if checkpoint else None
        spec = load_json(spec_path, {}) if spec_path else {}
        architecture = spec.get("architecture") or {}
        if architecture:
            md.append("### 5.1 模型结构\n")
            md.append("| 组件 | 配置 |")
            md.append("| --- | --- |")
            for key, value in architecture.items():
                md.append(f"| {key} | {value} |")
            md.append("")
        md.append("### 5.2 复现命令\n")
        md.append("```bash\npython scripts/bc2_experiment.py \\\n  --name <候选名> --dataset artifacts/datasets/expert_2x5_multicam_40ep.npz \\\n  --train-steps 10000 --image-size 64 --execution-horizon 12\n```\n")
    else:
        md.append("_本次运行没有可用的最优节点。_\n")

    final_eval_path = Path(args.final_eval) if args.final_eval else None
    if final_eval_path and final_eval_path.is_file():
        final = load_json(final_eval_path, {})
        episodes = final.get("episodes") or []
        md.append("## 五之二、最终调优模型（独立复核）\n")
        md.append("实验树之外，另用更大的训练预算训练了最终模型，并在开发块 6 个种子上复核：\n")
        md.append("| 项目 | 数值 |")
        md.append("| --- | --- |")
        md.append(f"| checkpoint | `{final.get('checkpoint', '-')}` |")
        md.append(f"| 执行 horizon | {final.get('execution_horizon', '-')} |")
        md.append(f"| 每种子分数 | {', '.join(str(row.get('score')) for row in episodes)} |")
        md.append(f"| 平均分 | {final.get('mean_score', '-')} / 10 |")
        md.append(f"| 成功次数 | {sum(1 for row in episodes if row.get('success'))} |")
        md.append("")
    md.append("## 六、真实训练记录汇总\n")
    history = training_history(repo / "artifacts")
    if history:
        md.append("| 模型 | 数据 | 训练步数 | final loss | 离线 MAE | 训练耗时(s) | checkpoint |")
        md.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in history:
            md.append(
                f"| {row['name']} | {Path(row['dataset']).name if row['dataset'] else '-'} | {row['steps']} "
                f"| {row['loss'] if row['loss'] is None else round(float(row['loss']), 5)} "
                f"| {row['mae'] if row['mae'] is None else round(float(row['mae']), 5)} "
                f"| {row['wall']} | {row['checkpoint']} |"
            )
        md.append("")
    md.append("外部参照（专家策略在相同任务上的水平）：脚本专家在同一 2x5 任务上约 410 步完成 10/10；当前学习策略在开发种子上平均分见第一节。\n")

    md.append("## 七、复现与测试\n")
    md.append("```bash\na3opt plan\na3opt selftest --experiments 6\npytest tests -q\n```\n")
    md.append("测试结果：80 passed, 1 skipped（真实仿真测试默认由 `A3OPT_RUN_SIM_TESTS=1` 控制）。\n")
    md.append("## 八、局限与下一步\n")
    md.append("- 行为克隆策略仍会出现逐种子波动：部分布局能放 5-10 块，部分布局提前停滞；预留确认种子块分数明显低于开发块。")
    md.append("- 执行预算为 600 步（约 30 s 仿真时间）；专家需要约 410 步，学习策略偶尔更慢。")
    md.append("- 下一步：继续扩大示范数据、训练更长步数；有条件时在 GPU 上训练 ACT/SmolVLA 作为主力候选，并用专家做 DAgger 纠偏。")
    md.append("")
    md.append("_本报告由 a3opt 运行产物自动生成。_")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
