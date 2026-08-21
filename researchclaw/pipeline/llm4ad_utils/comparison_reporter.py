"""对比报告生成器 - 生成 baseline vs evolved 的对比报告"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def generate_comparison_report(
    algorithm_info: Any,
    evolution_result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """生成单个算法的对比报告

    Args:
        algorithm_info: AlgorithmInfo 对象
        evolution_result: LLM4AD 演化结果
        output_dir: 输出目录（algorithm_X/）

    Returns:
        对比数据字典
    """
    if evolution_result.get("status") != "success":
        return _generate_failed_report(algorithm_info, evolution_result, output_dir)

    # 基本信息
    baseline_score = evolution_result["baseline_performance"]
    evolved_score = evolution_result["best_performance"]
    improvement_pct = evolution_result["improvement_pct"]

    # 构建对比数据
    comparison = {
        "algorithm_name": algorithm_info.condition_name,
        "status": "success",
        "baseline": {
            "score": float(baseline_score),
            "metrics": algorithm_info.baseline_metrics,
        },
        "evolved": {
            "score": float(evolved_score),
            "generation": evolution_result.get("best_generation", 0),
            "code_path": str(evolution_result.get("evolved_code_path", "")),
        },
        "improvement": {
            "absolute": float(evolved_score - baseline_score),
            "percentage": float(improvement_pct),
            "is_better": improvement_pct > 0,
        },
        "evolution_log": evolution_result.get("evolution_log", {}),
    }

    # 保存 JSON
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    # 生成 Markdown 报告
    markdown = _generate_markdown_report(comparison)
    md_path = output_dir / "report.md"
    md_path.write_text(markdown, encoding="utf-8")

    logger.info(f"对比报告已生成: {output_dir}")

    return comparison


def _generate_failed_report(
    algorithm_info: Any,
    evolution_result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """生成失败的对比报告"""
    comparison = {
        "algorithm_name": algorithm_info.condition_name,
        "status": "failed",
        "error": evolution_result.get("error", "Unknown error"),
        "baseline": {
            "score": algorithm_info.baseline_metrics.get("primary_metric", 0.0),
            "metrics": algorithm_info.baseline_metrics,
        },
    }

    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    markdown = f"""# 演化失败报告

## 算法
- **名称**: {algorithm_info.condition_name}

## 状态
❌ 演化失败

## 错误信息
```
{evolution_result.get('error', 'Unknown error')}
```

## Baseline 性能
- Score: {algorithm_info.baseline_metrics.get('primary_metric', 0.0):.4f}
"""

    md_path = output_dir / "report.md"
    md_path.write_text(markdown, encoding="utf-8")

    return comparison


def _generate_markdown_report(comparison: dict[str, Any]) -> str:
    """生成 Markdown 格式的报告"""

    algo_name = comparison["algorithm_name"]
    baseline = comparison["baseline"]
    evolved = comparison["evolved"]
    improvement = comparison["improvement"]

    status_icon = "✅" if improvement["is_better"] else "⚠️"
    improvement_text = f"+{improvement['percentage']:.2f}%" if improvement["is_better"] else f"{improvement['percentage']:.2f}%"

    markdown = f"""# {algo_name} 演化报告

## 状态
{status_icon} {"改进" if improvement["is_better"] else "无改进"}

## 性能对比

| 指标 | Baseline | Evolved | 提升 |
|------|----------|---------|------|
| Score | {baseline['score']:.4f} | {evolved['score']:.4f} | **{improvement_text}** |

## Baseline 指标

```json
{json.dumps(baseline['metrics'], indent=2)}
```

## 演化信息

- **最佳代数**: {evolved['generation']}
- **演化代码**: `{evolved['code_path']}`

## 提升分析

- **绝对提升**: {improvement['absolute']:.4f}
- **相对提升**: {improvement['percentage']:.2f}%
- **是否改进**: {"是" if improvement['is_better'] else "否"}
"""

    return markdown


def generate_summary_report(
    algorithms: list,
    all_comparisons: dict[str, Any],
    boost_dir: Path,
) -> dict[str, Any]:
    """生成汇总报告

    Args:
        algorithms: AlgorithmInfo 列表
        all_comparisons: 所有算法的对比结果
        boost_dir: llm4ad_boost 目录

    Returns:
        汇总数据字典
    """
    n_total = len(algorithms)
    n_success = sum(
        1 for comp in all_comparisons.values()
        if comp.get("status") == "success"
    )
    n_improved = sum(
        1 for comp in all_comparisons.values()
        if comp.get("status") == "success" and comp["improvement"]["is_better"]
    )
    n_failed = n_total - n_success

    # 收集提升数据
    improvements = []
    for comp in all_comparisons.values():
        if comp.get("status") == "success":
            improvements.append(comp["improvement"]["percentage"])

    avg_improvement = float(np.mean(improvements)) if improvements else 0.0
    median_improvement = float(np.median(improvements)) if improvements else 0.0

    summary = {
        "total_algorithms": n_total,
        "successful": n_success,
        "improved": n_improved,
        "failed": n_failed,
        "avg_improvement_pct": avg_improvement,
        "median_improvement_pct": median_improvement,
        "algorithms": {},
    }

    # 添加每个算法的简要信息
    for algo_name, comp in all_comparisons.items():
        if comp.get("status") == "success":
            summary["algorithms"][algo_name] = {
                "status": "success",
                "baseline_score": comp["baseline"]["score"],
                "evolved_score": comp["evolved"]["score"],
                "improvement_pct": comp["improvement"]["percentage"],
                "is_better": comp["improvement"]["is_better"],
            }
        else:
            summary["algorithms"][algo_name] = {
                "status": "failed",
                "error": comp.get("error", "Unknown"),
            }

    # 保存汇总 JSON
    summary_json = boost_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 生成汇总 Markdown
    markdown = _generate_summary_markdown(summary, all_comparisons)
    summary_md = boost_dir / "summary_report.md"
    summary_md.write_text(markdown, encoding="utf-8")

    logger.info(f"汇总报告已生成: {boost_dir}")

    return summary


def _generate_summary_markdown(
    summary: dict[str, Any],
    all_comparisons: dict[str, Any],
) -> str:
    """生成汇总 Markdown 报告"""

    markdown_lines = [
        "# LLM4AD 多算法演化汇总报告",
        "",
        "## 概览",
        "",
        f"- **总算法数**: {summary['total_algorithms']}",
        f"- **成功演化**: {summary['successful']}",
        f"- **性能改进**: {summary['improved']}",
        f"- **演化失败**: {summary['failed']}",
        f"- **平均提升**: {summary['avg_improvement_pct']:.2f}%",
        f"- **中位提升**: {summary['median_improvement_pct']:.2f}%",
        "",
        "## 算法对比表",
        "",
        "| 算法 | Baseline Score | Evolved Score | 提升 | 状态 |",
        "|------|----------------|---------------|------|------|",
    ]

    # 添加每个算法的行
    for algo_name, comp in all_comparisons.items():
        if comp.get("status") == "success":
            baseline = comp["baseline"]["score"]
            evolved = comp["evolved"]["score"]
            improvement = comp["improvement"]["percentage"]
            is_better = comp["improvement"]["is_better"]

            status = "✅ 改进" if is_better else "⚠️ 无改进"
            improvement_text = f"{improvement:+.2f}%"

            markdown_lines.append(
                f"| {algo_name} | {baseline:.4f} | {evolved:.4f} | "
                f"**{improvement_text}** | {status} |"
            )
        else:
            error = comp.get("error", "Unknown")[:30]
            markdown_lines.append(
                f"| {algo_name} | - | - | - | ❌ 失败: {error} |"
            )

    markdown_lines.extend([
        "",
        "## 详细报告",
        "",
        "每个算法的详细演化报告见各自目录：",
        "",
    ])

    for algo_name in all_comparisons.keys():
        markdown_lines.append(f"- `{algo_name}/report.md`")

    markdown_lines.extend([
        "",
        "---",
        "",
        "*此报告由 ResearchClaw LLM4AD Boost 自动生成*",
    ])

    return "\n".join(markdown_lines)
