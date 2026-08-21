"""Stage 13 LLM4AD Boost 后处理 — 在 Stage 13 内部调用的演化优化层

作为 Stage 13 的可选后处理，产物输出到 stage-13/llm4ad_boost/。
不创建独立 Stage；演化产物以候选版本的形式交回 Stage 13 择优，不原地改文件。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline._helpers import _utcnow_iso
from researchclaw.pipeline.llm4ad_utils import (
    extract_proposed_algorithms,
    build_task_package,
)
from researchclaw.pipeline.llm4ad_utils.comparison_reporter import (
    generate_comparison_report,
    generate_summary_report,
)

logger = logging.getLogger(__name__)


def run_llm4ad_boost_inline(
    boost_dir: Path,
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    base_files: dict[str, str] | None = None,
    adapters: AdapterBundle | None = None,  # 改为可选，实际上不需要
) -> dict[str, Any] | None:
    """Stage 13 内联 LLM4AD boost 后处理

    在 Stage 13 择优完成后、写入 experiment_final/ 之前运行，产物输出到
    stage-13/llm4ad_boost/。

    本函数**不修改任何已落盘的实验代码**：演化后的算法合并进 base_files 的
    副本后一并返回，由 Stage 13 当作一个普通候选版本重跑实验、按真实指标
    择优，胜出才会进入 experiment_final/。

    Args:
        boost_dir: boost 输出目录 (stage-13/llm4ad_boost/)
        stage_dir: Stage 13 目录 (stage-13/)
        run_dir: 运行根目录
        config: 全局配置
        base_files: Stage 13 当前最优文件集；为空则只出报告不产候选
        adapters: 适配器束（可选，当前未使用）

    Returns:
        汇总字典，额外带 "evolved_files"（合并后的候选文件集，无改动时为
        None）与 "merge"（每个算法的合并记录）；整体失败返回 skip 汇总。
    """
    boost_dir.mkdir(parents=True, exist_ok=True)
    boost_config = config.experiment.llm4ad_boost

    # fail_silently 只有一个来源：扁平字段。_execution.py 的 except 分支读的是
    # 同一个属性，两处口径必须一致。
    fail_silently = bool(boost_config.fail_silently)

    logger.info("=" * 60)
    logger.info("LLM4AD Boost 启动")
    logger.info("=" * 60)

    try:
        # 1. 提取 proposed 算法（LLM 三路分类：baseline/ablation/proposed）
        #    分类结果由 extract_proposed_algorithms 写入 boost_summary.json
        logger.info("步骤 1: 提取 proposed 算法...")
        try:
            algorithms = extract_proposed_algorithms(
                stage_dir, run_dir, boost_dir=boost_dir, config=config
            )
        except Exception as e:
            logger.error(f"提取算法失败: {e}", exc_info=True)
            return _skip_boost(boost_dir, f"extract_failed: {e}")

        if not algorithms:
            logger.warning("未找到任何 proposed 算法")
            return _skip_boost(boost_dir, "no_proposed_algorithms")

        logger.info(f"找到 {len(algorithms)} 个 proposed 算法")

        # 2. 为每个算法构建任务包
        logger.info("步骤 2: 构建任务包...")
        task_packages = []
        for algo in algorithms:
            try:
                task_dir = build_task_package(algo, boost_dir, run_dir, config)
                task_packages.append((algo, task_dir))
                logger.info(f"  ✓ {algo.condition_name}: {task_dir}")
            except Exception as e:
                logger.error(f"构建任务包失败 ({algo.condition_name}): {e}", exc_info=True)
                if not fail_silently:
                    raise

        if not task_packages:
            return _skip_boost(boost_dir, "task_package_build_failed")

        # 3. 演化算法
        logger.info("步骤 3: 运行 LLM4AD 演化...")
        evolution_results = {}

        try:
            if boost_config.parallel_evolution:
                # 并行演化
                max_parallel = boost_config.max_parallel_algorithms
                logger.info(f"并行演化模式 (最多 {max_parallel} 个并行)")

                # 分批处理
                for i in range(0, len(task_packages), max_parallel):
                    batch = task_packages[i:i + max_parallel]
                    batch_results = asyncio.run(
                        _evolve_batch_parallel(batch, boost_config, stage_dir, boost_dir)
                    )
                    evolution_results.update(batch_results)
            else:
                # 串行演化
                logger.info("串行演化模式")
                for algo, task_dir in task_packages:
                    try:
                        result = _run_llm4ad_evolution_single(
                            algo, task_dir, boost_config, stage_dir, boost_dir
                        )
                        evolution_results[algo.condition_name] = result
                    except Exception as e:
                        logger.error(f"演化失败 ({algo.condition_name}): {e}", exc_info=True)
                        evolution_results[algo.condition_name] = {
                            "status": "failed",
                            "error": str(e),
                        }
                        if not fail_silently:
                            raise
        except Exception as e:
            logger.error(f"演化过程异常: {e}", exc_info=True)
            if not fail_silently:
                raise
            # 继续生成已有结果的报告

        # 4. 生成对比报告
        logger.info("步骤 4: 生成对比报告...")
        all_comparisons = {}

        for algo, task_dir in task_packages:
            algo_dir = boost_dir / algo.condition_name
            result = evolution_results.get(algo.condition_name, {})

            try:
                comparison = generate_comparison_report(algo, result, algo_dir)
                all_comparisons[algo.condition_name] = comparison
            except Exception as e:
                logger.error(f"生成对比报告失败 ({algo.condition_name}): {e}", exc_info=True)
                all_comparisons[algo.condition_name] = {
                    "status": "failed",
                    "error": f"Report generation failed: {e}",
                }

        # 5. 生成汇总报告
        logger.info("步骤 5: 生成汇总报告...")
        summary = generate_summary_report(algorithms, all_comparisons, boost_dir)
        summary.setdefault("status", "success")

        # 6. 合并演化产物为 Stage 13 的候选文件集（不落盘）
        logger.info("步骤 6: 合并演化产物为候选版本...")
        merge_records: list[dict[str, Any]] = []
        evolved_files: dict[str, str] | None = None
        if not base_files:
            logger.info("未提供 base_files，跳过合并（仅产出报告）")
        else:
            working = dict(base_files)
            for algo, _task_dir in task_packages:
                res = evolution_results.get(algo.condition_name) or {}
                if res.get("status") != "success":
                    merge_records.append({
                        "condition": algo.condition_name,
                        "status": "skipped",
                        "reason": f"evolution_{res.get('status', 'missing')}",
                    })
                    continue
                if res.get("used_baseline_fallback"):
                    merge_records.append({
                        "condition": algo.condition_name,
                        "status": "skipped",
                        "reason": "baseline_fallback",
                    })
                    continue
                merged, info = merge_evolved_into_files(
                    algo, res.get("evolved_code_path"), working
                )
                merge_records.append(info)
                if merged is not None:
                    working = merged
            if any(r.get("status") == "merged" for r in merge_records):
                evolved_files = working

        summary["merge"] = merge_records
        summary["evolved_files"] = evolved_files

        logger.info("=" * 60)
        logger.info("LLM4AD Boost 完成")
        logger.info(f"成功演化: {summary['successful']} / {summary['total_algorithms']}")
        logger.info(f"性能改进: {summary['improved']} / {summary['successful']}")
        logger.info(f"平均提升: {summary['avg_improvement_pct']:.2f}%")
        logger.info("=" * 60)

        return summary

    except Exception as e:
        logger.error(f"LLM4AD Boost 执行异常: {e}", exc_info=True)
        return _skip_boost(boost_dir, f"unexpected_error: {e}")


def _skip_boost(boost_dir: Path, reason: str) -> dict[str, Any]:
    """跳过 boost，记录原因

    boost_summary.json 由 algorithm_extractor 先行写入分类审计记录，这里必须
    合并而非全量覆盖，否则会抹掉那份记录。
    """
    summary_path = boost_dir / "boost_summary.json"
    existing: dict[str, Any] = {}
    if summary_path.exists():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception as e:  # 损坏的旧文件不应阻断跳过路径
            logger.warning(f"读取既有 boost_summary.json 失败，将覆盖: {e}")

    existing.update({
        "status": "skipped",
        "reason": reason,
        "timestamp": _utcnow_iso(),
    })
    summary_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    logger.info(f"LLM4AD boost 跳过: {reason}")
    return existing


async def _evolve_batch_parallel(
    batch: list[tuple],
    boost_config: Any,
    stage_dir: Path,
    boost_dir: Path,
) -> dict[str, Any]:
    """并行演化一批算法"""
    tasks = []
    for algo, task_dir in batch:
        task = _run_llm4ad_evolution_async(
            algo, task_dir, boost_config, stage_dir, boost_dir
        )
        tasks.append((algo.condition_name, task))

    results = {}
    gathered = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

    for (condition_name, _), result in zip(tasks, gathered):
        if isinstance(result, Exception):
            logger.error(f"演化失败 ({condition_name}): {result}")
            results[condition_name] = {
                "status": "failed",
                "error": str(result),
            }
        else:
            results[condition_name] = result

    return results


async def _run_llm4ad_evolution_async(
    algo: Any,
    task_dir: Path,
    boost_config: Any,
    stage_dir: Path,
    boost_dir: Path,
) -> dict[str, Any]:
    """异步运行 LLM4AD 演化（单个算法）"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _run_llm4ad_evolution_single, algo, task_dir, boost_config, stage_dir, boost_dir
    )


def _run_llm4ad_evolution_single(
    algo: Any,
    task_dir: Path,
    boost_config: Any,
    stage_dir: Path,
    boost_dir: Path,
) -> dict[str, Any]:
    """运行 LLM4AD 演化（单个算法）

    Returns:
        {
            "status": "success" | "failed",
            "best_generation": int,
            "best_performance": float,
            "baseline_performance": float,
            "improvement_pct": float,
            "evolved_code_path": Path,
            "evolution_log": dict,
        }
    """
    logger.info(f"开始演化: {algo.condition_name}")

    try:
        # 检查 llm4ad 是否安装
        try:
            from llm4ad import LLM4AD
            from llm4ad.config import AppConfig
        except ImportError:
            raise RuntimeError(
                "llm4ad 未安装。请运行: pip install llm4ad"
            )

        # 读取 LLM4AD 配置
        config_path = task_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"LLM4AD 配置文件不存在: {config_path}")

        logger.info(f"加载 LLM4AD 配置: {config_path}")

        # 创建 LLM4AD 实例
        llm4ad = LLM4AD(str(config_path))

        logger.info(f"LLM4AD 初始化完成，开始演化...")
        llm4ad.print_run_summary()

        # 运行演化（asyncio 已在模块顶部导入）
        result = asyncio.run(llm4ad.run(resume_from_checkpoint=None))

        logger.info(f"演化完成: {result.state.value}")

        # 提取结果
        state = llm4ad.export_state()

        # 获取最佳个体
        best_individual = result.best_individual
        if best_individual is None:
            raise RuntimeError("演化未产生有效个体")

        best_score = best_individual.score
        baseline_score = algo.baseline_metrics.get("primary_metric", float("inf"))

        # 计算提升（primary_only 口径，仅用于报告；是否采纳由 Stage 13 重跑决定）
        #
        # 评估器返回 score = -primary_metric（minimize 方向取负转最大化），
        # 因此演化后的原始 primary_metric 值 best_raw = -best_score。
        # baseline_score 是 Stage 12 的原始 primary_metric（未取负）。
        #
        # minimize 方向下改进定义为 (baseline_raw - best_raw) / |baseline_raw|：
        #   improvement > 0 表示 best_raw < baseline_raw（更优），符合"越小越好"。
        # maximize 方向下则为 (best_raw - baseline_raw) / |baseline_raw|，
        #   此时 score == primary_metric（评估器不取负），best_raw = best_score。
        direction = (
            (boost_config.metric_aggregation or {})
            .get("primary_only", {})
            .get("direction", "minimize")
        )
        best_raw = -best_score if direction == "minimize" else best_score
        # baseline 缺失时是 inf，0 会除零；两种情况都算不出百分比，记 None 而
        # 不是 nan——nan 参与任何比较都为 False，会让下游判断静默走错分支。
        if not math.isfinite(baseline_score) or baseline_score == 0:
            improvement_pct = None
            logger.warning(
                f"{algo.condition_name}: baseline primary_metric 不可用 "
                f"({baseline_score})，无法计算提升百分比"
            )
        elif direction == "minimize":
            improvement_pct = (baseline_score - best_raw) / abs(baseline_score) * 100
        else:
            improvement_pct = (best_raw - baseline_score) / abs(baseline_score) * 100

        # 找到演化后的代码文件
        # 尝试多个可能的位置
        evolved_code_path = None

        # 位置 1: 从 LLM4AD 配置读取 workspace 路径
        try:
            config_path = task_dir / "config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    llm4ad_config = yaml.safe_load(f)

                base_dir = llm4ad_config.get("base_dir", str(task_dir.parent.parent / "llm4ad"))
                workspace_root = Path(base_dir) / "run"

                # 检查 generated 目录
                generated_dir = workspace_root / "generated"
                if generated_dir.exists():
                    json_files = sorted(generated_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
                    if json_files:
                        latest_json = json_files[-1]
                        with open(latest_json, "r", encoding="utf-8") as f:
                            gen_data = json.load(f)

                        if "code" in gen_data:
                            evolved_code_path = task_dir.parent / "evolved_solve.py"
                            evolved_code_path.write_text(gen_data["code"], encoding="utf-8")
                            logger.info(f"演化代码已保存（位置1）: {evolved_code_path}")
        except Exception as e:
            logger.warning(f"从 workspace/generated 提取代码失败: {e}")

        # 位置 2: 从 best_individual 获取
        if evolved_code_path is None:
            logger.info("尝试从 best_individual 获取代码...")
            if hasattr(best_individual, "code") and best_individual.code:
                evolved_code_path = task_dir.parent / "evolved_solve.py"
                evolved_code_path.write_text(best_individual.code, encoding="utf-8")
                logger.info(f"从 best_individual 提取代码: {evolved_code_path}")
            else:
                logger.warning("best_individual 没有 code 属性")

        # 回退检测：拿不到演化代码时用 baseline 顶替，此时没有任何可合并的改动
        used_baseline_fallback = False
        if evolved_code_path is None or not evolved_code_path.exists():
            logger.error("无法获取演化后的代码，使用 baseline 作为回退")
            evolved_code_path = task_dir / "solve.py"
            used_baseline_fallback = True

        # 构建结果
        # 注意：不在此处回写文件。演化产物交回 Stage 13，作为一个普通候选版本
        # 重跑实验、按 ARC 自己的真实指标择优——这里的 best_score 只是 LLM4AD
        # 在其任务包 benchmark 上的代理分数，两个口径不必然一致。
        result_dict = {
            "status": "success",
            "best_generation": state.get("generation", 0),
            "best_performance": best_score,
            "baseline_performance": baseline_score,
            "improvement_pct": improvement_pct,
            "evolved_code_path": evolved_code_path,
            "used_baseline_fallback": used_baseline_fallback,
            "evolution_log": {
                "state": result.state.value,
                "total_evaluations": state.get("total_evaluations", 0),
                "generation": state.get("generation", 0),
                "best_score": best_score,
            },
        }

        # 保存演化日志
        log_path = task_dir.parent / "evolution_log.json"
        log_path.write_text(
            json.dumps(result_dict, indent=2, default=str),
            encoding="utf-8"
        )

        _imp = "未知" if improvement_pct is None else f"{improvement_pct:.2f}%"
        logger.info(
            f"演化完成: {algo.condition_name}, 提升 {_imp}, 最佳分数: {best_score:.4f}"
        )

        return result_dict

    except Exception as e:
        logger.error(f"演化异常: {algo.condition_name}: {e}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
        }


# 辅助函数已移至 comparison_reporter.py
# _generate_summary_report() 已废弃


# ---------------------------------------------------------------------------
# 合并：把演化后的目标类并入 Stage 13 的候选文件集
# ---------------------------------------------------------------------------

def _class_span(node: Any) -> tuple[int, int]:
    """返回类定义在源码中的行区间 [start, end)（0-based，含装饰器）

    ClassDef.lineno 指向 `class` 关键字，不含装饰器；直接用它会在提取时丢掉
    @dataclass 之类的装饰器，在替换时把旧装饰器留给新类体。
    """
    first = node.lineno
    for deco in node.decorator_list:
        first = min(first, deco.lineno)
    end = node.end_lineno if getattr(node, "end_lineno", None) else node.lineno
    return first - 1, end


def _find_top_level_class(code: str, class_name: str) -> Any | None:
    """在模块顶层查找指定类定义；解析失败或不存在返回 None

    只扫 tree.body 而非 ast.walk：后者会匹配嵌套在函数/类内部的同名类，
    替换那种类会破坏外层结构。
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"解析源码失败（查找 {class_name}）: {e}")
        return None

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _extract_class_source(code: str, class_name: str) -> str | None:
    """用 AST 从源码中提取指定类的完整源码片段（含装饰器）"""
    node = _find_top_level_class(code, class_name)
    if node is None:
        return None
    start, end = _class_span(node)
    return "\n".join(code.split("\n")[start:end])


def _strip_evolve_markers(code: str) -> str:
    """移除 EVOLVE_START / EVOLVE_END 标记行（合并后不再需要）"""
    kept = []
    for line in code.split("\n"):
        stripped = line.strip()
        if "EVOLVE_START" in stripped or "EVOLVE_END" in stripped:
            # 仅当整行是标记注释时才删除（避免误删含该子串的代码）
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
        kept.append(line)
    return "\n".join(kept)


def _replace_class_in_source(
    full_code: str, class_name: str, new_class_source: str
) -> str | None:
    """用 AST 定位并替换 full_code 中的指定类定义为 new_class_source

    返回替换后的完整源码；若顶层不存在该类或解析失败返回 None。
    """
    node = _find_top_level_class(full_code, class_name)
    if node is None:
        return None
    start, end = _class_span(node)
    lines = full_code.split("\n")
    return "\n".join(lines[:start] + new_class_source.split("\n") + lines[end:])


def merge_evolved_into_files(
    algo: Any,
    evolved_code_path: Path | None,
    files: dict[str, str],
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """把演化后的目标类并入一份候选文件集，返回新的 files dict

    与旧的 write_back_evolved 的区别：**不落盘**。产物交回 Stage 13，由后者
    当作一个普通候选版本重跑实验、按真实指标择优，胜出才写入
    experiment_final/。因此这里不需要备份/回滚，也不需要用 LLM4AD 的代理
    分数判断"要不要采纳"。

    Args:
        algo: AlgorithmInfo（提供 class_name / condition_name）
        evolved_code_path: 演化后代码文件（evolved_solve.py）
        files: 基线文件集（Stage 13 的 best_files），不会被修改

    Returns:
        (merged_files | None, info)。失败时 merged_files 为 None。
    """
    import ast

    info: dict[str, Any] = {
        "condition": getattr(algo, "condition_name", "?"),
        "class_name": getattr(algo, "class_name", "?"),
    }

    if not evolved_code_path or not Path(evolved_code_path).exists():
        info.update(status="skipped", reason="evolved_code_missing")
        return None, info

    try:
        evolved_code = Path(evolved_code_path).read_text(encoding="utf-8")

        # 1. 提取目标类并剥离 EVOLVE 标记
        evolved_class = _extract_class_source(evolved_code, algo.class_name)
        if evolved_class is None:
            info.update(status="failed", reason=f"class_not_in_evolved: {algo.class_name}")
            return None, info
        evolved_class = _strip_evolve_markers(evolved_class)

        # 2. 定位承载该类的文件（不再硬编码 optimizers.py）
        target_name = next(
            (
                name
                for name, src in files.items()
                if name.endswith(".py")
                and _find_top_level_class(src, algo.class_name) is not None
            ),
            None,
        )
        if target_name is None:
            info.update(status="failed", reason=f"class_not_in_target: {algo.class_name}")
            return None, info
        info["target_file"] = target_name

        # 3. 替换目标类
        merged_src = _replace_class_in_source(
            files[target_name], algo.class_name, evolved_class
        )
        if merged_src is None:
            info.update(status="failed", reason=f"replace_failed: {algo.class_name}")
            return None, info

        # 4. AST 校验合并结果
        try:
            ast.parse(merged_src)
        except SyntaxError as e:
            info.update(status="failed", reason=f"merged_syntax_error: {e}")
            logger.error(f"合并后 AST 校验失败，放弃该算法: {e}")
            return None, info

        merged_files = dict(files)
        merged_files[target_name] = merged_src
        info.update(status="merged", reason="ok")
        logger.info(
            f"已合并: {algo.condition_name}.{algo.class_name} → {target_name}"
        )
        return merged_files, info

    except Exception as e:
        logger.error(f"合并异常 ({getattr(algo, 'condition_name', '?')}): {e}", exc_info=True)
        info.update(status="failed", reason=f"exception: {e}")
        return None, info
