"""评估器生成器 - 生成 LLM4AD 兼容的评估器代码

v3.2（Q3 简化）：
- **零文件 dataset**：实例在内存生成。dispatcher 对 `mode: files + files: []`
  的每个算法只调用一次 `evaluate(cfg)`（`data_path=""`，`project_root` 可能为空串），
  因此评估器不读 data/*.json，而是遍历 `Config.FUNCTIONS × DIMENSIONS × SEEDS`
  在内存构造实例，也**不依赖 `cfg.project_root`**。
- **实例配置来源**：直接从 task_package 内的 `experiment_config` import `Config`
  （task_builder 已把 stage-10 的 experiment_config.py 复制进任务包），
  不需要正则提取，不需要 config.yaml 传参，无内联配置。
- **primary_metric 取值**：优化器返回 dict 的 `best_f` 键（record_result 契约），
  不是 `primary_metric` 键——两个评估模式统一从 `r["best_f"]` 取值。
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 生成代码中实例配置的 import 语句（零文件 dataset 模式使用）
_INSTANCE_CONFIG_IMPORT = """\
# 实例生成配置（来自 stage-10 的 experiment_config.py，task_builder 已复制进任务包）
from experiment_config import Config as ExperimentConfig
"""


def generate_evaluator_code(
    algorithm_info: Any,
    aggregation_config: dict[str, Any],
    baseline_stats: dict[str, dict],
) -> str:
    """生成 LLM4AD 兼容的评估器代码

    Args:
        algorithm_info: AlgorithmInfo 对象
        aggregation_config: 指标聚合配置
        baseline_stats: 基线统计信息（用于归一化）

    Returns:
        evaluator.py 的完整代码
    """
    method = aggregation_config.get("method", "primary_only")

    if method == "primary_only":
        return _generate_primary_only_evaluator(algorithm_info, aggregation_config)
    elif method == "weighted_sum":
        return _generate_weighted_sum_evaluator(
            algorithm_info, aggregation_config, baseline_stats
        )
    else:
        raise ValueError(f"未知的聚合方法: {method}")


def _generate_in_memory_instances_code() -> str:
    """生成在内存构造测试实例列表的代码（零文件 dataset 核心）

    遍历 `ExperimentConfig.FUNCTIONS × DIMENSIONS × SEEDS`，每实例包含
    dimension/function/seed/budget。当 `cfg.data_path` 非空且指向真实目录时
    仍回退读取 data/instance_*.json（兼容旧数据包）。
    """
    return f'''\
def _load_instances(cfg):
    """加载测试实例：优先内存生成（零文件 dataset），回退读 data/*.json。

    v3.2: LLM4AD dispatcher 在 `dataset.mode: files + files: []` 时对每个算法
    只调用一次 evaluate(cfg)，此时 cfg.data_path 为空字符串、cfg.project_root
    可能为空。评估器必须在内存中生成 FUNCTIONS×DIMENSIONS×SEEDS 全部实例，
    不依赖任何磁盘文件。
    """
    # cfg.data_path 为空/不存在 → 内存生成
    has_data_path = bool(cfg.data_path) and cfg.data_path != "."
    if not has_data_path:
        instances = []
        for func in ExperimentConfig.FUNCTIONS:
            for dim in ExperimentConfig.DIMENSIONS:
                for seed in ExperimentConfig.SEEDS:
                    instances.append({{
                        "dimension": dim,
                        "function": func,
                        "seed": seed,
                        "budget": ExperimentConfig.EVALUATION_BUDGET,
                    }})
        return instances

    # 回退：旧数据包模式（data/instance_*.json）
    import json as _json
    from pathlib import Path as _Path
    data_dir = _Path(cfg.data_path)
    if data_dir.is_dir():
        inst_files = sorted(data_dir.glob("instance_*.json"))
        if inst_files:
            return [_json.loads(p.read_text(encoding="utf-8")) for p in inst_files]
    raise FileNotFoundError(f"No test instances found (data_path={{cfg.data_path}})")
'''


def _generate_primary_only_evaluator(
    algorithm_info: Any,
    aggregation_config: dict[str, Any],
) -> str:
    """生成 primary_only 模式的评估器"""

    primary_config = aggregation_config.get("primary_only", {})
    metric_key = primary_config.get("metric_key", "primary_metric")
    direction = primary_config.get("direction", "minimize")

    template = f'''"""自动生成的 LLM4AD 评估器 - Primary Only 模式"""
import sys
import time

import numpy as np

# 导入 LLM4AD 基类
try:
    from llm4ad.evaluator.base import (
        BaseEvaluator,
        EvalContext,
        EvaluationResult,
        Metric,
        MetricType,
    )
except ImportError:
    print("Error: llm4ad not installed. Please run: pip install llm4ad", file=sys.stderr)
    sys.exit(1)

# 导入算法、测试函数与实例配置
from solve import {algorithm_info.class_name}
from benchmark_suite import create_benchmark_suite
{_INSTANCE_CONFIG_IMPORT}

{_generate_in_memory_instances_code()}


@BaseEvaluator.register("arc_optimizer_evaluator")
class ARCOptimizerEvaluator(BaseEvaluator):
    """ARC 优化器评估器 - Primary Only 模式"""

    def __init__(self):
        """初始化评估器"""
        self._metrics = [
            Metric(
                name="{metric_key}",
                type=MetricType.{"MINIMIZE" if direction == "minimize" else "MAXIMIZE"},
                weight=1.0,
                description="Primary optimization metric",
            ),
        ]

    @property
    def name(self) -> str:
        return "arc_optimizer_evaluator"

    @property
    def metrics(self) -> list[Metric]:
        return self._metrics

    async def evaluate(self, cfg: EvalContext) -> EvaluationResult:
        """评估优化器性能"""
        start_time = time.time()

        try:
            # 1. 加载测试实例（零文件 dataset：内存生成）
            instances = _load_instances(cfg)
            if not instances:
                return EvaluationResult(
                    score=0.0,
                    metrics={{}},
                    success=False,
                    error_message="No test instances found",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # 2. 运行所有测试实例
            results = []
            for inst in instances:
                # 创建 benchmark 函数
                benchmark = create_benchmark_suite(
                    dimension=inst["dimension"],
                    function_name=inst["function"],
                )

                # 运行优化器
                optimizer = {algorithm_info.class_name}(
                    dimension=inst["dimension"],
                    lb=benchmark.lb,
                    ub=benchmark.ub,
                    maxfevals=inst["budget"],
                    seed=inst["seed"],
                )

                x0 = benchmark.generate_initial_guess(inst["seed"])
                result = optimizer.optimize(benchmark.evaluate, x0)

                results.append(result)

            # 3. 提取主指标（优化器返回 dict 的 best_f 键）
            primary_values = [r["best_f"] for r in results]
            primary_values = [v for v in primary_values if np.isfinite(v)]

            if not primary_values:
                return EvaluationResult(
                    score=0.0,
                    metrics={{}},
                    success=False,
                    error_message="All primary metric values are invalid",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # 4. 计算平均值
            avg_primary = float(np.mean(primary_values))

            # 5. 转为 score（最大化）
            {"score = -avg_primary" if direction == "minimize" else "score = avg_primary"}

            duration_ms = (time.time() - start_time) * 1000

            return EvaluationResult(
                score=score,
                metrics={{
                    "{metric_key}": avg_primary,
                }},
                success=True,
                duration_ms=duration_ms,
                metadata={{
                    "n_instances": len(instances),
                    "aggregation": "primary_only",
                }},
            )

        except Exception as e:
            return EvaluationResult(
                score=0.0,
                metrics={{}},
                success=False,
                error_message=f"Evaluation error: {{str(e)}}",
                duration_ms=(time.time() - start_time) * 1000,
            )
'''

    return template


def _generate_weighted_sum_evaluator(
    algorithm_info: Any,
    aggregation_config: dict[str, Any],
    baseline_stats: dict[str, dict],
) -> str:
    """生成 weighted_sum 模式的评估器"""

    weights = aggregation_config.get("weights", {"primary_metric": 1.0})
    normalization = aggregation_config.get("normalization", "zscore")

    # 序列化配置为 JSON（用于模板）
    weights_json = json.dumps(weights, indent=8)
    baseline_stats_json = json.dumps(baseline_stats, indent=8)

    template = f'''"""自动生成的 LLM4AD 评估器 - Weighted Sum 模式"""
import sys
import time

import numpy as np

# 导入 LLM4AD 基类
try:
    from llm4ad.evaluator.base import (
        BaseEvaluator,
        EvalContext,
        EvaluationResult,
        Metric,
        MetricType,
    )
except ImportError:
    print("Error: llm4ad not installed. Please run: pip install llm4ad", file=sys.stderr)
    sys.exit(1)

# 导入算法、测试函数与实例配置
from solve import {algorithm_info.class_name}
from benchmark_suite import create_benchmark_suite
{_INSTANCE_CONFIG_IMPORT}

{_generate_in_memory_instances_code()}


@BaseEvaluator.register("arc_optimizer_evaluator")
class ARCOptimizerEvaluator(BaseEvaluator):
    """ARC 优化器评估器 - Weighted Sum 模式"""

    def __init__(self):
        """初始化评估器"""
        # 定义跟踪的指标
        self._metrics = [
            Metric(
                name="primary_metric",
                type=MetricType.MINIMIZE,
                weight=1.0,
                description="Primary optimization metric",
            ),
        ]

        # 权重配置
        self.weights = {weights_json}

        # 归一化方法
        self.normalization = "{normalization}"

        # 基线统计信息（用于归一化）
        self.baseline_stats = {baseline_stats_json}

    @property
    def name(self) -> str:
        return "arc_optimizer_evaluator"

    @property
    def metrics(self) -> list[Metric]:
        return self._metrics

    def _normalize_value(self, metric_name: str, value: float) -> float:
        """归一化单个指标值"""
        if self.normalization == "none":
            return value

        stats = self.baseline_stats.get(metric_name, {{}})

        if self.normalization == "zscore":
            # Z-score 归一化
            mean = stats.get("mean", value)
            std = stats.get("std", 1.0)
            std = max(std, 1e-8)
            return (value - mean) / std

        elif self.normalization == "minmax":
            # Min-max 归一化
            min_val = stats.get("min", value)
            max_val = stats.get("max", value)
            range_val = max(max_val - min_val, 1e-8)
            return (value - min_val) / range_val

        return value

    async def evaluate(self, cfg: EvalContext) -> EvaluationResult:
        """评估优化器性能"""
        start_time = time.time()

        try:
            # 1. 加载测试实例（零文件 dataset：内存生成）
            instances = _load_instances(cfg)
            if not instances:
                return EvaluationResult(
                    score=0.0,
                    metrics={{}},
                    success=False,
                    error_message="No test instances found",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # 2. 运行所有测试实例
            all_metrics = []
            for inst in instances:
                # 创建 benchmark 函数
                benchmark = create_benchmark_suite(
                    dimension=inst["dimension"],
                    function_name=inst["function"],
                )

                # 运行优化器
                optimizer = {algorithm_info.class_name}(
                    dimension=inst["dimension"],
                    lb=benchmark.lb,
                    ub=benchmark.ub,
                    maxfevals=inst["budget"],
                    seed=inst["seed"],
                )

                x0 = benchmark.generate_initial_guess(inst["seed"])
                result = optimizer.optimize(benchmark.evaluate, x0)

                # 收集指标（优化器返回 dict 的 best_f 键）
                metrics = {{
                    "primary_metric": result.get("best_f", float("inf")),
                    "wall_time": result.get("wall_time", 0.0),
                    "success": float(result.get("success", False)),
                    "n_evals": result.get("n_evals", 0),
                }}
                all_metrics.append(metrics)

            # 3. 聚合指标（取平均）
            avg_metrics = {{}}
            for key in self.weights.keys():
                values = [m[key] for m in all_metrics if key in m and np.isfinite(m[key])]
                if values:
                    avg_metrics[key] = float(np.mean(values))
                else:
                    avg_metrics[key] = float("inf") if key == "primary_metric" else 0.0

            # 4. 归一化
            normalized = {{}}
            for metric_name in self.weights.keys():
                if metric_name in avg_metrics:
                    normalized[metric_name] = self._normalize_value(
                        metric_name, avg_metrics[metric_name]
                    )

            # 5. 加权求和
            score = 0.0
            for metric_name, weight in self.weights.items():
                if metric_name in normalized:
                    score += weight * normalized[metric_name]

            duration_ms = (time.time() - start_time) * 1000

            return EvaluationResult(
                score=score,
                metrics=avg_metrics,
                success=True,
                duration_ms=duration_ms,
                metadata={{
                    "n_instances": len(instances),
                    "normalized_metrics": normalized,
                    "aggregation": "weighted_sum",
                }},
            )

        except Exception as e:
            return EvaluationResult(
                score=0.0,
                metrics={{}},
                success=False,
                error_message=f"Evaluation error: {{str(e)}}",
                duration_ms=(time.time() - start_time) * 1000,
            )
'''

    return template
