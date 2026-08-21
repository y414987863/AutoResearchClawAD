"""指标聚合器 - 将多个指标聚合成单一 score"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class MetricAggregator:
    """指标聚合器基类"""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def aggregate(self, metrics: dict[str, float]) -> float:
        """聚合多个指标成单一 score

        Args:
            metrics: {
                "primary_metric": float,
                "wall_time": float,
                "success_rate": float,
                ...
            }

        Returns:
            单一聚合分数（越大越好）
        """
        raise NotImplementedError


class PrimaryOnlyAggregator(MetricAggregator):
    """仅使用主指标"""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.metric_key = config.get("metric_key", "primary_metric")
        self.direction = config.get("direction", "minimize")

    def aggregate(self, metrics: dict[str, float]) -> float:
        value = metrics.get(self.metric_key, float("inf"))

        if not np.isfinite(value):
            return -float("inf")  # 无效值返回最差分数

        # 转为最大化问题
        if self.direction == "minimize":
            return -value
        else:
            return value


class WeightedSumAggregator(MetricAggregator):
    """加权求和聚合器（支持归一化）"""

    def __init__(self, config: dict[str, Any], baseline_stats: dict[str, dict] = None):
        super().__init__(config)
        self.weights = config.get("weights", {"primary_metric": 1.0})
        self.normalization = config.get("normalization", "zscore")
        self.baseline_stats = baseline_stats or {}

        logger.info(f"WeightedSumAggregator: weights={self.weights}, norm={self.normalization}")

    def aggregate(self, metrics: dict[str, float]) -> float:
        """加权求和聚合

        Steps:
        1. 归一化每个指标
        2. 加权求和
        """
        normalized = {}
        for metric_name in self.weights.keys():
            if metric_name not in metrics:
                logger.warning(f"指标 {metric_name} 不存在于 metrics 中")
                continue

            value = metrics[metric_name]
            if not np.isfinite(value):
                logger.warning(f"指标 {metric_name} 值无效: {value}")
                normalized[metric_name] = 0.0
                continue

            # 归一化
            normalized[metric_name] = self._normalize_value(metric_name, value)

        # 加权求和
        score = 0.0
        for metric_name, weight in self.weights.items():
            if metric_name in normalized:
                score += weight * normalized[metric_name]

        return score

    def _normalize_value(self, metric_name: str, value: float) -> float:
        """归一化单个指标值"""
        if self.normalization == "none":
            return value

        stats = self.baseline_stats.get(metric_name, {})

        if self.normalization == "zscore":
            # Z-score 归一化: (x - mean) / std
            mean = stats.get("mean", value)
            std = stats.get("std", 1.0)
            std = max(std, 1e-8)  # 避免除零
            return (value - mean) / std

        elif self.normalization == "minmax":
            # Min-max 归一化: (x - min) / (max - min)
            min_val = stats.get("min", value)
            max_val = stats.get("max", value)
            range_val = max(max_val - min_val, 1e-8)
            return (value - min_val) / range_val

        else:
            logger.warning(f"未知的归一化方法: {self.normalization}")
            return value


def create_metric_aggregator(
    aggregation_config: dict[str, Any],
    baseline_stats: dict[str, dict] = None,
) -> MetricAggregator:
    """创建指标聚合器

    Args:
        aggregation_config: {
            "method": "weighted_sum" | "primary_only",
            "weights": {...},  # for weighted_sum
            "normalization": "zscore" | "minmax" | "none",
            "primary_only": {...},  # for primary_only
        }
        baseline_stats: 用于归一化的基线统计信息

    Returns:
        MetricAggregator 实例
    """
    method = aggregation_config.get("method", "primary_only")

    if method == "primary_only":
        config = aggregation_config.get("primary_only", {})
        return PrimaryOnlyAggregator(config)

    elif method == "weighted_sum":
        return WeightedSumAggregator(aggregation_config, baseline_stats)

    else:
        raise ValueError(f"未知的聚合方法: {method}")


def compute_baseline_stats(metrics_list: list[dict[str, float]]) -> dict[str, dict]:
    """计算基线统计信息（用于归一化）

    Args:
        metrics_list: 多次运行的指标列表

    Returns:
        {
            "metric_name": {
                "mean": float,
                "std": float,
                "min": float,
                "max": float,
            },
            ...
        }
    """
    if not metrics_list:
        return {}

    # 收集每个指标的所有值
    metric_values = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key not in metric_values:
                metric_values[key] = []
            if np.isfinite(value):
                metric_values[key].append(value)

    # 计算统计量
    stats = {}
    for metric_name, values in metric_values.items():
        if not values:
            continue

        values_array = np.array(values)
        stats[metric_name] = {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
        }

    return stats
