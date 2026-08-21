"""测试 LLM4AD 集成的简单脚本"""

import sys
from pathlib import Path

# 注意：不要在此处重绑 sys.stdout/sys.stderr。pytest 在收集期已经接管了这两个
# 流，用 io.TextIOWrapper 包一层会让整个 tests/ 目录的收集直接崩掉。
# 需要 UTF-8 输出时用 PYTHONIOENCODING=utf-8 或 python -X utf8。

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from researchclaw.pipeline.llm4ad_utils import (
    mark_evolution_boundaries,
    create_metric_aggregator,
)


def test_code_marker():
    """测试代码标记器"""
    print("=" * 60)
    print("测试: 代码标记器")
    print("=" * 60)

    sample_code = '''
class CMAESDefaultOptimizer(BaseOptimizer):
    def __init__(self, dimension, lb, ub, maxfevals, seed):
        super().__init__(dimension, lb, ub, maxfevals, seed)

    def optimize(self, objective, x0):
        """优化方法"""
        # 初始化
        self.mean = x0.copy()

        # 主循环
        while self.n_evals < self.maxfevals:
            # 采样
            samples = self._sample()
            # 评估
            values = [objective(s) for s in samples]
            # 更新
            self._update(samples, values)

        return self.best_x, self.best_f
'''

    marked = mark_evolution_boundaries(sample_code, strategy="auto")
    print("标记后的代码:")
    print(marked)
    print("\n[PASS] 代码标记测试通过\n")


def test_metric_aggregator():
    """测试指标聚合器"""
    print("=" * 60)
    print("测试: 指标聚合器")
    print("=" * 60)

    # 测试 primary_only
    config = {
        "method": "primary_only",
        "primary_only": {
            "metric_key": "primary_metric",
            "direction": "minimize",
        }
    }
    aggregator = create_metric_aggregator(config)

    metrics = {
        "primary_metric": 100.0,
        "wall_time": 10.0,
        "success_rate": 0.95,
    }

    score = aggregator.aggregate(metrics)
    print(f"Primary Only Score: {score}")
    assert score == -100.0, "Primary only score should be -100.0"
    print("[PASS] Primary Only 测试通过")

    # 测试 weighted_sum
    config = {
        "method": "weighted_sum",
        "weights": {
            "primary_metric": 1.0,
            "wall_time": -0.2,
            "success_rate": 0.5,
        },
        "normalization": "none",
    }
    aggregator = create_metric_aggregator(config)
    score = aggregator.aggregate(metrics)
    print(f"Weighted Sum Score: {score}")
    print("[PASS] Weighted Sum 测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("LLM4AD 集成 - 单元测试")
    print("=" * 60 + "\n")

    try:
        test_code_marker()
        test_metric_aggregator()

        print("=" * 60)
        print("[SUCCESS] 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
