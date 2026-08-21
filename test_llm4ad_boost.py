#!/usr/bin/env python3
"""LLM4AD Boost 独立测试脚本

直接在已有的 Stage 13 产物上运行 LLM4AD boost，无需重跑前面的步骤。
用于快速测试和调试 LLM4AD 集成。

使用方法:
    python test_llm4ad_boost.py <run_dir>

示例:
    python test_llm4ad_boost.py artifacts/ab-rc_full-ML03-fix-ds-1
"""

import sys
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from researchclaw.config import load_config
from researchclaw.pipeline.stage_impls._llm4ad_boost import run_llm4ad_boost_inline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('llm4ad_boost_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n错误: 请提供 run_dir 路径")
        print("\n示例:")
        print("  python test_llm4ad_boost.py artifacts/ab-rc_full-ML03-fix-ds-1")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    if not run_dir.exists():
        print(f"错误: 运行目录不存在: {run_dir}")
        sys.exit(1)

    # 检查 Stage 13 是否存在
    stage13_dir = run_dir / "stage-13"
    if not stage13_dir.exists():
        print(f"错误: Stage 13 目录不存在: {stage13_dir}")
        print("请确保提供的是包含 Stage 13 产物的完整运行目录")
        sys.exit(1)

    print("=" * 60)
    print("LLM4AD Boost 独立测试")
    print("=" * 60)
    print(f"运行目录: {run_dir}")
    print(f"Stage 13: {stage13_dir}")
    print()

    # 使用根目录的 config.arc.yaml（写死）
    config_file = Path(__file__).parent / "config.arc.yaml"

    if not config_file.exists():
        print(f"错误: 配置文件不存在: {config_file}")
        sys.exit(1)

    print(f"配置文件: {config_file}")

    # 加载配置
    try:
        config = load_config(config_file)
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"错误: 配置加载失败: {e}")
        sys.exit(1)

    # 检查 LLM4AD boost 是否启用
    if not hasattr(config.experiment, 'llm4ad_boost') or not config.experiment.llm4ad_boost.enabled:
        print("\n警告: LLM4AD boost 未启用")
        print("请在配置文件中设置:")
        print("  experiment:")
        print("    llm4ad_boost:")
        print("      enabled: true")

        response = input("\n是否继续运行? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    # 设置 boost 目录
    boost_dir = stage13_dir / "llm4ad_boost"

    print("\n开始运行 LLM4AD Boost...")
    print("-" * 60)

    try:
        # 运行 LLM4AD boost（不需要适配器参数）
        result = run_llm4ad_boost_inline(
            boost_dir=boost_dir,
            stage_dir=stage13_dir,
            run_dir=run_dir,
            config=config,
        )

        print("\n" + "=" * 60)
        if result and result.get("status") != "skipped":
            print("✓ LLM4AD Boost 完成")
            print("=" * 60)
            print(f"\n结果:")
            print(f"  总算法数: {result.get('total_algorithms', 0)}")
            print(f"  成功演化: {result.get('successful', 0)}")
            print(f"  性能改进: {result.get('improved', 0)}")
            print(f"  平均提升: {result.get('avg_improvement_pct', 0):.2f}%")
            print(f"\n产物位置: {boost_dir}")
            print(f"\n查看报告:")
            print(f"  汇总: {boost_dir / 'summary_report.md'}")
            print(f"  详细: {boost_dir / '<algorithm_name>' / 'report.md'}")
        else:
            print("⚠ LLM4AD Boost 跳过或失败")
            print("=" * 60)
            if result:
                print(f"原因: {result.get('reason', 'unknown')}")

    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
