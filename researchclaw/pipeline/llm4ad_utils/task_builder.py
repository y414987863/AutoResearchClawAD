"""任务包构建器 - 构建 LLM4AD 的完整任务包

v3.2（Q3 简化）：
- **删除 data/ 目录生成** — evaluator 通过零文件 dataset（`mode: files` + `files: []`）
  在内存生成所有 FUNCTIONS×DIMENSIONS×SEEDS 实例，无需 JSON 落盘
- **config.yaml 精简内置**（~50 行）— provider/evolution/workspace 等全部内置默认，
  只暴露 evaluator 与少量会变的演化参数
- `coder.type: custom` + `providers[0].type: openai_compatible`（Q4：纯 prompt，零 CLI）
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .code_marker import mark_class_in_module
from .evaluator_generator import generate_evaluator_code
from .metric_aggregator import compute_baseline_stats

logger = logging.getLogger(__name__)


def build_task_package(
    algorithm_info: Any,
    boost_dir: Path,
    run_dir: Path,
    config: Any,
) -> Path:
    """构建 LLM4AD 任务包

    Args:
        algorithm_info: AlgorithmInfo 对象
        boost_dir: llm4ad_boost 目录
        run_dir: 运行根目录
        config: 全局配置

    Returns:
        任务包目录路径

    Raises:
        ValueError: 输入参数无效
        FileNotFoundError: 必需文件不存在
    """
    # 输入验证
    if not hasattr(algorithm_info, 'condition_name'):
        raise ValueError("algorithm_info 缺少 condition_name 属性")

    if not hasattr(algorithm_info, 'code'):
        raise ValueError("algorithm_info 缺少 code 属性")

    if not boost_dir.exists():
        raise ValueError(f"boost_dir 不存在: {boost_dir}")

    if not run_dir.exists():
        raise ValueError(f"run_dir 不存在: {run_dir}")

    task_dir = boost_dir / algorithm_info.condition_name / "task_package"

    # v3.5: 统一绝对路径 —— config.yaml 里 local_path/base_dir 写入绝对路径，
    # 避免 LLM4AD 按"相对 config.yaml 目录"语义拼接（llm4ad.py:84-87/270-273）
    # 时，相对 task_dir 被二次拼接成 <task_package>/<相对路径> 双重嵌套，指向
    # 不存在的目录 → repo 分析 0 个文件 → generation-0 全跳。
    # 真实流水线 boost_dir 是相对路径（run_dir/stage-13/llm4ad_boost），
    # 不 resolve 必踩此坑（探针传绝对路径故未暴露）。
    task_dir = task_dir.resolve()

    # v3.4: 构建前清空旧 task_dir —— 任务包 100% 由本函数生成，无人工文件，
    # rmtree 安全。防止旧构建残留（如旧版 config.yaml 的 local_path 指向
    # 空目录 abc/）被新 build 覆盖后继续存活，导致 repo 分析 0 个 evolvable
    # block、generation-0 全跳（island_ga: "Skipping individual ..."）。
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"构建任务包: {algorithm_info.condition_name}")

    try:
        # 1. 生成 solve.py（标记后的算法）
        #    v3.2: 通过 out_method_name 回传真实被标记方法名（继承式类可能
        #    不是 optimize），供 Coder prompt 模板使用
        method_name: list[str] = []
        _generate_solve_py(algorithm_info, task_dir, config, method_name)

        # 2. 复制 benchmark_suite.py（原始测试函数，含数据生成逻辑）
        #    v3.2: 同时复制 experiment_config.py —— evaluator 从
        #    experiment_config.Config 读取 FUNCTIONS/DIMENSIONS/SEEDS/BUDGET
        #    在内存生成实例（零文件 dataset）
        _copy_benchmark_suite(run_dir, task_dir)

        # v3.2: 不再生成 data/ 目录（零文件 dataset，evaluator 内存生成实例）

        # 3. 生成 evaluator.py（LLM4AD 适配评估器）
        _generate_evaluator_py(algorithm_info, task_dir, config)

        # 4. 生成 config.yaml（LLM4AD 配置，精简内置）
        _generate_llm4ad_config(
            algorithm_info,
            task_dir,
            config,
            run_dir,
            evolved_method_name=method_name[0] if method_name else "optimize",
        )

        logger.info(f"任务包构建完成: {task_dir}")
        return task_dir

    except Exception as e:
        logger.error(f"构建任务包失败 ({algorithm_info.condition_name}): {e}", exc_info=True)
        raise


def _generate_solve_py(
    algorithm_info: Any,
    task_dir: Path,
    config: Any,
    out_method_name: list[str] | None = None,
) -> None:
    """生成 solve.py —— 完整可运行的 optimizers.py，仅标记目标 proposed 类

    关键设计（继承式 proposed 算法）：proposed 算法可能是继承式实现，例如
    `CMAESDefaultOptimizer(CMAOptimizer)` 只重写 `_update_covariance`，其
    `optimize` 方法继承自基类 `CMAOptimizer`。若 solve.py 只放单个类源码
    （algorithm_info.code），则 `from solve import CMAESDefaultOptimizer` 后
    构造/调用 `.optimize()` 会因缺失基类而 NameError 或无法解析继承的方法。

    因此 solve.py 直接采用**完整 optimizers.py**（algorithm_info.file_path），
    保留全部基类与继承链；用 `mark_class_in_module` 仅在目标类
    `algorithm_info.class_name` 上打 EVOLVE 标记，绝不注入手写 BaseOptimizer
    模板（会与真实基类重复/冲突）。完整模块依赖 `numpy`/`time`/
    `experiment_config.Config`，其中 experiment_config.py 已由
    `_copy_benchmark_suite` 复制进任务包，故 solve.py 在任务包内可直接 import。

    v3.2: 通过 out_method_name 回传实际被标记的方法名（可能是 optimize，
    也可能是继承式类重写的 _update_covariance 等），供 _generate_llm4ad_config
    注入 Coder prompt 模板，避免模板硬编码 optimize 签名。
    """
    # 注意：marking 在 RCConfig 中是普通 dict（config.py Llm4adBoostConfig），
    # 必须用 .get() 访问，不能用属性访问。
    marking_strategy = config.experiment.llm4ad_boost.marking.get("strategy", "auto")

    # 读取完整模块源码（含基类与继承链）
    module_path = (
        Path(algorithm_info.file_path)
        if getattr(algorithm_info, "file_path", None)
        else None
    )
    if module_path is None or not module_path.exists():
        raise FileNotFoundError(
            f"未找到完整优化器模块 optimizers.py "
            f"(algorithm_info.file_path={module_path})，"
            f"无法为继承式 proposed 算法生成可运行的 solve.py"
        )
    full_source = module_path.read_text(encoding="utf-8")

    # 仅在目标类上打 EVOLVE 标记（保留其余基类/类不变）
    marked_source = mark_class_in_module(
        full_source, algorithm_info.class_name, marking_strategy, out_method_name
    )

    # 头部注释（置于模块 docstring 之前——注释不影响 docstring 作为首个语句）
    header = (
        f"# 优化器模块 - {algorithm_info.condition_name}\n"
        f"# LLM4AD 演化目标类: {algorithm_info.class_name}\n"
        f"# 此文件由 ResearchClaw 自动生成：完整可运行的 optimizers.py 副本，\n"
        f"# 仅 {algorithm_info.class_name} 的 EVOLVE_START/END 之间代码会被演化，\n"
        f"# 其余基类与继承链保持不变，以保证 "
        f"from solve import {algorithm_info.class_name} 可运行。\n\n"
    )

    solve_code = header + marked_source
    solve_path = task_dir / "solve.py"
    solve_path.write_text(solve_code, encoding="utf-8")
    logger.info(
        f"生成 solve.py（完整模块 + {algorithm_info.class_name} EVOLVE 标记）: {solve_path}"
    )


def _copy_benchmark_suite(run_dir: Path, task_dir: Path) -> None:
    """复制 benchmark_suite.py（原始测试函数 + 数据生成逻辑）

    v3.2: evaluator 直接 import benchmark_suite 在内存生成实例。
    benchmark_suite.py 依赖 `experiment_config.py`（get_search_domain），
    因此同时复制 experiment_config.py 到任务包。
    """
    # 从 Stage 10 复制
    stage10_dir = run_dir / "stage-10" / "experiment"
    benchmark_src = stage10_dir / "benchmark_functions.py"

    if not benchmark_src.exists():
        logger.error(f"未找到 benchmark_functions.py: {benchmark_src}")
        raise FileNotFoundError(f"benchmark_functions.py not found: {benchmark_src}")

    benchmark_dst = task_dir / "benchmark_suite.py"
    shutil.copy(benchmark_src, benchmark_dst)

    # 复制 experiment_config.py（benchmark_suite 的 get_search_domain 依赖）
    config_src = stage10_dir / "experiment_config.py"
    if config_src.exists():
        shutil.copy(config_src, task_dir / "experiment_config.py")
        logger.info(f"复制 experiment_config.py: {task_dir / 'experiment_config.py'}")
    else:
        logger.warning(f"未找到 experiment_config.py: {config_src}（benchmark_suite 可能无法导入）")

    logger.info(f"复制 benchmark_suite.py: {benchmark_dst}")


def _generate_evaluator_py(
    algorithm_info: Any,
    task_dir: Path,
    config: Any,
) -> None:
    """生成 evaluator.py（LLM4AD 适配评估器）"""

    aggregation_config = config.experiment.llm4ad_boost.metric_aggregation

    # 计算基线统计信息（用于归一化）
    baseline_metrics = algorithm_info.baseline_metrics
    baseline_stats = {}

    if aggregation_config.get("method") == "weighted_sum":
        # 为每个指标创建统计信息
        for metric_name in aggregation_config.get("weights", {}).keys():
            if metric_name in baseline_metrics:
                value = baseline_metrics[metric_name]
                # 简化：使用单个值的统计（实际应该用多次运行）
                baseline_stats[metric_name] = {
                    "mean": value,
                    "std": value * 0.1,  # 估计 10% 的标准差
                    "min": value * 0.9,
                    "max": value * 1.1,
                }

    # 生成评估器代码
    evaluator_code = generate_evaluator_code(
        algorithm_info,
        aggregation_config,
        baseline_stats,
    )

    evaluator_path = task_dir / "evaluator.py"
    evaluator_path.write_text(evaluator_code, encoding="utf-8")

    logger.info(f"生成 evaluator.py: {evaluator_path}")


def _generate_llm4ad_config(
    algorithm_info: Any,
    task_dir: Path,
    config: Any,
    run_dir: Path | None = None,
    evolved_method_name: str = "optimize",
) -> None:
    """生成 LLM4AD 配置文件（v3.2 精简内置版，~50 行）

    只暴露用户会变的字段：
    - `evaluator`（module 路径、metrics、dataset 零文件）
    - 少量演化参数（max_generations/population_size）

    内置默认（用户无需关心）：
    - providers（从 config.arc.yaml 的 llm. 段读取）
    - coder（type: custom 纯 prompt EVOLVE 替换）
    - planner / evolution / workspace / version_control / repo_analyzer / logging

    v3.2: 实例生成配置（FUNCTIONS/DIMENSIONS/SEEDS/BUDGET）不写在 config.yaml
    里 —— 评估器直接从 task_package 内的 experiment_config.py import Config
    （_copy_benchmark_suite 已复制）。config.yaml 仅保留 LLM4AD 自身需要的字段，
    避免未知键被 AppConfig（pydantic，默认忽略未知键）静默吞掉造成数据流断裂。

    v3.2 generation-0 必须的内置段：
    - `version_control`：local_path 指向 task_package 绝对路径。llm4ad.py 只有在
      version_control.enabled 且 local_path 存在时才会对 task_package 做 repo 分析
      （否则 repo_path=None → _analyzed_repository=None → init_sampler 抛 ValueError）。
      git_worktree 在 local_path 处 auto_initialize（git init + 首次 commit，包含
      solve.py/evaluator.py/benchmark_suite.py/experiment_config.py），每次演化的
      独立 worktree 基于该 commit 创建。
    - `repo_analyzer`：type=evolve_detector，扫描 task_package 找到 solve.py 的
      EVOLVE 标记块，供 init_sampler/planner 构造演化 prompt。

    Args:
        algorithm_info: AlgorithmInfo 对象
        task_dir: 任务包目录
        config: 全局配置
        run_dir: 运行根目录（可选，用于校验 stage-10 实验配置已复制）
    """
    boost_config = config.experiment.llm4ad_boost
    llm_config = config.llm

    # 嵌套子配置在 RCConfig 中是普通 dict（config.py Llm4adBoostConfig），
    # 必须用 .get() 访问并提供默认值，兼容用户 yaml 省略部分字段的情况。
    resources = boost_config.resources or {}
    evolution = boost_config.evolution or {}
    island = evolution.get("island", {}) or {}

    # 校验实验配置已复制（evaluator 内存生成实例的依赖）
    if run_dir is not None:
        stage10_config = run_dir / "stage-10" / "experiment" / "experiment_config.py"
        if not stage10_config.exists():
            logger.warning(
                f"未找到 experiment_config.py: {stage10_config}，"
                f"evaluator 内存生成实例将依赖 task_package 内副本"
            )

    # 构建精简配置（v3.2：只保留会变的字段）
    llm4ad_config = {
        "project_name": f"arc_optimize_{algorithm_info.condition_name}",
        "description_en": f"Evolve {algorithm_info.condition_name} optimizer for ARC-Bench",
        "description_zh": f"演化优化 {algorithm_info.condition_name} 算法",
        "background": f"Optimize the {algorithm_info.condition_name} algorithm for non-convex optimization problems.",
        "random_seed": 42,
        "base_dir": str(task_dir.parent.parent / "llm4ad"),

        # LLM Provider（从 config.arc.yaml 的 llm. 段读取；type 为 openai_compatible）
        "providers": [
            {
                "name": "default",
                "type": "openai_compatible",
                "base_url": llm_config.base_url,
                "api_key": llm_config.api_key,
                "model": llm_config.primary_model,
                "temperature": 0.7,
                "max_tokens": 8192,
                "timeout": float(llm_config.timeout_sec),
            }
        ],

        # Coder（Q4：LLM4AD 内置 custom，纯 prompt EVOLVE 块替换）
        # v3.2: prompt_template 注入真实被演化方法名（可能不是 optimize，
        # 而是继承式类重写的 _update_covariance 等），模板不再硬编码签名。
        "coder": {
            "type": "custom",
            "provider": "default",
            "temperature": 0.7,
            "max_gen_tokens": 4096,
            "context_max_tokens": 4096,
            "prompt_template": _generate_coder_prompt_template(
                algorithm_info, evolved_method_name
            ),
        },

        # Planner（内置默认）
        "planner": {
            "type": "llm_evolution",
            "provider": "default",
            "samplers": [
                {"name": "init_sampler"},
                {"name": "mutation_sampler"},
                {"name": "crossover_sampler"},
            ],
        },

        # Version control（generation-0 必需：local_path 指向 task_package）
        # llm4ad.py 依赖 version_control.enabled + local_path 才能对 task_package
        # 做 repo 分析（否则 _analyzed_repository=None → init_sampler 抛 ValueError）。
        # git_worktree 在 local_path 处 auto_initialize（git init + 首次 commit，
        # 包含 task 全部文件），每次演化的独立 worktree 基于该 commit 创建。
        "version_control": {
            "enabled": True,
            "type": "git_worktree",
            "local_path": str(task_dir),
            "auto_initialize": True,
        },

        # Repo analyzer（generation-0 必需：扫描 task_package 的 EVOLVE 标记）
        "repo_analyzer": {
            "type": "evolve_detector",
            "context_lines_before": 5,
            "context_lines_after": 5,
        },

        # Evaluator（v3.2：零文件 dataset — mode: files + files: []）
        "evaluator": {
            "type": "custom",
            "module": "evaluator.py:ARCOptimizerEvaluator",
            "timeout": float(resources.get("eval_timeout_sec", 120)),
            "max_retries": 2,
            "parallel": True,
            "batch_size": min(int(resources.get("parallel_workers", 4)), 6),
            "dataset": {
                "mode": "files",
                "files": [],
            },
            "metrics": ["primary_metric", "wall_time", "success"],
        },

        # Evolution（暴露会变的参数，其余内置默认）
        "evolution": {
            "type": evolution.get("method", "island_ga"),
            "max_generations": evolution.get("max_generations", 20),
            "population_size": evolution.get("population_size", 10),
            "elite_ratio": evolution.get("elite_ratio", 0.2),
            "mutation_rate": evolution.get("mutation_rate", 0.6),
            "crossover_rate": evolution.get("crossover_rate", 0.3),
            "island": {
                "count": island.get("count", 4),
                "migration_interval": island.get("migration_interval", 5),
                "migration_rate": island.get("migration_rate", 0.1),
            },
        },
    }

    config_path = task_dir / "config.yaml"
    import yaml
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(llm4ad_config, f, indent=2, allow_unicode=True, sort_keys=False)

    logger.info(f"生成 config.yaml: {config_path}")


def _generate_coder_prompt_template(algorithm_info: Any, method_name: str) -> str:
    """生成 Coder 的 prompt 模板（initial 生成用，覆盖全部 generation）

    关键约束（custom_naive_coder._generate_initial，LLM4AD v3.x）：
    - 模板必须含 `{{insight}}` 或 `{{project_context}}` 占位符，否则 _generate_initial
      直接使用 planner 的原始 prompt（模板被忽略）。本模板两者都含：
      - `{{insight}}` ← 演化洞察（init_sampler 的算法描述 / mutation_sampler 的
        变异洞察；llm_evolution.implement() 只传 description，parent_code 从不设置，
        因此**每一代**都走 _generate_initial 全量重写，模板对所有 generation 生效）
      - `{{project_context}}` ← coder 收集的项目上下文（EVOLVE 块已掩码，
        LLM 看不到现有实现，只能看到类结构与方法签名）
    - `.format()` 只传 6 个具名键（insight/language/task_description/constraints/
      project_context/file_name），模板中**不得**出现其他 `{name}` 占位符。
      因此方法名/类名以字面量嵌入（用普通字符串拼接，不用 f-string 的 {method_name}
      表达式——那会在 .format() 时 KeyError）。

    v3.2 继承式修复：演化目标不一定是 optimize —— 继承式 proposed 类（如
    CMAESDefaultOptimizer(CMAOptimizer)）只重写 _update_covariance，EVOLVE
    标记包裹的是该方法。模板把真实演化方法名写进指令，并明确：
    - EVOLVE 块内是**完整方法定义**（含 def 签名行）——code_marker 的 fallback
      方法块策略把首个非 __init__ 方法至类结束整段包进标记；方法体策略则是 def
      行之后到方法结束。无论哪种，LLM 输出都必须保持标记位置、保持 def 签名与
      基类调用契约，只改进方法体逻辑。
    - solve.py 是完整 optimizers.py（基类+继承链），LLM 必须原样复刻标记外的
      全部代码，保证 `from solve import {class_name}` 可运行。

    Args:
        algorithm_info: AlgorithmInfo（class_name / condition_name）
        method_name: 被演化方法名（solve.py 中 EVOLVE 标记包裹的方法；如
            optimize 或 _update_covariance）
    """

    class_name = algorithm_info.class_name
    template = (
        "You are an expert optimization algorithm engineer. Improve the algorithm "
        + class_name
        + " for non-convex black-box optimization (Rastrigin, Rosenbrock, Ackley, etc.).\n"
        "\n"
        "Task:\n"
        "{insight}\n"
        "\n"
        "Project context below shows the full optimizer module `solve.py` exactly as it exists on disk.\n"
        "The code between `# EVOLVE_START` and `# EVOLVE_END` markers is the evolvable region — it is the\n"
        "method `"
        + class_name
        + "."
        + method_name
        + "` and is currently masked. Everything outside the markers (imports, other classes, base\n"
        "classes and the rest of the inheritance chain) must be reproduced EXACTLY as shown — do not rename,\n"
        "reorder or rewrite it, since the module is imported as a whole and `"
        + class_name
        + "` inherits from its base class.\n"
        "\n"
        "Rules:\n"
        "1. Output the COMPLETE `solve.py` module (imports, all classes, helpers) in a fenced code block\n"
        '   annotated ```python:solve.py```\n'
        "2. Keep the `# EVOLVE_START` / `# EVOLVE_END` markers at their current positions and write your\n"
        "   improved implementation of `"
        + method_name
        + "` (the ENTIRE method, including its `def "
        + method_name
        + "(self, ...)` signature line and body) between them — this is the only part that will change.\n"
        "3. Do NOT modify anything outside the EVOLVE markers, including the class definition line, the base\n"
        "   classes, or other methods of "
        + class_name
        + ".\n"
        "4. If `"
        + method_name
        + "` is an internal helper (e.g. `_update_covariance`), keep its signature and side effects\n"
        + "   compatible with the base class callers; if it is `optimize`, return a dict with keys\n"
        '   {{"best_x", "best_f", "n_evals", "wall_time", "success"}}\n'
        "5. Use the evaluation budget efficiently (self.maxfevals)\n"
        "6. Handle both low and high dimensional problems (5-20 dimensions)\n"
        "\n"
        "Consider improvements like:\n"
        "- Better initialization strategies\n"
        "- Adaptive parameter tuning\n"
        "- Hybrid approaches combining multiple strategies\n"
        "- Smart restart mechanisms\n"
        "- Improved exploration/exploitation balance\n"
        "\n"
        "{project_context}\n"
        "\n"
        "Write the complete improved `solve.py` now, with the improved `"
        + method_name
        + "` inside the EVOLVE markers.\n"
    )
    return template
