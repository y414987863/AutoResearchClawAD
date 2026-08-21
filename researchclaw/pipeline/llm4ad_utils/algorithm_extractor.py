"""算法提取器 - 从 Stage 13 提取 proposed 算法（LLM 三路分类 + 启发式回退）

v3.2:
- `_find_latest_experiment_version` 只认 `experiment_final/`（experiment_v* 是中间产物）
- `_load_stage12_results` 重写：优先 results.json，回退 run-*.json，扁平 metrics 解析
- LLM 三路分类（baseline/ablation/proposed，结构化 JSON）+ 代码启发式回退
- 分类结果写入 boost_summary.json（classified: {baseline, ablation, proposed}）
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# 回退启发式：scipy 直接搜索方法名（仅在 LLM 分类不可用时使用）
_SCIPY_DIRECT_SEARCH_CONDITIONS = {
    "nelder_mead",
    "powell",
    "cobyla",
    "slsqp",
    "l_bfgs_b",
    "tnc",
    "bfgs",
    "cg",
    "newton_cg",
    "dogleg",
    "trust_ncg",
    "trust_exact",
    "trust_krylov",
    "baseline",
}

# 扁平 metrics 键模式
# <condition>/<function>_<dim>d/<metric>           (单元格均值/汇总)
# <condition>/<function>_<dim>d/<seed>/<metric>     (单种子)
# <condition>/<function>_<dim>d/<metric>_mean|_std   (聚合统计)
_CELL_KEY_RE = re.compile(
    r"^(?P<cond>[^/]+)/(?P<cell>[^/]+)/(?P<metric>[^/]+)$"
)
_SEED_KEY_RE = re.compile(
    r"^(?P<cond>[^/]+)/(?P<cell>[^/]+)/(?P<seed>\d+)/(?P<metric>[^/]+)$"
)

# 汇总指标名（条件级）
# 与 Stage 12 报告口径一致：单元格级聚合的均值跨 cell 再平均
_COND_AGG_METRICS = (
    "primary_metric",
    "median_wall_clock_runtime_seconds",
    "completion_success_rate",
)

# 跳过 _mean/_std 聚合变体键（primary_metric_mean 等，与不带后缀的键重复）
_AGG_SUFFIXES = ("_mean", "_std")

# LLM 分类方法（写入 boost_summary.json 的 source 字段）
_last_classify_method = "heuristic"


class AlgorithmInfo:
    """算法信息数据类"""

    def __init__(
        self,
        condition_name: str,
        class_name: str,
        file_path: Path,
        code: str,
        baseline_metrics: dict[str, Any],
        rank: int = 0,
        category: str = "proposed",
    ):
        self.condition_name = condition_name
        self.class_name = class_name
        self.file_path = file_path
        self.code = code
        self.baseline_metrics = baseline_metrics
        self.rank = rank
        self.category = category

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_name": self.condition_name,
            "class_name": self.class_name,
            "file_path": str(self.file_path),
            "baseline_metrics": self.baseline_metrics,
            "rank": self.rank,
            "category": self.category,
        }


def extract_proposed_algorithms(
    stage_dir: Path,
    run_dir: Path,
    boost_dir: Path | None = None,
    config: Any = None,
) -> list[AlgorithmInfo]:
    """从 Stage 13 提取所有 proposed 算法（LLM 三路分类，排除 baseline 与消融）

    Args:
        stage_dir: Stage 13 目录
        run_dir: 运行根目录
        boost_dir: boost 输出目录（用于写 boost_summary.json；None 时写 stage_dir）
        config: RCConfig（LLM 分类用；None 时尝试自动定位 config.arc.yaml）

    Returns:
        AlgorithmInfo 列表，按性能排序
    """
    logger.info("开始提取 proposed 算法...")

    # 1. 找到 Stage 13 的规范最终版本目录（只认 experiment_final/）
    final_exp_dir = _find_latest_experiment_version(stage_dir)
    if not final_exp_dir:
        logger.warning("未找到 Stage 13 最终实验版本 (experiment_final/)")
        return []

    logger.info(f"使用 Stage 13 最终版本: {final_exp_dir.name}")

    # 2. 读取优化器代码文件
    optimizers_file = final_exp_dir / "optimizers.py"
    if not optimizers_file.exists():
        logger.error(f"未找到优化器文件: {optimizers_file}")
        return []

    optimizers_code = optimizers_file.read_text(encoding="utf-8")

    # 3. 从 Stage 12 读取所有条件的结果（扁平 metrics 解析）
    stage12_results = _load_stage12_results(run_dir / "stage-12")
    if not stage12_results:
        logger.warning("未找到 Stage 12 结果")
        return []

    # 4. 提取所有优化器类
    optimizer_classes = _parse_optimizer_classes(optimizers_code)

    # 5. LLM 三路分类（baseline / ablation / proposed）
    # 只对能匹配到 Stage-12 已知条件的类做分类（排除工具类/基类）
    known_conditions = set(stage12_results.keys())
    condition_classes = {
        cls: cond for cls, cond in (
            (c, _class_name_to_condition(c, list(known_conditions)))
            for c in optimizer_classes
        ) if cond in known_conditions
    }

    classifications = _classify_algorithms(
        optimizers_code=optimizers_code,
        class_names=list(condition_classes.keys()),
        final_exp_dir=final_exp_dir,
        config=config,
    )

    # 分类结果写入 boost_summary.json（可审计）
    _write_classification(
        boost_dir=boost_dir,
        classifications=classifications,
        stage_dir=stage_dir,
        known_conditions=list(known_conditions),
    )

    # 6. 匹配算法和结果，只保留 proposed
    algorithms = []
    for class_name, condition_name in condition_classes.items():
        class_code = optimizer_classes[class_name]
        category = classifications.get(class_name, "proposed")

        # 跳过 baseline 与消融
        if category != "proposed":
            logger.info(f"跳过 {category} 算法: {condition_name}")
            continue

        # 查找对应的结果
        result = stage12_results.get(condition_name)
        if not result:
            logger.warning(f"未找到条件 {condition_name} 的结果，跳过")
            continue

        algorithms.append(AlgorithmInfo(
            condition_name=condition_name,
            class_name=class_name,
            file_path=optimizers_file,
            code=class_code,
            baseline_metrics=result["metrics"],
            rank=result["rank"],
            category=category,
        ))

    # 7. 按性能排序（rank 越小越好）
    algorithms.sort(key=lambda a: a.rank)

    logger.info(f"成功提取 {len(algorithms)} 个 proposed 算法")
    for algo in algorithms:
        logger.info(f"  - {algo.condition_name} (rank={algo.rank})")

    return algorithms


def _class_name_to_condition(class_name: str, known_conditions: list[str] | None = None) -> str:
    """将类名转换为条件名

    例如:
    - CMAESDefaultOptimizer -> cma_es_default
    - NelderMeadOptimizer -> nelder_mead
    - CMAESIPOPRestartBaseline -> cma_es_ipop_restart

    已知条件列表存在时，用归一化（去下划线）模糊匹配避免缩写拆分错误
    （如 `CMAES` 被逐字符拆成 `c_m_a_e_s`）。
    """
    # 移除 "Optimizer" 后缀与 "Baseline" 后缀
    name = class_name.replace("Optimizer", "").replace("Baseline", "")

    # 驼峰转下划线
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())

    condition = "".join(result)

    # 已知条件列表存在时，用归一化模糊匹配修正缩写拆分
    if known_conditions:
        condition = _match_known_condition(condition, known_conditions)

    return condition


def _normalize_for_match(s: str) -> str:
    """归一化字符串用于条件名匹配（去下划线、去连字符、小写）"""
    return s.replace("_", "").replace("-", "").lower()


def _match_known_condition(condition: str, known_conditions: list[str]) -> str:
    """将解析出的条件名与已知条件列表做归一化模糊匹配

    匹配规则（按优先级）：
    1. 归一化后精确相等 → 命中
    2. 已知名归一化 ⊂ 解析名归一化（解析名更长，如 `cma_es_default`
       是从 `CMAESDefault` 解析来的，去掉下划线后与 `cma_es_default` 相等）→ 命中
    3. difflib 相似度 ≥ 0.75 → 命中
    """
    norm_condition = _normalize_for_match(condition)

    # 1. 精确匹配（归一化后）
    for known in known_conditions:
        if _normalize_for_match(known) == norm_condition:
            return known

    # 2. 已知名 ⊂ 解析名（解析名是已知名的超集）
    #    例如 `cma_es_ipop_restart`（解析自 CMAESIPOPRestartBaseline）
    #    归一化 `cmaesipoprestart` 包含已知名 `cma_es_ipop_restart` 的 `cmaesipoprestart`。
    #    但排除过短匹配：解析名去下划线后至少比已知名长 2 个字符（防 `cma` ⊂ `cmaesdefault`）
    for known in known_conditions:
        norm_known = _normalize_for_match(known)
        if len(norm_condition) >= len(norm_known) + 2 and norm_known in norm_condition:
            return known

    # 3. difflib 相似度
    import difflib

    best = difflib.get_close_matches(condition, known_conditions, n=1, cutoff=0.75)
    if best:
        return best[0]

    return condition


def _find_latest_experiment_version(stage_dir: Path) -> Path | None:
    """找到 Stage 13 的规范最终实验版本目录

    v3.2: **只认 `experiment_final/`**（`experiment_v1/`/`experiment_v2/`
    是中间迭代版本，仅参考，不作为 boost 输入）。
    """
    candidate = stage_dir / "experiment_final"
    if candidate.exists() and (candidate / "optimizers.py").exists():
        return candidate

    logger.warning(f"未找到 experiment_final/optimizers.py: {candidate}")
    return None


def _load_stage12_results(stage12_dir: Path) -> dict[str, Any]:
    """加载 Stage 12 所有条件的结果（扁平 metrics 解析）

    v3.2 重写：
    - 优先 `runs/results.json`，回退 `runs/run-*.json`（取最新）
    - 解析扁平 metrics 键（一个 run 包含所有条件）
    - 键模式见模块顶部 `_CELL_KEY_RE` / `_SEED_KEY_RE`

    Returns:
        {
            "condition_name": {
                "metrics": {
                    "primary_metric": float,      # 条件级均值（cell 均值再平均）
                    "wall_time": float,            # median_wall_clock_runtime_seconds 均值
                    "success_rate": float,         # completion_success_rate 均值
                    "std": float, "min": float, "max": float,
                },
                "rank": int,   # 按 primary_metric 升序（minimize）
            },
            ...
        }
    """
    if not stage12_dir.exists():
        logger.warning(f"Stage 12 目录不存在: {stage12_dir}")
        return {}

    runs_dir = stage12_dir / "runs"
    if not runs_dir.exists():
        logger.warning(f"Stage 12 runs 目录不存在: {runs_dir}")
        return {}

    # 优先 results.json（聚合结果，优先级最高）
    results_file = runs_dir / "results.json"
    if results_file.exists():
        logger.info(f"使用 Stage 12 结果文件: {results_file.name}")
        return _parse_metrics_payload(_safe_load_json(results_file), "results.json")

    # 回退 run-*.json（取最新）
    run_files = sorted(runs_dir.glob("run-*.json"), key=lambda f: f.stat().st_mtime)
    if not run_files:
        logger.warning(f"未找到任何 run 文件: {runs_dir}")
        return {}

    latest_run = run_files[-1]
    logger.info(f"使用 Stage 12 结果文件（回退）: {latest_run.name}")
    return _parse_metrics_payload(_safe_load_json(latest_run), latest_run.name)


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    """安全读取 JSON 文件"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"读取 {path.name} 失败 (JSON 解析错误): {e}")
    except Exception as e:
        logger.error(f"读取 {path.name} 失败: {e}")
    return None


def _parse_metrics_payload(data: dict[str, Any] | list, source: str) -> dict[str, Any]:
    """解析扁平 metrics payload（results.json 或 run-*.json 的 metrics 字典）

    单个文件包含所有条件的扁平 metrics：
    {
      "source": "stdout_parsed",
      "metrics": {
        "cma_es_default/rastrigin_2d/0/primary_metric": 4.97479,   # 单种子（跳过）
        "cma_es_default/rastrigin_2d/primary_metric": 1.39217,     # 单元格均值
        "cma_es_default/rastrigin_2d/primary_metric_mean": 1.39217,  # 变体（跳过）
        "cma_es_default/rastrigin_2d/primary_metric_std": 1.49337,   # 变体（跳过）
        ...
      },
      "run_id": "run-1", "task_id": "...", "status": "completed",  # run-*.json 额外字段
    }
    """
    if isinstance(data, list):
        # 可能是 run 文件数组（多 run），取第一个有 metrics 的
        for item in data:
            if isinstance(item, dict) and item.get("metrics"):
                return _parse_metrics_payload(item, source)
        logger.warning(f"{source}: 数组中没有含 metrics 的 run")
        return {}

    if not isinstance(data, dict):
        logger.warning(f"{source}: 未知格式 {type(data)}")
        return {}

    metrics = data.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        logger.warning(f"{source}: 无 metrics 字典")
        return {}

    # 按条件分组收集单元格指标（跳过种子级键与 _mean/_std 变体）
    # condition -> {metric -> [cell 均值]}
    cond_cells: dict[str, dict[str, list[float]]] = {}

    for key, value in metrics.items():
        if not isinstance(key, str):
            continue

        # 单种子键: cond/cell/seed/metric（跳过，均值键已覆盖单元格级）
        if _SEED_KEY_RE.match(key):
            continue

        # 单元格键: cond/cell/metric
        cell_m = _CELL_KEY_RE.match(key)
        if not cell_m:
            # 无条件前缀的全局汇总键（如顶层 primary_metric）跳过
            continue

        cond = cell_m.group("cond")
        metric = cell_m.group("metric")

        # 跳过 _mean/_std 聚合变体（与不带后缀的均值键重复）
        if metric.endswith(_AGG_SUFFIXES):
            continue

        if not _is_numeric(value):
            continue

        cond_cells.setdefault(cond, {}).setdefault(metric, []).append(float(value))

    # 计算每个条件的指标
    results: dict[str, Any] = {}
    for cond, metric_values in cond_cells.items():
        primary_values = metric_values.get("primary_metric", [])
        if not primary_values:
            logger.warning(f"条件 {cond} 无 primary_metric 单元格值")
            continue

        avg_primary = float(sum(primary_values) / len(primary_values))
        if len(primary_values) > 1:
            variance = sum((v - avg_primary) ** 2 for v in primary_values) / len(primary_values)
            std = variance ** 0.5
        else:
            std = 0.0

        wall_values = metric_values.get("median_wall_clock_runtime_seconds", [])
        success_values = metric_values.get("completion_success_rate", [])

        results[cond] = {
            "metrics": {
                "primary_metric": avg_primary,
                "wall_time": (
                    float(sum(wall_values) / len(wall_values)) if wall_values else 0.0
                ),
                "success_rate": (
                    float(sum(success_values) / len(success_values)) if success_values else 0.0
                ),
                "mean": avg_primary,
                "std": std,
                "min": min(primary_values),
                "max": max(primary_values),
            },
            "rank": 0,  # 稍后设置
        }

    if not results:
        logger.warning(f"{source}: 未能从扁平 metrics 提取任何条件")
        return {}

    # 排名（按 primary_metric 升序，minimize）
    sorted_conds = sorted(results.items(), key=lambda kv: kv[1]["metrics"]["primary_metric"])
    for rank, (cond, _) in enumerate(sorted_conds, start=1):
        results[cond]["rank"] = rank

    logger.info(f"加载了 {len(results)} 个条件的结果（扁平 metrics）")
    return results


def _is_numeric(value: Any) -> bool:
    """判断值是否为有限数值"""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == value  # NaN 检查
    return False


# --- LLM 三路分类 -----------------------------------------------------------

_CLASSIFY_PROMPT = """You are an expert research assistant classifying optimization algorithms in a benchmark study.

Analyze the following optimization algorithm classes from a research project and classify each into one of three categories:

- **baseline**: classic / direct-search reference algorithms (e.g. Nelder-Mead, Powell, scipy wrappers, simple grid/random search). These serve as comparison baselines and should NOT be evolved.
- **ablation**: variant of a main proposed algorithm created for ablation studies (e.g. removing a component like diagonal covariance, fixed covariance). These are referenced by hypothesis-testing comparisons as the "condition_b" (compared variant). Should NOT be evolved.
- **proposed**: the research's novel contribution algorithms — these ARE the evolution targets.

Consider these signals:
1. Class hierarchy / base classes (e.g. inheriting from a scipy wrapper base → likely baseline)
2. Class name and method names
3. The main.py condition registry: which conditions are compared against which in hypothesis tests (comparisons where a variant appears as condition_b suggests ablation)
4. Algorithmic content of the class body

Main.py condition registry and hypothesis comparisons (if any):
```python
{condition_registry}
```

Class source code:
```python
{class_sources}
```

Respond with ONLY a JSON object (no markdown fences, no commentary):
{{
  "classifications": [
    {{"class_name": "<ClassA>", "category": "baseline"}},
    {{"class_name": "<ClassB>", "category": "ablation"}},
    {{"class_name": "<ClassC>", "category": "proposed"}}
  ]
}}
"""


def _classify_algorithms(
    optimizers_code: str,
    class_names: list[str],
    final_exp_dir: Path,
    config: Any = None,
) -> dict[str, str]:
    """LLM 三路分类（baseline/ablation/proposed），失败回退代码启发式

    Returns:
        {class_name: "baseline" | "ablation" | "proposed", ...}
    """
    global _last_classify_method

    llm_classification = _classify_with_llm(
        optimizers_code, class_names, final_exp_dir, config
    )
    if llm_classification:
        _last_classify_method = "llm"
        logger.info(f"LLM 分类成功: {len(llm_classification)} 个类")
        return llm_classification

    logger.warning("LLM 分类不可用/失败，回退代码启发式")
    _last_classify_method = "heuristic"
    return _classify_heuristic(optimizers_code, class_names, final_exp_dir)


def _classify_with_llm(
    optimizers_code: str,
    class_names: list[str],
    final_exp_dir: Path,
    config: Any = None,
) -> dict[str, str]:
    """用 LLM 分类（结构化 JSON 输出）"""
    try:
        # 延迟导入，避免算法提取在没有 LLM 配置时失败
        if config is None:
            from researchclaw.config import load_config

            for candidate in (Path("config.arc.yaml"), Path.cwd() / "config.arc.yaml"):
                if candidate.exists():
                    config = load_config(candidate)
                    break
            if config is None:
                logger.warning("未找到 config.arc.yaml，无法 LLM 分类")
                return {}

        from researchclaw.llm.client import LLMClient

        # 读取 main.py 条件注册表（comparison 证据）
        condition_registry = _load_condition_registry(final_exp_dir)

        # 提取类源码（截断到合理长度）
        class_sources = _class_sources_to_string(optimizers_code, class_names)

        prompt = _CLASSIFY_PROMPT.format(
            condition_registry=condition_registry,
            class_sources=class_sources,
        )

        # 从配置加载 LLM 客户端
        llm_client = LLMClient.from_rc_config(config)

        response = llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            json_mode=True,
            max_tokens=2000,
        )

        # 解析 JSON（可能带 markdown fences）
        content = response.content.strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)
        data = json.loads(content)

        classifications = {}
        for entry in data.get("classifications", []):
            cls_name = entry.get("class_name")
            category = entry.get("category")
            if cls_name and category in ("baseline", "ablation", "proposed"):
                classifications[cls_name] = category

        # 校验：所有类都有分类才算成功
        if len(classifications) != len(class_names):
            missing = set(class_names) - set(classifications)
            logger.warning(f"LLM 分类不完整，缺少: {missing}")
            return {}

        return classifications

    except Exception as e:
        logger.warning(f"LLM 分类失败: {e}")
        return {}


def _classify_heuristic(
    optimizers_code: str,
    class_names: list[str],
    final_exp_dir: Path,
) -> dict[str, str]:
    """代码启发式回退分类（类继承 + comparison 解析）

    判定顺序（优先级从高到低）：
    1. baseline：继承 `ScipyOptimizerBase` 且条件名 ∈ scipy 直接搜索方法
    2. ablation：被 H4+ 假设检验 comparison 引用为 condition_b，
       且该条件**不是** baseline（H1-H3 的 condition_b 是 nelder_mead/powell
       等 baseline，不算消融）
    3. 其余 → proposed
    """
    classifications: dict[str, str] = {}

    # 1. 解析类继承关系
    base_classes = _parse_base_classes(optimizers_code)

    # 先判定 baseline（scipy 直接搜索方法）
    for class_name in class_names:
        condition = _class_name_to_condition(class_name)
        bases = base_classes.get(class_name, [])
        if "ScipyOptimizerBase" in bases and condition in _SCIPY_DIRECT_SEARCH_CONDITIONS:
            classifications[class_name] = "baseline"

    # 2. 解析 main.py comparisons（H4+ label 的 condition_b → ablation，且非 baseline）
    ablation_conditions = set()
    try:
        comparisons = _parse_comparisons(final_exp_dir / "main.py")
        for comp in comparisons:
            label = str(comp.get("label", ""))
            if re.match(r"^H\d+", label):
                cb = comp.get("condition_b")
                # condition_b 且不是 scipy baseline 才算消融变体
                if cb and cb not in _SCIPY_DIRECT_SEARCH_CONDITIONS:
                    ablation_conditions.add(cb)
    except Exception as e:
        logger.warning(f"解析 main.py comparisons 失败: {e}")

    for class_name in class_names:
        if class_name in classifications:
            continue  # 已判定 baseline

        condition = _class_name_to_condition(class_name)
        if condition in ablation_conditions:
            classifications[class_name] = "ablation"

    # 3. 其余 → proposed
    for class_name in class_names:
        if class_name not in classifications:
            classifications[class_name] = "proposed"

    return classifications


def _load_condition_registry(final_exp_dir: Path) -> str:
    """从 main.py 提取条件注册表 + comparisons 摘要（供 LLM 分类）"""
    main_file = final_exp_dir / "main.py"
    if not main_file.exists():
        return "(main.py 不存在)"

    code = main_file.read_text(encoding="utf-8", errors="replace")
    # 截断，避免 prompt 过大
    if len(code) > 6000:
        code = code[:6000] + "\n...(truncated)"
    return code


def _class_sources_to_string(optimizers_code: str, class_names: list[str]) -> str:
    """将类源码拼接为字符串（截断到合理长度）"""
    lines = optimizers_code.split("\n")
    tree = ast.parse(optimizers_code)
    node_map = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}

    parts = []
    for name in class_names:
        node = node_map.get(name)
        if not node:
            continue
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, "end_lineno") else start + 1
        code = "\n".join(lines[start:end])
        if len(code) > 3000:
            code = code[:3000] + "\n...(truncated)"
        parts.append(code)

    return "\n\n".join(parts)


def _parse_comparisons(main_file: Path) -> list[dict[str, Any]]:
    """从 main.py 解析假设检验 comparisons 列表"""
    if not main_file.exists():
        return []

    code = main_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    comparisons = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                if isinstance(elt, ast.Dict):
                    comp = {}
                    for key_node, value_node in zip(elt.keys, elt.values):
                        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                            key = key_node.value
                            if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                                comp[key] = value_node.value
                            elif isinstance(value_node, ast.Constant):
                                comp[key] = value_node.value
                    if "label" in comp or "condition_b" in comp:
                        comparisons.append(comp)
    return comparisons


def _parse_base_classes(code: str) -> dict[str, list[str]]:
    """解析每个类的基类列表"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            result[node.name] = bases
    return result


def _write_classification(
    boost_dir: Path | None,
    classifications: dict[str, str],
    stage_dir: Path,
    known_conditions: list[str] | None = None,
) -> None:
    """将分类结果写入 boost_summary.json（可审计，合并而非覆盖）"""
    # 按类别分组（类名 → 条件名展示，尽量用已知条件归一化匹配）
    grouped: dict[str, list[str]] = {"baseline": [], "ablation": [], "proposed": []}
    for class_name, category in classifications.items():
        condition = _class_name_to_condition(class_name, known_conditions)
        grouped.setdefault(category, []).append(condition)

    # 目标文件：boost_dir/boost_summary.json（boost 流程）或 stage_dir（独立调用）
    target = boost_dir or stage_dir
    summary_file = target / "boost_summary.json"

    # 合并：保留已有字段（如 status/reason/timestamp），新增/覆盖 classified
    existing: dict[str, Any] = {}
    if summary_file.exists():
        try:
            existing = json.loads(summary_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing["classified"] = grouped
    existing["classification_source"] = _last_classify_method
    existing["by_class"] = classifications

    summary_file.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"分类结果已写入: {summary_file}")


def _parse_optimizer_classes(code: str) -> dict[str, str]:
    """解析优化器代码，提取所有优化器类

    Returns:
        {
            "ClassName": "class ClassName: ...",
            ...
        }
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"代码解析失败: {e}")
        return {}

    classes = {}
    lines = code.split('\n')

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name

            # 提取类的完整代码
            start_line = node.lineno - 1  # AST 行号从 1 开始
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1

            class_code = '\n'.join(lines[start_line:end_line])
            classes[class_name] = class_code

    logger.info(f"解析出 {len(classes)} 个优化器类")
    return classes
