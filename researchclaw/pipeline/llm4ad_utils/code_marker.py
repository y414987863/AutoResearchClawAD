"""代码标记器 - 为算法代码添加 EVOLVE_START/END 标记"""

from __future__ import annotations

import ast
import logging

logger = logging.getLogger(__name__)


def mark_evolution_boundaries(
    code: str,
    strategy: str = "auto",
) -> str:
    """为算法代码添加 EVOLVE_START/END 标记

    Args:
        code: 算法类的完整代码
        strategy: 标记策略
            - "auto": 自动识别 optimize 方法的核心逻辑
            - "optimize_method": 标记整个 optimize 方法
            - "full": 标记整个类（除了 __init__）

    Returns:
        标记后的代码
    """
    if strategy == "auto":
        return _mark_auto(code)
    elif strategy == "optimize_method":
        return _mark_optimize_method(code)
    elif strategy == "full":
        return _mark_full_class(code)
    else:
        raise ValueError(f"Unknown marking strategy: {strategy}")


def mark_class_in_module(
    module_code: str,
    class_name: str,
    strategy: str = "auto",
    out_method_name: list[str] | None = None,
) -> str:
    """在完整模块源码中，仅为指定类添加 EVOLVE_START/END 标记

    与 mark_evolution_boundaries（面向单类源码，`ast.walk` 取第一个 ClassDef/
    optimize）不同，本函数用于"继承式 proposed 算法"场景：solve.py 是**完整可
    运行的 optimizers.py**（保留基类与继承链，如 CMAESDefaultOptimizer(CMAOptimizer)
    需要基类 CMAOptimizer 的 optimize 才能运行），但只有目标 proposed 类
    `class_name` 的核心逻辑可演化，绝不能误标基类。

    策略：
        - "auto": 目标类若定义了自己的 optimize 方法 → 标记该方法体；否则
          （optimize 继承自基类，本类只重写了如 _update_covariance 等方法）→
          标记本类自身定义的方法块（首个非 __init__ 方法起至类结束）。
        - "optimize_method": 标记目标类自身的 optimize 方法体；本类无自身
          optimize 时回退到方法块。
        - "full": 直接标记目标类自身方法块（首个非 __init__ 方法至类结束）。

    Args:
        module_code: 完整模块源码（如 optimizers.py 全文）
        class_name: 目标 proposed 类名（只标记这一个类）
        strategy: 标记策略（见上）
        out_method_name: 可选 out-param（list[str]）：标记成功时回传被标记的方法名
            （有自身 optimize 时为 "optimize"，否则为首个非 __init__ 方法名，如
            "_update_covariance"）。用于让调用方把真实演化方法名注入 Coder prompt
            模板，避免模板硬编码 optimize 签名与继承式类不匹配。

    Returns:
        带 EVOLVE 标记的完整模块源码；类不存在或解析失败时返回原始源码（不标记）。
    """
    try:
        tree = ast.parse(module_code)
    except SyntaxError as e:
        logger.error(f"模块解析失败（标记类 {class_name}）: {e}")
        return module_code

    # 只定位目标类，绝不误标基类/其他类
    target: ast.ClassDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            target = node
            break
    if target is None:
        logger.warning(f"未在模块中找到类 {class_name}，返回未标记源码")
        return module_code

    # 目标类自身定义的方法（不含继承）
    own_optimize: ast.FunctionDef | None = None
    first_non_init: ast.FunctionDef | None = None
    for item in target.body:
        if isinstance(item, ast.FunctionDef):
            if item.name == "optimize" and own_optimize is None:
                own_optimize = item
            if item.name != "__init__" and first_non_init is None:
                first_non_init = item

    lines = module_code.split("\n")

    # 决定标记目标：方法体 or 类自身方法块
    if strategy == "full":
        mark_method = None
    elif strategy == "optimize_method":
        mark_method = own_optimize  # None 时回退方法块
    else:  # auto
        mark_method = own_optimize  # 有自身 optimize 则标之，否则回退方法块

    if mark_method is not None:
        # 标记方法体：def 行之后 → 方法结束之后。
        # 先插入 END（较大索引），再插入 START，避免行号偏移。
        indent = _get_indent(lines[mark_method.lineno - 1]) + "    "
        lines.insert(mark_method.end_lineno, f"{indent}# EVOLVE_END")
        lines.insert(mark_method.lineno, f"{indent}# EVOLVE_START")
        logger.info(
            f"标记 {class_name}.{mark_method.name} 方法体: "
            f"行 {mark_method.lineno}-{mark_method.end_lineno}"
        )
        if out_method_name is not None:
            out_method_name.clear()
            out_method_name.append(mark_method.name)
        return "\n".join(lines)

    # 继承式（无自身 optimize）或 full：标记本类自身方法块
    if first_non_init is None:
        logger.warning(f"类 {class_name} 无可标记方法（仅 __init__），返回未标记源码")
        return module_code

    indent = _get_indent(lines[first_non_init.lineno - 1])
    # 先插入 END（类结束之后），再插入 START（首个非 __init__ 方法之前）
    lines.insert(target.end_lineno, f"{indent}# EVOLVE_END")
    lines.insert(first_non_init.lineno - 1, f"{indent}# EVOLVE_START")
    logger.info(
        f"标记 {class_name} 自身方法块: 行 {first_non_init.lineno}-{target.end_lineno}"
    )
    if out_method_name is not None:
        out_method_name.clear()
        out_method_name.append(first_non_init.name)
    return "\n".join(lines)


def _mark_auto(code: str) -> str:
    """自动识别并标记 optimize 方法的核心逻辑

    策略：
    1. 找到 optimize() 方法
    2. 跳过方法开头的参数验证和初始化
    3. 标记主循环和核心计算逻辑
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"代码解析失败: {e}")
        return _mark_optimize_method(code)  # 回退到标记整个方法

    lines = code.split('\n')

    # 找到 optimize 方法
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "optimize":
            # 找到方法的主循环起点
            start_line = _find_main_loop_start(node, lines)
            if start_line is None:
                # 如果找不到主循环，标记整个方法体
                start_line = node.lineno  # 方法定义的下一行

            end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)

            # 插入标记
            indent = _get_indent(lines[start_line - 1])
            lines.insert(start_line, f"{indent}# EVOLVE_START")
            lines.insert(end_line + 1, f"{indent}# EVOLVE_END")

            logger.info(f"标记 optimize 方法核心逻辑: 行 {start_line} - {end_line}")
            return '\n'.join(lines)

    # 如果找不到 optimize 方法，标记整个类
    logger.warning("未找到 optimize 方法，标记整个类")
    return _mark_full_class(code)


def _mark_optimize_method(code: str) -> str:
    """标记整个 optimize 方法"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"代码解析失败: {e}")
        return code

    lines = code.split('\n')

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "optimize":
            start_line = node.lineno
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)

            # 获取方法体的缩进
            indent = _get_indent(lines[start_line])

            # 在方法定义后插入 EVOLVE_START
            lines.insert(start_line, f"{indent}    # EVOLVE_START")

            # 在方法结束前插入 EVOLVE_END
            lines.insert(end_line + 1, f"{indent}    # EVOLVE_END")

            logger.info(f"标记整个 optimize 方法: 行 {start_line} - {end_line}")
            return '\n'.join(lines)

    logger.warning("未找到 optimize 方法")
    return code


def _mark_full_class(code: str) -> str:
    """标记整个类（除了 __init__）"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"代码解析失败: {e}")
        return code

    lines = code.split('\n')

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # 找到第一个非 __init__ 方法
            first_method_line = None
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name != "__init__":
                    first_method_line = item.lineno - 1
                    break

            if first_method_line is None:
                logger.warning("类中没有可标记的方法")
                return code

            # 找到类的结束位置
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)

            indent = _get_indent(lines[first_method_line])
            lines.insert(first_method_line, f"{indent}# EVOLVE_START")
            lines.insert(end_line, f"{indent}# EVOLVE_END")

            logger.info(f"标记整个类: 行 {first_method_line} - {end_line}")
            return '\n'.join(lines)

    logger.warning("未找到类定义")
    return code


def _find_main_loop_start(method_node: ast.FunctionDef, lines: list[str]) -> int | None:
    """在 optimize 方法中找到主循环的起始行

    启发式规则:
    1. 查找 while 或 for 循环
    2. 跳过前面的参数验证和初始化代码
    """
    for stmt in method_node.body:
        if isinstance(stmt, (ast.While, ast.For)):
            # 找到主循环
            return stmt.lineno - 1  # AST 行号从 1 开始，列表索引从 0 开始

    # 如果没有循环，返回方法体的第一行
    if method_node.body:
        return method_node.body[0].lineno - 1

    return None


def _get_indent(line: str) -> str:
    """获取行的缩进"""
    return line[:len(line) - len(line.lstrip())]


def has_evolution_markers(code: str) -> bool:
    """检查代码是否已经有 EVOLVE 标记"""
    return "# EVOLVE_START" in code and "# EVOLVE_END" in code
