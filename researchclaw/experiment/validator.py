"""Experiment code validation: syntax, security, and import checks.

This module provides pre-execution validation for LLM-generated experiment
code.  It catches common issues *before* running code in the sandbox,
enabling automated repair via LLM re-generation.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import sys
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: str  # "error" | "warning"
    category: str  # "syntax" | "security" | "import" | "style"
    message: str
    line: int | None = None
    col: int | None = None


@dataclass
class CodeValidation:
    """Aggregated validation result for a code snippet."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        errs = len(self.errors)
        warns = len(self.warnings)
        if errs == 0 and warns == 0:
            return "Code validation passed."
        parts: list[str] = []
        if errs:
            parts.append(f"{errs} error(s)")
        if warns:
            parts.append(f"{warns} warning(s)")
        return "Code validation: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Dangerous call patterns (security scan)
# ---------------------------------------------------------------------------

# Fully-qualified call names that are forbidden in experiment code.
DANGEROUS_CALLS: frozenset[str] = frozenset(
    {
        "os.system",
        "os.popen",
        "os.exec",
        "os.execl",
        "os.execle",
        "os.execlp",
        "os.execlpe",
        "os.execv",
        "os.execve",
        "os.execvp",
        "os.execvpe",
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.removedirs",
        "subprocess.call",
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
        "shutil.rmtree",
    }
)

# Bare built-in names that should never appear in experiment code.
DANGEROUS_BUILTINS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
    }
)

# Modules that should not be imported at all.
BANNED_MODULES: frozenset[str] = frozenset(
    {
        "subprocess",
        "shutil",
        "socket",
        "http",
        "urllib",
        "requests",
        "ftplib",
        "smtplib",
        "ctypes",
        "signal",
    }
)

# Packages considered safe / always available in experiment sandbox.
SAFE_STDLIB: frozenset[str] = frozenset(
    {
        "abc",
        "ast",
        "bisect",
        "builtins",
        "collections",
        "contextlib",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "glob",
        "gzip",
        "hashlib",
        "heapq",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "operator",
        "os",  # os itself is ok, certain calls aren't
        "pathlib",
        "pickle",
        "pprint",
        "random",
        "re",
        "statistics",
        "string",
        "struct",
        "sys",
        "tempfile",
        "textwrap",
        "time",
        "traceback",
        "typing",
        "unittest",
        "uuid",
        "warnings",
        "zipfile",
    }
)

COMMON_SCIENCE: frozenset[str] = frozenset(
    {
        "numpy",
        "np",
        "pandas",
        "pd",
        "scipy",
        "sklearn",
        "matplotlib",
        "plt",
        "seaborn",
        "torch",
        "tensorflow",
        "tf",
        "jax",
        "transformers",
        "datasets",
        "tqdm",
        "yaml",
        "pyyaml",
        "rich",
        # LLM training stack
        "peft",
        "trl",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "tokenizers",
        "safetensors",
        "evaluate",
        "rouge_score",
        # Runtime-injected by the experiment harness
        "experiment_harness",
    }
)


# ---------------------------------------------------------------------------
# AST visitor for security checks
# ---------------------------------------------------------------------------


class _SecurityVisitor(ast.NodeVisitor):
    """Walk AST to detect dangerous calls and imports."""

    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    # -- function calls --

    def visit_Call(self, node: ast.Call) -> None:
        name = _resolve_call_name(node.func)
        if name in DANGEROUS_BUILTINS:
            self.issues.append(
                ValidationIssue(
                    severity="error",
                    category="security",
                    message=f"Dangerous built-in call: {name}()",
                    line=node.lineno,
                    col=node.col_offset,
                )
            )
        elif name in DANGEROUS_CALLS:
            self.issues.append(
                ValidationIssue(
                    severity="error",
                    category="security",
                    message=f"Dangerous call: {name}()",
                    line=node.lineno,
                    col=node.col_offset,
                )
            )
        self.generic_visit(node)

    # -- import statements --

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in BANNED_MODULES:
                self.issues.append(
                    ValidationIssue(
                        severity="error",
                        category="security",
                        message=f"Banned module import: {alias.name}",
                        line=node.lineno,
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            top = node.module.split(".")[0]
            if top in BANNED_MODULES:
                self.issues.append(
                    ValidationIssue(
                        severity="error",
                        category="security",
                        message=f"Banned module import: from {node.module}",
                        line=node.lineno,
                    )
                )
        self.generic_visit(node)


def _resolve_call_name(node: ast.expr) -> str:
    """Best-effort name resolution for a Call node's func."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _resolve_call_name(node.value)
        if prefix:
            return f"{prefix}.{node.attr}"
        return node.attr
    return ""


# ---------------------------------------------------------------------------
# Import extractor
# ---------------------------------------------------------------------------


def extract_imports(code: str) -> set[str]:
    """Return top-level module names imported by *code*.

    Returns an empty set if the code can't be parsed.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


# ---------------------------------------------------------------------------
# Public validation functions
# ---------------------------------------------------------------------------


def validate_syntax(code: str) -> CodeValidation:
    """Check *code* parses as valid Python."""
    result = CodeValidation()
    try:
        ast.parse(code)
    except SyntaxError as exc:
        result.issues.append(
            ValidationIssue(
                severity="error",
                category="syntax",
                message=str(exc.msg) if exc.msg else str(exc),
                line=exc.lineno,
                col=exc.offset,
            )
        )
    return result


def validate_security(code: str) -> CodeValidation:
    """Scan *code* AST for dangerous calls and imports."""
    result = CodeValidation()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # If can't parse, skip security — syntax check will catch it.
        return result
    visitor = _SecurityVisitor()
    visitor.visit(tree)
    result.issues.extend(visitor.issues)
    return result


def validate_imports(
    code: str,
    available: set[str] | None = None,
) -> CodeValidation:
    """Check that all imported modules are available.

    *available* defaults to ``SAFE_STDLIB | COMMON_SCIENCE`` plus any
    modules already in ``sys.modules``.
    """
    result = CodeValidation()
    if available is None:
        available = set(SAFE_STDLIB) | set(COMMON_SCIENCE) | set(sys.modules.keys())

    imports = extract_imports(code)
    for mod in sorted(imports):
        if mod not in available:
            result.issues.append(
                ValidationIssue(
                    severity="warning",
                    category="import",
                    message=f"Module '{mod}' may not be available in sandbox",
                )
            )
    return result


def validate_code(
    code: str,
    *,
    available_packages: set[str] | None = None,
    skip_security: bool = False,
    skip_imports: bool = False,
) -> CodeValidation:
    """Run all validations and return a combined :class:`CodeValidation`.

    1. Syntax check (always)
    2. Security scan (unless *skip_security*)
    3. Import availability (unless *skip_imports*)
    """
    combined = CodeValidation()

    # 1. Syntax
    syntax = validate_syntax(code)
    combined.issues.extend(syntax.issues)
    if not syntax.ok:
        # No point running further checks if code doesn't parse
        return combined

    # 2. Security
    if not skip_security:
        security = validate_security(code)
        combined.issues.extend(security.issues)

    # 3. Import availability
    if not skip_imports:
        imp = validate_imports(code, available=available_packages)
        combined.issues.extend(imp.issues)

    return combined


# ---------------------------------------------------------------------------
# Error description helper (for LLM repair prompt)
# ---------------------------------------------------------------------------


def format_issues_for_llm(validation: CodeValidation) -> str:
    """Format validation issues as a concise error report for LLM repair."""
    if validation.ok and not validation.warnings:
        return "No issues found."
    lines: list[str] = []
    for issue in validation.issues:
        loc = f"line {issue.line}" if issue.line else "unknown location"
        lines.append(
            f"- [{issue.severity.upper()}] ({issue.category}) {issue.message} @ {loc}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Code complexity and quality checks (R10-Fix6)
# ---------------------------------------------------------------------------


def check_code_complexity(code: str) -> list[str]:
    """Check whether generated experiment code is too simplistic.

    Returns a list of warning strings.  Empty list means no quality concerns.
    """
    warnings: list[str] = []

    # Count non-blank, non-comment, non-import lines
    effective_lines = [
        l
        for l in code.splitlines()
        if l.strip()
        and not l.strip().startswith("#")
        and not l.strip().startswith(("import ", "from "))
    ]

    if len(effective_lines) < 10:
        warnings.append(
            f"Code has only {len(effective_lines)} effective lines "
            f"(excluding blanks/comments/imports) — likely too simple for "
            f"a research experiment"
        )

    # Check for trivially short functions/methods
    try:
        tree = ast.parse(code)
        func_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_count += 1
        if func_count == 0 and len(effective_lines) > 5:
            warnings.append(
                "Code has no function definitions — research experiments "
                "should be structured with reusable functions"
            )
    except SyntaxError:
        pass

    # Check for hardcoded metrics (a common LLM failure mode)
    import re

    hardcoded_patterns = [
        (r"print\(['\"].*:\s*0\.\d+['\"]\)", "print statement with hardcoded metric value"),
        (r"metric.*=\s*0\.\d{2,}", "hardcoded metric assignment"),
    ]
    for pattern, desc in hardcoded_patterns:
        if re.search(pattern, code):
            warnings.append(f"Possible hardcoded metric: {desc}")

    # Check for trivial computation patterns
    trivial_patterns = [
        ("sum(x**2)", "trivial sum-of-squares computation"),
        ("np.sum(x**2)", "trivial sum-of-squares computation"),
        ("0.3 + idx * 0.03", "formulaic/simulated metric generation"),
    ]
    for pattern, desc in trivial_patterns:
        if pattern in code:
            warnings.append(f"Trivial computation detected: {desc}")

    return warnings


# ---------------------------------------------------------------------------
# Deep code quality analysis (Phase 1 / P1.1 + P1.2)
# ---------------------------------------------------------------------------


def check_class_quality(all_files: dict[str, str]) -> list[str]:
    """Analyze class implementations across all experiment files.

    Detects:
    - Empty or trivial class inheritance (class B(A): pass)
    - Classes with too few methods (< 2 non-dunder)
    - Duplicate class bodies (identical forward/train logic across variants)
    - nn.Module created inside forward() instead of __init__()
    """
    warnings: list[str] = []

    class_info: dict[str, dict[str, Any]] = {}

    for fname, code in all_files.items():
        if not fname.endswith(".py"):
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            cls_name = node.name
            methods: list[str] = []
            method_sources: dict[str, str] = {}
            has_forward_new_module = False
            body_lines = 0

            for item in ast.walk(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)
                    # Approximate method body size
                    m_start = item.lineno
                    m_end = item.end_lineno or item.lineno
                    body_len = m_end - m_start
                    method_sources[item.name] = f"{fname}:{m_start}-{m_end}"

                    # Check for nn.Module creation inside forward()
                    if item.name in ("forward", "__call__"):
                        for sub in ast.walk(item):
                            if isinstance(sub, ast.Call):
                                call_name = _resolve_call_name(sub.func)
                                if call_name.startswith("nn.") and call_name != "nn.Module":
                                    has_forward_new_module = True

            # Count effective body lines
            code_lines = code.splitlines()
            if node.end_lineno and node.lineno:
                cls_body = code_lines[node.lineno - 1 : node.end_lineno]
                body_lines = sum(
                    1 for l in cls_body
                    if l.strip() and not l.strip().startswith("#")
                    and not l.strip().startswith(("import ", "from "))
                )

            non_dunder = [m for m in methods if not m.startswith("__")]

            has_explicit_bases = bool(node.bases)

            class_info[f"{fname}:{cls_name}"] = {
                "methods": methods,
                "non_dunder": non_dunder,
                "body_lines": body_lines,
                "file": fname,
                "has_forward_new_module": has_forward_new_module,
                "class_name": cls_name,
                "has_explicit_bases": has_explicit_bases,
            }

            # --- Check 1: Empty or trivial class ---
            if body_lines <= 2:
                warnings.append(
                    f"[{fname}] Class '{cls_name}' has only {body_lines} body lines "
                    f"— likely an empty or trivial subclass (class B(A): pass)"
                )

            # --- Check 2: Too few methods for an algorithm class ---
            if (
                body_lines > 5
                and len(non_dunder) < 2
                and not has_explicit_bases
            ):
                warnings.append(
                    f"[{fname}] Class '{cls_name}' has only {len(non_dunder)} "
                    f"non-dunder method(s) — algorithm classes should have at "
                    f"least __init__ + one core method (forward/train_step/predict)"
                )

            # --- Check 3: nn.Module created in forward() ---
            if has_forward_new_module:
                warnings.append(
                    f"[{fname}] Class '{cls_name}' creates nn.Module (nn.Linear etc.) "
                    f"inside forward() — these modules are unregistered and untrained. "
                    f"Move to __init__() and register as submodules."
                )

    # --- Check 4: Duplicate class names across files ---
    duplicated_class_names: set[str] = set()
    classes_by_name: dict[str, list[dict[str, Any]]] = {}
    for info in class_info.values():
        classes_by_name.setdefault(str(info["class_name"]), []).append(info)

    for cls_name, entries in classes_by_name.items():
        non_trivial = [entry for entry in entries if int(entry["body_lines"]) > 5]
        files = sorted({str(entry["file"]) for entry in non_trivial})
        if len(files) >= 2:
            duplicated_class_names.add(cls_name)
            warnings.append(
                f"Class '{cls_name}' is defined in multiple files "
                f"({', '.join(files)}). Keep each algorithm/helper class in one "
                f"canonical module and import it elsewhere instead of duplicating "
                f"the definition."
            )

    # --- Check 5: Duplicate class implementations ---
    # Compare class body hashes to find copy-paste variants
    class_names = list(class_info.keys())
    for i, name_a in enumerate(class_names):
        info_a = class_info[name_a]
        for name_b in class_names[i + 1:]:
            info_b = class_info[name_b]
            if (
                str(info_a["class_name"]) == str(info_b["class_name"])
                and str(info_a["class_name"]) in duplicated_class_names
            ):
                continue
            if (
                info_a["body_lines"] > 5
                and info_b["body_lines"] > 5
                and info_a["non_dunder"] == info_b["non_dunder"]
                and abs(info_a["body_lines"] - info_b["body_lines"]) <= 2
            ):
                # Same methods, same body size — likely duplicates
                warnings.append(
                    f"Classes '{name_a.split(':')[1]}' and '{name_b.split(':')[1]}' "
                    f"have identical method signatures and similar body sizes "
                    f"({info_a['body_lines']} vs {info_b['body_lines']} lines) — "
                    f"may be copy-paste variants with no real algorithmic difference"
                )

    # --- Check 6: Ablation subclasses must override with different logic ---
    # Parse inheritance relationships and compare method ASTs
    for fname_code, code in all_files.items():
        if not fname_code.endswith(".py"):
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue

        # Build {class_name: ClassDef} map for this file
        file_classes: dict[str, ast.ClassDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                file_classes[node.name] = node

        for cls_name, cls_node in file_classes.items():
            # Check if this class inherits from another class in the same file
            for base in cls_node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if not base_name or base_name not in file_classes:
                    continue

                parent_node = file_classes[base_name]
                # Get method bodies as AST dumps for comparison
                child_methods = {
                    m.name: ast.dump(m)
                    for m in cls_node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not m.name.startswith("__")
                }
                parent_methods = {
                    m.name: ast.dump(m)
                    for m in parent_node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not m.name.startswith("__")
                }

                if not child_methods:
                    # Already caught by Check 1 (empty class)
                    continue

                # Check if all overridden methods have identical AST to parent
                identical_count = 0
                override_count = 0
                for method_name, method_dump in child_methods.items():
                    if method_name in parent_methods:
                        override_count += 1
                        if method_dump == parent_methods[method_name]:
                            identical_count += 1

                if override_count > 0 and identical_count == override_count:
                    warnings.append(
                        f"[{fname_code}] Class '{cls_name}' inherits from "
                        f"'{base_name}' and overrides {override_count} method(s), "
                        f"but ALL overridden methods have identical AST to parent "
                        f"— this is NOT a real ablation. Methods must differ."
                    )
                elif override_count == 0 and len(child_methods) > 0:
                    # Has methods but none override parent — might be fine
                    # (new methods that parent doesn't have)
                    pass

                # --- Check 7: Ablation subclass must override >=1 parent method ---
                _lname = cls_name.lower()
                if ("ablation" in _lname or "no_" in _lname or "without" in _lname):
                    parent_non_dunder = {
                        m.name
                        for m in parent_node.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and not m.name.startswith("__")
                    }
                    child_overrides = set(child_methods.keys()) & parent_non_dunder
                    if not child_overrides and parent_non_dunder:
                        warnings.append(
                            f"[{fname_code}] Ablation class '{cls_name}' inherits "
                            f"from '{base_name}' but does NOT override any of its "
                            f"methods ({', '.join(sorted(parent_non_dunder))}). "
                            f"An ablation MUST override the method that removes "
                            f"the ablated component."
                        )

    return warnings


def check_variable_scoping(code: str, fname: str = "main.py") -> list[str]:
    """Detect common variable scoping bugs in experiment code.

    Catches the pattern where a variable is defined inside an if-branch
    but used outside that branch (UnboundLocalError at runtime).
    """
    warnings: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return warnings

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Collect variables assigned only inside if/elif/else branches
        if_only_vars: dict[str, int] = {}
        top_level_vars: set[str] = set()

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                _collect_if_only_assignments(child, if_only_vars)
            elif isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                for target in _extract_assign_targets(child):
                    top_level_vars.add(target)

        # Check for variables used after the if block but only defined inside it
        for var_name, var_line in if_only_vars.items():
            if var_name not in top_level_vars:
                # Check if this variable is used later in the function
                for later_node in ast.walk(node):
                    if (
                        isinstance(later_node, ast.Name)
                        and later_node.id == var_name
                        and isinstance(later_node.ctx, ast.Load)
                        and later_node.lineno > var_line
                    ):
                        warnings.append(
                            f"[{fname}:{var_line}] Variable '{var_name}' is assigned "
                            f"only inside an if-branch but used at line "
                            f"{later_node.lineno} — will cause UnboundLocalError "
                            f"if the branch is not taken"
                        )
                        break

    return warnings


def _collect_if_only_assignments(
    if_node: ast.If, result: dict[str, int]
) -> None:
    """Collect variables assigned only inside if/elif branches."""
    for child in ast.iter_child_nodes(if_node):
        if isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for target in _extract_assign_targets(child):
                result[target] = child.lineno
        elif isinstance(child, ast.If):
            _collect_if_only_assignments(child, result)


def _extract_assign_targets(node: ast.AST) -> list[str]:
    """Extract variable names from assignment targets."""
    names: list[str] = []
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
    elif isinstance(node, ast.AugAssign):
        if isinstance(node.target, ast.Name):
            names.append(node.target.id)
    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def auto_fix_unbound_locals(code: str) -> tuple[str, int]:
    """Programmatically fix UnboundLocalError patterns.

    For each variable assigned only inside an if-branch but used later,
    insert ``var = None`` before the if-statement.

    Returns (fixed_code, num_fixes).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0

    lines = code.splitlines(keepends=True)
    insertions: dict[int, list[str]] = {}  # lineno -> lines to insert before

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        if_only_vars: dict[str, int] = {}
        top_level_vars: set[str] = set()
        if_line_map: dict[str, int] = {}  # var -> if-statement lineno

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                before: dict[str, int] = {}
                _collect_if_only_assignments(child, before)
                for var_name, var_line in before.items():
                    if_only_vars[var_name] = var_line
                    if_line_map[var_name] = child.lineno
            else:
                # Any binding that is NOT exclusively inside a top-level
                # if-statement means the variable is not "if-only" and must
                # not be pre-seeded with ``= None`` (doing so would clobber a
                # value assigned in a for/while/with/try body). Collect every
                # Store-context name and except-handler binding from these
                # non-if children so such variables are excluded below.
                for sub in ast.walk(child):
                    if isinstance(sub, ast.Name) and isinstance(
                        sub.ctx, ast.Store
                    ):
                        top_level_vars.add(sub.id)
                    elif isinstance(sub, ast.ExceptHandler) and sub.name:
                        top_level_vars.add(sub.name)

        for var_name, var_line in if_only_vars.items():
            if var_name in top_level_vars:
                continue
            # Confirm it's actually used later
            used_later = False
            for later_node in ast.walk(node):
                if (
                    isinstance(later_node, ast.Name)
                    and later_node.id == var_name
                    and isinstance(later_node.ctx, ast.Load)
                    and later_node.lineno > var_line
                ):
                    used_later = True
                    break
            if not used_later:
                continue

            if_lineno = if_line_map.get(var_name)
            if if_lineno is None:
                continue
            # Determine indentation of the if-statement
            if if_lineno <= len(lines):
                if_line = lines[if_lineno - 1]
                indent = if_line[: len(if_line) - len(if_line.lstrip())]
            else:
                indent = "    "
            insertions.setdefault(if_lineno, [])
            fix_line = f"{indent}{var_name} = None\n"
            if fix_line not in insertions[if_lineno]:
                insertions[if_lineno].append(fix_line)

    if not insertions:
        return code, 0

    # Apply insertions in reverse line order to keep line numbers stable
    num_fixes = sum(len(v) for v in insertions.values())
    for lineno in sorted(insertions, reverse=True):
        idx = lineno - 1
        for fix_line in reversed(insertions[lineno]):
            lines.insert(idx, fix_line)

    return "".join(lines), num_fixes


class _Masked(str):
    """A copy of the source with prose blanked out, carrying line numbers.

    Subclasses ``str`` so every regex in the checks below runs against it
    unchanged, while ``_keep`` remembers the original lines the mask came from.
    Blanking (rather than deleting) preserves both offsets and the line count,
    so a reported line number still points at the real line.
    """

    _keep: list[str]

    def line(self, lineno: int) -> str:
        """The original text of 1-based *lineno*, for diagnostics."""
        try:
            return self._keep[lineno - 1]
        except IndexError:
            return ""


def _blank_prose(code: str) -> _Masked:
    """Return *code* with every comment and string literal blanked to spaces.

    The API checks are text patterns, so they match a name *mentioned* in prose
    exactly as if it were called. Every false positive found in this area came
    from that, and each one cost something:

    * a docstring reading ``NumPy 2.x: uses scipy.special.erfinv (np.erfinv
      removed)`` — written by a model explaining its own correct fix, then
      reported as the very defect it had repaired;
    * ``x = erfinv(u)  # legacy np.erfinv(2*u-1)`` — a trailing comment;
    * ``MSG = "call np.erfinv to invert"`` and a triple-quoted SQL snippet —
      strings are data, and no name inside one is ever looked up.

    Masking spans rather than whole lines is what keeps real code reportable:
    ``z = np.erfinv(u)  # compute it`` still reports (only the comment is
    blanked), and so does a call that follows a multi-line string's closing
    quotes on the same line. A per-line skip gets both of those wrong.

    f-strings are deliberately NOT blanked: their interpolations are
    executable, so ``f"{np.erfinv(x)}"`` holds a real reference that must stay
    visible. Masking the whole literal to hide its text would also hide that
    call, and a missed crash is worse than a rare false positive.

    Best-effort by construction: if tokenizing fails the original text is
    returned and every check behaves as it did before.
    """
    import io
    import tokenize

    # FSTRING_MIDDLE exists from 3.12's PEP 701 tokenizer; on older
    # interpreters an f-string arrives as a single STRING token (blanked in
    # full), and this sentinel simply never matches.
    _FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", -1)

    try:
        source_lines = code.splitlines(keepends=True)
    except (ValueError, MemoryError):  # pragma: no cover - defensive
        return _Masked(code)

    chars = list(code)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        tokens = []

    # Precomputed once: the character offset each line starts at. Recomputing
    # it inside the blanking loop made masking quadratic in file size.
    line_offsets: list[int] = []
    _running = 0
    for _line in source_lines:
        line_offsets.append(_running)
        _running += len(_line)

    def _blank(start: tuple[int, int], end: tuple[int, int]) -> None:
        """Space out the (row, col) half-open span, keeping newlines."""
        s_row, s_col = start
        e_row, e_col = end
        for row in range(s_row, e_row + 1):
            if row - 1 >= len(source_lines):
                break
            line = source_lines[row - 1]
            lo = s_col if row == s_row else 0
            hi = e_col if row == e_row else len(line)
            base = line_offsets[row - 1]
            for i in range(lo, min(hi, len(line))):
                off = base + i
                if off < len(chars) and chars[off] not in "\r\n":
                    chars[off] = " "

    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            _blank(tok.start, tok.end)
        elif tok.type == tokenize.STRING:
            _blank(tok.start, tok.end)
        elif tok.type == _FSTRING_MIDDLE:
            # The literal run inside an f-string. Its interpolations are
            # separate tokens and stay visible, so `f"{np.erfinv(x)}"` still
            # reports while `f"np.erfinv is gone"` does not.
            _blank(tok.start, tok.end)

    masked = _Masked("".join(chars))
    masked._keep = code.splitlines()
    return masked


def check_api_correctness(code: str, fname: str = "main.py") -> list[str]:
    """Detect common API misuse patterns.

    Catches:
    - np.erf()/np.erfc()/np.erfinv()/np.erfcinv() (should be scipy.special.*)
    - nn.Linear/nn.Conv2d inside forward() (unregistered module)
    - random.seed() without numpy.random.seed() (incomplete seeding)
    - NumPy 2.0 removed APIs (.ptp(), np.bool, etc.)
    """
    import re as _re

    warnings: list[str] = []

    # NumPy 2.0 removed names -> replacement. Matched as a whole attribute so
    # the live aliases (np.float_, np.int_, np.complex_, np.float64/32) are NOT
    # caught: the trailing-underscore names fail the "(?![_\w\d])" bound.
    _NP_REMOVED = {
        "np.bool": "bool",
        "np.int": "int",
        "np.float": "float",
        "np.complex": "complex",
        "np.object": "object",
        "np.str": "str",
        "np.trapz": "np.trapezoid",
        "np.product": "np.prod",
        "np.in1d": "np.isin",
        "np.row_stack": "np.vstack",
        "np.cumproduct": "np.cumprod",
        "np.round_": "np.round",
        "np.alltrue": "np.all",
        "np.sometrue": "np.any",
        "np.NaN": "np.nan",
        "np.Inf": "np.inf",
        "np.PINF": "np.inf",
        "np.NINF": "-np.inf",
        "np.PZERO": "0.0",
        "np.NZERO": "-0.0",
        # NOTE: these two were never in NumPy at any version (unlike the entries
        # above, which NumPy 2.0 removed). They are confabulated names — the
        # model knows the erf family and NumPy's np.* namespace and fuses them —
        # so no amount of "please target NumPy 2.x" in the prompt prevents them,
        # and only a static catch does. erfinv reached a real run inside
        # algorithms/cma_es/cma_es.py and the evaluator's bare `except
        # Exception: v = inf` swallowed the AttributeError, so it scored inf and
        # stayed invisible until Stage 13's fitness gate rejected the whole
        # evolution for "no finite primary metric".
        "np.erfinv": "scipy.special.erfinv",
        "np.erfcinv": "scipy.special.erfcinv",
    }
    # Subset of _NP_REMOVED that no NumPy release ever exposed, so the message
    # does not point at a 2.0 migration note for a name that never had one.
    _NP_NEVER_EXISTED = frozenset({"np.erfinv", "np.erfcinv"})

    # Prose (comments, string literals) is blanked to spaces in this copy, so
    # the patterns below see code only. Line numbers still line up with the
    # original, and `_masked._keep` holds the original text for messages.
    _masked = _blank_prose(code)
    lines = _masked.splitlines()
    _keep = _masked._keep
    _has_pandas = bool(
        _re.search(r"^\s*(?:import\s+pandas|from\s+pandas)\b", code, _re.MULTILINE)
    )
    for i, stripped in enumerate(lines, 1):
        stripped = stripped.strip()

        # np.erf()/np.erfc() don't exist; nor do np.erfinv()/np.erfcinv(), which
        # _NP_REMOVED reports below with a scipy.special replacement. The
        # negative lookahead is what keeps the two apart: `\bnp\.erf\b` looks
        # like it covers the whole family but a word boundary does not exist
        # between "erf" and "inv" — both are word characters — so np.erfinv
        # passed this check untouched. `(?![A-Za-z0-9_])` matches erfc (no
        # boundary inside) while leaving erfinv/erfcinv to the dict, so a line
        # never gets both messages.
        if _re.search(r"np\.erf(?:c)?(?![A-Za-z0-9_])", stripped):
            warnings.append(
                f"[{fname}:{i}] np.erf()/np.erfc() do not exist — use "
                f"scipy.special.erf()/erfc() or math.erf() instead"
            )

        # NumPy 2.0 removed ndarray methods
        if _re.search(r"\.ptp\s*\(", stripped):
            warnings.append(
                f"[{fname}:{i}] ndarray.ptp() was removed in NumPy 2.0 — "
                f"use np.ptp(arr) or arr.max() - arr.min() instead"
            )

        # NumPy 2.0 removed names (types, functions, constants). The regex
        # requires a non-identifier boundary immediately after the name, so
        # live aliases like np.float_ / np.int_ / np.complex_ (which end in an
        # underscore, still valid) and current np.float64 / np.float32 are NOT
        # caught, and the literal dotted name matches only the removed spelling.
        for old_name, replacement in _NP_REMOVED.items():
            pattern = _re.escape(old_name) + r"(?![_\w\d])"
            if _re.search(pattern, stripped):
                # Two different defects share this table, and telling them
                # apart matters to whoever reads the message: `np.trapz` moved
                # (the code was correct once, under an older NumPy), whereas
                # `np.erfinv` never existed under any version. Calling the
                # latter "removed" sends the reader looking for a migration
                # note that does not exist.
                if old_name in _NP_NEVER_EXISTED:
                    _why = "does not exist in NumPy (any version) — it is a SciPy name"
                else:
                    _why = "was removed in NumPy 2.0"
                warnings.append(
                    f"[{fname}:{i}] {old_name} {_why} — "
                    f"use {replacement} instead"
                )

        # pandas 2.0 removals. `.ix[]` and `.iteritems()` are unambiguous, so
        # they are flagged regardless of context. `.append()` is NOT — a plain
        # Python list.append is legal, so only flag it in a file that imports
        # pandas, and even then only on a bare `.append(` (a `list.append(...)`
        # qualified call is explicitly excluded).
        # `.ix[` is a pd.DataFrame/Series selector removed in pandas 2.0.
        if _re.search(r"\.ix\s*\[", stripped):
            warnings.append(
                f"[{fname}:{i}] `.ix[]` was removed in pandas 2.0 — "
                f"use `.loc[]` or `.iloc[]` instead"
            )
        if _re.search(r"\.iteritems\s*\(", stripped):
            warnings.append(
                f"[{fname}:{i}] `.iteritems()` was removed in pandas 2.0 — "
                f"use `.items()` instead"
            )
        if _has_pandas and "list.append" not in stripped and _re.search(r"\.append\s*\(", stripped):
            warnings.append(
                f"[{fname}:{i}] DataFrame/Series.append() was removed in pandas 2.0 — "
                f"use pd.concat() instead"
            )

        # np.random.RandomState with hardcoded seed in a function called multiple times
        if _re.search(r"RandomState\(\s*\d+\s*\)", stripped) and "def " not in stripped:
            warnings.append(
                f"[{fname}:{i}] Hardcoded RandomState seed inside a loop/function "
                f"may produce identical results across calls — pass seed as parameter"
            )

    # --- Import-usage mismatch detection ---
    # Detect `from X import Y` followed by `X.Y(...)` — guaranteed NameError
    import_from_map: dict[str, set[str]] = {}  # module -> {names}
    import_module_set: set[str] = set()  # modules imported with `import X`
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        m = _re.match(r"from\s+([\w.]+)\s+import\s+(.+)", stripped)
        if m:
            mod = m.group(1)
            names = {n.strip().split(" as ")[-1].strip()
                     for n in m.group(2).split(",")}
            import_from_map.setdefault(mod, set()).update(names)
        elif _re.match(r"import\s+([\w.]+)", stripped) and "from" not in stripped:
            m2 = _re.match(r"import\s+([\w.]+)", stripped)
            if m2:
                import_module_set.add(m2.group(1).split(".")[0])

    # Now scan for qualified calls to modules that were only from-imported
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        for mod, _names in import_from_map.items():
            top_mod = mod.split(".")[0]
            # Only flag if the module was NOT also imported via `import X`
            if top_mod in import_module_set:
                continue
            # Check for `module.name(...)` usage when `name` was from-imported
            for name in _names:
                pattern = _re.escape(f"{mod}.{name}") + r"\s*\("
                if _re.search(pattern, stripped):
                    warnings.append(
                        f"[{fname}:{i}] Import-usage mismatch: '{name}' was imported "
                        f"via `from {mod} import {name}` but called as `{mod}.{name}()` "
                        f"— this will raise NameError. Use `{name}()` directly."
                    )

    return warnings


def check_numpy_attribute_exists(code: str, fname: str = "main.py") -> list[str]:
    """Flag ``np.<name>`` references that the installed NumPy does not define.

    The complement of ``check_api_correctness``'s hand-written tables. Those
    tables can only catch a name someone already knew to write down, which is
    how ``np.erfinv`` reached a real run: the model fused "erf is a maths
    function" with "NumPy has np.* scientific functions" into a plausible name
    that has never existed in any NumPy release, and the tables — being a list
    of *known* defects — said nothing.

    Asking the namespace instead inverts that: the reference set is
    ``dir(numpy)`` from the live interpreter, so a *new* confabulated name is
    caught the first time without anyone predicting it. Measured over 3161
    generated files (30 artifact roots, 122 distinct ``np.*`` names used), a
    whitelist flagged exactly the 2 genuine defects and nothing else — against
    21 entries the hand-written table must maintain by hand.

    ``hasattr`` rather than ``in dir(np)``: ``np.matrixlib`` and
    ``np.version`` are reachable attributes that ``dir()`` omits, and
    reporting a working import as a NameError is the exact false positive this
    check exists to avoid.

    Scoped to plain ``np.<name>`` reads only — the alias must literally be
    ``np``, so a project that binds numpy to another name is simply not
    checked rather than mis-checked. The one case that still slips through is
    a *rebinding* of ``np`` (``np = MyShim()`` or a parameter named ``np``)
    inside a file that also imports numpy; that reads as a report on code
    whose ``np`` is not numpy. It is left unhandled deliberately — resolving
    it needs scope analysis, and of the two possible errors, flagging a
    shim's missing attribute is the cheaper one to be wrong about. Prose
    lines are skipped through the same helper ``check_api_correctness`` uses,
    since a docstring explaining that ``np.erfinv`` was removed must not
    itself be reported.
    """
    import re as _re

    if not _re.search(r"^\s*(?:import\s+numpy(?:\s+as\s+np)?|from\s+numpy)", code, _re.MULTILINE):
        return []

    # `np` rebound to something else means every `np.x` below is not a numpy
    # lookup. Detected from the AST and scoped to the span of the binding that
    # does the shadowing: a parameter named `np` in one function must not
    # silence a real `np.erfinv` two functions further down, while a
    # module-level `np = shim` legitimately covers the whole file.
    _shadowed: list[tuple[int, int]] = []
    try:
        _tree: ast.Module | None = ast.parse(code)
    except (SyntaxError, ValueError):
        _tree = None

    if _tree is not None:
        for node in ast.walk(_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _args = node.args
                if any(a.arg == "np" for a in (*_args.posonlyargs, *_args.args, *_args.kwonlyargs)):
                    _shadowed.append((node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno))
        for node in _tree.body:
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "np" for t in node.targets)) or (
                    isinstance(node, (ast.AnnAssign, ast.AugAssign))
                    and isinstance(node.target, ast.Name) and node.target.id == "np"):
                _shadowed.append((0, 10**9))  # module level: covers the file
                break
    elif _re.search(r"^\s*np\s*=\s*(?!=)", code, _re.MULTILINE):
        _shadowed.append((0, 10**9))

    def _is_shadowed(lineno: int) -> bool:
        return any(lo <= lineno <= hi for lo, hi in _shadowed)

    try:
        import numpy as _np
    except ImportError:  # pragma: no cover - numpy is a core dependency
        return []

    warnings: list[str] = []
    _masked = _blank_prose(code)
    lines = _masked.splitlines()
    seen: set[str] = set()
    for i, line in enumerate(lines, 1):
        if _is_shadowed(i):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        for name in _re.findall(r"\bnp\.([A-Za-z_][A-Za-z0-9_]*)", stripped):
            if name in seen or hasattr(_np, name):
                continue
            seen.add(name)
            warnings.append(
                f"[{fname}:{i}] np.{name} is not defined by the installed NumPy "
                f"({_np.__version__}) — this raises AttributeError at runtime. "
                f"Check the name, or import it from scipy/math."
            )

    return warnings


def check_undefined_calls(code: str, fname: str = "main.py") -> list[str]:
    """Detect calls to undefined functions/names in experiment code.

    Catches the pattern where a function is called but never defined or imported,
    which would cause NameError at runtime.
    """
    warnings: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return warnings

    # Common builtins that are always available.
    #
    # The exception classes come from the interpreter rather than being typed
    # out: a hand-written list is always incomplete, and every omission is a
    # false positive that sends the repair loop after correct code. `raise
    # FloatingPointError(...)` was reported as "Call to undefined function" for
    # exactly that reason — 38 builtin exceptions were missing, among them
    # NameError, TimeoutError, PermissionError and ConnectionError.
    builtins = {
        "print", "len", "range", "enumerate", "zip", "map", "filter", "sorted",
        "list", "dict", "set", "tuple", "str", "int", "float", "bool", "bytes",
        "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
        "delattr", "callable", "iter", "next", "reversed", "slice", "super",
        "property", "staticmethod", "classmethod", "abs", "all", "any", "bin",
        "chr", "ord", "hex", "oct", "pow", "round", "sum", "min", "max", "open",
        "input", "repr", "hash", "id", "dir", "vars", "globals", "locals",
        "format", "ascii", "object",
        "breakpoint", "memoryview", "bytearray", "frozenset", "complex",
        "divmod", "eval", "exec", "compile", "__import__", "help", "exit", "quit",
    }
    # Every builtin exception/warning class this interpreter defines.
    builtins.update(
        _n for _n in dir(_builtins)
        if isinstance(getattr(_builtins, _n, None), type)
        and issubclass(getattr(_builtins, _n), BaseException)
    )

    # Collect all defined names in the module
    defined_names: set[str] = set()

    for node in ast.walk(tree):
        # Function definitions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_names.add(node.name)
        # Class definitions
        elif isinstance(node, ast.ClassDef):
            defined_names.add(node.name)
        # Imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                defined_names.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if name != "*":
                    defined_names.add(name)
        # Assignments (including comprehensions)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            defined_names.add(elt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                defined_names.add(node.target.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                defined_names.add(node.target.id)
        # For loop targets
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                defined_names.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        defined_names.add(elt.id)
        # With statement targets
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    defined_names.add(item.optional_vars.id)
        # Exception handlers
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                defined_names.add(node.name)
        # Named expressions (walrus operator)
        elif isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name):
                defined_names.add(node.target.id)

    # Also collect function parameters
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                defined_names.add(arg.arg)
            for arg in node.args.posonlyargs:
                defined_names.add(arg.arg)
            for arg in node.args.kwonlyargs:
                defined_names.add(arg.arg)
            if node.args.vararg:
                defined_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined_names.add(node.args.kwarg.arg)

    # Now find all function calls to bare names (not attributes like obj.method())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Only check bare name calls, not attribute calls (obj.method())
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
                if (
                    call_name not in defined_names
                    and call_name not in builtins
                ):
                    warnings.append(
                        f"[{fname}:{node.lineno}] Call to undefined function "
                        f"'{call_name}()' — this will raise NameError at runtime. "
                        f"Either define the function or remove the call."
                    )

    return warnings


def check_filename_collisions(files: dict[str, str]) -> list[str]:
    """BUG-202: Detect local .py filenames that shadow pip/stdlib packages.

    The LLM commonly generates ``config.py``, ``models.py``, etc. which get
    shadowed by pip-installed packages (e.g. ``pip install config``).  The
    result is an import crash at runtime.
    """
    # Filenames (without .py) that are known to collide with pip/stdlib packages.
    _SHADOW_RISK: set[str] = {
        # pip packages frequently installed as transitive deps
        "config", "test", "tests", "types", "typing_extensions",
        # stdlib modules the LLM might accidentally shadow
        "io", "logging", "json", "time", "random", "copy", "math",
        "os", "sys", "collections", "functools", "abc", "re",
        "statistics", "signal", "pickle", "itertools",
        "string", "tokenize", "token", "email", "calendar",
        "numbers", "operator", "queue", "code", "profile",
    }
    warnings: list[str] = []
    for fname in files:
        stem = fname.removesuffix(".py") if fname.endswith(".py") else None
        if stem and stem in _SHADOW_RISK:
            warnings.append(
                f"[{fname}] Filename shadows stdlib/pip package '{stem}'. "
                f"Rename to e.g. '{stem}_config.py' or 'experiment_{stem}.py' "
                f"to avoid import collisions at runtime."
            )
    return warnings


def deep_validate_files(
    files: dict[str, str],
) -> list[str]:
    """Run all deep quality checks across all experiment files.

    Returns a list of warning strings. Empty = no concerns.

    ``check_numpy_attribute_exists`` is included here as a *warning* only. The
    callers separate critical from advisory by keyword (``_code_generation``
    looks for "does not exist" / "NameError" / …), and this check's message
    says "is not defined by the installed NumPy … AttributeError", which
    matches none of them. That is deliberate: a wrong name in a rarely-taken
    branch is not worth an LLM repair cycle, but it is worth a log line, since
    the alternative is what produced the 4c5e70cf run — a silent `inf` that
    surfaced only as "evolution refused to start".

    ``np.trapz`` is reported by both API checks, and by design they are not the
    same finding: the hand-written table knows the *replacement*
    (``np.trapezoid``), while the namespace check is what catches a name nobody
    tabulated. So the pair is deduplicated here, at the point they meet, rather
    than by deleting either check — dropping the table's entry would trade a
    duplicate line for a worse message, and dropping the namespace check would
    reopen the hole it exists to close.
    """
    warnings: list[str] = []
    warnings.extend(check_class_quality(files))
    warnings.extend(check_filename_collisions(files))
    _seen: set[tuple[str, str, str]] = set()

    def _emit(msg: str) -> None:
        """Append *msg* unless the same name was already reported at that site.

        Identity is (file, line, np-name), not the message text: the two API
        checks word the same defect differently, so comparing strings would let
        the duplicate through.
        """
        _key = (*_site_of(msg), _np_name_in(msg))
        if _np_name_in(msg) and _key in _seen:
            return
        _seen.add(_key)
        warnings.append(msg)

    for fname, code in files.items():
        if not fname.endswith(".py"):
            continue
        warnings.extend(check_variable_scoping(code, fname))
        for w in check_api_correctness(code, fname):
            _emit(w)
        for w in check_numpy_attribute_exists(code, fname):
            _emit(w)
        warnings.extend(check_undefined_calls(code, fname))
    return warnings


def _site_of(warning: str) -> tuple[str, str]:
    """The ``(file, line)`` a warning's ``[...]`` prefix names.

    The checks do not agree on the prefix shape — ``check_api_correctness``
    writes ``[main.py:12]`` while ``check_class_quality`` writes ``[main.py]``
    — so both are normalised to file plus line ("" when absent)."""
    import re as _re

    _m = _re.match(r"\[([^\]:]+)(?::(\d+))?\]", warning.strip())
    return (_m.group(1), _m.group(2) or "") if _m else ("", "")


def _np_name_in(warning: str) -> str:
    """The ``np.<name>`` a warning is about, or "" when it names none."""
    import re as _re

    _m = _re.search(r"\bnp\.[A-Za-z_][A-Za-z0-9_]*", warning)
    return _m.group(0) if _m else ""
