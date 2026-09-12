# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportUnknownLambdaType=false
from __future__ import annotations

import pytest

from researchclaw.experiment.validator import (
    BANNED_MODULES,
    DANGEROUS_BUILTINS,
    DANGEROUS_CALLS,
    CodeValidation,
    ValidationIssue,
    auto_fix_unbound_locals,
    check_api_correctness,
    check_filename_collisions,
    extract_imports,
    format_issues_for_llm,
    validate_code,
    validate_imports,
    validate_security,
    validate_syntax,
)


def _call_source(name: str) -> str:
    top = name.split(".")[0]
    lines: list[str] = []
    if top in {"os", "subprocess", "shutil"}:
        lines.append(f"import {top}")
    lines.append(f"{name}()")
    return "\n".join(lines)


def test_validate_syntax_accepts_valid_code():
    result = validate_syntax("x = 1\nif x > 0:\n    x += 1")

    assert result.ok is True
    assert result.issues == []


def test_validate_syntax_reports_syntax_error_with_location():
    result = validate_syntax("def bad(:\n    pass")

    assert result.ok is False
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.severity == "error"
    assert issue.category == "syntax"
    assert issue.line == 1
    assert issue.col is not None
    assert issue.message


@pytest.mark.parametrize("code", ["", "   \n\t  ", "# comment only\n# still comment"])
def test_validate_syntax_accepts_empty_whitespace_and_comment_only(code: str):
    result = validate_syntax(code)

    assert result.ok is True
    assert result.issues == []


def test_validate_security_accepts_safe_code():
    code = 'import os\nvalue = os.path.join("a", "b")\nprint(value)'
    result = validate_security(code)

    assert result.ok is True
    assert result.issues == []


def test_validate_security_skips_when_code_has_syntax_error():
    result = validate_security("def broken(:\n    pass")

    assert result.ok is True
    assert result.issues == []


@pytest.mark.parametrize("builtin_name", sorted(DANGEROUS_BUILTINS))
def test_validate_security_flags_every_dangerous_builtin_call(builtin_name: str):
    if builtin_name == "__import__":
        code = '__import__("os")'
    elif builtin_name == "compile":
        code = 'compile("x = 1", "<string>", "exec")'
    else:
        code = f'{builtin_name}("print(1)")'

    result = validate_security(code)

    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.severity == "error"
    assert issue.category == "security"
    assert issue.message == f"Dangerous built-in call: {builtin_name}()"


@pytest.mark.parametrize("call_name", sorted(DANGEROUS_CALLS))
def test_validate_security_flags_every_dangerous_call(call_name: str):
    result = validate_security(_call_source(call_name))

    messages = [issue.message for issue in result.issues]
    assert f"Dangerous call: {call_name}()" in messages
    assert all(issue.severity == "error" for issue in result.issues)
    assert all(issue.category == "security" for issue in result.issues)


@pytest.mark.parametrize("module_name", sorted(BANNED_MODULES))
def test_validate_security_flags_every_banned_import(module_name: str):
    result = validate_security(f"import {module_name}")

    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.severity == "error"
    assert issue.category == "security"
    assert issue.message == f"Banned module import: {module_name}"


@pytest.mark.parametrize("module_name", sorted(BANNED_MODULES))
def test_validate_security_flags_every_banned_from_import(module_name: str):
    result = validate_security(f"from {module_name} import x")

    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.severity == "error"
    assert issue.category == "security"
    assert issue.message == f"Banned module import: from {module_name}"


def test_validate_imports_recognizes_stdlib_modules_by_default():
    result = validate_imports("import json\nfrom math import sqrt")

    assert result.ok is True
    assert result.warnings == []


def test_validate_imports_warns_for_unavailable_package():
    result = validate_imports("import totally_missing_pkg")

    assert result.ok is True
    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert warning.severity == "warning"
    assert warning.category == "import"
    assert (
        warning.message
        == "Module 'totally_missing_pkg' may not be available in sandbox"
    )


def test_validate_imports_respects_custom_available_set():
    result = validate_imports(
        "import alpha\nimport beta\nimport gamma",
        available={"alpha", "gamma"},
    )

    assert [w.message for w in result.warnings] == [
        "Module 'beta' may not be available in sandbox",
    ]


def test_validate_imports_returns_no_warnings_for_syntax_error_input():
    result = validate_imports("def bad(:\n    pass", available=set())

    assert result.ok is True
    assert result.warnings == []


@pytest.mark.parametrize("code", ["", "   \n\t  ", "# comment only"])
def test_validate_imports_handles_empty_like_inputs(code: str):
    result = validate_imports(code, available=set())

    assert result.ok is True
    assert result.warnings == []


def test_validate_code_combines_security_and_import_issues_in_order():
    code = 'import os\nos.system("echo hi")\nimport unknown_mod'
    result = validate_code(code, available_packages={"os"})

    assert result.ok is False
    assert [i.category for i in result.issues] == ["security", "import"]
    assert result.issues[0].message == "Dangerous call: os.system()"
    assert (
        result.issues[1].message
        == "Module 'unknown_mod' may not be available in sandbox"
    )


def test_validate_code_short_circuits_after_syntax_error():
    result = validate_code("def bad(:\n    pass")

    assert len(result.issues) == 1
    assert result.issues[0].category == "syntax"


def test_validate_code_skip_security_excludes_security_issues():
    code = 'import os\nos.system("echo hi")\nimport unknown_mod'
    result = validate_code(code, available_packages={"os"}, skip_security=True)

    assert [i.category for i in result.issues] == ["import"]


def test_validate_code_skip_imports_excludes_import_warnings():
    code = 'import os\nos.system("echo hi")\nimport unknown_mod'
    result = validate_code(code, available_packages={"os"}, skip_imports=True)

    assert all(issue.category == "security" for issue in result.issues)
    assert len(result.issues) == 1


def test_validate_code_skip_both_returns_clean_for_safe_code():
    result = validate_code("x = 1", skip_security=True, skip_imports=True)

    assert result.ok is True
    assert result.issues == []


def test_validate_code_uses_available_packages_for_import_validation():
    code = "import alpha\nimport beta"
    result = validate_code(code, available_packages={"alpha"})

    assert [i.message for i in result.issues] == [
        "Module 'beta' may not be available in sandbox",
    ]


def test_extract_imports_supports_import_and_from_import_styles():
    code = (
        "import os\nimport numpy as np\nfrom pandas import DataFrame\nfrom x.y import z"
    )

    assert extract_imports(code) == {"os", "numpy", "pandas", "x"}


def test_extract_imports_supports_multiple_aliases_and_dedupes():
    code = "import os.path, os, json as js\nfrom json import loads"

    assert extract_imports(code) == {"os", "json"}


def test_extract_imports_ignores_relative_import_without_module_name():
    assert extract_imports("from . import local_mod") == set()


def test_extract_imports_includes_relative_import_with_module_name():
    assert extract_imports("from ..pkg.sub import thing") == {"pkg"}


def test_extract_imports_returns_empty_set_for_syntax_error():
    assert extract_imports("def bad(:\n    pass") == set()


@pytest.mark.parametrize("code", ["", "   \n\t", "# comment only"])
def test_extract_imports_handles_empty_like_inputs(code: str):
    assert extract_imports(code) == set()


def test_format_issues_for_llm_returns_no_issues_message_when_clean():
    assert format_issues_for_llm(CodeValidation()) == "No issues found."


def test_format_issues_for_llm_formats_issues_with_and_without_line():
    validation = CodeValidation(
        issues=[
            ValidationIssue(
                severity="error",
                category="syntax",
                message="invalid syntax",
                line=3,
            ),
            ValidationIssue(
                severity="warning",
                category="import",
                message="Module 'x' may be missing",
                line=None,
            ),
        ]
    )

    formatted = format_issues_for_llm(validation)

    assert "- [ERROR] (syntax) invalid syntax @ line 3" in formatted
    assert (
        "- [WARNING] (import) Module 'x' may be missing @ unknown location" in formatted
    )


def test_format_issues_for_llm_preserves_issue_order():
    validation = CodeValidation(
        issues=[
            ValidationIssue(severity="warning", category="import", message="first"),
            ValidationIssue(
                severity="error", category="security", message="second", line=9
            ),
        ]
    )

    formatted = format_issues_for_llm(validation).splitlines()

    assert formatted[0] == "- [WARNING] (import) first @ unknown location"
    assert formatted[1] == "- [ERROR] (security) second @ line 9"


def test_code_validation_ok_true_when_no_errors_present():
    validation = CodeValidation(
        issues=[ValidationIssue(severity="warning", category="import", message="warn")]
    )

    assert validation.ok is True


def test_code_validation_ok_false_when_error_present():
    validation = CodeValidation(
        issues=[ValidationIssue(severity="error", category="syntax", message="bad")]
    )

    assert validation.ok is False


def test_code_validation_errors_and_warnings_filter_correctly():
    err = ValidationIssue(severity="error", category="security", message="danger")
    warn = ValidationIssue(
        severity="warning", category="import", message="maybe missing"
    )
    validation = CodeValidation(issues=[err, warn])

    assert validation.errors == [err]
    assert validation.warnings == [warn]


def test_code_validation_summary_for_no_issues():
    assert CodeValidation().summary() == "Code validation passed."


def test_code_validation_summary_for_errors_only():
    validation = CodeValidation(
        issues=[ValidationIssue(severity="error", category="syntax", message="bad")]
    )

    assert validation.summary() == "Code validation: 1 error(s)"


def test_code_validation_summary_for_warnings_only():
    validation = CodeValidation(
        issues=[ValidationIssue(severity="warning", category="import", message="warn")]
    )

    assert validation.summary() == "Code validation: 1 warning(s)"


def test_code_validation_summary_for_errors_and_warnings():
    validation = CodeValidation(
        issues=[
            ValidationIssue(severity="error", category="syntax", message="bad"),
            ValidationIssue(severity="warning", category="import", message="warn"),
        ]
    )

    assert validation.summary() == "Code validation: 1 error(s), 1 warning(s)"


# ---------------------------------------------------------------------------
# check_filename_collisions (BUG-202)
# ---------------------------------------------------------------------------


def test_filename_collision_detects_config_py():
    """BUG-202: config.py shadows pip 'config' package."""
    warnings = check_filename_collisions({"config.py": "x = 1", "main.py": "print(1)"})
    assert len(warnings) == 1
    assert "shadows stdlib/pip" in warnings[0]
    assert "config" in warnings[0]


def test_filename_collision_detects_stdlib_shadows():
    """Filenames shadowing stdlib modules should be flagged."""
    warnings = check_filename_collisions({"json.py": "x = 1"})
    assert len(warnings) == 1
    assert "json" in warnings[0]


def test_filename_collision_allows_safe_names():
    """Normal experiment filenames should not trigger warnings."""
    files = {
        "main.py": "print(1)",
        "models.py": "class M: pass",
        "training.py": "def train(): pass",
        "data_loader.py": "def load(): pass",
        "experiment_config.py": "LR = 0.01",
        "requirements.txt": "torch",
    }
    warnings = check_filename_collisions(files)
    assert warnings == []


def test_filename_collision_multiple_shadows():
    """Multiple shadowing files should each produce a warning."""
    files = {"config.py": "", "logging.py": "", "main.py": ""}
    warnings = check_filename_collisions(files)
    assert len(warnings) == 2


def test_auto_fix_does_not_clobber_loop_assigned_var():
    """Regression: a variable assigned in a for-loop AND an if-branch must NOT
    be pre-seeded with ``= None`` (that would clobber the loop's value)."""
    code = (
        "def compute(items, flag):\n"
        "    for x in items:\n"
        "        total = x * 2\n"
        "    if flag:\n"
        "        total = 0\n"
        "    return total\n"
    )
    fixed, n = auto_fix_unbound_locals(code)
    assert n == 0
    assert "total = None" not in fixed


def test_auto_fix_does_not_clobber_with_bound_var():
    """A variable bound by a with-statement must not be pre-seeded either."""
    code = (
        "def g(flag, path):\n"
        "    with open(path) as fh:\n"
        "        data = fh.read()\n"
        "    if flag:\n"
        "        data = ''\n"
        "    return data\n"
    )
    fixed, n = auto_fix_unbound_locals(code)
    assert n == 0
    assert "data = None" not in fixed


def test_auto_fix_still_seeds_genuine_if_only_var():
    """A variable assigned only inside an if-branch must still be fixed."""
    code = (
        "def f(flag):\n"
        "    if flag:\n"
        "        result = 1\n"
        "    return result\n"
    )
    fixed, n = auto_fix_unbound_locals(code)
    assert n >= 1
    assert "result = None" in fixed


# ---------------------------------------------------------------------------
# check_api_correctness — NumPy 2.0 / pandas 2.0 removed APIs
# ---------------------------------------------------------------------------

def test_api_correctness_flags_removed_numpy_trapz():
    warnings = check_api_correctness("x = np.trapz(y, dx=0.1)\n", "m.py")
    assert any("np.trapz" in w and "np.trapezoid" in w for w in warnings)


def test_api_correctness_flags_removed_numpy_names():
    cases = [
        ("np.product", "np.prod"),
        ("np.in1d", "np.isin"),
        ("np.row_stack", "np.vstack"),
        ("np.cumproduct", "np.cumprod"),
        ("np.round_", "np.round"),
        ("np.alltrue", "np.all"),
        ("np.sometrue", "np.any"),
        ("np.NaN", "np.nan"),
        ("np.Inf", "np.inf"),
    ]
    for removed, replacement in cases:
        warnings = check_api_correctness(f"v = {removed}\n", "m.py")
        assert any(removed in w and replacement in w for w in warnings), (
            f"expected {removed} -> {replacement} to be flagged"
        )


def test_api_correctness_does_not_flag_live_numpy_aliases():
    code = (
        "import numpy as np\n"
        "a = np.float_(1.0)\n"
        "b = np.int_(2)\n"
        "c = np.complex_(3j)\n"
        "d = np.float64(4.0)\n"
        "e = np.float32(5.0)\n"
        "f = np.mat([[1, 2], [3, 4]])\n"
    )
    warnings = check_api_correctness(code, "m.py")
    # None of these live aliases should be reported as removed-in-2.0.
    assert not any("was removed in NumPy 2.0" in w for w in warnings)


def test_api_correctness_flags_pandas_removals():
    code = (
        "import pandas as pd\n"
        "df = df.append(other)\n"
        "row = df.ix[0]\n"
        "for k, v in series.iteritems():\n"
        "    pass\n"
    )
    warnings = check_api_correctness(code, "m.py")
    assert any("DataFrame/Series.append() was removed in pandas 2.0" in w for w in warnings)
    assert any("`.ix[]` was removed in pandas 2.0" in w for w in warnings)
    assert any("`.iteritems()` was removed in pandas 2.0" in w for w in warnings)


def test_api_correctness_does_not_flag_list_append_when_no_pandas():
    # A pure-python list.append is legal and must not be flagged.
    code = "xs = []\nfor i in range(3):\n    xs.append(i)\n"
    warnings = check_api_correctness(code, "m.py")
    assert not any("DataFrame/Series.append() was removed in pandas 2.0" in w for w in warnings)


# ---------------------------------------------------------------------------
# code-generation helpers — llm4ad constraint text + degenerate-instance scan
# ---------------------------------------------------------------------------

class _Attr:
    """Minimal attribute holder so a plain config object can be faked."""

    def __init__(self, **kw):
        self._kw = kw

    def __getattr__(self, name):
        try:
            return self._kw[name]
        except KeyError:
            raise AttributeError(name) from None


class _Cfg:
    """Config whose .experiment may be absent or a plain object."""

    def __init__(self, experiment=None):
        self.experiment = experiment


def test_llm4ad_constraint_text_empty_when_disabled():
    from researchclaw.pipeline.stage_impls import _code_generation as cg

    assert cg._llm4ad_constraint_text(_Cfg(experiment=None)) == ""
    assert cg._llm4ad_constraint_text(
        _Cfg(experiment=_Attr(llm4ad_boost=_Attr(enabled=False)))
    ) == ""


def test_llm4ad_constraint_text_populated_when_enabled():
    from researchclaw.pipeline.stage_impls import _code_generation as cg

    text = cg._llm4ad_constraint_text(
        _Cfg(experiment=_Attr(llm4ad_boost=_Attr(enabled=True)))
    )
    assert "EVOLVE_START" in text
    assert "PRESERVE these markers" in text


def test_warn_degenerate_instances(tmp_path):
    from researchclaw.pipeline.stage_impls import _code_generation as cg

    results = {
        "by_phase": {
            "A": {
                "by_instance": {
                    "Masked_d10": {
                        "algorithms": [
                            {"algo": "x", "n_total": 5, "n_valid": 0},
                            {"algo": "y", "n_total": 5, "n_valid": 0},
                        ]
                    },
                    "Healthy_d5": {
                        "algorithms": [
                            {"algo": "x", "n_total": 5, "n_valid": 3},
                        ]
                    },
                }
            }
        }
    }
    (tmp_path / "results.json").write_text(
        __import__("json").dumps(results), encoding="utf-8"
    )
    warnings = cg._warn_degenerate_instances(tmp_path)
    assert any("Masked_d10" in w for w in warnings)
    assert not any("Healthy_d5" in w for w in warnings)


def test_revert_marker_dropped_files_keeps_neutral_fix():
    """A repair that strips EVOLVE markers is reverted; a neutral fix is kept."""
    from researchclaw.pipeline.stage_impls import _code_generation as cg

    original = {
        "algorithms/a/a.py": "def optimize(i, s):\n    # EVOLVE_START\n    x = i\n    # EVOLVE_END\n    return x\n",
        "algorithms/b/b.py": "def optimize(i, s):\n    return i\n",
    }
    applied = {
        "algorithms/a/a.py": "def optimize(i, s):\n    x = i\n    return x\n",  # dropped markers
        "algorithms/b/b.py": "def optimize(i, s):\n    return i*2\n",          # never had markers
    }

    files = dict(original)
    prev = dict(files)
    files, app = cg._merge_repaired_files(files, applied, label="test")
    reverted = cg._revert_marker_dropped_files(prev, app, label="test")

    # a dropped its markers -> must be reverted to the original
    assert reverted == ["algorithms/a/a.py"]
    for fname in reverted:
        files[fname] = prev[fname]
    assert "EVOLVE_START" in files["algorithms/a/a.py"]
    # b never had markers -> the neutral fix is preserved
    assert "i*2" in files["algorithms/b/b.py"]


def test_every_repair_channel_reverts_dropped_markers():
    """Every `_merge_repaired_files` call site must be paired with the guard.

    Stage 10 rewrites project files through four LLM repair channels (deep
    repair, review-fix, OpenCode repair, smoke fix). Each one hands the model the
    current code and merges back whatever it returns — so each one can also drop
    the `# EVOLVE_START` / `# EVOLVE_END` pair, which makes the algorithm
    non-evolvable and silently wastes the evolution stage.

    Two of the four were missing this guard. One real run had the model write all
    four algorithms correctly, markers included, and verify them itself — then a
    repair pass rewrote the files without the markers, and Stage 13's analysis
    found zero evolvable blocks.

    The channels are inline inside `_execute_code_generation`, so their behaviour
    is not reachable from a unit test; this checks the invariant that must hold
    for each, which is what regressed.
    """
    import inspect

    from researchclaw.pipeline.stage_impls import _code_generation as cg

    src = inspect.getsource(cg._execute_code_generation)
    lines = src.splitlines()

    merges = [i for i, ln in enumerate(lines) if "_merge_repaired_files(" in ln]
    assert merges, "expected the repair channels to call _merge_repaired_files"

    for i in merges:
        # The guard is invoked a few lines after the merge, within the same
        # block. A channel that merges but never reverts is the bug.
        window = "\n".join(lines[i:i + 25])
        assert "_revert_marker_dropped_files(" in window, (
            "repair channel at line %d merges repaired files without reverting "
            "dropped EVOLVE markers:\n%s" % (i, lines[i].strip())
        )


def test_builtin_exceptions_are_not_undefined_functions():
    """`raise FloatingPointError(...)` must not read as an undefined function.

    The undefined-call check kept its own hardcoded list of "common builtins",
    and the exception classes in it were incomplete — 38 builtin exceptions were
    missing, including FloatingPointError, NameError, TimeoutError and
    PermissionError. Every omission turned correct `raise`/`except` code into a
    "Call to undefined function" warning, which is a repair-loop trigger: one
    real Stage-10 run was flagged for `raise FloatingPointError(...)` and spent a
    deep-repair round on code that was already right.

    The list now comes from the interpreter, so it cannot drift again.
    """
    from researchclaw.experiment.validator import deep_validate_files

    code = (
        "import numpy as np\n"
        "def evaluate_instance(instance, solve):\n"
        "    try:\n"
        "        w = np.asarray(solve(instance, 0)['w'], dtype=np.float64)\n"
        "        if not np.all(np.isfinite(w)):\n"
        "            raise FloatingPointError('non-finite')\n"
        "    except FloatingPointError:\n"
        "        pass\n"
        "    except (TimeoutError, PermissionError, NameError,\n"
        "            ArithmeticError, ConnectionError, LookupError):\n"
        "        pass\n"
        "    return {'m': 1.0}\n"
    )
    warnings = deep_validate_files({"evaluator.py": code})
    flagged = [w for w in warnings if "undefined function" in w.lower()]
    assert not flagged, flagged


def test_a_genuinely_undefined_call_is_still_reported():
    """Widening the builtin set must not disable the check itself."""
    from researchclaw.experiment.validator import deep_validate_files

    code = (
        "def evaluate_instance(instance, solve):\n"
        "    return {'m': genuinely_undefined_fn(1)}\n"
    )
    warnings = deep_validate_files({"evaluator.py": code})
    assert any("undefined function" in w.lower() for w in warnings), warnings
