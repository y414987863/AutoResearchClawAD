"""Tests for Stage-13's EVOLVE-marker restoration.

Stage 10 wraps each algorithm's `optimize` in `# EVOLVE_START` / `# EVOLVE_END`
so the evolution stage can rewrite that function. Refining the project rewrites
algorithms, and a model that rewrites the body rarely re-emits the markers.

When they vanish, LLM4AD's repo analyzer finds zero evolvable blocks and every
candidate is skipped:

    InitSampler requires analyzed_repository with at least one evolvable block

The stage then burns its whole budget producing nothing. One real run reached
this through exactly that path: stage 10 emitted the markers, stage 13's refine
deleted them, and the packaged task had `files_with_blocks: 0`.

`_restore_evolve_markers` is a closure inside `_execute_iterative_refine`, so
these tests extract it by source and execute it standalone rather than driving
the whole stage.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_EXECUTION_PY = (
    Path(__file__).resolve().parent.parent
    / "researchclaw" / "pipeline" / "stage_impls" / "_execution.py"
)


def _load_restore_helper():
    """Pull the nested helper out of the stage function and exec it."""
    src = _EXECUTION_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(
        (
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_restore_evolve_markers"
        ),
        None,
    )
    if node is None:
        pytest.skip("_restore_evolve_markers not present")
    segment = ast.get_source_segment(src, node)
    # The helper is nested one level deep; dedent it to module scope.
    body = "\n".join(
        line[4:] if line.startswith("    ") else line
        for line in segment.split("\n")
    )
    namespace: dict = {"logger": _NullLogger()}
    exec("import ast as _ast_markers\n" + body, namespace)  # noqa: S102
    return namespace["_restore_evolve_markers"]


class _NullLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, msg: str, *args) -> None:
        self.messages.append(msg % args if args else msg)


_OLD_WITH_MARKERS = '''\
import numpy as np


# EVOLVE_START
def optimize(instance: dict, seed: int) -> dict:
    """Original, evolvable."""
    return {"best_value": 1.0}
# EVOLVE_END
'''

# The refine output: same module, body rewritten, markers gone.
_NEW_WITHOUT_MARKERS = '''\
import math

import numpy as np


def optimize(instance: dict, seed: int) -> dict:
    """Rewritten by refine."""
    total = sum(range(10))
    return {"best_value": float(total)}
'''


def test_restores_markers_a_rewrite_dropped():
    restore = _load_restore_helper()
    new = {"algorithms/a/a.py": _NEW_WITHOUT_MARKERS}
    fixed = restore(new, {"algorithms/a/a.py": _OLD_WITH_MARKERS})

    assert fixed == ["algorithms/a/a.py"]
    code = new["algorithms/a/a.py"]
    assert code.count("EVOLVE_START") == 1
    assert code.count("EVOLVE_END") == 1
    # The markers must sit OUTSIDE `optimize`, as the reference packages do:
    # everything from the `def` line through the final `return` is evolvable.
    lines = code.splitlines()
    start = lines.index("# EVOLVE_START")
    end = lines.index("# EVOLVE_END")
    def_line = next(i for i, l in enumerate(lines) if l.startswith("def optimize"))
    assert start < def_line < end
    # Still valid Python, and the body refine produced is untouched.
    ast.parse(code)
    assert "total = sum(range(10))" in code


def test_markers_wrap_the_whole_function():
    """The block must reach the final `return`, not stop at the docstring."""
    restore = _load_restore_helper()
    code_in = (
        "def optimize(instance: dict, seed: int) -> dict:\n"
        "    x = 1\n"
        "    return {'best_value': x}\n"
    )
    new = {"algorithms/a/a.py": code_in}
    restore(new, {"algorithms/a/a.py": _OLD_WITH_MARKERS})
    lines = new["algorithms/a/a.py"].splitlines()
    end = lines.index("# EVOLVE_END")
    assert lines[end - 1].strip() == "return {'best_value': x}"


def test_file_without_a_newline_is_still_valid():
    """A file whose last line has no newline must not merge with the marker."""
    restore = _load_restore_helper()
    new = {"algorithms/a/a.py": "def optimize(i, s):\n    return {'v': 1}"}
    fixed = restore(new, {"algorithms/a/a.py": _OLD_WITH_MARKERS})
    assert fixed == ["algorithms/a/a.py"]
    code = new["algorithms/a/a.py"]
    assert "}# EVOLVE" not in code
    ast.parse(code)


def test_leaves_files_that_never_had_markers():
    """A helper module was never evolvable; inventing a block there is wrong."""
    restore = _load_restore_helper()
    helper = "def clip(x):\n    return x\n"
    new = {"helpers.py": helper}
    assert restore(new, {"helpers.py": helper}) == []
    assert new["helpers.py"] == helper


def test_leaves_files_that_kept_their_markers():
    restore = _load_restore_helper()
    new = {"algorithms/a/a.py": _OLD_WITH_MARKERS}
    assert restore(new, {"algorithms/a/a.py": _OLD_WITH_MARKERS}) == []


def test_ignores_unparsable_rewrite():
    """Broken syntax is the validation path's problem, not this one's."""
    restore = _load_restore_helper()
    broken = "def optimize(:\n    return\n"
    new = {"algorithms/a/a.py": broken}
    assert restore(new, {"algorithms/a/a.py": _OLD_WITH_MARKERS}) == []
    assert new["algorithms/a/a.py"] == broken


def test_ignores_file_without_optimize():
    restore = _load_restore_helper()
    no_fn = "x = 1\n"
    new = {"algorithms/a/a.py": no_fn}
    assert restore(new, {"algorithms/a/a.py": _OLD_WITH_MARKERS}) == []


def test_only_touches_nested_algorithm_paths():
    """Root-level modules are fixed infrastructure, never evolvable units."""
    restore = _load_restore_helper()
    new = {"main.py": _NEW_WITHOUT_MARKERS}
    assert restore(new, {"main.py": _OLD_WITH_MARKERS}) == []
