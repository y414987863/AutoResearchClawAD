"""Unit tests for Stage-10 generated-metric consistency and task-package cleanup.

Two silent-production bugs in the pipeline:

- The codegen LLM picks PRIMARY_METRIC freely; config ``metric_key`` is only an
  advisory in the prompt. When the two dissent (ml25 chose ``valid_prediction_time``
  while the plan meant an RMSE), the pipeline reports one number while the paper
  reads another, with no sign. ``_metric_mismatch_problem`` surfaces that dissent.

- ``generate_task_packages`` rebuilt only the packages in the current scope and
  left stale ones from a prior generation, so ``run_evolution_on_packages``
  globbed and ran *more* packages than were built — the ml03 run "built 4, ran 10".
  The out-dir is now cleared wholesale first.
"""

from pathlib import Path

from researchclaw.pipeline.stage_impls._code_generation import (
    _generated_primary_metric,
    _metric_mismatch_problem,
)
from researchclaw.pipeline.llm4ad_task_packages import generate_task_packages


# --- generated-primary-metric extraction ------------------------------------

def test_reads_primary_metric_from_evaluator():
    files = {
        "evaluator.py": 'PRIMARY_METRIC = "one_step_rmse"\ndef evaluate_instance(i, s): pass',
        "main.py": "x=1",
    }
    assert _generated_primary_metric(files) == "one_step_rmse"


def test_reads_primary_metric_single_quote():
    files = {"evaluator.py": "PRIMARY_METRIC='nDCG@10'\n", "main.py": ""}
    assert _generated_primary_metric(files) == "nDCG@10"


def test_no_primary_metric_returns_none():
    assert _generated_primary_metric({"evaluator.py": "def f(): pass"}) is None


# --- metric mismatch is advisory, and only when real dissent -----------------

def test_mismatch_reported():
    p = _metric_mismatch_problem(
        {"evaluator.py": 'PRIMARY_METRIC = "valid_prediction_time"'},
        "one_step_rmse",
    )
    assert p and "METRIC_MISMATCH" in p and "valid_prediction_time" in p


def test_match_not_reported():
    p = _metric_mismatch_problem(
        {"evaluator.py": 'PRIMARY_METRIC = "one_step_rmse"'}, "one_step_rmse"
    )
    assert p is None


def test_placeholder_metric_key_is_ignored():
    # A generic default carries no topic intent, so dissent says nothing.
    p = _metric_mismatch_problem(
        {"evaluator.py": 'PRIMARY_METRIC = "valid_prediction_time"'},
        "mean_best_objective_value",
    )
    assert p is None


def test_missing_metric_key_is_ignored():
    assert _metric_mismatch_problem(
        {"evaluator.py": 'PRIMARY_METRIC = "x"'}, ""
    ) is None


# --- stale-package cleanup --------------------------------------------------

def _exp_with(algo_src, primary="m"):
    import tempfile

    exp = Path(tempfile.mkdtemp()) / "e"
    (exp / "algorithms" / "nm").mkdir(parents=True)
    (exp / "algorithms" / "nm" / "nm.py").write_text(algo_src)
    (exp / "evaluator.py").write_text(
        f'PRIMARY_METRIC="{primary}"\ndef evaluate_instance(i, s): return {{"{primary}": 1.0}}\n'
    )
    (exp / "main.py").write_text("x=1\n")
    (exp / "data").mkdir()
    (exp / "data" / "i.json").write_text('{"coords":[[0,0],[1,1]]}')
    return exp


def test_generate_clears_stale_packages():
    """A package left over from a previous (scoped or unscopped) generation must
    not survive into the new output set — else evolution runs packages that were
    never built this run (ml03: built 4, ran 10)."""
    import tempfile

    exp = _exp_with("def optimize(i,s):\n    # EVOLVE_START\n    return {}\n    # EVOLVE_END\n")
    out = Path(tempfile.mkdtemp()) / "tp"
    # Simulate a stale package from an earlier run.
    stale = out / "esn"
    stale.mkdir(parents=True)
    (out / "manifest.json").write_text("[]")
    (stale / "config.yaml").write_text("x: 1")
    generate_task_packages(exp, out, None, None, None, background="t")
    assert not (stale / "config.yaml").exists()
    assert (out / "nm" / "config.yaml").exists()
    # Only the rebuilt package remains.
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["nm"]
