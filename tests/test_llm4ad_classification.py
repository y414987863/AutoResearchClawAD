"""Unit tests for ``_read_algorithms_classification``.

The scope filter (``evolve_scope``) selects which algorithms get evolved from
stage-10's ``algorithms_classification.json``. That file is a *category*
classification only when it maps each algorithm to a proposed/baseline/ablation
role; real products also ship *method-family groupings* whose values name the
algorithm set rather than a role. Flattening those into ``{algo: family}`` made
a ``{"categories": ["proposed"]}`` run match every algorithm against the literal
string ``"proposed"``, get the empty set, and silently skip evolution.

These tests pin the reader's contract: the two canonical shapes decode to a
per-algorithm role mapping, and every non-canonical grouping makes it return
``None`` so :func:`_filter_algorithms_by_scope` fails closed instead of
classifying nothing.
"""

from pathlib import Path

from researchclaw.pipeline.llm4ad_task_packages import _read_algorithms_classification


def _exp_with(cls_text: str) -> Path:
    import tempfile

    exp = Path(tempfile.mkdtemp()) / "e"
    (exp / "algorithms").mkdir(parents=True)
    (exp / "algorithms_classification.json").write_text(cls_text, encoding="utf-8")
    return exp


# --- canonical shapes decode to a role mapping ------------------------------

def test_canonical_wrapper_shape():
    exp = _exp_with(
        '{"classification": {"nelder_mead": "proposed", "random": "baseline", "gp": "ablation"}}'
    )
    assert _read_algorithms_classification(exp) == {
        "nelder_mead": "proposed", "random": "baseline", "gp": "ablation",
    }


def test_flat_top_level_scalar_shape():
    exp = _exp_with('{"nelder_mead": "proposed", "baseline_solver": "baseline"}')
    assert _read_algorithms_classification(exp) == {
        "nelder_mead": "proposed", "baseline_solver": "baseline",
    }


# --- non-canonical groupings return None (fail-closed) -----------------------

def test_family_grouping_dict_of_lists():
    """ml25: ``{"reservoir": ["esn"], "mlp": ["mlp"], ...}`` — the VALUES name
    the algorithm set, not a proposed/baseline/ablation role."""
    exp = _exp_with('{"reservoir": ["esn"], "gaussian_process": ["gp"], "baseline": ["persistence"]}')
    assert _read_algorithms_classification(exp) is None


def test_family_grouping_wrapper():
    """ml25: ``{"forecasting": {esn: "reservoir_computing", ...}}`` nested under a
    family name."""
    exp = _exp_with('{"forecasting": {"esn": "reservoir_computing", "linear_ar": "autoregressive"}}')
    assert _read_algorithms_classification(exp) is None


def test_type_grouping_dict_of_lists():
    """ml23: ``{"algorithm_types": {"pointwise": [...]}}`` — family grouping."""
    exp = _exp_with('{"algorithm_types": {"pointwise": ["pointwise_linear", "pointwise_ridge"]}}')
    assert _read_algorithms_classification(exp) is None


# --- invented / non-whitelisted categories are dropped ----------------------

def test_invented_category_values_dropped():
    """A classifier that invents a role (``"sota"``) instead of one of the three
    whitelisted categories is unusable; if nothing valid remains, None."""
    exp = _exp_with('{"nelder_mead": "sota", "gp": "custom"}')
    assert _read_algorithms_classification(exp) is None


def test_invented_category_mixed_with_valid_keeps_valid():
    exp = _exp_with('{"nelder_mead": "sota", "gp": "proposed"}')
    assert _read_algorithms_classification(exp) == {"gp": "proposed"}
