"""Unit tests for the LLM4AD pre-evolution fitness sanity gate.

The gate exists to stop evolution from burning LLM budget on a metric that
cannot be improved. Three degeneracies are covered, each observed on a real
ResearchClaw product earlier in the audit:

- ml23: a model-invariant ``valid_prediction_time`` — constant *per instance*
  (every algorithm scores the same value on a given instance) even though the
  value changes across instances. A global min/max check would MISS this, so the
  ``_constant`` test is deliberately per-instance.
- ml03: ``mean_time`` from ``time.perf_counter()`` — wall-clock, nondeterministic
  and gameable (an algorithm that returns without solving "wins").
- TSP: ``evaluate_instance`` computes ``cost`` but never writes ``instance["cost"]``
  before calling ``solve``, so every algorithm hits ``KeyError`` and scores ``inf``.
"""

from pathlib import Path

from researchclaw.pipeline.llm4ad_utils import fitness_gate as fg


def _entry(algo, values):
    return {"algo": algo, "values": values, "detail": "", "num_instances": len(values)}


# --- per-instance constant detection ----------------------------------------

def test_constant_across_instances_is_still_constant():
    """ml23: algorithm sets vary by instance but never by algorithm.

    The global min (min over 1) == global max (max over 2809.99) would look
    non-constant, yet no algorithm ever beats another. The gate must flag it.
    """
    entries = [
        _entry("listmle_lite", {"a": 1237.59, "b": 2663.60}),
        _entry("pointwise_linear", {"a": 1237.59, "b": 2663.60}),
        _entry("pointwise_ridge", {"a": 1237.59, "b": 2663.60}),
        _entry("ranknet_lite", {"a": 1237.59, "b": 2663.60}),
    ]
    assert fg._constant(entries) is True


def test_varied_metric_is_not_constant():
    """ml25: a genuine, algorithm-differentiating metric passes the gate."""
    entries = [
        _entry("esn_best", {"a": 0.12, "b": 0.34}),
        _entry("esn_wide", {"a": 0.20, "b": 0.41}),
        _entry("mlp", {"a": 0.09, "b": 0.28}),
    ]
    assert fg._constant(entries) is False


def test_constant_within_tolerance_is_flagged():
    """Values differing only by float noise are still 'constant'."""
    entries = [
        _entry("algo_a", {"a": 1.0, "b": 2.0}),
        _entry("algo_b", {"a": 1.0 + 1e-9, "b": 2.0 - 1e-9}),
    ]
    assert fg._constant(entries) is True


def test_single_algo_not_constant():
    """No cross-algorithm comparison possible — cannot call it constant."""
    assert fg._constant([_entry("only", {"a": 1.0, "b": 2.0})]) is False
    assert fg._constant([]) is False


# --- non-finite (TSP instance-contract break) --------------------------------

def test_non_finite_only_when_no_values():
    assert fg._non_finite(_entry("x", {})) is True
    # A dict non-empty is non-finite only if it holds no value — the gate's
    # definition is "exposes no finite score on ANY instance", i.e. empty values.
    assert fg._non_finite(_entry("x", {"a": float("inf")})) is False
    assert fg._non_finite(_entry("x", {"a": 1.0})) is False


# --- gate wiring: violation strings -----------------------------------------

def test_gate_raises_on_constant_violation(monkeypatch, tmp_path):
    def fake_score(exp_dir, algo_files, *, timeout_sec=300):
        return [
            _entry("a", {"i": 5.0, "j": 9.0}),
            _entry("b", {"i": 5.0, "j": 9.0}),
        ]
    monkeypatch.setattr(fg, "score_algorithms", fake_score)
    viz = fg.fitness_sanity_gate(tmp_path, [("a", Path("a.py")), ("b", Path("b.py"))])
    assert viz and "constant" in viz[0]


def test_gate_raises_on_no_finite_metric(monkeypatch, tmp_path):
    def fake_score(exp_dir, algo_files, *, timeout_sec=300):
        return [
            _entry("tsp_a", {}),
            _entry("tsp_b", {}),
        ]
    monkeypatch.setattr(fg, "score_algorithms", fake_score)
    viz = fg.fitness_sanity_gate(tmp_path, [("tsp_a", Path("a.py")), ("tsp_b", Path("b.py"))])
    assert viz and "no finite primary metric" in viz[0]
