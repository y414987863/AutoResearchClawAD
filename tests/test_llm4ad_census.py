"""Unit tests for the Stage-13 completeness census.

P0 defect: an algorithm that was *attempted* but produced no evolution_results/
entry vanished from llm4ad_comparison.json with no trace, so a short comparison
list read as "evolution skipped those" (ml25 silently dropped esn and gp).
_failed_census :: _promote_llm4ad_to_experiment_final must emit a ``failed:true``
entry for every attempted algorithm that did not land in evolution_results/ —
even when the whole run failed and no evolution_results/ dir exists at all.
"""

from pathlib import Path

from researchclaw.pipeline.stage_impls._execution import (
    _failed_census,
    _promote_llm4ad_to_experiment_final,
)


def _mk_base(tmp: Path) -> Path:
    """A minimal clean stage-10 experiment: main.py + one algorithm + evaluator."""
    base = tmp / "experiment"
    (base / "algorithms" / "nm").mkdir(parents=True)
    (base / "algorithms" / "nm" / "nm.py").write_text(
        "def optimize(inst, subtree):\n    return 1.0\n"
    )
    (base / "evaluator.py").write_text(
        'PRIMARY_METRIC="m"\ndef load_instance(p):\n    return {}\n'
        'def evaluate_instance(i, s):\n    return {"m": 0.1}\n'
    )
    (base / "main.py").write_text("x=1\n")
    return base


def test_failed_census_marks_all_attempted():
    """A pure mapper: every attempted algorithm gets a failed entry."""
    attempted = {"esn": "package timed out", "gp": "", "mlp": ""}
    census = _failed_census(attempted)
    assert set(census) == {"esn", "gp", "mlp"}
    assert census["esn"]["failed"] is True
    assert census["esn"]["reason"] == "package timed out"
    # A blank message falls back to the generic explanation, not an empty string.
    assert census["gp"]["reason"].startswith("no evolution result")
    assert census["mlp"]["reason"].startswith("no evolution result")


def test_failed_census_empty_attemptset():
    """No attempted algorithms → no census (nothing to be faithful to)."""
    assert _failed_census(None) == {}
    assert _failed_census({}) == {}


def test_promote_missing_evolution_dir_still_reports_failures(tmp_path):
    """Wholesale failure: no evolution_results/ at all, but the run tried 2 algos.

    The old code returned ``(0, {})``; the pipeline then wrote an empty
    comparison. Must now report both algorithms as failed so the comparison is a
    complete census of the run.
    """
    base = _mk_base(tmp_path)
    n, comparison = _promote_llm4ad_to_experiment_final(
        base,
        tmp_path / "evolution_results",  # does not exist
        tmp_path / "experiment_final",
        metric_direction="minimize",
        attempted_algos={"esn": "scoring failed: timed out", "gp": ""},
    )
    assert n == 0
    assert set(comparison) == {"esn", "gp"}
    assert all(v["failed"] is True for v in comparison.values())
    assert comparison["esn"]["reason"] == "scoring failed: timed out"


def test_promote_unions_evolution_dir_into_wholesale_failure(tmp_path):
    """evolution_results/ exists but every package is empty (no <algo>.py).

    The loop skips dirs without an evolvable module, so nothing is recorded; the
    census must pick those dirs up and mark them failed rather than drop them.
    """
    base = _mk_base(tmp_path)
    evo = tmp_path / "evolution_results"
    (evo / "esn").mkdir(parents=True)  # empty — no esn.py to promote
    n, comparison = _promote_llm4ad_to_experiment_final(
        base, evo, tmp_path / "experiment_final",
        metric_direction="minimize",
        attempted_algos={"esn": ""},  # ran, but nothing materialised
    )
    assert n == 0
    assert set(comparison) == {"esn"}
    assert comparison["esn"]["failed"] is True
    assert comparison["esn"]["reason"].startswith("no evolution result")
