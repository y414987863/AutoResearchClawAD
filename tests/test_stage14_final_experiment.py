"""Stage 14 must report Stage 13's decision, not re-derive it.

The defect these cover: Stage 14 used to rebuild four packages from
``stage-13/legacy_refine_baseline`` and re-score them. The refine loop rewrites
``objectives.py``, so code evolved against the stage-10 API could not be
imported into the refined project (``make_objective`` had become
``get_objective``), every evolved algorithm was recorded as ``scoring_failed``,
and the re-run's numbers overwrote the promotion decision Stage 13 had already
made correctly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from researchclaw.pipeline.llm4ad_utils.package_scoring import (
    final_experiment_for_config,
)


# ── Config stubs ────────────────────────────────────────────────────────────

@dataclass
class _Boost:
    enabled: bool = True


@dataclass
class _Exp:
    metric_key: str = "primary_metric"
    metric_direction: str = "minimize"
    llm4ad_boost: _Boost | None = field(default_factory=_Boost)


@dataclass
class _Cfg:
    experiment: _Exp = field(default_factory=_Exp)


# Shaped like the stage-10 code these topics generate: per-seed lines carrying
# the metric name, a per-condition `_mean` line, and a headline global. It is
# NOT shaped like `ndcg_at_10: 0.954850` alone — that variant prints the metric
# name from a variable, and a bare alias for a name the parser sees elsewhere is
# deliberately dropped as "the last seed's value".
_MAIN_PY = """\
print("condition=alpha instance=i0 seed=0 primary_metric: 5.0")
print("condition=alpha instance=i0 primary_metric_mean: 5.0")
print("condition=beta instance=i1 seed=0 primary_metric: 9.0")
print("condition=beta instance=i1 primary_metric_mean: 9.0")
print("primary_metric: 7.0")
"""

_COMPARISON = {
    "generated": "2026-01-01T00:00:00Z",
    "metric_direction": "minimize",
    "n_promoted": 1,
    "algorithms": {
        "alpha": {
            "baseline": 9.0,
            "evolved": 5.0,
            "delta_pct": -44.4,
            "promoted": True,
            "failed": False,
            "reason": "improved",
        },
        "beta": {
            "baseline": 9.0,
            "evolved": 9.5,
            "delta_pct": 5.5,
            "promoted": False,
            "failed": False,
            "reason": "worse",
        },
    },
}


def _make_run(tmp_path: Path, *, with_comparison: bool = True) -> Path:
    run_dir = tmp_path / "run"
    final = run_dir / "stage-13" / "experiment_final"
    final.mkdir(parents=True)
    (final / "main.py").write_text(_MAIN_PY, encoding="utf-8")
    if with_comparison:
        (run_dir / "stage-13" / "llm4ad_comparison.json").write_text(
            json.dumps(_COMPARISON), encoding="utf-8",
        )
    return run_dir


# ── the reader ──────────────────────────────────────────────────────────────

def test_reads_stage13_verdicts_and_runs_the_delivered_project(
    tmp_path: Path,
) -> None:
    run_dir = _make_run(tmp_path)
    out = final_experiment_for_config(
        run_dir, tmp_path / "stage-14", _Cfg(), timeout_sec=120,
    )
    assert out is not None
    assert out["results_status"] == "ok"
    assert out["n_promoted"] == 1
    assert out["n_algorithms"] == 2
    assert set(out["algorithms"]) == {"alpha", "beta"}
    # The verdicts are Stage 13's, read not recomputed.
    assert out["algorithms"]["alpha"]["promoted"] is True
    assert out["algorithms"]["beta"]["promoted"] is False
    assert out["metrics"]["primary_metric"] == pytest.approx(7.0)
    assert out["metrics"]["alpha/i0/0/primary_metric"] == pytest.approx(5.0)


def test_attribution_points_at_stage13_not_a_rebuilt_package(
    tmp_path: Path,
) -> None:
    """The four-package rebuild is what could not import the evolved code, so
    nothing under stage-14/packages/ may be consulted."""
    run_dir = _make_run(tmp_path)
    stage14 = tmp_path / "stage-14"
    out = final_experiment_for_config(run_dir, stage14, _Cfg(), timeout_sec=120)
    assert out is not None
    assert Path(out["package_dir"]) == (
        run_dir / "stage-13" / "experiment_final"
    )
    assert not (stage14 / "packages").exists()
    assert not (stage14 / "scores").exists()


def test_boost_off_is_a_no_op(tmp_path: Path) -> None:
    """Off means the caller keeps Stage 14's long-standing behaviour."""
    run_dir = _make_run(tmp_path)
    cfg = _Cfg(experiment=_Exp(llm4ad_boost=_Boost(enabled=False)))
    assert final_experiment_for_config(
        run_dir, tmp_path / "stage-14", cfg, timeout_sec=120,
    ) is None


def test_missing_experiment_final_is_reported_not_raised(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "stage-13").mkdir(parents=True)
    assert final_experiment_for_config(
        run_dir, tmp_path / "stage-14", _Cfg(), timeout_sec=120,
    ) is None


def test_missing_comparison_yields_no_attribution_but_keeps_metrics(
    tmp_path: Path,
) -> None:
    """A Stage 13 predating llm4ad_comparison.json still gets its numbers read."""
    run_dir = _make_run(tmp_path, with_comparison=False)
    out = final_experiment_for_config(
        run_dir, tmp_path / "stage-14", _Cfg(), timeout_sec=120,
    )
    assert out is not None
    assert out["n_promoted"] == 0
    assert out["algorithms"] == {}
    assert out["metrics"]["primary_metric"] == pytest.approx(7.0)


def test_a_project_that_does_not_run_reports_failure(tmp_path: Path) -> None:
    """The status carries the failure; nothing is substituted for it."""
    run_dir = _make_run(tmp_path)
    (run_dir / "stage-13" / "experiment_final" / "main.py").write_text(
        "import sys\nsys.exit(3)\n", encoding="utf-8",
    )
    out = final_experiment_for_config(
        run_dir, tmp_path / "stage-14", _Cfg(), timeout_sec=120,
    )
    assert out is not None
    assert out["results_status"] != "ok"
    assert out["metrics"] == {}
    # The verdicts are still reported — they are what Stage 13 decided.
    assert out["n_promoted"] == 1


def test_the_artifact_directory_is_not_executed_in_place(tmp_path: Path) -> None:
    """`__pycache__` must not be left in the directory a reviewer is pointed at."""
    run_dir = _make_run(tmp_path)
    final = run_dir / "stage-13" / "experiment_final"
    final_experiment_for_config(
        run_dir, tmp_path / "stage-14", _Cfg(), timeout_sec=120,
    )
    assert not (final / "__pycache__").exists()
