"""Unit tests for the LLM4AD comparison → paper integration.

Two defects these pin down, both rooted in the same thing — three consumers
each globbing ``stage-13*/llm4ad_comparison.json`` with their own selection
rule:

_live_only :: ``stage-13_vN/`` are snapshots of attempts a rollback superseded
(``_snapshot_stages()`` renames the old stage-13/ aside before the re-run), so
the unsuffixed dir is the run's real artifact.  Reading the _vN dirs let a
promotion that was rolled back reappear in the paper.

_notice_agrees_with_data :: ``_llm4ad_was_run`` matched *any* stage-13* while
``_collect_llm4ad_comparison`` returned the block for a different one.  When
only a stale _vN existed, Stages 19/22 were ordered "RESTORE this section" and
handed no numbers to restore it from — a direct invitation to fabricate.
"""

from __future__ import annotations

import json
from pathlib import Path

from researchclaw.pipeline.stage_impls._paper_writing import (
    _collect_llm4ad_comparison,
)
from researchclaw.pipeline.stage_impls._review_publish import (
    _llm4ad_preservation_notice,
    _llm4ad_was_run,
)
from researchclaw.pipeline.verified_registry import (
    VerifiedRegistry,
    load_llm4ad_comparison,
)


def _write_cmp(run_dir: Path, stage_dir: str, baseline: float) -> None:
    """Write a one-algorithm comparison artifact under *stage_dir*."""
    d = run_dir / stage_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "llm4ad_comparison.json").write_text(
        json.dumps(
            {
                "metric_direction": "MAXIMIZE",
                "algorithms": {
                    "nelder_mead": {
                        "baseline": baseline,
                        "evolved": baseline * 2,
                        "delta_pct": 100.0,
                        "promoted": True,
                        "reason": "improved",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_loader_reads_live_stage13_not_versioned_snapshots(tmp_path: Path):
    """stage-13/ wins; stage-13_vN/ are superseded attempts and stay invisible."""
    _write_cmp(tmp_path, "stage-13_v2", 1.0)
    _write_cmp(tmp_path, "stage-13_v3", 2.0)
    _write_cmp(tmp_path, "stage-13", 3.0)

    data = load_llm4ad_comparison(tmp_path)
    assert data is not None
    assert data["algorithms"]["nelder_mead"]["baseline"] == 3.0

    block = _collect_llm4ad_comparison(tmp_path)
    assert "baseline=3 -> evolved=6" in block
    assert "baseline=1" not in block and "baseline=2" not in block


def test_rolled_back_promotion_does_not_reach_the_paper(tmp_path: Path):
    """Only a stale _vN survives: no data, and therefore no "preserve" order.

    This is the pairing that matters — a notice without numbers is worse than
    neither, because Stage 19 is told to restore a section it has no source for.
    """
    _write_cmp(tmp_path, "stage-13_v2", 1.0)
    (tmp_path / "stage-13").mkdir()  # re-ran, evolution produced nothing

    assert load_llm4ad_comparison(tmp_path) is None
    assert _collect_llm4ad_comparison(tmp_path) == ""
    assert _llm4ad_was_run(tmp_path) is False
    assert _llm4ad_preservation_notice(tmp_path, audience="reviser") == ""


def test_notice_and_numbers_appear_together(tmp_path: Path):
    """With live data, both the order and its source material are emitted."""
    _write_cmp(tmp_path, "stage-13", 3.0)

    assert _llm4ad_was_run(tmp_path) is True
    for audience in ("reviewer", "reviser", "exporter"):
        assert _llm4ad_preservation_notice(tmp_path, audience=audience)
    assert "baseline=3" in _collect_llm4ad_comparison(tmp_path)


def test_malformed_artifact_is_not_a_run(tmp_path: Path):
    """Unparseable / algorithms-less files must not trigger the preserve order."""
    d = tmp_path / "stage-13"
    d.mkdir()
    (d / "llm4ad_comparison.json").write_text("{not json", encoding="utf-8")
    assert load_llm4ad_comparison(tmp_path) is None

    (d / "llm4ad_comparison.json").write_text('{"algorithms": []}', encoding="utf-8")
    assert load_llm4ad_comparison(tmp_path) is None
    assert _llm4ad_was_run(tmp_path) is False


def test_comparison_numbers_are_registered_as_verifiable(tmp_path: Path):
    """baseline/evolved/delta must verify, or paper_verifier blocks the table."""
    _write_cmp(tmp_path, "stage-13", 3.0)
    data = load_llm4ad_comparison(tmp_path)
    reg = VerifiedRegistry.from_llm4ad_comparison(data, metric_direction="maximize")

    assert reg.is_verified(3.0)      # baseline
    assert reg.is_verified(6.0)      # evolved
    assert reg.is_verified(100.0)    # delta_pct, signed and absolute coincide here
