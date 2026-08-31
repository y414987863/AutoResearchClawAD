#!/usr/bin/env python3
"""ARC-Bench dispatcher — runs one or all topics in a given mode.

Modes:
  rc_full     — autoclaw full-auto (--auto-approve)
  rc_copilot  — autoclaw co-pilot with interventions/<Txx>.json

For each run:
  1. prepare_run.py injects stage-07/08/09 + checkpoint
  2. python -m researchclaw run --from-stage CODE_GENERATION --to-stage RESULT_ANALYSIS
  3. paper_replication/scripts/paperbench_finalize.py produces submission/*
  4. paper_replication/scripts/judge.py (llm backend) grades the submission
  5. results/<mode>/<Txx>/<run_id>/ keeps EVAL_KEEP only; log/ archives full run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent.parent
BASE_CONFIG = REPO_ROOT / "config.arc.yaml"
INTERVENTIONS_DIR = ROOT / "baseline" / "interventions"
RESULTS_DIR = REPO_ROOT / "artifacts"
LOG_DIR = ROOT / "log"

# Paper-replication harness we reuse. We import the finalizer directly; the
# local judge is invoked as a subprocess because it is self-contained.
PR_ROOT = REPO_ROOT / "experiments" / "paper_replication"
sys.path.insert(0, str(PR_ROOT / "scripts"))

sys.path.insert(0, str(ROOT / "scripts"))
from prepare_run import load_manifest, prepare  # noqa: E402

EVAL_KEEP = {
    "submission",
    "judge_result.json",
    "bench_meta.json",
    "claims.json",
    "RESULTS_README.md",
}


def _load_topics() -> list[dict[str, Any]]:
    """Aggregate topics from every sub-domain registry.

    Post-rename the layout is split: ML topics live in ``config/ml/topics.yaml``,
    physics in ``config/physics/topics.yaml``, biology in ``config/biology/topics.yaml``,
    statistics in ``config/statistics/topics.yaml``, quantum in
    ``config/quantum/topics.yaml``. Legacy ``config/topics.yaml`` (pre-2026-05)
    is honoured if present so old branches keep working.
    """
    registries = [
        ROOT / "config" / "ml" / "topics.yaml",
        ROOT / "config" / "physics" / "topics.yaml",
        ROOT / "config" / "biology" / "topics.yaml",
        ROOT / "config" / "statistics" / "topics.yaml",
        ROOT / "config" / "quantum" / "topics.yaml",
        ROOT / "config" / "topics.yaml",  # legacy fallback
    ]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reg in registries:
        if not reg.is_file():
            continue
        data = yaml.safe_load(reg.read_text(encoding="utf-8")) or {}
        for t in data.get("topics", []):
            tid = t.get("id")
            if tid and tid not in seen:
                seen.add(tid)
                out.append(t)
    return out


def materialize_config(manifest: dict[str, Any], run_dir: Path, mode: str) -> Path:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["project"]["name"] = f"arc-{mode}-{manifest['id']}"
    cfg["project"]["mode"] = "full-auto" if mode == "rc_full" else "semi-auto"

    synthesis = (manifest.get("synthesis") or "").strip().splitlines()
    synopsis_head = " ".join(synthesis[:4])[:400]
    design = manifest.get("experiment_design") or {}
    cfg["research"]["topic"] = (
        f"ARC-Bench {manifest['id']}: {manifest['title']}. "
        f"Research question: {design.get('research_question', manifest['title'])}. "
        f"Context: {synopsis_head}"
    )
    cfg["research"]["domains"] = ["machine-learning", "arc-bench"]

    metrics = design.get("metrics") or []
    if metrics:
        primary = metrics[0]
        cfg["experiment"]["metric_key"] = primary.get("name", "primary_metric")
        cfg["experiment"]["metric_direction"] = primary.get("direction", "maximize")

    if mode == "rc_copilot":
        cfg["project"]["mode"] = "semi-auto"
        cfg["hitl"] = {
            "enabled": True,
            "mode": "co-pilot",
            "timeouts": {
                "default_human_timeout_sec": 86400,
                "auto_proceed_on_timeout": False,
            },
        }

    path = run_dir / "config.yaml"
    path.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
    return path


def run_one(topic_id: str, mode: str, *, dry_run: bool) -> dict[str, Any]:
    manifest = load_manifest(topic_id)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"ab-{mode}-{topic_id}-{ts}"
    run_dir = RESULTS_DIR / mode / topic_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  {run_id}")
    print(f"  Mode:   {mode}")
    print(f"  Topic:  {topic_id} — {manifest.get('title', '?')[:64]}")
    print(f"  Inject: stage-07/08/09 + checkpoint")
    print(f"  Range:  stage 10 (CODE_GENERATION) → stage 14 (RESULT_ANALYSIS)")
    print(f"{'='*72}")

    prepare(topic_id, run_dir)
    config_path = materialize_config(manifest, run_dir, mode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an ARC-Bench cell")
    parser.add_argument("--mode", required=True, choices=["rc_full", "rc_copilot"])
    parser.add_argument("--topic", help="single topic id, e.g. T01")
    parser.add_argument("--all", action="store_true", help="run every topic in topics.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.all:
        topic_ids = [t["id"] for t in _load_topics()]
    elif args.topic:
        topic_ids = [args.topic]
    else:
        parser.error("--topic or --all required")

    for tid in topic_ids:
        run_one(tid, args.mode, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
