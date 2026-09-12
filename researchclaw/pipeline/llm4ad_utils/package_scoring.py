"""Stage-14 LLM4AD attribution: score four candidate packages, pick per algorithm.

Stage 13 leaves three things on disk when ``llm4ad_boost`` is on:

* ``legacy_refine_baseline/`` — the refined project, snapshotted *before*
  promotion. The refinement pass rewrote every algorithm (proposed, baseline,
  ablation alike).
* ``evolution_results/<algo>/<algo>.py`` — the winning individual per proposed
  algorithm, evolved from the **clean stage-10** code.
* ``experiment_final/`` — stage-10 plus the evolved modules overlaid.

``experiment_final`` is an unfair comparison: the proposed algorithms carry an
evolution gain while the baselines were reset to un-refined stage-10 code. And
nothing measures the evolved module *against the refined one* — the number the
paper actually needs, because the refined project is the system being reported.

This module answers that, at Stage 14, without touching Stage 13:

    A  stage-10 clean                    → the original baseline
    B  legacy_refine_baseline            → refinement alone
    C  legacy_refine_baseline + evolved  → refinement + LLM4AD
    D  B, with each algorithm replaced by its C counterpart **only if C scored
       strictly better on the shared instances** → the reported system

Every package is scored through the experiment's own ``evaluator.py`` via
``comparison_runner``, so all four numbers come from one code path and are
comparable by construction rather than by inspection.

Everything is additive. A missing ``evolution_results/``, a failed score, an
instance set that does not overlap — each degrades to "keep the refined
algorithm" and is recorded with a reason. The module never invents a number and
never promotes a candidate it could not measure.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess as _sp
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Scoring runs the experiment's evaluator over every instance; the same budget
# Stage 13 allows. A value the caller may override but that should not silently
# differ between the two stages.
_SCORING_TIMEOUT_SEC = 1800

# Package directory names. The letters are load-bearing in the artifact paths
# reviewers are pointed at, so they are named here once.
PKG_CLEAN = "A_stage10"
PKG_REFINE = "B_refine"
PKG_REFINE_LLM4AD = "C_refine_llm4ad"
PKG_FINAL = "D_final"

# Why a per-algorithm decision went the way it did. A closed set so downstream
# code can branch on the cause instead of substring-matching a sentence.
REASON_LLM4AD_BETTER = "llm4ad_better"
REASON_LLM4AD_NOT_BETTER = "llm4ad_not_better"
REASON_NO_EVOLUTION = "no_evolved_candidate"
REASON_IDENTICAL = "evolved_source_identical_to_refined"
REASON_SCORING_FAILED = "scoring_failed"
REASON_NO_SHARED_INSTANCE = "no_instance_scored_by_both"
REASON_COULD_NOT_SCORE_REFINED = "refined_scoring_failed"
#: The refined package could not be scored in full, so its aggregate is a mean
#: over a subset and cannot be compared against a complete one.
REASON_REFINED_INCOMPLETE = "refined_package_incompletely_scored"


@dataclass
class ScoreResult:
    """One package's score, or the reason it has none."""

    # Per-algorithm means — what the package as a whole is worth. Keys are
    # algorithm names.
    values: dict[str, float] | None = None
    # Per-algorithm, per-instance values, kept so a decision between two
    # packages can be made on the instances both of them actually scored.
    # Collapsing to the mean above first would silently compare a mean over
    # one instance set against a mean over another.
    per_algo: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-algorithm instances the evaluator could not score. Kept per algorithm
    # rather than as one package-wide set: an instance name is shared across
    # algorithms, so a union would charge one algorithm's crash to every other
    # algorithm that scored that instance fine.
    per_algo_failures: dict[str, set[str]] = field(default_factory=dict)
    # Algorithms that could not be scored at all. Non-empty means the aggregate
    # is over a *subset*, which is a different quantity from a complete score —
    # a candidate whose module fails to import drops out of its own average.
    unscored_algos: list[str] = field(default_factory=list)
    detail: str = ""
    failures: set[str] | None = None

    @property
    def ok(self) -> bool:
        return bool(self.values)

    @property
    def complete(self) -> bool:
        """True when every algorithm the package holds was scored."""
        return bool(self.values) and not self.unscored_algos

    def mean(self, keys: list[str] | None = None) -> float | None:
        """Mean over *keys* (all values when omitted), or None when unscored."""
        if not self.values:
            return None
        selected = self.values if keys is None else {k: self.values[k] for k in keys}
        if not selected:
            return None
        return sum(selected.values()) / len(selected)


@dataclass
class PackageOutcome:
    """What happened to one of the four packages."""

    name: str
    path: Path | None = None
    built: bool = False
    score: ScoreResult = field(default_factory=ScoreResult)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _runner_path() -> Path:
    return Path(__file__).resolve().parent / "comparison_runner.py"


def score_algorithm(
    exp_dir: Path,
    algo_name: str,
    algo_file: Path,
    *,
    instance_argv: str,
    timeout_sec: int = _SCORING_TIMEOUT_SEC,
    python: str | None = None,
) -> ScoreResult:
    """Score one algorithm file with *exp_dir*'s evaluator, per instance.

    Mirrors Stage 13's scorer: the instance list is passed on stdin so
    enumeration is single-sourced, and the payload is located by marker because
    the generated evaluator may legitimately print of its own accord.

    Never raises — a subprocess that cannot start, times out, or returns
    garbage is a ScoreResult with a reason, because every caller's response is
    the same ("keep the refined code, say why").
    """
    runner = _runner_path()
    if not runner.is_file():  # pragma: no cover - ships beside this module
        return ScoreResult(detail=f"comparison runner missing at {runner}")
    if not algo_file.is_file():
        return ScoreResult(detail=f"algorithm file missing: {algo_file}")

    try:
        proc = _sp.run(
            [python or sys.executable, str(runner), str(exp_dir), algo_name, str(algo_file)],
            input=instance_argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
    except (OSError, _sp.TimeoutExpired) as exc:
        return ScoreResult(detail=f"scoring failed to run: {exc}")

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-200:]
        return ScoreResult(detail=f"scoring exited {proc.returncode}: {tail}")

    payload = _extract_payload(proc.stdout or "")
    if payload is None:
        return ScoreResult(detail="scoring returned unparsable stdout")
    if payload.get("error"):
        return ScoreResult(detail=str(payload["error"]))

    raw = payload.get("values")
    failures = payload.get("failures")
    fset = set(failures) if isinstance(failures, dict) else None
    if not isinstance(raw, dict) or not raw:
        first = next(iter(failures.values()), "") if isinstance(failures, dict) else ""
        return ScoreResult(
            detail=f"no finite primary metric on any instance{f'; e.g. {first}' if first else ''}",
            failures=fset,
        )
    try:
        return ScoreResult(values={str(k): float(v) for k, v in raw.items()}, failures=fset)
    except (TypeError, ValueError):
        return ScoreResult(detail="scoring returned non-numeric values", failures=fset)


def _extract_payload(stdout: str) -> dict[str, Any] | None:
    """Find the runner's JSON object in *stdout*.

    Searched by marker rather than taken as the whole stream: ``evaluator.py``
    is generated code shared with ``main.py``, whose contract requires it to
    print, so stdout legitimately carries the experiment's own chatter. The
    last-line fallback keeps an older runner (which emitted no marker) working.
    """
    from researchclaw.pipeline.llm4ad_task_packages import _RESULT_MARKER

    for line in stdout.splitlines():
        idx = line.find(_RESULT_MARKER)
        if idx == -1:
            continue
        try:
            return json.loads(line[idx + len(_RESULT_MARKER):].strip())
        except json.JSONDecodeError:
            continue

    last = ""
    for line in stdout.splitlines():
        if line.strip():
            last = line.strip()
    if not last:
        return None
    try:
        parsed = json.loads(last)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Package assembly
# ---------------------------------------------------------------------------

def _replace_tree(src: Path, dst: Path) -> None:
    """Copy *src* onto *dst*, replacing anything already there.

    Idempotent by construction: Stage 15 sends the run back to Stage 13 on
    REFINE, so Stage 14 may execute repeatedly over the same directories and a
    stale package must not survive a rerun.
    """
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, symlinks=False)


def _overlay_evolved(
    base_dir: Path,
    dest_dir: Path,
    evolved: dict[str, Path],
) -> list[str]:
    """Copy *base_dir* to *dest_dir*, then overlay *evolved* algorithm modules.

    Returns the algorithms actually overlaid. Only files under
    ``algorithms/<algo>/<algo>.py`` are touched — everything else in the
    project (evaluator, data, helpers) stays exactly as the refined pass left
    it, so the comparison isolates the algorithm module.
    """
    _replace_tree(base_dir, dest_dir)
    applied: list[str] = []
    for algo, src in sorted(evolved.items()):
        if not src.is_file():
            continue
        dst = dest_dir / "algorithms" / algo / f"{algo}.py"
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except OSError as exc:
            logger.warning("Stage 14: could not overlay %s: %s", algo, exc)
            continue
        applied.append(algo)
    return applied


def _discover_evolved(evolution_dir: Path) -> dict[str, Path]:
    """Map algorithm name → evolved module path under ``evolution_results/``.

    The winning individual's worktree holds its module flat at the package
    root because ``version_control.local_path`` is ``algorithms/<algo>``; the
    nested layout is the pre-flattening shape, still read so older runs work.
    """
    found: dict[str, Path] = {}
    if not evolution_dir.is_dir():
        return found
    for pkg in sorted(evolution_dir.iterdir()):
        if not pkg.is_dir():
            continue
        algo = pkg.name
        for candidate in (pkg / f"{algo}.py", pkg / "algorithms" / algo / f"{algo}.py"):
            if candidate.is_file():
                found[algo] = candidate
                break
    return found


def _refined_algo_files(project_dir: Path) -> set[str]:
    """Algorithm names that have a module in *project_dir*."""
    algo_root = project_dir / "algorithms"
    if not algo_root.is_dir():
        return set()
    return {
        d.name
        for d in algo_root.iterdir()
        if d.is_dir() and (d / f"{d.name}.py").is_file()
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_and_score_packages(
    run_dir: Path,
    stage_dir: Path,
    *,
    metric_direction: str = "minimize",
    metric_key: str = "",
    timeout_sec: int = _SCORING_TIMEOUT_SEC,
    python: str | None = None,
    instances: list[Path] | None = None,
) -> dict[str, Any] | None:
    """Build packages A–D, score them, and choose per algorithm.

    Returns the provenance dict (also written to ``algorithm_provenance.json``),
    or ``None`` when the comparison does not apply — ``llm4ad_boost`` was off,
    so no ``legacy_refine_baseline/`` exists and Stage 13 already produced the
    only artifact there is. ``None`` means "leave Stage 14 exactly as it was".

    The returned dict is the single source for the paper's method table, the
    choose-per-algorithm decision, and the WeChat write-up, so it carries the
    four scores as well as the verdict.
    """
    legacy_dir = _find_legacy_baseline(run_dir)
    clean_dir = run_dir / "stage-10" / "experiment"
    # Evolved modules always come from the live Stage 13 alongside the baseline
    # being reported — never a pivot workspace, whose evolution ran against a
    # different starting point.
    _stage13 = _newest_stage13(run_dir)
    evolution_dir = (_stage13 / "evolution_results") if _stage13 else None

    if legacy_dir is None:
        logger.info(
            "Stage 14: no legacy_refine_baseline for this run — llm4ad comparison "
            "does not apply (llm4ad_boost off, or Stage 13 predates the snapshot).",
        )
        return None
    if not clean_dir.is_dir():
        logger.warning(
            "Stage 14: %s missing; cannot build the clean baseline package. "
            "Proceeding with the refined project only.", clean_dir,
        )

    evolved = _discover_evolved(evolution_dir) if evolution_dir else {}
    if evolution_dir is None or not evolution_dir.is_dir():
        logger.info(
            "Stage 14: no evolution_results/ under %s — the evolved package will "
            "equal the refined one and nothing will be replaced.",
            _stage13 or "the live stage-13",
        )
    elif not evolved:
        logger.warning(
            "Stage 14: %s exists but holds no usable algorithm module — treating "
            "this as 'LLM4AD produced nothing'.", evolution_dir,
        )

    packages_root = stage_dir / "packages"
    scores_dir = stage_dir / "scores"
    packages_root.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Instances are resolved per package inside _score_project — each project's
    # evaluator reads its own data/ schema. This only reports the clean
    # project's count, which is what the caller's override refers to.
    if instances is None and not _instances_of(clean_dir) and not _instances_of(legacy_dir):
        logger.warning(
            "Stage 14: no instance files under data/ in %s or %s; the llm4ad "
            "comparison cannot be scored and is skipped.", clean_dir, legacy_dir,
        )
        return None

    direction = metric_direction if metric_direction in ("minimize", "maximize") else "minimize"
    maximize = direction == "maximize"
    refined_algos = _refined_algo_files(legacy_dir)

    outcome: dict[str, PackageOutcome] = {}

    # ── A: the clean stage-10 project ────────────────────────────────────
    if clean_dir.is_dir():
        dst = packages_root / PKG_CLEAN
        _replace_tree(clean_dir, dst)
        outcome[PKG_CLEAN] = PackageOutcome(PKG_CLEAN, dst, built=True)
    else:
        outcome[PKG_CLEAN] = PackageOutcome(PKG_CLEAN, None, built=False)

    # ── B: the refined project, as Stage 13 left it ──────────────────────
    dst_b = packages_root / PKG_REFINE
    _replace_tree(legacy_dir, dst_b)
    outcome[PKG_REFINE] = PackageOutcome(PKG_REFINE, dst_b, built=True)

    # ── C: refined project with the evolved modules overlaid ─────────────
    # Only algorithms the refined project actually defines are overlaid; an
    # evolved module for an algorithm that no longer exists would be copied
    # into a project whose evaluator never refers to it, scoring as dead code.
    overlay = {a: p for a, p in evolved.items() if a in refined_algos}
    skipped = sorted(set(evolved) - set(overlay))
    if skipped:
        logger.warning(
            "Stage 14: evolved module(s) %s have no counterpart in the refined "
            "project — not overlaid.", skipped,
        )
    dst_c = packages_root / PKG_REFINE_LLM4AD
    applied = _overlay_evolved(legacy_dir, dst_c, overlay)
    _overlay_failed = sorted(set(overlay) - set(applied))
    if _overlay_failed:
        logger.warning(
            "Stage 14: could not overlay %s into the evolved package — their "
            "evolved code is absent from package %s.", _overlay_failed, PKG_REFINE_LLM4AD,
        )
    outcome[PKG_REFINE_LLM4AD] = PackageOutcome(PKG_REFINE_LLM4AD, dst_c, built=True)

    # ── Score A, B, C ────────────────────────────────────────────────────
    # Scored from the copied packages, not from the source directories, so the
    # numbers in the provenance belong to the artifacts a reviewer will re-run.
    # A copy that silently came out wrong then shows up as a scoring failure
    # rather than as a score attributed to a package that is not on disk.
    #
    # Each package is scored against its OWN data/: package A holds the clean
    # project and B/C/D the refined one, and the two need not describe instances
    # the same way. The caller's `instances` override applies to A only, since
    # that is the project whose instance set the caller can know about.
    for name in (PKG_CLEAN, PKG_REFINE, PKG_REFINE_LLM4AD):
        out = outcome[name]
        if not out.built or out.path is None:
            continue
        out.score = _score_project(
            out.path, sorted(_refined_algo_files(out.path)) or sorted(refined_algos),
            timeout_sec=timeout_sec, python=python,
            instances=instances if name == PKG_CLEAN else None,
        )

    # ── Per-algorithm decision ───────────────────────────────────────────
    chosen_dir = packages_root / PKG_FINAL
    _replace_tree(legacy_dir, chosen_dir)

    score_b = outcome[PKG_REFINE].score
    score_c = outcome[PKG_REFINE_LLM4AD].score
    algo_records: dict[str, dict[str, Any]] = {}
    n_replaced = 0

    # If the refined package itself did not score in full, nothing in it is
    # comparable: its aggregate covers a subset of algorithms, so a per-algorithm
    # decision would be made against a baseline that was never fully measured.
    # Reported on every algorithm rather than silently proceeding.
    _refine_incomplete = bool(score_b.values) and not score_b.complete
    if _refine_incomplete:
        logger.warning(
            "Stage 14: the refined package scored only %d of its algorithms "
            "(%s unscored) — no per-algorithm decision is possible against an "
            "incomplete baseline. Keeping the refined implementations.",
            len(score_b.values), ", ".join(score_b.unscored_algos),
        )

    for algo in sorted(refined_algos):
        # Per-algorithm, per-instance values for this one algorithm. The
        # decision compares two implementations of the *same* algorithm, so
        # only this algorithm's instances are in play — using the package-wide
        # means here would compare every algorithm against the same number and
        # decide them all identically.
        b_inst = score_b.per_algo.get(algo, {})
        c_inst = score_c.per_algo.get(algo, {})
        a_inst = outcome[PKG_CLEAN].score.per_algo.get(algo, {})

        # `applied`, not `overlay`: an algorithm whose copy failed is not in the
        # evolved package, so whether it is judged "available" must follow what
        # actually landed rather than what was intended.
        evolved_src = overlay.get(algo) if algo in applied else None

        record: dict[str, Any] = {
            "refined": _finite_mean(b_inst),
            "llm4ad": _finite_mean(c_inst),
            "stage10": _finite_mean(a_inst),
            "evolved_available": evolved_src is not None,
        }
        refined_src = legacy_dir / "algorithms" / algo / f"{algo}.py"

        # Ordered cheapest-and-most-definitive first, so the reason recorded is
        # the most specific one that applies.
        if _refine_incomplete and algo in score_b.unscored_algos:
            record["source"] = "refine"
            record["replaced"] = False
            record["reason"] = REASON_REFINED_INCOMPLETE
            algo_records[algo] = record
            continue
        if evolved_src is None:
            record["source"] = "refine"
            record["replaced"] = False
            record["reason"] = REASON_NO_EVOLUTION
        elif _identical_files(evolved_src, refined_src):
            # The evolution returned the refined code unchanged (or the refined
            # pass already converged on it). Nothing to gain, nothing to prove.
            record["source"] = "refine"
            record["replaced"] = False
            record["reason"] = REASON_IDENTICAL
        elif not b_inst:
            record["source"] = "refine"
            record["replaced"] = False
            record["reason"] = REASON_COULD_NOT_SCORE_REFINED
        elif algo in score_c.unscored_algos:
            # The evolved implementation could not be scored *at all* while the
            # refined one could. Refinement may have rebuilt the project — new
            # module names, a different instance schema — and code evolved from
            # the pre-refinement project then simply cannot be loaded into it.
            # That is a portability failure, not a worse algorithm: keep the
            # refined implementation and say so, rather than reporting a
            # comparison that never happened.
            record["source"] = "refine"
            record["replaced"] = False
            record["reason"] = REASON_SCORING_FAILED
        elif not c_inst:
            record["source"] = "refine"
            record["replaced"] = False
            record["reason"] = REASON_SCORING_FAILED
        else:
            # Compare like with like: an algorithm that crashed on instances
            # the refined version solved must not win on the rest.
            common = sorted(set(b_inst) & set(c_inst))
            if not common:
                record["source"] = "refine"
                record["replaced"] = False
                record["reason"] = REASON_NO_SHARED_INSTANCE
                record["n_instances_compared"] = 0
            else:
                ref_score = sum(b_inst[k] for k in common) / len(common)
                l4a_score = sum(c_inst[k] for k in common) / len(common)
                better = l4a_score > ref_score if maximize else l4a_score < ref_score
                record.update({
                    "refined": ref_score,
                    "llm4ad": l4a_score,
                    "delta_pct": (
                        (l4a_score - ref_score) / abs(ref_score) * 100.0
                        if ref_score else None
                    ),
                    "n_instances_compared": len(common),
                    "n_instances_total": max(len(b_inst), len(c_inst)),
                    # Instances this algorithm's evolved version failed to
                    # score but the refined version solved. Taken from THIS
                    # algorithm's failures, not the package's: instance names
                    # are shared across algorithms, so a package-wide set would
                    # report a crash belonging to another algorithm here.
                    "lost_instances": len(
                        score_c.per_algo_failures.get(algo, set()) & set(b_inst)
                    ),
                    "source": "llm4ad" if better else "refine",
                    "replaced": better,
                    "reason": REASON_LLM4AD_BETTER if better else REASON_LLM4AD_NOT_BETTER,
                })
                if better:
                    dst_file = chosen_dir / "algorithms" / algo / f"{algo}.py"
                    try:
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(evolved_src, dst_file)
                        n_replaced += 1
                    except OSError as exc:
                        record["replaced"] = False
                        record["source"] = "refine"
                        record["reason"] = f"copy failed: {exc}"
                        logger.warning("Stage 14: could not replace %s: %s", algo, exc)
        algo_records[algo] = record

    # ── D: score the chosen project ──────────────────────────────────────
    outcome[PKG_FINAL] = PackageOutcome(PKG_FINAL, chosen_dir, built=True)
    outcome[PKG_FINAL].score = _score_project(
        chosen_dir, sorted(refined_algos),
        timeout_sec=timeout_sec, python=python,
    )
    # Run the chosen project so its own numbers exist. Its stdout is parsed by
    # the shared Stage 12 parser, giving canonical condition/metric keys that
    # every downstream reader already understands.
    _run_status, _final_metrics = run_final_package(
        chosen_dir, timeout_sec=timeout_sec, python=python,
    )
    if _run_status != "ok":
        logger.warning(
            "Stage 14: package %s produced no metrics of its own (%s); "
            "downstream stages keep the numbers they already had.",
            PKG_FINAL, _run_status,
        )

    provenance = _assemble_provenance(
        outcome=outcome,
        algo_records=algo_records,
        metric_key=metric_key,
        metric_direction=direction,
        n_replaced=n_replaced,
        n_evolved_available=len(overlay),
        skipped_evolved=skipped,
        overlay_failed=_overlay_failed,
        final_run_status=_run_status,
        instance_count=len(_instances_of(legacy_dir)),
    )
    # Carried back to the caller rather than written as another stage run
    # artifact: a second runs/ directory would compete with Stage 12's in
    # _collect_experiment_results, which picks the best run by primary metric
    # and averages all of them into metrics_summary — so the delivered
    # project's numbers would be blended with, or displaced by, the stale ones.
    if _final_metrics:
        provenance["final_metrics"] = _final_metrics
    else:
        provenance["final_metrics_error"] = _run_status
    _write(packages_root, scores_dir, provenance)
    logger.info(
        "Stage 14: llm4ad attribution — %d/%d algorithm(s) replaced; "
        "refined=%s final=%s",
        n_replaced, len(refined_algos),
        _fmt(score_b.mean()), _fmt(outcome[PKG_FINAL].score.mean()),
    )
    return provenance


def _score_project(
    project_dir: Path,
    algos: list[str],
    *,
    timeout_sec: int,
    python: str | None,
    instances: list[Path] | None = None,
) -> ScoreResult:
    """Score every algorithm in *project_dir* and reduce to one mean.

    The package's own evaluator does the aggregation (it may average over
    seeds internally); this averages the per-instance values it returns, which
    is the same reduction Stage 13's comparison uses.

    Instances come from *project_dir*'s own ``data/`` unless *instances* is
    given, and they must: an algorithm is scored by ITS project's evaluator,
    which reads ITS project's instance schema. Refinement may rewrite the
    instance files outright — a cost budget replacing an evaluation budget, say
    — so feeding one project's instances to another's evaluator fails with a
    KeyError that reads like an algorithm defect rather than a mismatched
    pairing.
    """
    if instances is None:
        instances = _instances_of(project_dir)
    if not instances:
        return ScoreResult(detail=f"no instance files under {project_dir / 'data'}")
    instance_argv = json.dumps([str(p) for p in instances])

    per_algo: dict[str, float] = {}
    per_algo_instances: dict[str, dict[str, float]] = {}
    per_algo_failures: dict[str, set[str]] = {}
    failures: set[str] = set()
    problems: list[str] = []
    unscored: list[str] = []

    for algo in algos:
        algo_file = project_dir / "algorithms" / algo / f"{algo}.py"
        if not algo_file.is_file():
            problems.append(f"{algo}: module missing")
            unscored.append(algo)
            continue
        res = score_algorithm(
            project_dir, algo, algo_file,
            instance_argv=instance_argv, timeout_sec=timeout_sec, python=python,
        )
        if not res.ok:
            problems.append(f"{algo}: {res.detail}")
            unscored.append(algo)
            continue
        if res.failures:
            failures |= res.failures
            per_algo_failures[algo] = set(res.failures)
        if res.detail:
            # score_algorithm puts a reason here for a partial score too, and
            # a partial result that is reported as a clean one is exactly the
            # silence this collector must not produce.
            problems.append(f"{algo}: {res.detail}")
        mean = res.mean()
        if mean is None or not math.isfinite(mean):
            # Never drop an algorithm without saying so: it would shrink
            # n_algorithms with no explanation and make a package look smaller
            # than it is.
            problems.append(f"{algo}: non-finite mean ({_fmt(mean)})")
            unscored.append(algo)
            continue
        per_algo[algo] = mean
        per_algo_instances[algo] = dict(res.values)

    if not per_algo:
        return ScoreResult(detail="; ".join(problems) or "no algorithm could be scored",
                           failures=failures or None)

    # A package that could not be scored *in full* is not comparable with one
    # that was. Its aggregate is the mean over whichever algorithms happened to
    # run, which is not the same quantity: a candidate whose code does not even
    # import leaves its own (bad) number out of the average and so looks better
    # than the baseline it fails to beat. Reported through `incomplete` so the
    # caller can refuse to decide on it rather than silently prefer it.
    detail = "" if not problems else "partial: " + "; ".join(problems)
    return ScoreResult(
        values=per_algo,
        per_algo=per_algo_instances,
        per_algo_failures=per_algo_failures,
        detail=detail,
        failures=failures or None,
        unscored_algos=unscored,
    )


def metrics_from_stdout(stdout: str) -> dict[str, float]:
    """Canonical metric keys from a project's stdout, via the shared parser.

    ``sandbox.parse_metrics`` is the same parser Stage 12 uses on the
    experiment's own output, so the chosen package's numbers land as
    ``condition/.../metric`` in exactly the shape ``VerifiedRegistry`` and the
    table builder already read. Reusing it (rather than reshaping
    ``results.json`` by hand) is what keeps a newly written experiment — whose
    stdout format this parser already understands — working unchanged.

    The parser also emits short alias keys (``cond/metric``, ``metric``) for a
    per-seed line, so each seed overwrites the previous one and what survives is
    the *last seed's* value sitting under a name that reads like the condition's
    average. Those aliases are dropped and rebuilt here from the per-instance
    values, which is the reduction the alias is meant to denote.
    """
    from researchclaw.experiment.sandbox import parse_metrics

    parsed = parse_metrics(stdout)

    # Group the per-instance values (``cond/instance/seed/metric`` in this
    # parser's output) by (cond, metric) — the reduction the short aliases
    # below are meant to denote.
    per_condition: dict[str, dict[str, list[float]]] = {}
    per_metric: dict[str, list[float]] = {}
    for key, value in parsed.items():
        parts = key.split("/")
        if len(parts) != 4:
            continue
        cond, _instance, _seed, metric = parts
        if not math.isfinite(value):
            continue
        per_condition.setdefault(cond, {}).setdefault(metric, []).append(value)
        per_metric.setdefault(metric, []).append(value)

    out: dict[str, float] = {}
    for key, value in parsed.items():
        parts = key.split("/")
        if len(parts) >= 3:
            # Per-instance value: the measurement every aggregate is built from.
            out[key] = value
        elif len(parts) == 2:
            # ``condition/metric``. An alias of the condition's last per-seed
            # line when the metric carries no suffix — dropped, and rebuilt
            # below from that condition's per-instance values. A suffixed name
            # (``cond/metric_mean``) is the project's own stated aggregate.
            if parts[1].endswith(("_mean", "_std", "_median", "_best", "_final")):
                out[key] = value
        elif parts[0].endswith(("_mean", "_std", "_median", "_best", "_final")):
            out[key] = value  # a global aggregate the project stated
        # else: a global alias holding one seed's number; rebuilt below.

    for cond, metrics in per_condition.items():
        for metric, values in metrics.items():
            out[f"{cond}/{metric}"] = sum(values) / len(values)
    for metric, values in per_metric.items():
        out[metric] = sum(values) / len(values)
    return out


def run_final_package(
    project_dir: Path,
    *,
    timeout_sec: int,
    python: str | None,
) -> tuple[str, dict[str, float]]:
    """Run *project_dir*'s own ``main.py`` and return its canonical metrics.

    The per-instance scoring above yields one number per algorithm — enough to
    choose between two implementations, but not the per-condition breakdown the
    paper's tables are built from. The project's ``main.py`` already emits it,
    so the chosen package is run here and its stdout parsed with the *same*
    parser Stage 12 uses. That keeps this path working for an experiment whose
    metrics or naming this code has never seen.

    Returns ``(status, metrics)``. *status* is a human-readable reason for the
    provenance record; *metrics* is empty when the run produced none. Never
    raises — the caller decides what to do with a package that would not run.
    """
    entry = project_dir / "main.py"
    if not entry.is_file():
        return "no main.py in the chosen package", {}
    try:
        proc = _sp.run(
            [python or sys.executable, "main.py"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
    except (OSError, _sp.TimeoutExpired) as exc:
        return f"running main.py failed: {exc}", {}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-200:]
        return f"main.py exited {proc.returncode}: {tail}", {}

    metrics = metrics_from_stdout(proc.stdout or "")
    if not metrics:
        return "main.py produced no parsable metrics", {}
    return "ok", metrics


def _assemble_provenance(
    *,
    outcome: dict[str, PackageOutcome],
    algo_records: dict[str, dict[str, Any]],
    metric_key: str,
    metric_direction: str,
    n_replaced: int,
    n_evolved_available: int,
    skipped_evolved: list[str],
    overlay_failed: list[str],
    final_run_status: str,
    instance_count: int,
) -> dict[str, Any]:
    return {
        "metric_key": metric_key,
        "metric_direction": metric_direction,
        "n_instances": instance_count,
        "decision_rule": (
            "per-algorithm: keep the better of {refined, llm4ad} on the shared "
            "instances, scored by the experiment's own evaluator"
        ),
        "n_algorithms_replaced": n_replaced,
        "n_evolved_modules_available": n_evolved_available,
        "skipped_evolved_modules": skipped_evolved,
        "overlay_failed_modules": overlay_failed,
        "final_results_status": final_run_status,
        "packages": {
            name: {
                "dir": str(out.path) if out.path else None,
                "built": out.built,
                "scored": out.score.ok,
                "score": out.score.mean(),
                "n_algorithms": len(out.score.values) if out.score.values else 0,
                "detail": out.score.detail,
            }
            for name, out in outcome.items()
        },
        "algorithms": algo_records,
        # One flag for the paper and the write-up: did llm4ad change anything?
        "llm4ad_used": n_replaced > 0,
    }


def _write(packages_root: Path, scores_dir: Path, provenance: dict[str, Any]) -> None:
    """Persist the per-package scores and the provenance record."""
    for name, meta in provenance["packages"].items():
        payload = {"package": name, **meta}
        try:
            (scores_dir / f"score_{name}.json").write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - disk failure
            logger.warning("Stage 14: could not write score for %s: %s", name, exc)
    try:
        (packages_root.parent / "algorithm_provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str), encoding="utf-8",
        )
    except OSError as exc:  # pragma: no cover - disk failure
        logger.warning("Stage 14: could not write algorithm_provenance.json: %s", exc)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _newest_stage13(run_dir: Path) -> Path | None:
    """The current ``stage-13`` directory, excluding versioned workspaces.

    ``stage-13_v1``/``_v2`` are *earlier* pivot rounds, not newer ones, and a
    plain reverse name sort puts them after ``stage-13`` — so this applies the
    same rule ``_read_prior_artifact`` does (descending stage number, with
    ``_repair*`` and ``_vN`` filtered out; ``_vN`` sorts *before* the bare name
    because ``ver=0`` comes first under the reverse sort). Picking a pivot
    workspace here would snapshot a superseded refinement as the baseline.
    """
    from researchclaw.pipeline._helpers import _STAGE_NAME_RE, _STAGE_VER_RE

    def _key(p: Path) -> tuple[float, int, str]:
        m = _STAGE_NAME_RE.match(p.name)
        num = float(m.group(1)) if m else float("inf")
        m2 = _STAGE_VER_RE.search(p.name)
        return (num, int(m2.group(1)) if m2 else 0, p.name)

    for candidate in sorted(run_dir.glob("stage-13*"), key=_key, reverse=True):
        if "_repair" in candidate.name or _STAGE_VER_RE.search(candidate.name):
            continue
        if candidate.is_dir():
            return candidate
    return None


def _find_legacy_baseline(run_dir: Path) -> Path | None:
    """The current ``stage-13/legacy_refine_baseline``, when it holds a project.

    Only the live Stage 13 counts: its baseline is the refinement this run is
    actually reporting. A pivot round's snapshot belongs to a superseded
    attempt and comparing against it would measure the wrong thing.
    """
    stage13 = _newest_stage13(run_dir)
    if stage13 is None:
        return None
    candidate = stage13 / "legacy_refine_baseline"
    if candidate.is_dir() and (candidate / "evaluator.py").is_file():
        return candidate
    return None


def _instances_of(project_dir: Path) -> list[Path]:
    """Instance files for *project_dir*, via the shared discovery rule."""
    from researchclaw.pipeline.llm4ad_task_packages import _discover_instances

    return list(_discover_instances(project_dir))


def _identical_files(a: Path, b: Path) -> bool:
    """True when both paths exist and hold the same bytes.

    An unreadable file is reported as *different* so the caller falls through
    to scoring, which is the safe direction: a spurious scoring run costs
    seconds, a spurious "identical" verdict silently drops a real candidate.
    """
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def _finite_mean(values: dict[str, float]) -> float | None:
    """Mean of the finite values, or None when there are none.

    A single non-finite score must not turn a whole algorithm's mean into
    ``nan``: the evaluator already drops those per instance, but a value can
    still arrive as ``inf`` from a package assembled by hand.
    """
    finite = [v for v in values.values() if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


__all__ = [
    "PKG_CLEAN",
    "PKG_FINAL",
    "PKG_REFINE",
    "PKG_REFINE_LLM4AD",
    "ScoreResult",
    "build_and_score_packages",
    "metrics_from_stdout",
    "run_final_package",
    "package_scoring_for_config",
    "score_algorithm",
]


def package_scoring_for_config(
    run_dir: Path,
    stage_dir: Path,
    config: Any,
    *,
    on_error: Callable[[BaseException], None] | None = None,
) -> dict[str, Any] | None:
    """Entry point for Stage 14: run the comparison when ``llm4ad_boost`` is on.

    Returns the provenance dict, or ``None`` when the comparison does not
    apply. A failure inside the comparison is reported through *on_error* and
    swallowed — this is an additive analysis, and losing it must not cost the
    run the paper it was about to write.
    """
    boost = getattr(getattr(config, "experiment", None), "llm4ad_boost", None)
    if not (boost is not None and getattr(boost, "enabled", False)):
        return None

    exp_cfg = getattr(config, "experiment", None)
    try:
        return build_and_score_packages(
            run_dir,
            stage_dir,
            metric_direction=str(getattr(exp_cfg, "metric_direction", "") or "minimize"),
            metric_key=str(getattr(exp_cfg, "metric_key", "") or ""),
        )
    except Exception as exc:  # noqa: BLE001 — analysis is additive, never fatal
        logger.warning("Stage 14: llm4ad attribution failed: %s", exc, exc_info=True)
        if on_error is not None:
            on_error(exc)
        return None
