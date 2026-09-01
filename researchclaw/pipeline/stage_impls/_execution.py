"""Stages 11-13: Resource planning, experiment execution, and iterative refinement."""

from __future__ import annotations

import json
import logging
import math
import re
import tempfile
import time as _time
import uuid
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.experiment.validator import (
    CodeValidation,
    format_issues_for_llm,
    validate_code,
)
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._domain import _detect_domain
from researchclaw.pipeline._helpers import (
    StageResult,
    _chat_with_prompt,
    _detect_runtime_issues,
    _ensure_sandbox_deps,
    _extract_code_block,
    _extract_multi_file_blocks,
    _get_evolution_overlay,
    _load_hardware_profile,
    _parse_metrics_from_stdout,
    _read_prior_artifact,
    _safe_filename,
    _safe_json_loads,
    _utcnow_iso,
    _write_stage_meta,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)

# Wall-clock ceiling for a single LLM4AD scoring subprocess (one algorithm
# scored across all instances). 600s was too tight for O(N^3) non-parametric
# candidates (e.g. a GP with gradient ascent on the Gram matrix): the whole
# evaluation has to finish inside this window, so a slow candidate was
# mis-classified as "failed" rather than "slow". 1800s covers a full sweep.
_SCORING_TIMEOUT_SEC = 1800

# Per-file ceiling when a project file is rendered into a refinement prompt.
# Sized just above the largest source file observed across generated projects
# (~50KB), so real code passes through untouched and only a pathological file
# is trimmed.
_CONTEXT_FILE_MAX_CHARS = 60_000


def _execute_resource_planning(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    schedule: dict[str, Any] | None = None
    schedule_source = "template"
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "resource_planning")
        sp = _pm.for_stage("resource_planning", evolution_overlay=_overlay, exp_plan=exp_plan)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        parsed = _safe_json_loads(resp.content, {})
        if (
            isinstance(parsed, dict)
            and isinstance(parsed.get("tasks"), list)
            and parsed["tasks"]
        ):
            schedule = parsed
            schedule_source = "model"
        elif isinstance(parsed, dict):
            logger.warning(
                "Stage 11: model response missing/empty 'tasks' list "
                "(received keys: %s); falling back to template",
                sorted(parsed.keys()),
            )
    if schedule is None:
        schedule = {
            "tasks": [
                {
                    "id": "baseline",
                    "name": "Run baseline",
                    "depends_on": [],
                    "gpu_count": 1,
                    "estimated_minutes": 20,
                    "priority": "high",
                },
                {
                    "id": "proposed",
                    "name": "Run proposed method",
                    "depends_on": ["baseline"],
                    "gpu_count": 1,
                    "estimated_minutes": 30,
                    "priority": "high",
                },
            ],
            "total_gpu_budget": 1,
            "generated": _utcnow_iso(),
        }
    schedule.setdefault("generated", _utcnow_iso())
    schedule["_meta"] = {"source": schedule_source}
    (stage_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2), encoding="utf-8"
    )
    return StageResult(
        stage=Stage.RESOURCE_PLANNING,
        status=StageStatus.DONE,
        artifacts=("schedule.json",),
        evidence_refs=("stage-11/schedule.json",),
    )


def _estimate_stage12_footprint_bytes(run_dir: Path) -> int:
    """Sum the on-disk size of stage-12 and any stage-12_v* siblings."""
    total = 0
    for d in run_dir.glob("stage-12*"):
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    return total


def _execute_experiment_run(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    from researchclaw.experiment.factory import create_sandbox
    from researchclaw.experiment.runner import ExperimentRunner

    schedule_text = _read_prior_artifact(run_dir, "schedule.json") or "{}"
    # Try multi-file experiment directory first, fall back to single file
    exp_dir_path = _read_prior_artifact(run_dir, "experiment/")
    code_text = ""
    if exp_dir_path and Path(exp_dir_path).is_dir():
        main_path = Path(exp_dir_path) / "main.py"
        if main_path.exists():
            try:
                code_text = main_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                code_text = ""
    if not code_text:
        code_text = _read_prior_artifact(run_dir, "experiment.py") or ""

    runs_dir = stage_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    mode = config.experiment.mode

    # ── ColliderAgent physics mode ─────────────────────────────────────
    if mode == "collider_agent":
        from researchclaw.experiment.collider_agent_sandbox import ColliderAgentSandbox

        # Read physics prompt from Stage 10 artifact (collider_plan.md)
        # or fall back to the experiment design plan
        prompt_text = _read_prior_artifact(run_dir, "collider_plan.md") or ""
        if not prompt_text:
            # Try exp_plan.yaml as fallback — Stage 9 artifact
            prompt_text = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
        if not prompt_text:
            logger.warning(
                "Stage 12 (collider_agent): no collider_plan.md found — "
                "using generic placeholder prompt"
            )
            prompt_text = (
                "# Physics Analysis Task\n\n"
                "Run the collider physics pipeline for the configured topic.\n"
                "Generate exclusion contours and output figures to output/figures/.\n"
            )

        ca_cfg = config.experiment.collider_agent
        workspace = runs_dir / (ca_cfg.working_dir or "collider_workspace")

        # Incremental re-entry: snapshot prior workspace under stage-12_v{N}
        # BEFORE the sandbox prepares the new prompt, so the merge step can
        # recover the previous results.json. Only fires when prior workspace
        # is non-empty (models/ or events/ contain artifacts).
        if (
            getattr(ca_cfg, "incremental", False)
            and workspace.is_dir()
            and (
                ((workspace / "models").is_dir() and any((workspace / "models").iterdir()))
                or ((workspace / "events").is_dir() and any((workspace / "events").iterdir()))
            )
        ):
            import shutil as _shutil_inc

            existing_versions = sorted(
                p for p in run_dir.glob("stage-12_v*")
                if p.is_dir() and p.name.replace("stage-12_v", "").isdigit()
            )
            next_v = (
                int(existing_versions[-1].name.replace("stage-12_v", "")) + 1
                if existing_versions
                else 1
            )
            snap_dir = run_dir / f"stage-12_v{next_v}"
            try:
                _shutil_inc.copytree(stage_dir, snap_dir, symlinks=False)
                logger.info(
                    "Incremental snapshot: %s → %s",
                    stage_dir.name,
                    snap_dir.name,
                )
            except OSError as _snap_err:
                logger.warning(
                    "Incremental snapshot failed: %s — proceeding without history",
                    _snap_err,
                )
            else:
                _summary_lines = [
                    f"timestamp: {_utcnow_iso()}",
                    "trigger: incremental re-entry",
                ]
                _prev_results = runs_dir / "results.json"
                if _prev_results.is_file():
                    try:
                        _pr = json.loads(_prev_results.read_text(encoding="utf-8"))
                        _summary_lines.append(
                            f"prior_metrics: {json.dumps(_pr.get('metrics', {}))[:300]}"
                        )
                    except (OSError, json.JSONDecodeError):
                        pass
                (snap_dir / "INCREMENTAL_SNAPSHOT.txt").write_text(
                    "\n".join(_summary_lines) + "\n", encoding="utf-8"
                )
                # Disk-guard: warn (do not abort) when cumulative footprint > 20 GB
                _footprint = _estimate_stage12_footprint_bytes(run_dir)
                _GB = 1024 * 1024 * 1024
                if _footprint > 20 * _GB:
                    logger.warning(
                        "Incremental footprint cumulative across stage-12*/ is "
                        "%.1f GB. Consider `rm -rf %s/stage-12_v*` to reclaim space.",
                        _footprint / _GB,
                        run_dir,
                    )

        workspace.mkdir(parents=True, exist_ok=True)

        sandbox = ColliderAgentSandbox(ca_cfg, workspace)
        result = sandbox.run(prompt_text, timeout_sec=ca_cfg.timeout_sec)

        # Read structured results.json written by ColliderAgentSandbox
        structured_results = None
        results_json_path = workspace / "results.json"
        if results_json_path.exists():
            try:
                import json as _json
                structured_results = _json.loads(results_json_path.read_text(encoding="utf-8"))
                # Copy to runs dir for easy access
                (runs_dir / "results.json").write_text(
                    results_json_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
            except Exception:  # noqa: BLE001
                structured_results = None

        if result.returncode == 0 and not result.timed_out:
            run_status = "completed"
        elif result.timed_out and result.metrics:
            run_status = "partial"
        else:
            run_status = "failed"

        run_payload: dict[str, Any] = {
            "run_id": "run-1",
            "task_id": "collider-agent-main",
            "status": run_status,
            "metrics": result.metrics,
            "elapsed_sec": result.elapsed_sec,
            "stdout": result.stdout[:4000] if result.stdout else "",
            "stderr": result.stderr[:2000] if result.stderr else "",
            "timed_out": result.timed_out,
            "completed_at": _utcnow_iso(),
        }
        if structured_results is not None:
            run_payload["structured_results"] = structured_results

        import json as _json_io
        (runs_dir / "run-1.json").write_text(
            _json_io.dumps(run_payload, indent=2), encoding="utf-8"
        )

        return StageResult(
            stage=Stage.EXPERIMENT_RUN,
            status=StageStatus.DONE,
            artifacts=("runs/",),
            evidence_refs=("stage-12/runs/",),
        )
    # ── End ColliderAgent mode ──────────────────────────────────────────

    if mode in ("sandbox", "docker"):
        # P7: Auto-install missing dependencies before subprocess sandbox
        if mode == "sandbox":
            _all_code = code_text
            if exp_dir_path and Path(exp_dir_path).is_dir():
                for _pyf in Path(exp_dir_path).glob("*.py"):
                    try:
                        _all_code += "\n" + _pyf.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        pass
            _ensure_sandbox_deps(_all_code, config.experiment.sandbox.python_path)

        sandbox = create_sandbox(config.experiment, runs_dir / "sandbox")
        # Use run_project for multi-file, run for single-file
        if exp_dir_path and Path(exp_dir_path).is_dir():
            result = sandbox.run_project(
                Path(exp_dir_path), timeout_sec=config.experiment.time_budget_sec
            )
        else:
            result = sandbox.run(
                code_text, timeout_sec=config.experiment.time_budget_sec
            )
        # Try to read structured results.json from sandbox working dir
        structured_results: dict[str, Any] | None = None
        sandbox_project = runs_dir / "sandbox" / "_project"
        results_json_path = sandbox_project / "results.json"
        if results_json_path.exists():
            try:
                structured_results = json.loads(
                    results_json_path.read_text(encoding="utf-8")
                )
                # Copy results.json to runs dir for easy access
                (runs_dir / "results.json").write_text(
                    results_json_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            except (json.JSONDecodeError, OSError):
                structured_results = None

        # If sandbox metrics are empty, try to parse from stdout
        effective_metrics = result.metrics
        if not effective_metrics and result.stdout:
            effective_metrics = _parse_metrics_from_stdout(result.stdout)

        # Determine run status: completed / partial (timed out with data) / failed
        # R6-2: Detect stdout failure signals even when exit code is 0
        _stdout_has_failure = bool(
            result.stdout
            and not effective_metrics
            and any(
                sig in result.stdout
                for sig in ("FAIL:", "NaN/divergence", "Traceback (most recent")
            )
        )
        if result.returncode == 0 and not result.timed_out and not _stdout_has_failure:
            run_status = "completed"
        elif result.timed_out and effective_metrics:
            run_status = "partial"
            logger.warning(
                "Experiment timed out but captured %d partial metrics",
                len(effective_metrics),
            )
        else:
            run_status = "failed"
            if _stdout_has_failure:
                logger.warning(
                    "Experiment exited cleanly but stdout contains failure signals"
                )

        # P1: Warn if experiment completed suspiciously fast (trivially easy benchmark)
        if run_status == "completed" and result.elapsed_sec and result.elapsed_sec < 5.0:
            logger.warning(
                "Stage 12: Experiment completed in %.2fs — benchmark may be trivially easy. "
                "Consider increasing task difficulty.",
                result.elapsed_sec,
            )

        run_payload: dict[str, Any] = {
            "run_id": "run-1",
            "task_id": "sandbox-main",
            "status": run_status,
            "metrics": effective_metrics,
            "elapsed_sec": result.elapsed_sec,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "completed_at": _utcnow_iso(),
        }
        if structured_results is not None:
            run_payload["structured_results"] = structured_results
        # Auto-generate results.json from parsed metrics if sandbox didn't produce one
        if structured_results is None and effective_metrics:
            auto_results = {"source": "stdout_parsed", "metrics": effective_metrics}
            (runs_dir / "results.json").write_text(
                json.dumps(auto_results, indent=2), encoding="utf-8"
            )
            logger.info("Stage 12: Auto-generated results.json from stdout metrics (%d keys)", len(effective_metrics))
        (runs_dir / "run-1.json").write_text(
            json.dumps(run_payload, indent=2), encoding="utf-8"
        )

        # R11-6: Time budget adequacy check
        if result.timed_out or (result.elapsed_sec and result.elapsed_sec > config.experiment.time_budget_sec * 0.9):
            # Parse stdout to estimate how many conditions/seeds completed
            _stdout = result.stdout or ""
            _completed_conditions = set()
            _completed_seeds = 0
            for _line in _stdout.splitlines():
                if "condition=" in _line and "seed=" in _line:
                    _completed_seeds += 1
                    _cond_match = re.match(r".*condition=(\S+)", _line)
                    if _cond_match:
                        _completed_conditions.add(_cond_match.group(1))
            _time_budget_warning = {
                "timed_out": result.timed_out,
                "elapsed_sec": result.elapsed_sec,
                "budget_sec": config.experiment.time_budget_sec,
                "conditions_completed": sorted(_completed_conditions),
                "total_seed_runs": _completed_seeds,
                "warning": (
                    f"Experiment used {result.elapsed_sec:.0f}s of "
                    f"{config.experiment.time_budget_sec}s budget. "
                    f"Only {len(_completed_conditions)} conditions completed "
                    f"({_completed_seeds} seed-runs). Consider increasing "
                    f"time_budget_sec for more complete results."
                ),
            }
            logger.warning(
                "Stage 12: %s", _time_budget_warning["warning"]
            )
            (stage_dir / "time_budget_warning.json").write_text(
                json.dumps(_time_budget_warning, indent=2), encoding="utf-8"
            )

        # FIX-8: Validate seed count from structured results
        if structured_results and isinstance(structured_results, dict):
            _sr_conditions = structured_results.get("conditions", structured_results.get("per_condition", {}))
            if isinstance(_sr_conditions, dict):
                for _cname, _cdata in _sr_conditions.items():
                    if isinstance(_cdata, dict):
                        _seeds_run = _cdata.get("seeds_run", _cdata.get("n_seeds", 0))
                        if isinstance(_seeds_run, (int, float)) and 0 < _seeds_run < 3:
                            logger.warning(
                                "Stage 12: Condition '%s' ran only %d seed(s) — "
                                "minimum 3 required for statistical validity",
                                _cname, int(_seeds_run),
                            )

    elif mode == "simulated":
        schedule = _safe_json_loads(schedule_text, {})
        tasks = schedule.get("tasks", []) if isinstance(schedule, dict) else []
        if not isinstance(tasks, list):
            tasks = []
        for idx, task in enumerate(tasks or [{"id": "task-1", "name": "simulated"}]):
            task_id = (
                str(task.get("id", f"task-{idx + 1}"))
                if isinstance(task, dict)
                else f"task-{idx + 1}"
            )
            payload = {
                "run_id": f"run-{idx + 1}",
                "task_id": task_id,
                "status": "simulated",
                "key_metrics": {
                    config.experiment.metric_key: round(0.3 + idx * 0.03, 4),
                    "secondary_metric": round(0.6 - idx * 0.04, 4),
                },
                "notes": "Simulated run result",
                "completed_at": _utcnow_iso(),
            }
            run_id = str(payload["run_id"])
            (runs_dir / f"{_safe_filename(run_id)}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
    else:
        runner = ExperimentRunner(config.experiment, runs_dir / "workspace")
        history = runner.run_loop(code_text, run_id=f"exp-{run_dir.name}", llm=llm)
        runner.save_history(stage_dir / "experiment_history.json")
        for item in history.results:
            payload = {
                "run_id": f"run-{item.iteration}",
                "task_id": item.run_id,
                "status": "completed" if item.error is None else "failed",
                "metrics": item.metrics,
                "primary_metric": item.primary_metric,
                "improved": item.improved,
                "kept": item.kept,
                "elapsed_sec": item.elapsed_sec,
                "error": item.error,
                "completed_at": _utcnow_iso(),
            }
            run_id = str(payload["run_id"])
            (runs_dir / f"{_safe_filename(run_id)}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
    # ---- Hard guard: block pipeline when experiment produced no real data ----
    # Issue #165 / fabrication-guard: An experiment that completes in seconds
    # with zero metrics (or only noise) must NOT proceed to paper writing.
    # The old code always returned DONE, which let fabricated papers through.
    _has_real_metrics = False
    if mode in ("sandbox", "docker"):
        # Check that we have at least one non-trivial float metric
        _real_metric_count = sum(
            1 for k, v in (effective_metrics or {}).items()
            if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)
        )
        _has_real_metrics = _real_metric_count > 0
        if not _has_real_metrics and run_status == "failed":
            logger.error(
                "Stage 12: Experiment FAILED and produced zero real metrics. "
                "Refusing to mark as DONE to prevent fabricated results downstream."
            )
            return StageResult(
                stage=Stage.EXPERIMENT_RUN,
                status=StageStatus.FAILED,
                artifacts=("runs/",),
                evidence_refs=("stage-12/runs/",),
                error=(
                    f"Experiment failed with zero real metrics "
                    f"(status={run_status}, elapsed={result.elapsed_sec:.1f}s). "
                    f"Pipeline must not proceed to paper writing without experiment data."
                ),
            )
        if not _has_real_metrics and _stdout_has_failure:
            logger.error(
                "Stage 12: Experiment crashed (failure signals in stdout) with zero "
                "real metrics. Refusing to mark as DONE."
            )
            return StageResult(
                stage=Stage.EXPERIMENT_RUN,
                status=StageStatus.FAILED,
                artifacts=("runs/",),
                evidence_refs=("stage-12/runs/",),
                error=(
                    f"Experiment crashed with failure signals in stdout and zero "
                    f"real metrics (elapsed={result.elapsed_sec:.1f}s). "
                    f"Pipeline must not proceed without experiment data."
                ),
            )
        # Anomaly detection: suspiciously fast completion with empty metrics
        if (
            run_status == "completed"
            and not _has_real_metrics
            and result.elapsed_sec is not None
            and result.elapsed_sec < 30.0
        ):
            logger.error(
                "Stage 12: Experiment 'completed' in %.1fs with zero real metrics "
                "(time_budget=%ds). This is almost certainly a crash that was "
                "misclassified. Refusing to mark as DONE.",
                result.elapsed_sec,
                config.experiment.time_budget_sec,
            )
            return StageResult(
                stage=Stage.EXPERIMENT_RUN,
                status=StageStatus.FAILED,
                artifacts=("runs/",),
                evidence_refs=("stage-12/runs/",),
                error=(
                    f"Experiment 'completed' in {result.elapsed_sec:.1f}s with zero "
                    f"real metrics (budget was {config.experiment.time_budget_sec}s). "
                    f"Likely a misclassified crash. Pipeline must not proceed "
                    f"without experiment data."
                ),
            )
    return StageResult(
        stage=Stage.EXPERIMENT_RUN,
        status=StageStatus.DONE,
        artifacts=("runs/",),
        evidence_refs=("stage-12/runs/",),
    )


def _generate_llm4ad_task_packages(
    stage_dir: Path,
    run_dir: Path,
    config: Any,
    exp_dir_text: str | None,
    log: dict,
) -> tuple[str, ...]:
    """Generate LLM4AD task packages from the stage-10/12 experiment directory.

    Deterministic template generation (no LLM, no evolution). Returns extra
    artifact names to append to the StageResult; log is mutated in place.
    Non-fatal: any failure logs a warning and returns nothing.
    """
    _l4b = getattr(getattr(config, "experiment", None), "llm4ad_boost", None)
    if _l4b is None or not getattr(_l4b, "enabled", False):
        return ()

    try:
        from researchclaw.pipeline.llm4ad_task_packages import (
            generate_task_packages,
        )
    except Exception as _l4b_imp:  # pragma: no cover - defensive
        logger.warning(
            "Stage 13: llm4ad task-package generator unavailable: %s",
            _l4b_imp,
        )
        return ()

    # Inject the live LLM connection settings into each package's config.yaml.
    _llm = getattr(config, "llm", None)
    _llm_config: dict = {}
    if _llm is not None:
        try:
            _llm_config = {
                "base_url": getattr(_llm, "base_url", "") or "",
                "api_key": getattr(_llm, "api_key", "") or "",
                "model": getattr(_llm, "primary_model", "") or "",
                "provider": getattr(_llm, "provider", "") or "",
                "timeout": getattr(_llm, "timeout_sec", None),
            }
            # api_key may live in an env var and be empty on the config object.
            if not _llm_config.get("api_key"):
                _env = getattr(_llm, "api_key_env", "") or ""
                if _env:
                    import os as _os
                    _llm_config["api_key"] = _os.environ.get(_env, "") or ""
        except Exception as _l4b_cfg:
            logger.warning(
                "Stage 13: could not read LLM config for task package: %s",
                _l4b_cfg,
            )
            _llm_config = {}

    # Evolution must start from the CLEAN stage-10 experiment/, never the
    # refined experiment_final/ that ``exp_dir_text`` points at on PIVOT
    # rollback (BUG-58) — that copy may carry LLM-modified data/metrics.
    # Fall back to exp_dir_text only when no clean directory exists.
    _tp_exp = _read_prior_artifact(run_dir, "experiment/") or exp_dir_text
    if not _tp_exp or not Path(_tp_exp).is_dir():
        logger.warning(
            "Stage 13: no experiment directory found for LLM4AD task "
            "package generation (%s); skipping", _tp_exp,
        )
        return ()

    try:
        _tp_out = stage_dir / "task_packages"
        _evo_cfg = _l4b_dataclass_to_dict(getattr(_l4b, "evolution", None))
        _res_cfg = _l4b_dataclass_to_dict(getattr(_l4b, "resources", None))
        # Live topic from config.arc.yaml becomes the LLM4AD `background` the
        # sampler feeds to the LLM; the explicit metric_direction overrides the
        # name-based inference so evolution optimises the right way.
        _topic = getattr(getattr(config, "research", None), "topic", "") or ""
        _direction = getattr(getattr(config, "experiment", None), "metric_direction", "") or ""
        # Fresh per-call token: run_dir.name is stable across repeated runs, so a
        # module-level token would reuse the same temp workspace for every entry
        # into Stage 13 (including a same-process re-entry after a rollback),
        # making _resolve_run_best read a stale best. Generating it here ties
        # gen + collect to the same new token per package-generation call.
        _l4b_token = uuid.uuid4().hex[:8]
        _manifests = generate_task_packages(
            Path(_tp_exp), _tp_out, _llm_config, _evo_cfg, _res_cfg,
            background=_topic, metric_direction=_direction,
            # Worktrees live under the temp dir (not task_packages/, whose deep
            # path hits Windows' 260-char limit), scoped to this invocation.
            runs_base_dir=Path(tempfile.gettempdir()) / "rc_llm4ad" / run_dir.name / f"run_{_l4b_token}",
            run_id=_l4b_token,
        )
        logger.info(
            "Stage 13: generated %d LLM4AD task package(s) under %s",
            len(_manifests), _tp_out,
        )
        log["task_packages"] = {
            "dir": str(_tp_out),
            "count": len(_manifests),
            "packages": [
                {
                    "algo": m.algo,
                    "path": m.path,
                    "primary_metric": m.primary_metric,
                    "metric_direction": m.metric_direction,
                    "n_instances": m.n_instances,
                }
                for m in _manifests
            ],
        }

        # --- Run LLM4AD evolution on the generated packages (optional) ---
        artifact_extra = _run_llm4ad_evolution(stage_dir, _tp_out, config, log)
        return ("task_packages/",) + artifact_extra
    except Exception as _l4b_exc:
        # exc_info matters here: this handler previously swallowed a plain
        # NameError in the evolution path and reported it as a business
        # failure, so the bug survived every run without a traceback.
        logger.warning(
            "Stage 13: LLM4AD task-package generation failed: %s",
            _l4b_exc,
            exc_info=True,
        )
        log["task_packages_error"] = f"{type(_l4b_exc).__name__}: {_l4b_exc}"
        if not bool(getattr(_l4b, "fail_silently", True)):
            raise
        return ()


def _l4b_dataclass_to_dict(obj: Any) -> dict:
    """Shallow-convert a frozen llm4ad_boost sub-config to a plain dict."""
    if obj is None:
        return {}
    try:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(obj):
            return asdict(obj)
    except Exception:  # noqa: BLE001 - defensive, config shape is user-supplied
        pass
    return {
        k: getattr(obj, k)
        for k in dir(obj)
        if not k.startswith("_") and not callable(getattr(obj, k, None))
    }


def _resolve_llm4ad_cmd(config: Any) -> str:
    """Resolve the ``llm4ad`` executable.

    Prefers an on-PATH ``llm4ad``; on Windows, falls back to the venv's
    Scripts/llm4ad.exe if present. Returns the command name otherwise so the
    Runner reports a clean 'not found' error.
    """
    import shutil as _sh

    found = _sh.which("llm4ad")
    if found:
        return found
    # Windows venv layout: <venv>/Scripts/llm4ad.exe (or equivalent on PATH).
    _py = getattr(getattr(config, "experiment", None), "sandbox", None)
    _py_path = getattr(_py, "python_path", "") or ""
    if _py_path:
        exe = Path(_py_path).resolve().parent / ("llm4ad.exe" if Path(_py_path).suffix == ".exe" else "llm4ad")
        if exe.exists():
            return str(exe)
        # python_path may be a forward-slash or bare name; try the venv root.
        venv = Path(_py_path).resolve().parent
        for cand in (venv / "llm4ad.exe", venv / "llm4ad"):
            if cand.exists():
                return str(cand)
    return "llm4ad"


def _run_llm4ad_evolution(
    stage_dir: Path, packages_dir: Path, config: Any, log: dict
) -> tuple[str, ...]:
    """Run llm4ad evolution on task packages; degrade gracefully on failure.

    Returns extra artifact names ('' or ('evolution_results/',)). Mutates log.
    Honors ``config.experiment.llm4ad_boost.fail_silently`` and ``resources``.
    """
    import shutil

    _l4b = getattr(getattr(config, "experiment", None), "llm4ad_boost", None)
    if _l4b is None or not getattr(_l4b, "enabled", False):
        return ()
    if not packages_dir.is_dir() or not any(packages_dir.glob("*/config.yaml")):
        logger.info("Stage 13: no task packages to evolve; skipping LLM4AD evolution")
        return ()

    try:
        from researchclaw.pipeline.llm4ad_task_packages import (
            run_evolution_on_packages,
        )
    except Exception as _eval_imp:  # pragma: no cover - defensive
        logger.warning("Stage 13: llm4ad evolution runner unavailable: %s", _eval_imp)
        return ()

    cmd = _resolve_llm4ad_cmd(config)
    _res = getattr(_l4b, "resources", None)
    total_budget = int(getattr(_res, "time_budget_sec", 1800) or 1800)
    fail_silently = bool(getattr(_l4b, "fail_silently", True))

    # Per-package wall-clock budget. When ``per_package_timeout_sec`` is set
    # (> 0) it wins — the real failure mode last run was NOT a small budget but
    # every package being killed mid-generation because total_budget was divided
    # across N packages (1800//6 = 300s, less than one generation). A long
    # evolution deserves its own ceiling per package; otherwise fall back to the
    # old total-budget division so prior configs keep working unchanged.
    _n_pkgs = max(1, len(list(packages_dir.glob("*/config.yaml"))))
    _per_pkg = int(getattr(_res, "per_package_timeout_sec", 0) or 0)
    if _per_pkg > 0:
        timeout = max(60, _per_pkg)
    else:
        timeout = max(60, total_budget // _n_pkgs)

    # Provider env vars: pass through (base_url/api_key already in config.yaml,
    # but keep ambient keys available to the subprocess).
    env: dict[str, Any] = {}
    _llm = getattr(config, "llm", None)
    if _llm is not None and getattr(_llm, "api_key", ""):
        env["OPENAI_API_KEY"] = getattr(_llm, "api_key", "")
    if _llm is not None and getattr(_llm, "api_key_env", "") and getattr(_llm, "api_key_env", "") not in env:
        import os as _os
        _k = getattr(_llm, "api_key_env", "")
        if _os.environ.get(_k):
            env[_k] = _os.environ[_k]

    logger.info(
        "Stage 13: running LLM4AD evolution on %d package(s) under %s "
        "(cmd=%s, per-package timeout=%ds%s)",
        _n_pkgs, packages_dir, cmd, timeout,
        f" of {total_budget}s total" if _per_pkg <= 0 else "",
    )
    try:
        results = run_evolution_on_packages(
            packages_dir, llm4ad_cmd=cmd, timeout_sec=timeout, env=env or None
        )
    except Exception as _eval_exc:
        logger.warning(
            "Stage 13: LLM4AD evolution failed: %s", _eval_exc, exc_info=True
        )
        log["evolution_error"] = f"{type(_eval_exc).__name__}: {_eval_exc}"
        if not fail_silently:
            raise
        return ()

    ok = [r for r in results if r.success]
    # LLM4AD's MetricType.MINIMIZE negates the metric in compute_score, so the
    # logged best_score is sign-flipped vs the raw objective. Surface both + the
    # configured direction: a best_score far ABOVE the baseline (or 0.0 from a
    # failed run) means evolution optimised the WRONG way. Print the raw metric
    # so we can sanity-check the sign ourselves.
    _direction_cfg = str(
        getattr(getattr(config, "experiment", None), "metric_direction", "") or ""
    ).strip().upper()
    for r in ok:
        if r.best_score is None:
            continue
        # Log whatever metrics the evaluator reported rather than looking up one
        # hard-coded key: the metric name is the generated experiment's choice
        # (`tour_length`, `accuracy`, …), so naming one here printed "n/a" for
        # every task that is not continuous box-constrained optimisation.
        _raw = ", ".join(
            f"{k}={v:.6g}" for k, v in sorted((r.best_metrics or {}).items())
        )
        logger.info(
            "Stage 13: %s best_score=%.6g (metrics: %s; configured direction=%s) — "
            "MINIMIZE negates the metric, so best_score <= 0 is expected there",
            r.algo, r.best_score,
            _raw or "none reported",
            _direction_cfg or "inferred",
        )
    log["evolution"] = {
        "cmd": cmd,
        "timeout_sec": timeout,
        "total_budget_sec": total_budget,
        "metric_direction": _direction_cfg,
        "total": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "results": [
            {
                "algo": r.algo,
                "success": r.success,
                "best_score": r.best_score,
                "best_metrics": r.best_metrics,
                "best_code_dir": r.best_code_dir,
                "run_id": r.run_id,
                "error_message": r.error_message,
            }
            for r in results
        ],
    }
    if not ok:
        logger.warning(
            "Stage 13: LLM4AD evolution produced no successes (%d packages "
            "attempted); first error: %s",
            len(results),
            next((r.error_message for r in results if r.error_message), "n/a"),
        )
        return ()

    # Report the scores explicitly — a run whose best_score is None produced no
    # measurable evidence of improvement, which is worth surfacing loudly.
    _scored = [r for r in ok if r.best_score is not None]
    if not _scored:
        logger.warning(
            "Stage 13: LLM4AD evolution succeeded but no best_score was "
            "recovered (no best/metadata.json) — the run has no quantitative "
            "result to report downstream"
        )
    else:
        for r in _scored:
            logger.info(
                "Stage 13: LLM4AD best score for %s: %.6g (run_id=%s)",
                r.algo, r.best_score, r.run_id,
            )

    # Materialise best evolved code to a stable evolution_results/ dir.
    evo_out = stage_dir / "evolution_results"
    evo_out.mkdir(parents=True, exist_ok=True)
    # Copy the winning individual's CODE only: the source worktree may sit
    # under a runs/ tree carrying .git and the run history, none of which
    # belongs in the deliverable — and neither does __pycache__.
    _tree_ignore = shutil.ignore_patterns(".git", "runs", "best", "__pycache__", "*.pyc")
    for r in results:
        if not r.success or not r.best_code_dir:
            continue
        src = Path(r.best_code_dir)
        if not src.is_dir():
            continue
        dst = evo_out / r.algo
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=_tree_ignore)
    log["evolution"]["results_dir"] = str(evo_out)
    logger.info(
        "Stage 13: LLM4AD evolution succeeded for %d/package(s); results under %s",
        len(ok), evo_out,
    )
    return ("evolution_results/",)


def _promote_llm4ad_to_experiment_final(
    base_exp_dir: Path,
    evolution_dir: Path,
    final_dir: Path,
    *,
    metric_direction: str = "minimize",
) -> tuple[int, dict[str, Any]]:
    """Overlay only genuinely-improved evolved modules onto a clean project.

    Each ``evolution_results/<algo>/`` is a copy of the winning individual's git
    worktree — which, because ``version_control.local_path`` is
    ``algorithms/<algo>``, holds just the evolved ``<algo>.py``. Downstream stages
    consume ``experiment_final/`` as a top-level project (``main.py`` +
    ``algorithms/<algo>/<algo>.py`` + ``data/*.json``), so we rebuild ``final_dir``
    from ``base_exp_dir`` (the clean stage-10 code) and overlay ONLY the evolved
    modules that actually beat their clean baseline under the experiment's own
    evaluator protocol.  A module that fails to run, has no finite primary
    metric, or is equal/worse than the baseline is left as the clean code, so
    LLM4AD can only ever improve ``experiment_final/`` — never degrade it.

    ``metric_direction`` ("minimize"/"maximize") decides which score is better.

    Returns ``(n_promoted, comparison)`` where ``comparison`` is a dict of
    ``{algo: {baseline, evolved, delta_pct, promoted, failed, reason}}`` for
    every algo present in ``evolution_dir``.
    """
    import shutil as _sh_promote
    import subprocess as _sp
    import sys as _sys

    _dir = metric_direction.strip().lower()
    maximize = _dir == "maximize"
    if _dir not in ("minimize", "maximize"):
        logger.warning(
            "Stage 13: unknown metric_direction=%r, assuming minimize", metric_direction,
        )

    if not base_exp_dir.is_dir():
        logger.warning(
            "Stage 13: cannot promote llm4ad results — base experiment dir "
            "missing: %s", base_exp_dir,
        )
        return 0, {}

    # Rebuild final_dir from the clean base (remove errant leftover, if any).
    if final_dir.exists():
        _sh_promote.rmtree(final_dir, ignore_errors=True)
    _sh_promote.copytree(base_exp_dir, final_dir, symlinks=False)

    _runner = Path(__file__).resolve().parent.parent / "llm4ad_utils" / "comparison_runner.py"

    # Enumerating instances has one implementation, shared with the task
    # packager: every file directly under data/, whatever its extension.
    from researchclaw.pipeline.llm4ad_task_packages import _discover_instances
    _instance_argv = json.dumps([str(p) for p in _discover_instances(base_exp_dir)])

    def _score(algo_name: str, algo_file: Path) -> tuple[dict[str, float] | None, str]:
        """Per-instance primary-metric values via the experiment's own evaluator.

        Returns ``(values, detail)``.  ``values`` is None when scoring could not
        run at all; ``detail`` explains why, and is surfaced in the comparison
        artifact so a failure is not mistaken for "no improvement".
        """
        try:
            proc = _sp.run(
                [_sys.executable, str(_runner), str(base_exp_dir), algo_name, str(algo_file)],
                input=_instance_argv,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=_SCORING_TIMEOUT_SEC,
            )
        except (OSError, _sp.TimeoutExpired) as _pexc:
            return None, f"scoring failed to run: {_pexc}"
        if proc.returncode != 0:
            return None, (
                f"scoring exited {proc.returncode}: "
                f"{(proc.stderr or proc.stdout or '').strip()[-200:]}"
            )
        payload = _safe_json_loads(proc.stdout.strip(), None)
        if not isinstance(payload, dict):
            return None, "scoring returned unparsable stdout"
        if payload.get("error"):
            return None, str(payload["error"])
        values = payload.get("values")
        if not isinstance(values, dict) or not values:
            _failed = payload.get("failures") or {}
            _first = next(iter(_failed.values()), "") if isinstance(_failed, dict) else ""
            return None, f"no finite primary metric on any instance{f'; e.g. {_first}' if _first else ''}"
        try:
            return {str(k): float(v) for k, v in values.items()}, ""
        except (TypeError, ValueError):
            return None, "scoring returned non-numeric values"

    comparison: dict[str, Any] = {}
    n_promoted = 0
    if not evolution_dir.is_dir():
        return 0, comparison

    for algo_pkg in sorted(evolution_dir.iterdir()):
        if not algo_pkg.is_dir():
            continue
        algo_name = algo_pkg.name
        # `evolution_results/<algo>/` is a copy of the winning individual's git
        # worktree, and the worktree is a copy of `version_control.local_path` =
        # `algorithms/<algo>`. So the evolved module sits FLAT at the root here.
        # The nested path is the pre-flattening layout, kept so results produced
        # by an older package build still promote.
        evolved_src = algo_pkg / f"{algo_name}.py"
        if not evolved_src.is_file():
            evolved_src = algo_pkg / "algorithms" / algo_name / f"{algo_name}.py"
        if not evolved_src.is_file():
            continue

        # Map back to the original layout: algorithms/<algo>/<algo>.py.
        base_algo = base_exp_dir / "algorithms" / algo_name / f"{algo_name}.py"
        _dest = final_dir / "algorithms" / algo_name / f"{algo_name}.py"
        _dest.parent.mkdir(parents=True, exist_ok=True)

        if not base_algo.is_file():
            # No clean counterpart — cannot judge improvement, keep clean code.
            comparison[algo_name] = {
                "baseline": None, "evolved": None, "delta_pct": None,
                "promoted": False, "failed": True,
                "reason": "baseline algorithm file missing",
            }
            continue

        # Byte-identical source means no evolvable change: keep the clean code.
        try:
            identical = evolved_src.read_bytes() == base_algo.read_bytes()
        except OSError:
            identical = False

        if identical:
            comparison[algo_name] = {
                "baseline": None, "evolved": None, "delta_pct": None,
                "promoted": False, "failed": False,
                "reason": "evolved source identical to baseline; no change to promote",
            }
            logger.info("Stage 13: %s unchanged — keeping baseline", algo_name)
            continue

        baseline_values, baseline_detail = _score(algo_name, base_algo)
        evolved_values, evolved_detail = _score(algo_name, evolved_src)

        # A failed/unsaved evolution counts as "no improvement": never degrade.
        # Which side failed matters — "evolved failed" and "baseline failed"
        # need different follow-up, and reporting one as the other sent an
        # earlier investigation down the wrong path.
        if baseline_values is None or evolved_values is None:
            if baseline_values is None and evolved_values is None:
                _reason = (
                    f"scoring failed on both sides (baseline: {baseline_detail}; "
                    f"evolved: {evolved_detail})"
                )
            elif evolved_values is None:
                _reason = f"evolved scoring failed ({evolved_detail})"
            else:
                _reason = f"baseline scoring failed ({baseline_detail})"
            comparison[algo_name] = {
                "baseline": None, "evolved": None, "delta_pct": None,
                "promoted": False, "failed": True, "reason": _reason,
            }
            logger.warning("Stage 13: %s — %s; keeping baseline", algo_name, _reason)
            continue

        # Compare like with like.  If the candidate crashed on an instance the
        # baseline solved, averaging each over its own instance set compares two
        # different quantities and can promote a strictly worse algorithm.
        common = sorted(set(baseline_values) & set(evolved_values))
        if not common:
            comparison[algo_name] = {
                "baseline": None, "evolved": None, "delta_pct": None,
                "promoted": False, "failed": True,
                "reason": "no instance was scored by both baseline and evolved",
            }
            logger.warning(
                "Stage 13: %s — no shared scored instance; keeping baseline", algo_name,
            )
            continue

        baseline_score = sum(baseline_values[k] for k in common) / len(common)
        evolved_score = sum(evolved_values[k] for k in common) / len(common)
        n_total = max(len(baseline_values), len(evolved_values))

        # Direction-aware improvement test (minimize → lower is better).
        better = evolved_score < baseline_score if not maximize else evolved_score > baseline_score
        if better:
            _sh_promote.copy2(evolved_src, _dest)
            n_promoted += 1
        delta_pct = (
            (evolved_score - baseline_score) / baseline_score * 100.0
            if baseline_score
            else None
        )
        comparison[algo_name] = {
            "baseline": baseline_score,
            "evolved": evolved_score,
            "delta_pct": delta_pct,
            "promoted": better,
            "failed": False,
            "n_instances_compared": len(common),
            "n_instances_total": n_total,
            "reason": "improved" if better else "equal or worse than baseline",
        }
        logger.info(
            "Stage 13: %s %s (baseline=%.6g evolved=%.6g delta=%.3f%% over %d/%d instances)",
            algo_name, "promoted" if better else "kept baseline",
            baseline_score, evolved_score, delta_pct if delta_pct is not None else 0.0,
            len(common), n_total,
        )

    if not n_promoted:
        logger.warning(
            "Stage 13: no evolved algorithm beat its baseline — experiment_final/ "
            "left as the clean stage-10 code"
        )
    return n_promoted, comparison


def _execute_iterative_refine(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    from researchclaw.experiment.factory import create_sandbox

    def _to_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            f = float(value)
            # BUG-EX-01: NaN/Inf block all future improvement detection
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except (TypeError, ValueError):
            return None

    # Agent-based modes (collider_agent, biology_agent, stat_agent): no Python
    # refinement loop — the agent handled the full pipeline atomically in
    # Stage 12 and wrote a canonical results.json.  "Refining" python
    # source files that were never executed is wasted work; the only
    # meaningful refinement option is re-invoking the agent (which the
    # repair loop in pipeline/runner.py handles separately).  Create
    # placeholder artifacts and exit so downstream stages see a non-empty
    # experiment_final/.
    if config.experiment.mode in ("collider_agent", "biology_agent", "stat_agent"):
        agent_label = config.experiment.mode
        agent_pretty = {
            "collider_agent": "ColliderAgent",
            "biology_agent": "Biology-Agent",
            "stat_agent": "stat_research_agent",
        }.get(agent_label, agent_label)
        logger.info(
            "Stage 13: Skipping iterative refinement in %s mode "
            "(%s pipeline completed in Stage 12)",
            agent_label, agent_pretty,
        )
        import shutil as _shutil

        final_dir = stage_dir / "experiment_final"
        final_dir.mkdir(exist_ok=True)

        # Copy Stage 12 run artifacts into experiment_final/ for downstream stages
        runs_artifact = _read_prior_artifact(run_dir, "runs/")
        if runs_artifact and Path(runs_artifact).is_dir():
            for _item in Path(runs_artifact).iterdir():
                _dst = final_dir / _item.name
                if _item.is_file():
                    _shutil.copy2(_item, _dst)
        else:
            (final_dir / f"{agent_label}_results.md").write_text(
                f"# {agent_pretty} Results\n\nExperiment executed via {agent_pretty} in Stage 12.\n",
                encoding="utf-8",
            )

        log: dict[str, Any] = {
            "generated": _utcnow_iso(),
            "mode": agent_label,
            "skipped": True,
            "skip_reason": (
                f"Iterative refinement not applicable in {agent_label} mode — "
                f"{agent_pretty} ran the full pipeline in Stage 12"
            ),
            "metric_key": config.experiment.metric_key,
        }
        (stage_dir / "refinement_log.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )
        return StageResult(
            stage=Stage.ITERATIVE_REFINE,
            status=StageStatus.DONE,
            artifacts=("refinement_log.json", "experiment_final/"),
            evidence_refs=("stage-13/refinement_log.json",),
        )

    # R10-Fix3: Skip iterative refinement in simulated mode (no real execution)
    if config.experiment.mode == "simulated":
        logger.info(
            "Stage 13: Skipping iterative refinement in simulated mode "
            "(no real code execution available)"
        )
        import shutil

        final_dir = stage_dir / "experiment_final"
        # Copy latest experiment code as final (directory or single file)
        copied = False
        for stage_num in (12, 10):
            src_dir = run_dir / f"stage-{stage_num:02d}" / "experiment"
            if src_dir.is_dir():
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                shutil.copytree(src_dir, final_dir)
                copied = True
                break
            # Also check for single experiment.py
            src_file = run_dir / f"stage-{stage_num:02d}" / "experiment.py"
            if src_file.is_file():
                (stage_dir / "experiment_final.py").write_text(
                    src_file.read_text(encoding="utf-8"), encoding="utf-8"
                )
                copied = True
                break

        log: dict[str, Any] = {
            "generated": _utcnow_iso(),
            "mode": "simulated",
            "skipped": True,
            "skip_reason": "Iterative refinement not meaningful in simulated mode",
            "metric_key": config.experiment.metric_key,
        }
        (stage_dir / "refinement_log.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )
        return StageResult(
            stage=Stage.ITERATIVE_REFINE,
            status=StageStatus.DONE,
            artifacts=("refinement_log.json",),
            evidence_refs=(),
        )

    metric_key = config.experiment.metric_key
    metric_direction = config.experiment.metric_direction

    # P9: Detect metric direction mismatch between config and experiment code.
    # The code-gen stage instructs experiments to print a line like:
    #   METRIC_DEF: primary_metric | direction=higher | desc=...
    # Log a warning if mismatch is detected, but trust the config value
    # (BUG-06 fix: no longer auto-override, since Stage 9 and 12 now
    # explicitly enforce config.metric_direction in prompts).
    _runs_dir_detect = _read_prior_artifact(run_dir, "runs/")
    if _runs_dir_detect and Path(_runs_dir_detect).is_dir():
        import re as _re_detect

        for _rf in sorted(Path(_runs_dir_detect).glob("*.json"))[:5]:
            try:
                _rp = _safe_json_loads(_rf.read_text(encoding="utf-8"), {})
                _stdout = _rp.get("stdout", "") if isinstance(_rp, dict) else ""
                _match = _re_detect.search(
                    r"METRIC_DEF:.*direction\s*=\s*(higher|lower)", _stdout
                )
                if _match:
                    _detected = _match.group(1)
                    _detected_dir = "maximize" if _detected == "higher" else "minimize"
                    if _detected_dir != metric_direction:
                        logger.warning(
                            "P9: Metric direction mismatch — config says '%s' but "
                            "experiment code declares 'direction=%s'. "
                            "Keeping config value '%s'. Code will be "
                            "corrected in next refinement cycle.",
                            metric_direction,
                            _detected,
                            metric_direction,
                        )
                    break
            except OSError:
                pass

    maximize = metric_direction == "maximize"

    def _is_better(candidate: float | None, current: float | None) -> bool:
        if candidate is None:
            return False
        if current is None:
            return True
        return candidate > current if maximize else candidate < current

    def _find_metric(metrics: dict[str, object], key: str) -> float | None:
        """R13-4: Find metric value with fuzzy key matching.

        Tries exact match first, then looks for aggregate keys that contain
        the metric name (e.g. 'primary_metric_mean' when key='primary_metric').
        """
        # Exact match
        val = _to_float(metrics.get(key))
        if val is not None:
            return val
        # Try aggregate/mean keys containing the metric name
        # Prefer keys ending with the metric name or containing '_mean'
        candidates: list[tuple[str, float]] = []
        for mk, mv in metrics.items():
            fv = _to_float(mv)
            if fv is None:
                continue
            if mk == key or mk.endswith(f"/{key}"):
                return fv  # Exact match via condition prefix
            if key in mk and ("mean" in mk or "avg" in mk):
                candidates.append((mk, fv))
            elif mk.endswith(f"_{key}") or mk.endswith(f"/{key}_mean"):
                candidates.append((mk, fv))
        if candidates:
            # Take the aggregate mean if available, otherwise first match
            for ck, cv in candidates:
                if "mean" in ck:
                    return cv
            return candidates[0][1]
        # Last resort: if there's an "overall" or root-level aggregate
        for mk, mv in metrics.items():
            fv = _to_float(mv)
            if fv is not None and key in mk and "/" not in mk and "seed" not in mk:
                return fv
        return None

    requested_iterations = int(getattr(config.experiment, "max_iterations", 10) or 10)
    max_iterations = max(1, min(requested_iterations, 10))

    # BUG-57: Wall-clock time cap for the entire refinement stage.
    # Default: 3× the per-iteration time budget (e.g., 2400s → 7200s = 2h).
    import time as _time_bug57
    _refine_start_time = _time_bug57.monotonic()
    _per_iter_budget = int(getattr(config.experiment, "time_budget_sec", 2400) or 2400)
    _max_refine_wall_sec = int(
        getattr(config.experiment, "max_refine_duration_sec", 0) or 0
    ) or int(_per_iter_budget * 1.5)

    # --- Collect baseline metrics from prior runs ---
    runs_dir_path: Path | None = None
    runs_dir_text = _read_prior_artifact(run_dir, "runs/")
    if runs_dir_text:
        runs_dir_path = Path(runs_dir_text)

    run_summaries: list[str] = []
    baseline_metric: float | None = None
    if runs_dir_path is not None:
        for run_file in sorted(runs_dir_path.glob("*.json"))[:40]:
            payload = _safe_json_loads(run_file.read_text(encoding="utf-8"), {})
            if not isinstance(payload, dict):
                continue
            # R5-5: Truncate stdout/stderr for context efficiency
            summary = dict(payload)
            if "stdout" in summary and isinstance(summary["stdout"], str):
                lines = summary["stdout"].splitlines()
                if len(lines) > 30:
                    summary["stdout"] = (
                        f"[...truncated {len(lines) - 30} lines...]\n"
                        + "\n".join(lines[-30:])
                    )
                if len(summary["stdout"]) > 2000:
                    summary["stdout"] = summary["stdout"][-2000:]
            if "stderr" in summary and isinstance(summary["stderr"], str):
                lines = summary["stderr"].splitlines()
                if len(lines) > 50:
                    summary["stderr"] = "\n".join(lines[-50:])
                if len(summary["stderr"]) > 2000:
                    summary["stderr"] = summary["stderr"][-2000:]
            run_summaries.append(json.dumps(summary, ensure_ascii=False))
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                metrics = (
                    payload.get("key_metrics")
                    if isinstance(payload.get("key_metrics"), dict)
                    else {}
                )
            metric_val = (
                _find_metric(metrics, metric_key)
                if isinstance(metrics, dict)
                else None
            )
            if metric_val is None:
                metric_val = _to_float(payload.get("primary_metric"))
            if _is_better(metric_val, baseline_metric):
                baseline_metric = metric_val

    # --- Read experiment project (multi-file or single-file) ---
    # BUG-58: When PIVOT rolls back to Stage 13, prefer the best refined code
    # from a previous cycle (stage-13_vX/experiment_final/) over the original
    # unrefined code (stage-12/experiment/ or stage-10/experiment/).
    # Enhanced: try ALL versioned directories (latest first) with fallback chain.
    exp_dir_text: str | None = None
    _prev_refine_dirs = sorted(
        run_dir.glob("stage-13_v*/experiment_final"),
        key=lambda p: p.parent.name,
        reverse=True,  # latest version first
    )
    # BUG-58 fix: Find the best version across ALL cycles (not just latest)
    _best_prev_metric: float | None = None
    _best_prev_dir: Path | None = None
    for _prd in _prev_refine_dirs:
        if not _prd.is_dir():
            continue
        _prd_log = _prd.parent / "refinement_log.json"
        if _prd_log.is_file():
            _prd_data = _safe_json_loads(
                _prd_log.read_text(encoding="utf-8"), {}
            )
            _prd_metric = _prd_data.get("best_metric") if isinstance(_prd_data, dict) else None
            if isinstance(_prd_metric, (int, float)) and _is_better(_prd_metric, _best_prev_metric):
                _best_prev_metric = _prd_metric
                _best_prev_dir = _prd
        elif _best_prev_dir is None:
            # No log but directory exists — use as fallback
            _best_prev_dir = _prd
    if _best_prev_dir is not None:
        exp_dir_text = str(_best_prev_dir)
        logger.info(
            "BUG-58: Recovered best refined code from PIVOT cycle: %s (metric=%s)",
            _best_prev_dir.parent.name,
            f"{_best_prev_metric:.4f}" if _best_prev_metric is not None else "N/A",
        )
    if not exp_dir_text:
        exp_dir_text = _read_prior_artifact(run_dir, "experiment/")
    best_files: dict[str, str] = {}
    if exp_dir_text and Path(exp_dir_text).is_dir():
        # Load all text files (requirements.txt, setup.py, config etc. are
        # needed for Docker sandbox phases), recursing so nested modules and
        # data (algorithms/, data/*.json) survive into refinement.
        _exp_root = Path(exp_dir_text)
        for src_file in sorted(_exp_root.rglob("*")):
            if not src_file.is_file():
                continue
            if src_file.suffix.lower().lstrip(".") in (
                "py", "txt", "yaml", "yml", "json", "cfg", "ini", "sh",
            ):
                _rel = src_file.relative_to(_exp_root).as_posix()
                try:
                    best_files[_rel] = src_file.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    pass  # skip binary files
    if not best_files:
        # Backward compat: single experiment.py
        original_code = _read_prior_artifact(run_dir, "experiment.py") or ""
        if original_code:
            best_files = {"main.py": original_code}

    # --- Detect if prior experiment timed out ---
    prior_timed_out = False
    prior_time_budget = config.experiment.time_budget_sec
    if runs_dir_path is not None:
        for run_file in sorted(runs_dir_path.glob("*.json"))[:5]:
            try:
                payload = _safe_json_loads(run_file.read_text(encoding="utf-8"), {})
                if isinstance(payload, dict) and payload.get("timed_out"):
                    prior_timed_out = True
                    break
            except OSError:
                pass

    best_metric = baseline_metric
    best_version = "experiment/"
    # BUG-58: Recover best_metric from best previous PIVOT cycle
    if _best_prev_metric is not None and _is_better(_best_prev_metric, best_metric):
        best_metric = _best_prev_metric
        logger.info(
            "BUG-58: Recovered best_metric=%.4f from previous PIVOT",
            best_metric,
        )
    no_improve_streak = 0
    consecutive_no_metrics = 0

    log: dict[str, Any] = {
        "generated": _utcnow_iso(),
        "mode": config.experiment.mode,
        "metric_key": metric_key,
        "metric_direction": metric_direction,
        "max_iterations_requested": requested_iterations,
        "max_iterations_executed": max_iterations,
        "baseline_metric": baseline_metric,
        "project_files": list(best_files.keys()),
        "iterations": [],
        "converged": False,
        "stop_reason": "max_iterations_reached",
    }

    # --- Helper: write files to a directory ---
    def _write_project(target_dir: Path, project_files: dict[str, str]) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname, code in project_files.items():
            _wf = target_dir / fname
            _wf.parent.mkdir(parents=True, exist_ok=True)
            _wf.write_text(code, encoding="utf-8")

    # --- Helper: format all files for LLM context ---
    def _files_to_context(project_files: dict[str, str]) -> str:
        """Render the project as prompt context — source files only.

        ``best_files`` is the project payload as well as the prompt context, and
        the two need different contents.  ``_write_project`` must keep every
        collected file so the refined project stays runnable (data/*.json
        included), but the refinement prompt asks for ``filename:xxx.py`` back
        and never acts on a data file.  One ML23 run carried 3.8MB of
        data/*.json into the prompt — ~1M tokens — and every gateway rejected it
        with HTTP 400 across three vendors before the fallback chain gave up.
        Filtering here, not at collection time, keeps both uses correct.
        """
        parts = []
        for fname, code in sorted(project_files.items()):
            if not fname.endswith(".py"):
                continue
            if len(code) > _CONTEXT_FILE_MAX_CHARS:
                code = (
                    code[:_CONTEXT_FILE_MAX_CHARS]
                    + f"\n# ... [truncated, {len(code)} chars total]"
                )
            parts.append(f"```filename:{fname}\n{code}\n```")
        return "\n\n".join(parts)

    def _write_refinement_log() -> None:
        (stage_dir / "refinement_log.json").write_text(
            json.dumps(log, indent=2), encoding="utf-8"
        )

    def _pause_refinement(
        *,
        reason: str,
        stop_reason: str,
        iteration: int | None = None,
    ) -> StageResult:
        log.update(
            {
                "paused": True,
                "converged": False,
                "stop_reason": stop_reason,
                "pause_reason": reason,
                "best_metric": best_metric,
                "best_version": best_version,
                "iterations_completed": len(log["iterations"]),
            }
        )
        if iteration is not None:
            log["pause_iteration"] = iteration
        _write_refinement_log()
        artifacts = ("refinement_log.json",)
        return StageResult(
            stage=Stage.ITERATIVE_REFINE,
            status=StageStatus.PAUSED,
            artifacts=artifacts,
            error=reason,
            decision="resume",
            evidence_refs=tuple(f"stage-13/{a}" for a in artifacts),
        )

    if llm is None:
        logger.info("Stage 13: LLM unavailable, saving original experiment as final")
        final_dir = stage_dir / "experiment_final"
        _write_project(final_dir, best_files)
        # Backward compat
        if "main.py" in best_files:
            (stage_dir / "experiment_final.py").write_text(
                best_files["main.py"], encoding="utf-8"
            )
        log.update(
            {
                "converged": True,
                "stop_reason": "llm_unavailable",
                "best_metric": best_metric,
                "best_version": "experiment_final/",
                "iterations": [
                    {
                        "iteration": 0,
                        "version_dir": "experiment_final/",
                        "source": "fallback_original",
                        "metric": best_metric,
                    }
                ],
            }
        )
        _write_refinement_log()
        artifacts = ("refinement_log.json", "experiment_final/")
        artifacts += _generate_llm4ad_task_packages(
            stage_dir, run_dir, config, exp_dir_text, log
        )
        return StageResult(
            stage=Stage.ITERATIVE_REFINE,
            status=StageStatus.DONE,
            artifacts=artifacts,
            evidence_refs=tuple(f"stage-13/{a}" for a in artifacts),
        )

    _pm = prompts or PromptManager()
    timeout_refine_attempts = 0

    # R7-3: Read experiment plan to detect condition coverage gaps
    _exp_plan_text = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    _condition_coverage_hint = ""
    if _exp_plan_text and run_summaries:
        # Check if stdout contains condition labels
        _all_stdout = " ".join(run_summaries)
        _has_condition_labels = "condition=" in _all_stdout
        if not _has_condition_labels and _exp_plan_text.strip():
            _condition_coverage_hint = (
                "\nCONDITION COVERAGE GAP DETECTED:\n"
                "The experiment plan specifies multiple conditions/treatments, "
                "but the output contains NO condition labels (no 'condition=...' in stdout).\n"
                "You MUST:\n"
                "1. Run ALL conditions/treatments from the experiment plan independently\n"
                "2. Label each metric output: `condition=<name> {metric_key}: <value>`\n"
                "3. Print a SUMMARY line comparing all conditions after completion\n"
                "This is the MOST IMPORTANT improvement — a single unlabeled metric stream "
                "cannot support any comparative conclusions.\n\n"
            )
            logger.info(
                "Stage 13: condition coverage gap detected, injecting multi-condition hint"
            )

    # P1: Track metrics history for saturation detection
    _metrics_history: list[float | None] = [baseline_metric]

    for iteration in range(1, max_iterations + 1):
        # BUG-57: Check wall-clock time before starting a new iteration
        _elapsed = _time_bug57.monotonic() - _refine_start_time
        if _elapsed > _max_refine_wall_sec:
            logger.warning(
                "Stage 13: Wall-clock time cap reached (%.0fs > %ds). "
                "Stopping refinement after %d iterations.",
                _elapsed, _max_refine_wall_sec, iteration - 1,
            )
            log["stop_reason"] = "wall_clock_time_cap"
            break
        logger.info("Stage 13: refinement iteration %d/%d (%.0fs elapsed, cap %ds)",
                    iteration, max_iterations, _elapsed, _max_refine_wall_sec)

        # P1: Detect metric saturation and inject difficulty upgrade hint
        _saturation_hint = ""
        _valid_metrics = [m for m in _metrics_history if m is not None]
        if len(_valid_metrics) >= 2:
            _last_two = _valid_metrics[-2:]
            _saturated = False
            # Use relative change rate instead of hard-coded thresholds
            _change_rate = abs(_last_two[-1] - _last_two[-2]) / max(abs(_last_two[-2]), 1e-8)
            if metric_direction == "minimize":
                _saturated = all(m <= 0.001 for m in _last_two) or (
                    _change_rate < 0.001 and _last_two[-1] < 0.01
                )
            else:
                _saturated = all(m >= 0.999 for m in _last_two) or (
                    _change_rate < 0.001 and _last_two[-1] > 0.99
                )
            if _saturated:
                _saturation_hint = (
                    "\n\nWARNING — BENCHMARK SATURATION DETECTED:\n"
                    "All methods achieve near-perfect scores, making the task too easy "
                    "to discriminate between methods.\n"
                    "YOU MUST increase benchmark difficulty in this iteration:\n"
                    "1. Increase the number of actions/decisions from 8 to at least 20\n"
                    "2. Increase the horizon from 12-18 to at least 50-100 steps\n"
                    "3. Increase noise level to at least 0.3-0.5\n"
                    "4. Add partial observability (agent cannot see full state)\n"
                    "5. Add delayed rewards (reward only at episode end)\n"
                    "6. Ensure random search achieves < 50% success rate\n"
                    "Without this change, the experiment produces meaningless results.\n"
                )
                logger.warning("Stage 13: metric saturation detected, injecting difficulty upgrade hint")

        files_context = _files_to_context(best_files)
        # BUG-10 fix: anchor refinement to original experiment plan
        _exp_plan_anchor = ""
        if _exp_plan_text.strip():
            _exp_plan_anchor = (
                "Original experiment plan (exp_plan.yaml):\n"
                "```yaml\n" + _exp_plan_text[:4000] + "\n```\n"
                "You MUST preserve ALL condition names from this plan.\n\n"
            )
        ip = _pm.sub_prompt(
            "iterative_improve",
            metric_key=metric_key,
            metric_direction=metric_direction,
            files_context=files_context,
            run_summaries=chr(10).join(run_summaries[:20]),
            condition_coverage_hint=_condition_coverage_hint,
            topic=config.research.topic,
            exp_plan_anchor=_exp_plan_anchor,
        )

        # --- Timeout-aware prompt injection ---
        user_prompt = ip.user + _saturation_hint
        if prior_timed_out and baseline_metric is None:
            timeout_refine_attempts += 1
            timeout_hint = (
                f"\n\nCRITICAL: The experiment TIMED OUT after {prior_time_budget}s "
                f"with NO results. You MUST drastically reduce the experiment scale:\n"
                f"- Reduce total runs to ≤50\n"
                f"- Reduce steps per run to ≤2000\n"
                f"- Remove conditions that are not essential\n"
                f"- Add time.time() checks to stop gracefully before timeout\n"
                f"- Print intermediate metrics frequently so partial data is captured\n"
                f"- Time budget is {prior_time_budget}s — design for ≤{int(prior_time_budget * 0.7)}s\n"
            )
            user_prompt = user_prompt + timeout_hint
            logger.warning(
                "Stage 13: injecting timeout-aware prompt (attempt %d)",
                timeout_refine_attempts,
            )

        try:
            response = _chat_with_prompt(
                llm,
                ip.system,
                user_prompt,
                max_tokens=ip.max_tokens or 8192,
            )
        except RuntimeError as exc:
            if "ACP prompt timed out after" in str(exc):
                logger.warning(
                    "Stage 13: ACP prompt timed out during iteration %d; pausing for resume",
                    iteration,
                )
                return _pause_refinement(
                    reason=str(exc),
                    stop_reason="acp_prompt_timeout",
                    iteration=iteration,
                )
            raise
        extracted_files = _extract_multi_file_blocks(response.content)
        # If LLM returns only single block, treat as main.py update
        if not extracted_files:
            single_code = _extract_code_block(response.content)
            if single_code.strip():
                extracted_files = {"main.py": single_code}
        # R8-2: Merge with best_files to preserve supporting modules
        # (e.g., graphs.py, game.py) that the LLM didn't rewrite
        candidate_files = dict(best_files)
        if extracted_files:
            candidate_files.update(extracted_files)
        # If LLM returned nothing at all, candidate_files == best_files (unchanged)

        # BUG-R6-02: Preserve entry point when LLM strips main() function.
        # The LLM often returns only class/function improvements without the
        # main() entry point, causing the script to exit with no output.
        _new_main = candidate_files.get("main.py", "")
        _old_main = best_files.get("main.py", "")
        if (
            _new_main
            and _old_main
            and "if __name__" not in _new_main
            and "if __name__" in _old_main
        ):
            # Extract the entry-point block from original main.py
            _ep_idx = _old_main.rfind("\ndef main(")
            if _ep_idx == -1:
                _ep_idx = _old_main.rfind("\nif __name__")
            if _ep_idx != -1:
                _entry_block = _old_main[_ep_idx:]
                candidate_files["main.py"] = _new_main.rstrip() + "\n\n" + _entry_block
                logger.info(
                    "Stage 13 iter %d: restored entry point stripped by LLM "
                    "(%d chars appended from original main.py)",
                    iteration,
                    len(_entry_block),
                )

        # Validate main.py
        main_code = candidate_files.get("main.py", "")
        validation = validate_code(main_code)
        issue_text = ""
        repaired = False

        if not validation.ok:
            issue_text = format_issues_for_llm(validation)
            logger.info(
                "Stage 13 iteration %d validation failed: %s",
                iteration,
                validation.summary(),
            )
            irp = _pm.sub_prompt(
                "iterative_repair",
                issue_text=issue_text,
                all_files_ctx=_files_to_context(candidate_files),
            )
            try:
                repair_response = _chat_with_prompt(llm, irp.system, irp.user)
            except RuntimeError as exc:
                if "ACP prompt timed out after" in str(exc):
                    logger.warning(
                        "Stage 13: ACP repair prompt timed out during iteration %d; pausing for resume",
                        iteration,
                    )
                    return _pause_refinement(
                        reason=str(exc),
                        stop_reason="acp_prompt_timeout",
                        iteration=iteration,
                    )
                raise
            candidate_files["main.py"] = _extract_code_block(repair_response.content)
            validation = validate_code(candidate_files["main.py"])
            repaired = True

        # Save version directory
        version_dir = stage_dir / f"experiment_v{iteration}"
        _write_project(version_dir, candidate_files)

        iter_record: dict[str, Any] = {
            "iteration": iteration,
            "version_dir": f"experiment_v{iteration}/",
            "files": list(candidate_files.keys()),
            "validation_ok": validation.ok,
            "validation_summary": validation.summary(),
            "repaired": repaired,
            "metric": None,
            "improved": False,
        }
        if issue_text:
            iter_record["validation_issues"] = issue_text

        metric_val = None  # R6-3: initialize before conditional block
        if validation.ok and config.experiment.mode in ("sandbox", "docker"):
            # P7: Ensure deps for refined code (subprocess sandbox only)
            if config.experiment.mode == "sandbox":
                _refine_code = "\n".join(candidate_files.values())
                _ensure_sandbox_deps(_refine_code, config.experiment.sandbox.python_path)

            sandbox = create_sandbox(
                config.experiment,
                stage_dir / f"refine_sandbox_v{iteration}",
            )
            rerun = sandbox.run_project(
                version_dir,
                timeout_sec=config.experiment.time_budget_sec,
            )
            metric_val = _find_metric(rerun.metrics, metric_key)
            # R19-1: Store stdout (capped) so PAIRED lines survive for Stage 14
            _stdout_cap = rerun.stdout[:50000] if rerun.stdout else ""
            iter_record["sandbox"] = {
                "returncode": rerun.returncode,
                "metrics": rerun.metrics,
                "elapsed_sec": rerun.elapsed_sec,
                "timed_out": rerun.timed_out,
                "stderr": rerun.stderr[:2000] if rerun.stderr else "",
                "stdout": _stdout_cap,
            }
            iter_record["metric"] = metric_val

            # BUG-110: Parse ABLATION_CHECK lines from stdout
            if rerun.stdout:
                import re as _re_ablation
                _ablation_checks = _re_ablation.findall(
                    r"ABLATION_CHECK:\s*(\S+)\s+vs\s+(\S+)\s+outputs_differ=(True|False)",
                    rerun.stdout,
                )
                if _ablation_checks:
                    _identical_pairs = [
                        (c1, c2) for c1, c2, diff in _ablation_checks if diff == "False"
                    ]
                    iter_record["ablation_checks"] = [
                        {"cond1": c1, "cond2": c2, "differ": diff == "True"}
                        for c1, c2, diff in _ablation_checks
                    ]
                    if _identical_pairs:
                        _pairs_str = ", ".join(f"{c1} vs {c2}" for c1, c2 in _identical_pairs)
                        logger.warning(
                            "BUG-110: Identical ablation outputs detected: %s. "
                            "Ablation conditions may not be wired correctly.",
                            _pairs_str,
                        )
                        iter_record["ablation_identical"] = True

            # --- Track timeout in refine sandbox ---
            if rerun.timed_out:
                prior_timed_out = True
                timeout_refine_attempts += 1
                logger.warning(
                    "Stage 13 iteration %d: sandbox timed out after %.1fs",
                    iteration,
                    rerun.elapsed_sec,
                )
                # If still no metrics after timeout, use partial stdout metrics
                if not rerun.metrics and rerun.stdout:
                    from researchclaw.experiment.sandbox import parse_metrics as _parse_sb_metrics
                    partial = _parse_sb_metrics(rerun.stdout)
                    if partial:
                        iter_record["sandbox"]["metrics"] = partial
                        metric_val = _find_metric(partial, metric_key)
                        iter_record["metric"] = metric_val
                        logger.info(
                            "Stage 13 iteration %d: recovered %d partial metrics from timeout stdout",
                            iteration,
                            len(partial),
                        )

            # --- Detect runtime issues (NaN/Inf, stderr warnings) ---
            runtime_issues = _detect_runtime_issues(rerun)
            if runtime_issues:
                iter_record["runtime_issues"] = runtime_issues
                logger.info(
                    "Stage 13 iteration %d: runtime issues detected: %s",
                    iteration,
                    runtime_issues[:200],
                )
                # Attempt LLM repair with runtime context
                rrp = _pm.sub_prompt(
                    "iterative_repair",
                    issue_text=runtime_issues,
                    all_files_ctx=_files_to_context(candidate_files),
                )
                try:
                    repair_resp = _chat_with_prompt(llm, rrp.system, rrp.user)
                except RuntimeError as exc:
                    if "ACP prompt timed out after" in str(exc):
                        logger.warning(
                            "Stage 13: ACP runtime-repair prompt timed out during iteration %d; pausing for resume",
                            iteration,
                        )
                        return _pause_refinement(
                            reason=str(exc),
                            stop_reason="acp_prompt_timeout",
                            iteration=iteration,
                        )
                    raise
                repaired_files = _extract_multi_file_blocks(repair_resp.content)
                if not repaired_files:
                    single = _extract_code_block(repair_resp.content)
                    if single.strip():
                        repaired_files = dict(candidate_files)
                        repaired_files["main.py"] = single
                if repaired_files:
                    # BUG-106 fix: merge instead of replace to preserve
                    # supporting modules (trainers.py, utils.py, etc.)
                    merged = dict(candidate_files)
                    merged.update(repaired_files)
                    candidate_files = merged
                    _write_project(version_dir, candidate_files)
                    # Re-run after runtime fix
                    sandbox2 = create_sandbox(
                        config.experiment,
                        stage_dir / f"refine_sandbox_v{iteration}_fix",
                    )
                    rerun2 = sandbox2.run_project(
                        version_dir,
                        timeout_sec=config.experiment.time_budget_sec,
                    )
                    metric_val = _find_metric(rerun2.metrics, metric_key)
                    iter_record["sandbox_after_fix"] = {
                        "returncode": rerun2.returncode,
                        "metrics": rerun2.metrics,
                        "elapsed_sec": rerun2.elapsed_sec,
                        "timed_out": rerun2.timed_out,
                    }
                    iter_record["metric"] = metric_val
                    iter_record["runtime_repaired"] = True

            if metric_val is not None:
                consecutive_no_metrics = 0
                # R6-1: Only count toward no_improve_streak when we have real metrics
                if _is_better(metric_val, best_metric):
                    best_metric = metric_val
                    best_files = dict(candidate_files)
                    best_version = f"experiment_v{iteration}/"
                    iter_record["improved"] = True
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
            else:
                consecutive_no_metrics += 1
        elif validation.ok and best_version == "experiment/":
            best_files = dict(candidate_files)
            best_version = f"experiment_v{iteration}/"

        # P1: Track metric for saturation detection
        _metrics_history.append(metric_val)

        log["iterations"].append(iter_record)

        if consecutive_no_metrics >= 3:
            log["stop_reason"] = "consecutive_no_metrics"
            logger.warning("Stage 13: Aborting after %d consecutive iterations without metrics", consecutive_no_metrics)
            break

        if no_improve_streak >= 2:
            log["converged"] = True
            log["stop_reason"] = "no_improvement_for_2_iterations"
            logger.info(
                "Stage 13 converged after %d iterations (no improvement streak=%d)",
                iteration,
                no_improve_streak,
            )
            break

    # Write final experiment directory
    final_dir = stage_dir / "experiment_final"
    _write_project(final_dir, best_files)
    # Backward compat: also write experiment_final.py (copy of main.py)
    if "main.py" in best_files:
        (stage_dir / "experiment_final.py").write_text(
            best_files["main.py"], encoding="utf-8"
        )

    log["best_metric"] = best_metric
    log["best_version"] = best_version
    log["final_version"] = "experiment_final/"
    # BUG-110: Aggregate ablation check results across iterations
    _all_ablation_identical = any(
        iter_rec.get("ablation_identical", False)
        for iter_rec in log.get("iterations", [])
        if isinstance(iter_rec, dict)
    )
    if _all_ablation_identical:
        log["ablation_identical_warning"] = True
    _write_refinement_log()

    # ── LLM4AD comparison mode ─────────────────────────────────────────
    # When llm4ad_boost is on, the refined output above is the comparison
    # *baseline*, not the deliverable: snapshot it, let llm4ad evolve from the
    # clean stage-10 code, then overlay its best algorithms into
    # experiment_final/ for Stage 14+. Keeps a fabricated refinement pass from
    # ever reaching the paper.
    _l4b_final = getattr(getattr(config, "experiment", None), "llm4ad_boost", None)
    _l4b_enabled = bool(
        _l4b_final is not None and getattr(_l4b_final, "enabled", False)
    )
    _legacy_baseline = stage_dir / "legacy_refine_baseline"
    if _l4b_enabled:
        # 1. Snapshot the refined output as the comparison baseline.
        try:
            import shutil as _sh_clean_baseline

            if _legacy_baseline.exists():
                _sh_clean_baseline.rmtree(_legacy_baseline, ignore_errors=True)
            _sh_clean_baseline.copytree(final_dir, _legacy_baseline)
            _legacy_result = {
                "generated": _utcnow_iso(),
                "source": "iterative_refine",
                "metric_key": metric_key,
                "metric_direction": metric_direction,
                "best_metric": best_metric,
                "best_version": best_version,
                "refinement_log": str(stage_dir / "refinement_log.json"),
            }
            (stage_dir / "legacy_refine_result.json").write_text(
                json.dumps(_legacy_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log["legacy_refine_baseline"] = {
                "dir": str(_legacy_baseline),
                "result_json": str(stage_dir / "legacy_refine_result.json"),
                "best_metric": best_metric,
                "note": (
                    "Refinement output snapshotted before llm4ad promotion; it is "
                    "the comparison baseline, not the final artifact."
                ),
            }
            logger.info(
                "Stage 13: snapshotted refinement output as %s (best_metric=%s)",
                _legacy_baseline,
                f"{best_metric:.6g}" if best_metric is not None else "N/A",
            )
        except OSError as _baseline_err:
            logger.warning(
                "Stage 13: could not snapshot refinement baseline: %s",
                _baseline_err,
            )

    artifacts = ["refinement_log.json", "experiment_final/"]
    if _l4b_enabled and _legacy_baseline.is_dir():
        # Keep the refinement snapshot as a first-class artifact so the
        # comparison baseline survives into the run directory for review.
        artifacts += ["legacy_refine_baseline/", "legacy_refine_result.json"]
    artifacts += _generate_llm4ad_task_packages(
        stage_dir, run_dir, config, exp_dir_text, log
    )

    # 3. Promote evolved algorithms into experiment_final/ (llm4ad mode only,
    #    and only when evolution actually produced a usable package).
    if _l4b_enabled and "evolution_results/" in artifacts:
        _evo_dir = stage_dir / "evolution_results"
        # llm4ad evolved from the CLEAN stage-10 code, so the base must also be
        # the clean stage-10 code — NOT the refined experiment_final/ that was
        # just snapshotted. Rebuild from that clean source.
        _clean_exp = _read_prior_artifact(run_dir, "experiment/")
        if _clean_exp and Path(_clean_exp).is_dir():
            _metric_direction = getattr(
                config.experiment, "metric_direction", ""
            ) or "minimize"
            _n, _comparison = _promote_llm4ad_to_experiment_final(
                Path(_clean_exp), _evo_dir, final_dir,
                metric_direction=_metric_direction,
            )
            # Persist the per-algo evolvability comparison so reviewers can see
            # which evolved algorithms were actually promoted into experiment_final/.
            _cmp_path = stage_dir / "llm4ad_comparison.json"
            try:
                _cmp_path.write_text(
                    json.dumps(
                        {
                            "generated": _utcnow_iso(),
                            "metric_direction": _metric_direction,
                            "base": _clean_exp,
                            "n_promoted": _n,
                            "algorithms": _comparison,
                        },
                        ensure_ascii=False, indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError as _cmp_err:
                logger.warning(
                    "Stage 13: could not write llm4ad_comparison.json: %s", _cmp_err,
                )
            else:
                artifacts.append("llm4ad_comparison.json")
            log["llm4ad_promoted"] = {
                "n_algorithms_overlaid": _n,
                "n_algorithms_total": len(_comparison),
                "base": _clean_exp,
                "metric_direction": _metric_direction,
                "comparison": _comparison,
                "note": (
                    "only evolved <algo>.py that beat their clean baseline under the "
                    "experiment's own evaluator protocol are overlaid into "
                    "experiment_final/; main.py, data/, other algorithms come from "
                    "clean stage-10 code"
                ),
            }
            # Keep experiment_final.py consistent with the promoted main.py.
            _promoted_main = final_dir / "main.py"
            if _promoted_main.is_file():
                (stage_dir / "experiment_final.py").write_text(
                    _promoted_main.read_text(encoding="utf-8"), encoding="utf-8"
                )
            _write_refinement_log()

    artifacts.extend(
        entry["version_dir"]
        for entry in log["iterations"]
        if isinstance(entry, dict) and isinstance(entry.get("version_dir"), str)
    )
    return StageResult(
        stage=Stage.ITERATIVE_REFINE,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-13/{a}" for a in artifacts),
    )
