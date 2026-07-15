"""LLM4AD sandbox — evolves algorithms via the LLM4AD framework.

This backend integrates AutoResearchClaw with **LLM4AD** (an LLM-driven
automatic algorithm-design framework using island genetic algorithms).
Unlike the Claude-Code agent sandboxes, it does not shell out to
``claude``; instead it launches a small driver
(:mod:`researchclaw.experiment.llm4ad_driver`) in a child process.

Two-phase pipeline split
------------------------
LLM4AD naturally splits into two very different phases, which map cleanly
to the AutoResearchClaw 23-stage pipeline:

* **Stage 10 CODE_GENERATION** — :meth:`run_build` turns a natural-language
  problem description into a task directory (seed.py + evaluation.py +
  dataset + config.yaml).  Fast (minutes), one LLM invocation per repair.
* **Stage 12 EXPERIMENT_RUN** — :meth:`run_evolve` reads the built
  config.yaml and runs the island GA loop (many LLM calls, potentially
  hours), producing the canonical ``results.json`` at the workspace root.

For backward compatibility :meth:`run` still runs both phases in one shot
(``phase="both"``); new pipeline code prefers the split entrypoints.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from researchclaw.config import Llm4adAgentConfig
from researchclaw.experiment.sandbox import SandboxResult

logger = logging.getLogger(__name__)

# Prompt file written to the workspace (the assembled problem description).
_PROMPT_FILENAME = "llm4ad_plan.md"
# JSON job spec handed to the driver child process.
_JOB_FILENAME = "llm4ad_job.json"
# Filename the driver writes after a successful build phase.
_BUILD_RESULT_FILENAME = "build_result.json"
# Driver module shipped alongside this file.
_DRIVER_PATH = Path(__file__).with_name("llm4ad_driver.py")


class Llm4adAgentSandbox:
    """Run an LLM4AD build+evolve pipeline in a child process.

    Parameters
    ----------
    config:
        :class:`~researchclaw.config.Llm4adAgentConfig` with all LLM4AD
        build/evolve settings.
    workdir:
        Working directory for this experiment run.  The driver builds the
        task directory under ``<workdir>/task`` and writes ``results.json``
        at ``<workdir>`` root.
    """

    def __init__(self, config: Llm4adAgentConfig, workdir: Path) -> None:
        self.config = config
        self.workdir = workdir

    # ==================================================================
    # Public API — split pipeline (Stage 10 build, Stage 12 evolve)
    # ==================================================================

    def run_build(
        self,
        prompt_text: str,
        *,
        timeout_sec: int | None = None,
    ) -> SandboxResult:
        """Stage 10 — build the LLM4AD task directory only.

        ``prompt_text`` is the raw problem description; it is refined into
        an LLM4AD-friendly build description before being handed to the
        driver.  On success, the sandbox writes ``build_result.json`` at
        the workspace root (containing ``config_path`` and paths to
        ``seed.py`` / ``evaluation.py`` / ``dataset/``).

        This method does NOT run evolution.  Call :meth:`run_evolve`
        with the ``config_path`` from ``build_result.json`` to do that.
        """
        timeout_sec = (
            timeout_sec
            if timeout_sec is not None
            else (self.config.build_timeout_sec or self.config.timeout_sec)
        )
        workspace = self._prepare_workspace(prompt_text, phase="build")
        return self._invoke_driver(
            workspace, timeout_sec=timeout_sec, phase_label="build"
        )

    def run_evolve(
        self,
        config_path: Path | str,
        *,
        timeout_sec: int | None = None,
    ) -> SandboxResult:
        """Stage 12 — run the island GA on a previously built task.

        ``config_path`` must point at a ``config.yaml`` produced by a prior
        :meth:`run_build` call (or by any other means — the sandbox does
        not care who built it, only that it is a valid LLM4AD config).

        Writes the canonical ``results.json`` at the workspace root — the
        same schema Stage 14 / the requirements gate expect.
        """
        timeout_sec = (
            timeout_sec
            if timeout_sec is not None
            else (self.config.evolve_timeout_sec or self.config.timeout_sec)
        )
        workspace = self._prepare_workspace(
            prompt_text=None,
            phase="evolve",
            config_path=str(Path(config_path).resolve()),
        )
        return self._invoke_driver(
            workspace, timeout_sec=timeout_sec, phase_label="evolve"
        )

    # ==================================================================
    # Public API — legacy combined path (matches SandboxProtocol)
    # ==================================================================

    def run(
        self,
        prompt_text: str,
        *,
        timeout_sec: int | None = None,
    ) -> SandboxResult:
        """Legacy: run build+evolve in one shot.

        Kept for backward compatibility and for callers that still want
        the atomic "one sandbox call handles everything" semantic (e.g.
        the ``run_project`` requirements-gate rerun).  New pipeline code
        should use :meth:`run_build` / :meth:`run_evolve` so Stage 10 and
        Stage 12 each own their phase.
        """
        timeout_sec = (
            timeout_sec if timeout_sec is not None else self.config.timeout_sec
        )
        workspace = self._prepare_workspace(prompt_text, phase="both")
        return self._invoke_driver(
            workspace, timeout_sec=timeout_sec, phase_label="both"
        )

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 300,
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SandboxResult:
        """``SandboxProtocol.run_project`` adapter.

        LLM4AD runs atomically via :meth:`run` when the requirements gate
        rerun repair loop re-invokes us, so this shim dispatches to the
        combined path.  The pipeline runner routes the more nuanced Stage
        10 → Stage 12 flow directly through :meth:`run_build` /
        :meth:`run_evolve`; ``run_project`` only fires on the repair path.
        """
        del entry_point, args, env_overrides  # SandboxProtocol parity only

        candidates = (
            project_dir / "REPAIR_PROMPT.md",
            project_dir / _PROMPT_FILENAME,
            self.workdir / _PROMPT_FILENAME,
        )
        prompt_text = ""
        for cand in candidates:
            if cand.is_file():
                prompt_text = cand.read_text(encoding="utf-8")
                logger.info("Llm4adAgentSandbox.run_project: using prompt %s", cand)
                break
        if not prompt_text:
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=(
                    "Llm4adAgentSandbox.run_project: no llm4ad_plan.md found "
                    f"in project_dir={project_dir} or workspace={self.workdir}"
                ),
                elapsed_sec=0.0,
                metrics={},
                timed_out=False,
            )
        return self.run(prompt_text, timeout_sec=timeout_sec)

    # ==================================================================
    # Internal — driver subprocess invocation
    # ==================================================================

    def _invoke_driver(
        self,
        workspace: Path,
        *,
        timeout_sec: int,
        phase_label: str,
    ) -> SandboxResult:
        job_path = workspace / _JOB_FILENAME
        cmd = self._build_command(job_path)

        logger.info(
            "Llm4adAgentSandbox[%s]: running %r in %s (timeout=%ds)",
            phase_label, " ".join(cmd), workspace, timeout_sec,
        )

        start = time.monotonic()
        # Both build and evolve phases now use streaming output so users can
        # monitor long-running evolution progress in real time.  The evolve
        # phase can run for hours, but seeing periodic progress (generation
        # numbers, best scores) is essential for knowing the job is alive.
        stdout, stderr, returncode, timed_out = self._run_streaming(
            cmd, workspace, timeout_sec, phase_label
        )

        elapsed = time.monotonic() - start

        artifacts = self._collect_artifacts(workspace)
        metrics = self._build_metrics(
            returncode, timed_out, artifacts, workspace, phase_label
        )
        self._write_summary(
            workspace, returncode, elapsed, artifacts, timed_out, phase_label
        )

        succeeded = returncode == 0 and not timed_out
        log_at = logger.info if succeeded else logger.warning
        log_at(
            "Llm4adAgentSandbox[%s]: finished (rc=%d, elapsed=%.1fs, artifacts=%d)",
            phase_label, returncode, elapsed,
            sum(len(v) for v in artifacts.values()),
        )
        # For the non-streaming evolve/both path the driver output was captured
        # silently; surface stderr on failure so the error isn't swallowed.
        if not succeeded and phase_label != "build" and stderr.strip():
            logger.warning(
                "Llm4adAgentSandbox[%s]: driver stderr:\n%s",
                phase_label, stderr.strip(),
            )
        return SandboxResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_sec=elapsed,
            metrics=metrics,
            timed_out=timed_out,
        )

    def _run_captured(
        self,
        cmd: list[str],
        workspace: Path,
        timeout_sec: int,
        phase_label: str,
    ) -> tuple[str, str, int, bool]:
        """Run the driver capturing all output (no live streaming).

        Used for the evolve/both phases whose per-generation output would
        flood the console.  Returns ``(stdout, stderr, returncode, timed_out)``.
        """
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        timed_out = False
        returncode = -1
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace),
                env=self._build_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
            )
            returncode = proc.returncode
            stdout_parts.append(proc.stdout or "")
            stderr_parts.append(proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = -1
            stdout_parts.append(
                exc.stdout.decode("utf-8", "replace")
                if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            )
            stderr_parts.append(
                exc.stderr.decode("utf-8", "replace")
                if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            )
            logger.warning(
                "Llm4adAgentSandbox[%s]: timed out after %ds",
                phase_label, timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001
            returncode = -1
            stderr_parts.append(
                f"Llm4adAgentSandbox[{phase_label}] launch error: {exc}"
            )
            logger.exception("Llm4adAgentSandbox[%s]: unexpected error", phase_label)
        return (
            "\n".join(stdout_parts),
            "\n".join(stderr_parts),
            returncode,
            timed_out,
        )

    def _run_streaming(
        self,
        cmd: list[str],
        workspace: Path,
        timeout_sec: int,
        phase_label: str,
    ) -> tuple[str, str, int, bool]:
        """Run the driver, forwarding its output to the logger line-by-line.

        Used for the build phase so LLM4AD's ``build_task_sync`` progress /
        repair logs are visible live at INFO level (prefixed
        ``[llm4ad:<phase>]``).  stdout and stderr are read on separate
        threads (portable across platforms, including Windows) and also
        accumulated so they still land in the returned ``SandboxResult``.
        Returns ``(stdout, stderr, returncode, timed_out)``.
        """
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(workspace),
                env=self._build_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,  # line-buffered
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Llm4adAgentSandbox[%s]: unexpected error", phase_label)
            return (
                "",
                f"Llm4adAgentSandbox[{phase_label}] launch error: {exc}",
                -1,
                False,
            )

        prefix = f"[llm4ad:{phase_label}]"
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        def _pump(stream: Any, sink: list[str]) -> None:
            try:
                for line in iter(stream.readline, ""):
                    sink.append(line)
                    logger.info("%s %s", prefix, line.rstrip("\n"))
            finally:
                stream.close()

        t_out = threading.Thread(
            target=_pump, args=(proc.stdout, stdout_parts), daemon=True
        )
        t_err = threading.Thread(
            target=_pump, args=(proc.stderr, stderr_parts), daemon=True
        )
        t_out.start()
        t_err.start()

        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
            proc.kill()
            proc.wait()
            logger.warning(
                "Llm4adAgentSandbox[%s]: timed out after %ds",
                phase_label, timeout_sec,
            )

        # Ensure the reader threads have drained the pipes before we return.
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        return (
            "".join(stdout_parts),
            "".join(stderr_parts),
            returncode,
            timed_out,
        )

    # ==================================================================
    # Internal — workspace + job spec construction
    # ==================================================================

    def _prepare_workspace(
        self,
        prompt_text: str | None,
        *,
        phase: str,
        config_path: str | None = None,
    ) -> Path:
        """Create the workspace and write the phase-specific job spec.

        For build/both phases we also refine the raw prompt text into the
        LLM4AD-friendly template.  For evolve phase we just need
        ``config_path`` — the description was already consumed at build
        time.
        """
        workspace = self.workdir
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "task").mkdir(parents=True, exist_ok=True)

        description: str | None = None
        if phase in ("build", "both"):
            followup_delta = self._consume_followup_delta(workspace)
            description = self._refine_description(
                prompt_text or "", followup_delta
            )
            (workspace / _PROMPT_FILENAME).write_text(description, encoding="utf-8")

        job: dict[str, Any] = {
            "phase": phase,
            "workspace": str(workspace.resolve()),
            "metric_direction": self.config.metric_direction,
        }
        if phase in ("build", "both"):
            job.update(
                {
                    "description": description,
                    "output_dir": str((workspace / "task").resolve()),
                    "project_name": "evolved_task",
                    "build_api_key": self._resolve_build_api_key(),
                    "build_model": self._resolve_build_model(),
                    "build_base_url": self._resolve_build_base_url(),
                    "max_repair_attempts": self.config.max_repair_attempts,
                    "build_max_tries": self.config.build_max_tries,
                }
            )
        if phase in ("evolve", "both"):
            job.update(
                {
                    "resume_from_checkpoint": self.config.resume_from_checkpoint,
                }
            )
        if phase == "evolve":
            if not config_path:
                raise ValueError(
                    "Llm4adAgentSandbox._prepare_workspace(phase=evolve) "
                    "requires config_path"
                )
            job["config_path"] = config_path

        (workspace / _JOB_FILENAME).write_text(
            json.dumps(job, indent=2), encoding="utf-8"
        )
        return workspace

    def _consume_followup_delta(self, workspace: Path) -> str:
        """Pick up a requirements-gate rerun delta if present.

        Mirrors the other agent sandboxes, but does NOT assume a fixed
        workspace depth.  The requirements gate writes ``REPAIR_PROMPT.md``
        at the *run-dir root* (``run_dir/REPAIR_PROMPT.md``), and llm4ad's
        two phases live at different depths under that root:

        * Stage 10 build → ``run_dir/stage-10/llm4ad_workspace`` (run_dir
          is ``parents[1]``)
        * Stage 12 evolve/both → ``run_dir/stage-12/runs/llm4ad_workspace``
          (run_dir is ``parents[2]``)

        A fixed ``parents[3]`` (the ColliderAgent convention, whose
        workspace is one level deeper) would miss the file for both phases.
        Instead we walk up the ancestor chain and take the first
        ``REPAIR_PROMPT.md`` we find.  The delta is prepended to the build
        description so the build LLM sees the unmet requirements first, then
        consumed (deleted) so it doesn't leak into the next run.
        """
        candidates: list[Path] = [workspace / "REPAIR_PROMPT.md"]
        # Walk up a bounded number of ancestors so we find the run-dir-root
        # copy regardless of which phase (and thus which depth) we run at.
        for ancestor in list(workspace.parents)[:5]:
            candidates.append(ancestor / "REPAIR_PROMPT.md")

        followup_delta = ""
        found: Path | None = None
        for cand in candidates:
            if cand.is_file():
                try:
                    followup_delta = cand.read_text(encoding="utf-8")
                    found = cand
                    logger.info(
                        "Llm4adAgentSandbox: consumed %s (%d chars) — "
                        "requirements-gate rerun", cand, len(followup_delta),
                    )
                except OSError as exc:
                    logger.warning(
                        "Llm4adAgentSandbox: failed to read %s: %s", cand, exc
                    )
                break
        if found is not None:
            try:
                found.unlink()
            except OSError:
                pass
        return followup_delta

    def _refine_description(self, prompt_text: str, followup_delta: str) -> str:
        """Turn the pipeline's raw experiment plan into a clean LLM4AD build
        description.

        LLM4AD's ``build_task_sync`` is sensitive to description quality: a
        vague or ML-flavoured prompt makes the build LLM emit buggy
        seed/evaluator code that exhausts the repair budget.  We therefore
        wrap the raw plan in an explicit, structured template that spells out
        exactly what an algorithm-evolution task needs: the problem, the
        function signature to evolve, the evaluation criterion, and the I/O
        contract.  This is the single biggest lever on build success rate.
        """
        raw = (prompt_text or "").strip()
        # Keep the raw plan bounded — build LLMs do worse with very long,
        # rambling context, and LLM4AD only needs the essence.
        if len(raw) > 6000:
            raw = raw[:6000] + "\n...[truncated]..."

        header = (
            "# Algorithm Evolution Task (for LLM4AD)\n\n"
            "You are defining an automatic algorithm-design task to be solved "
            "by an island genetic algorithm. Produce a task that is concrete, "
            "self-contained, and unambiguous.\n\n"
            "## Objective\n"
            "Evolve an algorithm that solves the problem described below, "
            "maximising (or minimising, as stated) a single well-defined "
            "numeric score.\n\n"
        )
        requirements = (
            "\n\n## Task-definition requirements (follow exactly)\n"
            "1. Define ONE clear function to evolve, with a precise, typed "
            "signature and a docstring stating inputs and outputs.\n"
            "2. Provide a simple, CORRECT seed algorithm that runs without "
            "error on the evaluation data (it may be naive — evolution "
            "improves it).\n"
            "3. Define a deterministic evaluator that returns a single float "
            "score; higher is better unless the objective says otherwise.\n"
            "4. Use only standard scientific Python (numpy/scipy) unless the "
            "problem clearly needs more; avoid heavyweight/ML frameworks.\n"
            "5. Handle edge cases (empty input, ties) so the seed never "
            "raises — a crashing seed makes the whole task un-evolvable.\n"
            "6. Keep the evaluation dataset small and fast so many "
            "generations can run within the time budget.\n"
        )
        delta_block = ""
        if followup_delta:
            delta_block = (
                "\n\n## FOLLOWUP DELTA — requirements gate rerun (read FIRST)\n"
                "The previous run did not satisfy one or more requirements. "
                "Address the points below when redefining the task:\n\n"
                f"{followup_delta}\n"
            )
        return (
            header
            + delta_block
            + "## Problem description\n\n"
            + (raw or "No detailed plan was provided; infer a reasonable, "
                      "well-scoped algorithm-design problem from the objective.")
            + requirements
        )

    def _resolve_build_api_key(self) -> str:
        return (
            self.config.build_api_key
            or os.getenv("LLM4AD_BUILD_API_KEY")
            or os.getenv("LLM_API_KEY")
            or ""
        )

    def _resolve_build_model(self) -> str:
        return (
            self.config.build_model
            or os.getenv("LLM4AD_BUILD_MODEL")
            or os.getenv("LLM_MODEL")
            or ""
        )

    def _resolve_build_base_url(self) -> str:
        return (
            self.config.build_base_url
            or os.getenv("LLM4AD_BUILD_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or ""
        )

    def _build_command(self, job_path: Path) -> list[str]:
        binary = self.config.python_binary or sys.executable or "python"
        return [binary, str(_DRIVER_PATH.resolve()), str(job_path.resolve())]

    def _build_env(self) -> dict[str, str]:
        """Environment for the driver child process.

        Prepends the LLM4AD project dir to PYTHONPATH so ``import llm4ad``
        resolves, and passes the evolution-phase LLM credentials through so
        the generated ``config.yaml`` (which references ``LLM_*``) can be
        resolved.
        """
        env = os.environ.copy()
        # Force unbuffered child stdio so the build phase's streamed logs
        # appear live rather than in one block-buffered dump at exit.
        env["PYTHONUNBUFFERED"] = "1"
        # By default (llm4ad_dir empty) we run the driver with the SAME
        # interpreter/environment as the parent, which — when AutoResearchClaw
        # is installed inside the LLM4AD repo's environment — already has
        # ``llm4ad`` importable.  Only when llm4ad_dir is explicitly set do we
        # prepend it to PYTHONPATH.
        raw_dir = (self.config.llm4ad_dir or "").strip()
        if raw_dir:
            llm4ad_dir = Path(raw_dir).expanduser()
            if llm4ad_dir.is_dir():
                existing = env.get("PYTHONPATH", "")
                # LLM4AD packages typically live under <repo>/src or <repo> root.
                paths = [str(llm4ad_dir.resolve())]
                src = llm4ad_dir / "src"
                if src.is_dir():
                    paths.insert(0, str(src.resolve()))
                env["PYTHONPATH"] = os.pathsep.join(
                    [p for p in paths + [existing] if p]
                )

        # The evolution phase's generated config.yaml references LLM_BASE_URL /
        # LLM_API_KEY / LLM_MODEL.  If the user configured build credentials but
        # did NOT export those env vars, propagate the build credentials as the
        # LLM_* defaults so a single config point makes both phases work.  We
        # only FILL missing vars — never override an explicitly-set environment.
        key = self._resolve_build_api_key()
        base = self._resolve_build_base_url()
        model = self._resolve_build_model()
        if key and not env.get("LLM_API_KEY"):
            env["LLM_API_KEY"] = key
        if base and not env.get("LLM_BASE_URL"):
            env["LLM_BASE_URL"] = base
        if model and not env.get("LLM_MODEL"):
            env["LLM_MODEL"] = model
        return env

    def _collect_artifacts(self, workspace: Path) -> dict[str, list[str]]:
        """Scan the workspace for evolution artifacts."""
        artifacts: dict[str, list[str]] = {
            "figures": [], "data": [], "scripts": [], "logs": []
        }
        for pattern in ("**/*.png", "**/*.pdf", "**/*.svg"):
            for p in sorted(workspace.glob(pattern)):
                rel = str(p.relative_to(workspace))
                if rel not in artifacts["figures"]:
                    artifacts["figures"].append(rel)
        for pattern in ("**/*.json", "**/*.csv", "**/*.jsonl"):
            for p in sorted(workspace.glob(pattern)):
                rel = str(p.relative_to(workspace))
                # Skip our own control files
                if p.name in (_JOB_FILENAME, "results.json", _BUILD_RESULT_FILENAME):
                    continue
                if rel not in artifacts["data"]:
                    artifacts["data"].append(rel)
        for p in sorted((workspace / "task").glob("**/*.py")):
            artifacts["scripts"].append(str(p.relative_to(workspace)))
        return artifacts

    def _build_metrics(
        self,
        returncode: int,
        timed_out: bool,
        artifacts: dict[str, list[str]],
        workspace: Path,
        phase_label: str,
    ) -> dict[str, Any]:
        """Merge driver-written scientific metrics with sandbox coverage stats.

        For the build phase we don't have scientific metrics yet — Stage 10
        just cares that seed.py / evaluation.py / config.yaml were produced.
        For evolve / both we merge the driver's results.json.
        """
        success = returncode == 0 and not timed_out
        metrics: dict[str, Any] = {
            "llm4ad_agent_success": 1.0 if success else 0.0,
            f"llm4ad_{phase_label}_success": 1.0 if success else 0.0,
            "figures_produced": float(len(artifacts.get("figures", []))),
            "scripts_generated": float(len(artifacts.get("scripts", []))),
        }
        if phase_label == "build":
            # For the build phase there is no scientific "primary_metric" yet
            # — Stage 10 only cares that seed.py / evaluation.py / config.yaml
            # were produced.  We deliberately do NOT synthesize a
            # ``primary_metric`` here: emitting one would be a fake scientific
            # headline (the build success bit masquerading as a result) that
            # could pollute primary-metric selection if this phase's metrics
            # were ever collected as a run.  The success bits above are enough
            # for Stage 10 gating / observability.
            build_doc = self._read_build_result(workspace)
            if build_doc and build_doc.get("status") == "success":
                metrics["build_task_config_present"] = 1.0
            return metrics

        # evolve / both: merge canonical results.json
        agent_doc = self._read_agent_results(workspace)
        if agent_doc:
            for k, v in (agent_doc.get("metrics") or {}).items():
                if isinstance(v, (int, float, bool)):
                    metrics[k] = float(v)
            hyps = agent_doc.get("hypotheses") or {}
            if isinstance(hyps, dict):
                for hid, payload in hyps.items():
                    if isinstance(payload, dict) and "supported" in payload:
                        metrics[f"hypothesis_{hid}_supported"] = (
                            1.0 if payload["supported"] else 0.0
                        )
                    elif isinstance(payload, bool):
                        metrics[f"hypothesis_{hid}_supported"] = 1.0 if payload else 0.0
            agent_primary = agent_doc.get("primary_metric")
            if isinstance(agent_primary, (int, float)):
                metrics["primary_metric"] = float(agent_primary)
        if "primary_metric" not in metrics:
            metrics["primary_metric"] = metrics["llm4ad_agent_success"]
        return metrics

    @classmethod
    def _read_agent_results(cls, workspace: Path) -> dict[str, Any] | None:
        """Read the driver-written canonical results.json (skip sandbox stub).

        The sandbox's own meta stub (see :meth:`_write_summary`) contains only
        ``source`` / ``returncode`` / ``elapsed_sec`` / ``timed_out`` /
        ``artifacts`` / ``status``.  We skip it by requiring at least one
        genuine agent key (``metrics`` / ``primary_metric`` / ``hypotheses`` /
        ``structured_results``) — a positive check that is simpler and safer
        than subtracting a meta-key set.
        """
        path = workspace / "results.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if (
            "metrics" in data
            or "primary_metric" in data
            or "hypotheses" in data
            or "structured_results" in data
        ):
            return data
        return None

    @classmethod
    def _read_build_result(cls, workspace: Path) -> dict[str, Any] | None:
        """Read the driver-written build_result.json (Stage 10)."""
        path = workspace / _BUILD_RESULT_FILENAME
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _write_summary(
        self,
        workspace: Path,
        returncode: int,
        elapsed: float,
        artifacts: dict[str, list[str]],
        timed_out: bool,
        phase_label: str,
    ) -> None:
        """Merge sandbox metadata into results.json (evolve/both) or into
        build_result.json (build) without clobbering driver output.
        ``returncode``/``elapsed_sec``/``timed_out`` are always
        sandbox-authoritative."""
        sandbox_meta = {
            "source": f"llm4ad_agent:{phase_label}",
            "returncode": returncode,
            "elapsed_sec": round(elapsed, 2),
            "timed_out": timed_out,
            "artifacts": artifacts,
            "status": (
                "success" if returncode == 0 and not timed_out
                else ("timeout" if timed_out else "failed")
            ),
        }
        target = (
            workspace / _BUILD_RESULT_FILENAME
            if phase_label == "build"
            else workspace / "results.json"
        )
        existing: dict[str, Any] = {}
        if target.is_file():
            try:
                _data = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(_data, dict):
                    existing = _data
            except (OSError, json.JSONDecodeError):
                existing = {}
        merged = dict(existing)
        for k, v in sandbox_meta.items():
            if k in ("returncode", "elapsed_sec", "timed_out"):
                merged[k] = v
            elif k not in merged:
                merged[k] = v
        if "artifacts" not in existing:
            merged["artifacts"] = artifacts
        target.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        logger.debug(
            "Llm4adAgentSandbox[%s]: wrote summary to %s",
            phase_label, target,
        )
