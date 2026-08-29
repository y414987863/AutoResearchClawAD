"""OpenCode 'Beast Mode' bridge — routes complex code generation to OpenCode CLI.

OpenCode (https://github.com/anomalyco/opencode) is an external AI coding agent
invoked via ``opencode run --format json "prompt"``.  This module provides:

1. **ComplexityScore / score_complexity()** — analyses an experiment plan to
   decide whether beast mode is warranted.
2. **OpenCodeBridge** — manages workspace creation, OpenCode invocation, file
   collection, and cleanup.
"""

from __future__ import annotations

import ast
import itertools
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _rmtree_force(path: Path) -> None:
    """Delete ``path`` including read-only files (git worktrees).

    ``shutil.rmtree(..., ignore_errors=True)`` is not enough here: git marks
    every object under ``.git/objects`` read-only, so on Windows the unlink
    raises PermissionError and ``ignore_errors`` swallows it — the tree is left
    half-deleted with the old git history intact. A resurrected history is worse
    than no cleanup at all: the next run's ``auto_initialize`` sees the previous
    snapshot as HEAD. Clearing the read-only bit and retrying is the standard
    way to remove a git tree on Windows.
    """
    import stat as _stat

    def _clear_readonly(func, target, _exc):  # noqa: ANN001 - shutil callback shape
        try:
            os.chmod(target, _stat.S_IWRITE)
            func(target)
        except OSError:
            pass  # genuinely undeletable (locked by another process)

    try:
        shutil.rmtree(path, onexc=_clear_readonly)
    except TypeError:  # Python < 3.12 renamed the hook to onexc
        shutil.rmtree(path, onerror=_clear_readonly)


# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------

# Keywords that indicate multi-component architectures
_COMPONENT_KEYWORDS: tuple[str, ...] = (
    "encoder",
    "decoder",
    "discriminator",
    "generator",
    "critic",
    "actor",
    "teacher",
    "student",
    "backbone",
    "head",
    "neck",
    "classifier",
    "embedder",
    "attention",
    "transformer",
    "tokenizer",
    "vae",
    "autoencoder",
)

# Indicators that multi-file generation is needed
_FILE_HINT_KEYWORDS: tuple[str, ...] = (
    "model.py",
    "trainer.py",
    "dataset.py",
    "utils.py",
    "config.py",
    "multiple files",
    "modular",
    "separate module",
    "multi-file",
)

# Domain-complexity keywords
_DOMAIN_COMPLEX_KEYWORDS: tuple[str, ...] = (
    "multi-modal",
    "multimodal",
    "distributed",
    "gan",
    "diffusion",
    "nerf",
    "mixture of experts",
    "moe",
    "meta-learning",
    "meta learning",
    "maml",
    "neural ode",
    "neural sde",
    "physics-informed",
    "pinn",
    "graph neural",
    "gnn",
    "reinforcement learning",
    "multi-agent",
    "world model",
    "vision-language",
    "text-to-image",
    "image-to-text",
)

# Patterns suggesting deep dependency chains
_DEPENDENCY_KEYWORDS: tuple[str, ...] = (
    "custom layer",
    "custom loss",
    "wrapper",
    "registry",
    "hook",
    "callback",
    "scheduler",
    "custom optimizer",
    "custom dataset",
    "custom sampler",
    "custom transform",
)


@dataclass
class ComplexityScore:
    """Result of complexity analysis on an experiment plan."""

    score: float  # 0.0-1.0
    signals: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""  # "beast_mode" | "code_agent" | "legacy"
    reason: str = ""


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def score_complexity(
    exp_plan: str,
    topic: str = "",
    *,
    historical_failures: int = 0,
    threshold: float = 0.6,
) -> ComplexityScore:
    """Score the complexity of an experiment to determine if beast mode is warranted.

    Returns a ComplexityScore with score in [0.0, 1.0].
    """
    if not exp_plan and not topic:
        return ComplexityScore(
            score=0.0,
            signals={},
            recommendation="legacy",
            reason="Empty plan",
        )

    combined = f"{topic}\n{exp_plan}"

    # Signal 1: Component count (weight 0.25)
    comp_hits = _count_keyword_hits(combined, _COMPONENT_KEYWORDS)
    component_score = min(comp_hits / 5.0, 1.0)

    # Signal 2: File count hint (weight 0.20)
    file_hits = _count_keyword_hits(combined, _FILE_HINT_KEYWORDS)
    file_score = min(file_hits / 3.0, 1.0)

    # Signal 3: Domain complexity (weight 0.20)
    domain_hits = _count_keyword_hits(combined, _DOMAIN_COMPLEX_KEYWORDS)
    domain_score = min(domain_hits / 3.0, 1.0)

    # Signal 4: Condition count (weight 0.15)
    # Numbered conditions/ablations/variants, plus baseline mentions.
    condition_pattern = re.compile(
        r"(?:condition|ablation|variant|experiment)\s*[\-_:]?\s*\d+",
        re.IGNORECASE,
    )
    condition_matches = len(condition_pattern.findall(combined))
    condition_matches += combined.lower().count("baseline")
    condition_score = min(condition_matches / 8.0, 1.0)

    # Signal 5: Historical failures (weight 0.10)
    failure_score = min(historical_failures / 3.0, 1.0)

    # Signal 6: Dependency depth (weight 0.10)
    dep_hits = _count_keyword_hits(combined, _DEPENDENCY_KEYWORDS)
    dep_score = min(dep_hits / 3.0, 1.0)

    # Weighted sum
    weighted = (
        0.25 * component_score
        + 0.20 * file_score
        + 0.20 * domain_score
        + 0.15 * condition_score
        + 0.10 * failure_score
        + 0.10 * dep_score
    )
    final_score = min(max(weighted, 0.0), 1.0)

    signals = {
        "component_count": round(component_score, 3),
        "file_count_hint": round(file_score, 3),
        "domain_complexity": round(domain_score, 3),
        "condition_count": round(condition_score, 3),
        "historical_failure": round(failure_score, 3),
        "dependency_depth": round(dep_score, 3),
    }

    if final_score >= threshold:
        recommendation = "beast_mode"
        reason = (
            f"Complexity {final_score:.2f} >= threshold {threshold:.2f}: "
            f"top signals: "
            + ", ".join(
                f"{k}={v:.2f}"
                for k, v in sorted(signals.items(), key=lambda x: -x[1])[:3]
            )
        )
    else:
        recommendation = "code_agent"
        reason = f"Complexity {final_score:.2f} < threshold {threshold:.2f}"

    return ComplexityScore(
        score=round(final_score, 4),
        signals=signals,
        recommendation=recommendation,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# OpenCode bridge
# ---------------------------------------------------------------------------

@dataclass
class OpenCodeResult:
    """Result from an OpenCode invocation."""

    success: bool
    files: dict[str, str] = field(default_factory=dict)
    opencode_log: str = ""
    elapsed_sec: float = 0.0
    error: str = ""
    # True when ``files`` was salvaged from a workspace whose OpenCode run
    # actually FAILED (timeout / non-zero exit). ``success`` is still True
    # because a usable package was recovered, but the package was never validated
    # by the agent itself — callers must not treat it as a clean generation.
    recovered_from_failure: bool = False


# Top-level modules that must NEVER be swapped into ``main.py`` even if they
# carry a ``__main__`` guard. They have a fixed role in the LLM4AD layout —
# ``run_single.py`` imports ``benchmarks``/``stats_utils`` by name — so
# relocating their contents breaks the package. See _is_entry_swap_candidate.
_ENTRY_SWAP_EXCLUDED = frozenset({
    "benchmarks.py",
    "evaluator.py",
    "stats_utils.py",
    "run_single.py",
    "setup.py",
    "conftest.py",
})


# The CLI prompt is passed as a single argv element, so it MUST stay short:
# Windows caps a command line at 32767 chars and POSIX at ARG_MAX. The full
# instructions therefore live in TASK.md inside the workspace (see
# _TASK_MD_TEMPLATE) and this prompt only points at them. Keep additions here
# to a few lines; put anything substantial in TASK.md instead.
_MEGA_PROMPT_TEMPLATE = """\
Implement the experiment described in this repository. Work autonomously.

Read these workspace files FIRST, in order:
1. TASK.md — your task, requirements, and constraints (READ THIS FIRST)
2. EXPERIMENT_PLAN.yaml — the experiment design to implement
3. GUIDANCE.md — metric, environment, and domain constraints

Do NOT ask questions. No human will answer: this session is driven by an
automated pipeline and any reply that asks a question instead of writing code
is discarded as a failure. If a detail is unclear, pick a sensible default,
note it in a comment, and continue.

Your first action must be a tool call that reads TASK.md — never a text reply.

RULE — ONE file per step. Write exactly one file per step, then start the next
step for the next file. Never emit a whole multi-file project in a single step.
A single step's output is capped at a hard limit: if a step tries to write more
than one large file (or one very large file), the tail of that output is
truncated and you LOSE everything it was writing — the truncated step commits
no files. This is a requirement, not a suggestion. Keep each step focused,
small and complete: one file, written and closed, then continue.

If any single file is itself too large to fit in one step (roughly 400+ lines),
do not write a huge file in one step. Split it: write the smaller modules as
separate files, or grow a large file across steps by writing the first part
then appending the rest with a shell redirect (`>>`) in the next step.

You are NOT done when the files exist — you are done when you have RUN them and
they work. TASK.md tells you which interpreter to use and what to check.
Writing no files is a failure; so is reporting success for code you never ran.
"""

# Full instructions, written to the workspace as TASK.md. This is read by the
# agent via a tool call, so it is not subject to the command-line length limit.
_TASK_MD_TEMPLATE = """\
# Task

Implement a complete, runnable ML/science experiment in this repository.

Read `EXPERIMENT_PLAN.yaml` for the experiment design (conditions, baselines,
ablations, metrics) and `GUIDANCE.md` for the environment and domain
constraints. Implement what the plan specifies — do not substitute a different
method or dataset.

## Requirements

1. Design the file structure. `main.py` is the required entry point.
2. Implement ALL files with complete, runnable code. No placeholders or TODOs.
3. `main.py` must print the primary metric as: `{metric}: <value>`
4. Include numerical stability guards (gradient clipping, NaN detection, etc.).
5. Use multi-seed evaluation (seeds 0, 1, 2) and report mean ± std.
6. Each ablation/condition MUST be genuinely different — not a copy-paste with
   a renamed variable.
7. Implement a time guard: stop gracefully at 80% of the time budget
   ({time_budget_sec} seconds).
8. Write `requirements.txt` listing any extra pip packages needed — UNLESS
   `GUIDANCE.md` forbids it (offline runs cannot pip install; then use only the
   preinstalled packages it lists).
9. If the experiment needs dataset downloads AND `GUIDANCE.md` permits network
   access, write a `setup.py` that handles them. If downloads are forbidden, do
   NOT create `setup.py`.

## Constraints

- The code runs in an isolated container. `GUIDANCE.md` lists exactly which
  packages are available and is AUTHORITATIVE: where it and this file disagree,
  `GUIDANCE.md` wins.
- Do NOT use argparse or CLI arguments — hardcode all configuration, UNLESS
  `GUIDANCE.md` explicitly requires a CLI flag (e.g. an `--algorithm` selector).
- All results must go to stdout via print statements.
- Keep the experiment feasible within {time_budget_sec} seconds total.

## Verify before you finish (MANDATORY)

Writing the files is not the end of the task. Code that has never been executed
is not done — a single bad line (an accumulator reset to `None`, a key that no
data file carries) is invisible on reading and fatal on running, and nothing
downstream will fix it for you.

1. Always use this exact interpreter, never a bare `python`. It is the
   environment the experiment is executed in later, and the only one guaranteed
   to have the packages `GUIDANCE.md` lists. Keep it quoted — the path may
   contain spaces:

       "{python}"

2. Run the DEFAULT entry point — this is what the pipeline executes later, so it
   is the ONLY valid completion check:

       "{python}" main.py

   It must exit 0 and print a finite `{metric}`. A traceback, `nan`/`inf`, or a
   missing metric means the task is NOT done — no exception.

   If it fails, FIX the code and re-run the default. Keep fixing and re-running
   until the default passes. This is a loop, not a single try.

   If `GUIDANCE.md` defines a single-unit mode (e.g. an `--algorithm` selector),
   you may run `"{python}" main.py --algorithm <name>` to LOCATE which algorithm
   is at fault when the default fails — but a passing single-unit run is NOT a
   pass. Only the default run (no flags) counts as done.

3. Read the output. It must exit 0 and print a finite `{metric}`. A traceback,
   a `nan`/`inf`, or a missing metric means the task is NOT done.

4. Fix whatever you find and run it again. Repeat until it passes.

   One exception: if it fails only because a package `GUIDANCE.md` lists as
   available cannot be imported here, that is an environment gap, not a defect.
   The experiment may be executed elsewhere (a container) where the package does
   exist. Note it in a comment and move on — do NOT restructure the code, drop
   the dependency, or reimplement it by hand to make the import go away.

5. Keep the default configuration small enough that this verification finishes
   in seconds. Shrink the workload, not the correctness.

Never report success for code you did not run to completion. If it still fails
after your best effort, say so explicitly and leave the files in place.

## Working style

- Do not ask clarifying questions; no human is available to answer. Choose a
  reasonable default, record it in a comment, and keep going.
- Finish by writing the files to disk. Producing no files is a failure.
"""

# Prepended to the CLI prompt on a retry. Kept deliberately short for the same
# command-line length reason; the detail goes in RETRY.md.
#
# BUG-OB-X: the retry used to open a FRESH OpenCode session with no task
# description, so attempt 2+ replied "I don't have the original task description"
# and wrote nothing. The full task is always on disk in TASK.md (rewritten per
# attempt), so the retry prompt must point straight at it rather than assume the
# session carries the prior task context.
_RETRY_PROMPT_PREFIX = """\
RETRY: the previous attempt FAILED and produced no usable code.
Read RETRY.md for what went wrong. This is a brand-new session — it has NOT
seen the previous attempt or the task. The full task is in TASK.md; read it
FIRST (tool call, not text) and implement it. Do not reply with a question:
a text-only reply is itself the failure mode.
"""

_RETRY_MD_TEMPLATE = """\
# Retry {attempt} of {total} — the previous attempt FAILED

Reason: {reason}

{no_files_note}Do not repeat the previous mistake. Write the actual files this
time: create `main.py` and every supporting module with real, runnable code.

Write ONE file per step. Do not emit a whole project in a single step — a step
is capped and the tail of a big output is truncated. Keep each step small and
complete, then start the next step for the next file.

The previous attempt's output is in `PREVIOUS_ATTEMPT/` and its log in
`PREVIOUS_ATTEMPT_ERROR.txt`, for reference only.

This is a brand-new session. Re-read `TASK.md` FIRST and follow it — that is
the task; do not reply with a question.
"""

_NO_FILES_NOTE = (
    "The previous attempt produced NO code files at all. It replied with a "
    "question or with commentary instead of calling tools to write files. That "
    "is exactly the failure mode to avoid: ask nothing, write code.\n\n"
)

# Cap the context handed back to a retry. ``_invoke_opencode`` returns EVERY
# line opencode prints as a JSON event, and each event embeds whatever file the
# agent read (TASK.md, GUIDANCE.md, RETRY.md itself, ...) in full. Dumping all
# of it as the retry ``reason`` blew RETRY.md and PREVIOUS_ATTEMPT_ERROR.txt up
# to ~72KB each; and because the agent then READS RETRY.md, that content re-entered
# the event stream and the next retry's dump grew again — a compounding blow-up
# across retries that also bloated the model context. Only a short diagnostic
# tail is useful to the next attempt; the verbose event stream stays on disk in
# opencode_log.txt for humans.
_RETRY_REASON_MAX_CHARS = 2000


def _summarize_log(log: str, *, max_chars: int = _RETRY_REASON_MAX_CHARS) -> str:
    """Compress an OpenCode event stream into a short retry reason.

    The raw log is one JSON object per line. Most of it is tool-use noise with
    huge ``output``/``content`` payloads (the agent echoing every file it read),
    which carries no signal about WHY the attempt failed. This keeps only the
    diagnostic lines — the trailing failure marker (``TIMEOUT`` / ``error`` /
    ``exit`` / ``reason``) and the final events — and strips the embedded
    file-content fields, so a retry sees a compact reason rather than a ~70KB
    transcript.
    """
    if not log:
        return ""
    lines = [l.rstrip("\n") for l in log.splitlines()]

    # Best-effort over UNTRUSTED data: an arbitrary opencode event stream may be
    # truncated mid-JSON or carry circular/unpicklable objects, so json.loads /
    # json.dumps / _shrink can all throw. A raise here would mask the very error
    # we are summarising (callers assign ``last_error = _summarize_log(log)`` at
    # the top of the exception path), so fall back to a plain tail-of-log.
    try:
        return _summarize_log_impl(log, lines, max_chars=max_chars)
    except Exception:  # noqa: BLE001 — log summarisation must never raise
        fallback = "\n".join(lines[-6:])
        return fallback[-max_chars:]


def _summarize_log_impl(log: str, lines: list[str], *, max_chars: int) -> str:
    """Actual summarisation; wrapped by :func:`_summarize_log` for safety."""
    if not lines:
        return ""

    def _is_signal(line: str) -> bool:
        # Only a line that is a TOP-LEVEL event object counts. Matching signal
        # words anywhere is too loose: the agent reads GUIDANCE.md, whose prose
        # ("no network access", "do NOT use", ...) contains "error"/"failed",
        # echoed into a read event's content, which then passes the word match
        # while carrying zero diagnostic signal.
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            # Not JSON — keep only a trailing marker like the TIMEOUT line.
            return line.startswith("TIMEOUT") or (
                "timeout" in line.lower() and "after" in line.lower()
            )
        if not isinstance(obj, dict):
            return False
        # An event declaring a type like "error"/"failure" is a real signal.
        etype = str(obj.get("type", "")).lower()
        if any(tok in etype for tok in ("error", "fail", "timeout")):
            return True
        # A step that ended for a "length" / "error" / "max_tokens" / "stop"
        # reason is the failure mechanism (output truncated mid-write).
        reason = str(obj.get("part", {}).get("reason", "")).lower() if isinstance(
            obj.get("part"), dict
        ) else ""
        return any(tok in reason for tok in ("error", "length", "max_tokens", "stop"))

    # Prefer signal-carrying lines; fall back to the tail when none are present
    # (e.g. a plain tool-call loop that never raised, which is itself the bug).
    chosen = [l for l in lines if _is_signal(l)]
    if not chosen:
        chosen = lines[-6:]

    # Compress any embedded JSON deep in a line down to its bare text.
    compact: list[str] = []
    for line in chosen:
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            compact.append(line)
            continue

        def _shrink(v: Any) -> Any:
            if isinstance(v, dict):
                out = {}
                for k, val in v.items():
                    # File-body echoes are the bulk of the transcript; drop them.
                    if k in ("output", "content", "preview", "text"):
                        out[k] = _truncate_json_text(val)
                    else:
                        out[k] = _shrink(val)
                return out
            if isinstance(v, list):
                return [_shrink(item) for item in v]
            return v

        obj = _shrink(obj)
        # Keep the original single-line-per-event style (the ``sep`` output) so a
        # compacted reason stays one object per line, not pretty-printed.
        try:
            compact.append(json.dumps(obj, ensure_ascii=False))
        except (TypeError, ValueError, RecursionError):
            # A single un-serialisable line (circular ref, odd provider shape)
            # must not kill the whole summary — keep the raw line instead.
            compact.append(line)

    out = "\n".join(compact)
    return out[-max_chars:]


def _truncate_json_text(value: Any, limit: int = 160) -> Any:
    """Shrink an ``output``/``content``/``preview``/``text`` field.

    These hold the file text the agent read (or the code it wrote) verbatim,
    which is exactly the bulk that bloats the retry reason. Keep a short
    head+tail so the *kind* of content is still identifiable.
    """
    if not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return value[:limit] + f"...<+{len(value) - limit} chars truncated>"



class OpenCodeBridge:
    """Manages OpenCode CLI invocations for beast mode code generation."""

    def __init__(
        self,
        *,
        model: str = "",
        llm_base_url: str = "",
        api_key: str = "",
        api_key_env: str = "",
        llm_provider: str = "openai-compatible",
        timeout_sec: int = 600,
        max_retries: int = 1,
        workspace_cleanup: bool = True,
        python_path: str = "",
    ) -> None:
        self._model = model
        self._llm_base_url = llm_base_url
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._llm_provider = llm_provider
        self._timeout_sec = timeout_sec
        self._max_retries = max_retries
        self._workspace_cleanup = workspace_cleanup
        # Interpreter the generated code will actually be executed with later
        # (the sandbox's). The agent is told to verify with THIS one, because a
        # bare `python` on PATH is often a different environment without the
        # scientific packages, so "it ran for me" would not transfer.
        self._python_path = python_path

    # -- availability check ---------------------------------------------------

    @staticmethod
    def check_available() -> bool:
        """Return True if the ``opencode`` CLI is installed and callable."""
        opencode_cmd = shutil.which("opencode")
        if not opencode_cmd:
            return False
            
        try:
            result = subprocess.run(
                [opencode_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False
        except Exception:  # noqa: BLE001
            return False

    # -- workspace preparation ------------------------------------------------

    def _prepare_workspace(
        self,
        stage_dir: Path,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        extra_guidance: str,
        time_budget_sec: int,
        prev_error: str = "",
        prev_files: dict[str, str] | None = None,
        retry_note: str = "",
    ) -> Path:
        """Create a temporary workspace directory with context files."""
        # Each attempt needs its OWN workspace. The previous name mixed
        # time.time() with time.monotonic_ns() % 100000, but the monotonic clock
        # has ~15.6ms granularity on Windows, so that suffix is frequently 0 and
        # two attempts in the same second collided — mkdir(exist_ok=True) then
        # silently reused the directory and the retry inherited the old output.
        # An explicit counter makes each attempt unique.
        for _n in itertools.count():
            ws = stage_dir / f"opencode_beast_{int(time.time())}_{_n}"
            if not ws.exists():
                break
        ws.mkdir(parents=True, exist_ok=True)

        # Write the full task instructions. These live here rather than in the
        # CLI prompt because argv has a hard length limit (32767 on Windows).
        #
        # The interpreter is spelled out so the agent verifies with the SAME
        # environment the experiment later runs in; falling back to "python"
        # only when none is configured, which at least keeps the instruction
        # runnable.
        _py = self._python_path or "python"
        try:
            if self._python_path:
                # POSIX separators on purpose: the agent runs this through a
                # shell, and a Windows path like ``D:\4.work\...`` would have
                # ``\4`` eaten as an escape. Windows accepts forward slashes for
                # every API we need, so this form is safe on both platforms.
                _py = Path(self._python_path).resolve().as_posix()
        except OSError:
            pass
        (ws / "TASK.md").write_text(
            _TASK_MD_TEMPLATE
            .replace("{python}", _py)
            .replace("{metric}", metric)
            .replace("{time_budget_sec}", str(time_budget_sec)),
            encoding="utf-8",
        )

        # On a retry, spell out what went wrong last time.
        if retry_note:
            (ws / "RETRY.md").write_text(retry_note, encoding="utf-8")

        # Write experiment plan
        (ws / "EXPERIMENT_PLAN.yaml").write_text(
            exp_plan or "# No experiment plan provided\n",
            encoding="utf-8",
        )

        # Seed the retry with the previous attempt's partial output, so a new
        # session can resume/fix instead of starting from a blank repo.
        if prev_files:
            prev_dir = ws / "PREVIOUS_ATTEMPT"
            prev_dir.mkdir(parents=True, exist_ok=True)
            for fname, content in prev_files.items():
                p = prev_dir / fname
                p.parent.mkdir(parents=True, exist_ok=True)
                try:
                    p.write_text(content, encoding="utf-8")
                except OSError as exc:
                    logger.warning("Beast mode: failed to seed %s: %s", fname, exc)

        # Write the previous attempt's error/log so the retry knows what failed.
        if prev_error:
            (ws / "PREVIOUS_ATTEMPT_ERROR.txt").write_text(
                prev_error, encoding="utf-8",
            )

        # Write guidance document
        guidance_parts = [
            f"# Experiment Guidance\n",
            f"## Topic\n{topic}\n",
            f"## Primary Metric\n{metric}\n",
            f"## Time Budget\n{time_budget_sec} seconds\n",
        ]
        if pkg_hint:
            guidance_parts.append(f"## Environment\n{pkg_hint}\n")
        if extra_guidance:
            guidance_parts.append(f"## Additional Guidance\n{extra_guidance}\n")
        (ws / "GUIDANCE.md").write_text(
            "\n".join(guidance_parts), encoding="utf-8",
        )

        # Write opencode.json config
        opencode_cfg = self._build_opencode_config()
        (ws / "opencode.json").write_text(
            json.dumps(opencode_cfg, indent=2), encoding="utf-8",
        )

        # OpenCode requires a git repository — initialise one with
        # a single commit so that ``opencode run`` doesn't hang.
        # BUG-OB-01/OB-02: Check return codes and catch TimeoutExpired.
        try:
            r = subprocess.run(
                ["git", "init"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git init failed: {r.stderr}")
            r = subprocess.run(
                ["git", "add", "-A"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git add failed: {r.stderr}")
            r = subprocess.run(
                ["git", "-c", "user.email=beast@researchclaw",
                 "-c", "user.name=BeastMode",
                 "commit", "-m", "init workspace"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git commit failed: {r.stderr}")
        except subprocess.TimeoutExpired as exc:
            raise OSError(f"git workspace init timed out: {exc}") from exc

        return ws

    def _is_azure(self) -> bool:
        """Detect Azure OpenAI from base URL or provider string."""
        return (
            "azure" in (self._llm_base_url or "").lower()
            or "azure" in (self._llm_provider or "").lower()
        )

    def _build_opencode_config(self) -> dict[str, Any]:
        """Build the opencode.json configuration.

        Always uses the "openai" provider — this works for both standard
        OpenAI endpoints and Azure OpenAI (which accepts Bearer token auth
        on the ``/openai/v1`` path and now supports the Responses API).
        """
        cfg: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
        }

        if self._llm_base_url:
            if self._model:
                cfg["model"] = (
                    self._model if "/" in self._model
                    else f"openai/{self._model}"
                )
            cfg["provider"] = {
                "openai": {
                    "options": {
                        "baseURL": self._llm_base_url,
                        # ``_invoke_opencode`` exports the resolved key (from
                        # llm.api_key or llm.api_key_env) as OPENAI_API_KEY, so
                        # point at that one name instead of the configured env
                        # var — this also works when the key came from
                        # ``llm.api_key``, and keeps the literal secret out of
                        # the on-disk config.
                        "apiKey": "{env:OPENAI_API_KEY}"
                        if self._resolve_api_key()
                        else "",
                    },
                    "models": {},
                }
            }
            # Register the model so OpenCode knows it exists
            if self._model:
                model_name = self._model.split("/")[-1]
                cfg["provider"]["openai"]["models"] = {
                    model_name: {
                        "name": model_name,
                        "modalities": {
                            "input": ["text"],
                            "output": ["text"],
                        },
                    }
                }
        elif self._model:
            cfg["model"] = (
                self._model if "/" in self._model
                else f"openai/{self._model}"
            )

        return cfg

    # -- model resolution -------------------------------------------------------

    def _resolve_opencode_model(self) -> str:
        """Resolve the model identifier for OpenCode CLI's ``-m`` flag.

        Resolution order:
        1. If model already contains "/" (e.g. "anthropic/claude-sonnet-4-6") → use as-is
        2. Otherwise → "openai/{model}" (works for both Azure and standard OpenAI)

        Note: Azure AI Services now supports the Responses API with Bearer
        token auth via the OpenAI-compatible endpoint, so we use the "openai"
        provider universally — no Anthropic fallback needed.
        """
        if not self._model:
            return "anthropic/claude-sonnet-4-6"
        if "/" in self._model:
            return self._model
        return f"openai/{self._model}"

    # -- credentials -----------------------------------------------------------

    def _resolve_api_key(self) -> str:
        """Resolve the API key, preferring the configured value over the env var.

        Mirrors the resolution order used for the main LLM client
        (``llm.api_key or os.environ[llm.api_key_env]``): a key written directly
        into the config wins, and ``api_key_env`` names a variable to read when
        it isn't. Previously only the env-var path existed here, so a key set as
        ``llm.api_key`` never reached the CLI.
        """
        if self._api_key:
            return self._api_key
        if self._api_key_env:
            return os.environ.get(self._api_key_env, "")
        return ""

    # -- invocation ------------------------------------------------------------

    def _invoke_opencode(
        self,
        workspace: Path,
        prompt: str,
    ) -> tuple[bool, str, float]:
        """Run ``opencode run`` in the workspace. Returns (success, log, elapsed)."""
        env = os.environ.copy()
        # Pass API key via environment if configured. The key still has to reach
        # the CLI as an env var (that is opencode's only input for it), but we
        # take it straight from the resolved config when it was set there rather
        # than requiring a round-trip through the ambient environment.
        api_key = self._resolve_api_key()
        if api_key:
            # We always use the "openai" provider for OpenCode now,
            # which reads OPENAI_API_KEY (works for Azure too via
            # Bearer token auth on the OpenAI-compatible endpoint).
            env["OPENAI_API_KEY"] = api_key

        # Use -m flag to specify model (more reliable than opencode.json)
        resolved_model = self._resolve_opencode_model()
        opencode_cmd = shutil.which("opencode") or "opencode"
        cmd = self._build_opencode_command(opencode_cmd, resolved_model, prompt)

        t0 = time.monotonic()
        # Stream live: unlike subprocess.run(capture_output=True), which blocks
        # until the process exits, we read the merged stdout/stderr line-by-line
        # from a background thread and echo each line to the logger as it
        # arrives. The full text is still collected into ``log`` so the caller's
        # return value and opencode_log.txt are unchanged.
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(workspace),
                # ``opencode run`` reads stdin (to allow piped input) and only
                # starts work once it sees EOF. Whatever stdin the host process
                # happens to have is therefore load-bearing: a console or closed
                # handle gives EOF immediately, but an open-and-never-written
                # pipe (e.g. when launched from an IDE run configuration) leaves
                # opencode blocked forever. Pin it to DEVNULL so EOF is
                # immediate regardless of how the pipeline itself was launched.
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
        except FileNotFoundError:
            return False, "opencode CLI not found", 0.0
        except Exception as exc:  # noqa: BLE001
            return False, f"Unexpected error: {exc}", 0.0

        log_parts: list[str] = []

        def _reader() -> None:
            assert proc is not None and proc.stdout is not None
            for line in proc.stdout:
                _line = line.rstrip("\n")
                log_parts.append(line)
                if _line.strip():
                    logger.info("[opencode] %s", _line.rstrip())
            # On EOF the pipe closes even if we timeout-kill the process, so the
            # reader thread terminates cleanly either way.

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        try:
            proc.wait(timeout=self._timeout_sec)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            proc.kill()
            reader.join(timeout=2.0)
            log = "".join(log_parts)
            log += f"\nTIMEOUT after {elapsed:.1f}s"
            logger.warning("opencode timed out after %.1fs, killed", elapsed)
            return False, log, elapsed
        reader.join(timeout=2.0)
        elapsed = time.monotonic() - t0
        log = "".join(log_parts)
        return proc.returncode == 0, log, elapsed

    @staticmethod
    def _build_opencode_command(
        opencode_cmd: str, resolved_model: str, prompt: str
    ) -> list[str]:
        """Build the argv for ``opencode run``, wrapping with a pseudo-TTY on Linux.

        The ``opencode`` CLI requires a TTY: when invoked via ``subprocess.run``
        with piped stdout/stderr it can return exit 0 with empty output and no
        generated files. On Linux we wrap the call with util-linux
        ``script -q -e -c "<cmd>" /dev/null`` to provide a pseudo-TTY:

        * ``-q`` suppresses ``script``'s start/done messages,
        * ``-c`` runs the requested command,
        * ``-e`` returns the child's exit status (without it ``script`` can
          return 0 even when the child command fails, which would mask an
          ``opencode`` failure as success — the very silent-success behaviour
          this wrapper exists to avoid).

        The wrapper is gated on Linux specifically because the
        ``script -q -e -c ... /dev/null`` form is util-linux syntax; BSD/macOS
        ``script`` implementations are not compatible with it. On non-Linux
        platforms, and when ``script`` is unavailable, we fall back to invoking
        ``opencode`` directly — i.e. the prior behaviour, no regression.
        """
        direct = [opencode_cmd, "run", "-m", resolved_model, "--format", "json", prompt]

        script_path = shutil.which("script")
        if sys.platform.startswith("linux") and script_path:
            inner = " ".join(shlex.quote(part) for part in direct)
            return [script_path, "-q", "-e", "-c", inner, "/dev/null"]
        return direct

    # -- file collection -------------------------------------------------------

    # Scaffolding files written into the workspace as *input* for the agent.
    # These are not experiment code — leaking them downstream trips the stage-10
    # validation checks and pollutes the package, so they are never collected.
    _INPUT_SCAFFOLD_FILES = frozenset(
        {
            "TASK.md",
            "RETRY.md",
            "GUIDANCE.md",
            "EXPERIMENT_PLAN.yaml",
            "opencode.json",
            "PREVIOUS_ATTEMPT_ERROR.txt",
        }
    )
    _INPUT_SCAFFOLD_DIRS = frozenset({"PREVIOUS_ATTEMPT"})

    @staticmethod
    def _collect_files(workspace: Path) -> dict[str, str]:
        """Collect generated files (Python, JSON data, requirements, etc.).

        Relative subdirectory paths are preserved (e.g. ``algorithms/nm/nm.py``,
        ``data/rastrigin_d2_s0.json``) so nested structures produced by the
        LLM4AD task-package guidance survive.  Files are keyed by their path
        relative to the workspace root; when two files share that path the one
        closer to the root wins.

        Input scaffolding (TASK.md, GUIDANCE.md, the plan, PREVIOUS_ATTEMPT/…)
        is excluded — only the agent's own output is returned.
        """
        files: dict[str, str] = {}
        # Sort by depth (fewer parts first) so root-level files take priority.
        all_files = sorted(
            workspace.rglob("*"),
            key=lambda p: len(p.relative_to(workspace).parts),
        )
        for fpath in all_files:
            if not fpath.is_file():
                continue
            try:
                rel = fpath.relative_to(workspace)
            except ValueError:
                continue
            parts = rel.parts
            if any(p.startswith("__pycache__") or p.startswith(".") for p in parts):
                continue
            # Preserve relative subdirectory paths; skip dotfiles.
            if parts and parts[-1].startswith("."):
                continue
            # Skip the scaffolding we wrote in as input.
            if len(parts) == 1 and parts[0] in OpenCodeBridge._INPUT_SCAFFOLD_FILES:
                continue
            if parts[0] in OpenCodeBridge._INPUT_SCAFFOLD_DIRS:
                continue
            key = rel.as_posix()
            if key in files:
                continue
            try:
                files[key] = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Beast mode: failed to read %s: %s", fpath, exc)

        return files

    # -- entry-point validation ------------------------------------------------

    @staticmethod
    def _has_main_guard(source: str) -> bool:
        """Return True if *source* contains ``if __name__ == "__main__":``."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
                    if test.left.id == "__name__" and len(test.comparators) == 1:
                        comp = test.comparators[0]
                        if isinstance(comp, ast.Constant) and comp.value == "__main__":
                            return True
        return False

    @staticmethod
    def _is_entry_swap_candidate(fname: str) -> bool:
        """Whether ``fname`` may be swapped into ``main.py`` (BUG-17 guard).

        Only top-level ``.py`` files that are not fixed-role modules qualify.
        Anything nested (``algorithms/nm/nm.py``) or role-bound
        (``benchmarks.py``, ``evaluator.py``, ``stats_utils.py``, ``setup.py``,
        ``conftest.py``, ``test_*.py``) is off limits: those files have a
        contract with the rest of the package and moving their contents breaks
        both ends of the swap.
        """
        if fname == "main.py" or not fname.endswith(".py"):
            return False
        norm = fname.replace("\\", "/")
        if "/" in norm:
            return False  # nested module — never a top-level driver
        if norm in _ENTRY_SWAP_EXCLUDED:
            return False
        base = norm[:-3]
        return not (base.startswith("test_") or base.endswith("_test"))

    @staticmethod
    def _ensure_main_entry_point(files: dict[str, str]) -> dict[str, str]:
        """Ensure ``main.py`` has an ``if __name__ == "__main__"`` guard.

        Beast Mode often generates multi-file projects where ``main.py`` is a
        library module and the real entry point lives in another file (e.g.
        ``run_experiment.py``).  Since the Docker sandbox always executes
        ``python3 main.py``, a library-only ``main.py`` exits immediately with
        no output.

        Strategy:
        1. If ``main.py`` already has the guard → return unchanged.
        2. Find the first *eligible* other ``.py`` file that **does** have the
           guard.  Eligible means top-level (no ``/`` in the name) and not one
           of the fixed-role modules — see ``_ENTRY_SWAP_EXCLUDED``.
        3. Swap: rename that file to ``main.py`` and the old ``main.py`` to a
           helper module (its original basename, or ``_lib.py``).
        4. If no file has a guard, append a minimal stub to ``main.py`` that
           calls the most likely entry function (``main()``, ``run()``, etc.).

        BUG-17: strategies 2/3 used to consider **every** ``.py`` key. With the
        LLM4AD layout the dict carries ``algorithms/<algo>/<algo>.py``,
        ``benchmarks.py``, ``evaluator.py`` and ``stats_utils.py``, each of which
        may legitimately carry a ``__main__`` guard — swapping one into
        ``main.py`` corrupts both files. Restricting the candidate set to
        plausible top-level drivers makes that impossible.
        """
        main_code = files.get("main.py", "")
        if not main_code:
            return files

        if OpenCodeBridge._has_main_guard(main_code):
            return files

        # -- Strategy 2/3: find another file with the guard and swap -----------
        for fname, code in files.items():
            if not OpenCodeBridge._is_entry_swap_candidate(fname):
                continue
            if OpenCodeBridge._has_main_guard(code):
                logger.info(
                    "Beast mode: main.py lacks __main__ guard; swapping "
                    "entry point with %s",
                    fname,
                )
                new_files = dict(files)
                # Rename original main.py → helper module
                helper_name = fname  # reuse the other file's name for old main
                new_files[helper_name] = main_code
                new_files["main.py"] = code
                return new_files

        # -- Strategy 4: inject a minimal entry point into main.py -------------
        # Look for common entry functions defined in main.py
        entry_func: str | None = None
        try:
            tree = ast.parse(main_code)
            candidates = [
                n.name
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in ("main", "run", "run_experiment", "train",
                               "run_experiments", "experiment", "run_all")
            ]
            if candidates:
                entry_func = candidates[0]
        except SyntaxError:
            pass

        if entry_func:
            logger.info(
                "Beast mode: main.py lacks __main__ guard; injecting call "
                "to %s()",
                entry_func,
            )
            new_files = dict(files)
            new_files["main.py"] = (
                main_code.rstrip()
                + "\n\n\nif __name__ == \"__main__\":\n"
                + f"    {entry_func}()\n"
            )
            return new_files

        logger.warning(
            "Beast mode: main.py lacks __main__ guard and no known entry "
            "function found — experiment may exit without producing output",
        )
        return files

    # -- main entry point ------------------------------------------------------

    def generate(
        self,
        stage_dir: Path,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str = "",
        extra_guidance: str = "",
        time_budget_sec: int = 300,
    ) -> OpenCodeResult:
        """Run OpenCode to generate experiment code.

        Returns an OpenCodeResult with success status and generated files.
        """
        # Check availability first
        if not self.check_available():
            return OpenCodeResult(
                success=False,
                error="OpenCode CLI not installed or not callable",
            )

        workspace: Path | None = None
        last_error = ""
        last_log = ""
        files: dict[str, str] = {}
        elapsed = 0.0
        prev_error = ""
        prev_files: dict[str, str] = {}
        prev_had_no_files = False

        # BUG-03: files salvaged from a FAILED attempt are a last resort, not a
        # reason to stop retrying. Stash them and keep going — a later attempt
        # may still succeed cleanly, and it gets the salvaged output as
        # PREVIOUS_ATTEMPT context. Only if every attempt fails do we return the
        # stash, flagged via ``recovered_from_failure``.
        salvaged_files: dict[str, str] = {}
        salvaged_error = ""
        salvaged_elapsed = 0.0
        # BUG-07: keep at most ONE failed workspace for post-mortem, since retries
        # used to leak every workspace (each nesting the previous attempt's copy).
        kept_workspace: Path | None = None

        def _retire(ws: Path | None) -> None:
            """Drop ``ws``, keeping it only if it is the newest failed one."""
            nonlocal kept_workspace
            if ws is None:
                return
            if kept_workspace is not None and kept_workspace != ws:
                _rmtree_force(kept_workspace)
            kept_workspace = ws

        for attempt in range(1 + self._max_retries):
            # On a retry, tell the agent what failed last time — a byte-identical
            # prompt reliably reproduced the same failure (notably a clarifying
            # question and no written files).
            retry_note = ""
            if attempt > 0:
                retry_note = (
                    _RETRY_MD_TEMPLATE
                    .replace("{attempt}", str(attempt + 1))
                    .replace("{total}", str(1 + self._max_retries))
                    .replace(
                        "{no_files_note}",
                        _NO_FILES_NOTE if prev_had_no_files else "",
                    )
                    .replace("{reason}", last_error or "unknown")
                )

            # Prepare workspace
            try:
                workspace = self._prepare_workspace(
                    stage_dir=stage_dir,
                    topic=topic,
                    exp_plan=exp_plan,
                    metric=metric,
                    pkg_hint=pkg_hint,
                    extra_guidance=extra_guidance,
                    time_budget_sec=time_budget_sec,
                    prev_error=prev_error,
                    prev_files=prev_files or None,
                    retry_note=retry_note,
                )
            except OSError as exc:
                last_error = f"Failed to prepare workspace: {exc}"
                logger.warning("Beast mode: %s", last_error)
                continue

            # Build the CLI prompt. It only points at TASK.md, so it stays well
            # inside the argv length limit regardless of plan/guidance size. Use
            # replace, not .format(), to avoid KeyError when the metric contains
            # curly braces like "F{1}".
            prompt = _MEGA_PROMPT_TEMPLATE.replace(
                "{metric}", metric
            ).replace(
                "{time_budget_sec}", str(time_budget_sec)
            )
            if retry_note:
                prompt = _RETRY_PROMPT_PREFIX + prompt

            logger.info(
                "Beast mode: invoking OpenCode (attempt %d/%d, timeout=%ds, "
                "prompt=%d chars)",
                attempt + 1,
                1 + self._max_retries,
                self._timeout_sec,
                len(prompt),
            )

            success, log, elapsed = self._invoke_opencode(workspace, prompt)
            last_log = log

            if success:
                files = self._collect_files(workspace)
                if "main.py" not in files:
                    logger.warning(
                        "Beast mode: OpenCode succeeded but no main.py found "
                        "(files: %s)", list(files.keys()),
                    )
                    # Distinguish "wrote nothing at all" (replied with text
                    # instead of calling tools) from "wrote code but no main.py"
                    # — they need different retry guidance.
                    if not any(f.endswith(".py") for f in files):
                        last_error = (
                            "OpenCode wrote no code files at all — it replied "
                            "with text instead of generating the experiment"
                        )
                        prev_had_no_files = True
                    else:
                        last_error = "No main.py in OpenCode output"
                        prev_had_no_files = False
                    # Carry this attempt's output + reason to the next retry.
                    # Keep the reason COMPACT: a full log here re-enters RETRY.md
                    # and, once the agent reads RETRY.md, the next attempt's log.
                    prev_error = _summarize_log(log)
                    prev_files = files
                    # BUG-07: keep the workspace so partial output can be
                    # inspected — but only the newest one. This path used to
                    # skip cleanup entirely and leak one directory per retry.
                    _retire(workspace)
                    continue

                # BUG-R52-01: Ensure main.py has an entry point.
                files = self._ensure_main_entry_point(files)

                # Write log
                try:
                    (stage_dir / "opencode_log.txt").write_text(
                        log or "", encoding="utf-8",
                    )
                except OSError as _wexc:
                    logger.warning("Beast mode: failed to write log: %s", _wexc)

                # Cleanup workspace if configured. A clean success supersedes
                # every earlier failed attempt, so drop those too.
                if self._workspace_cleanup:
                    if workspace.exists():
                        _rmtree_force(workspace)
                    if kept_workspace is not None:
                        _rmtree_force(kept_workspace)
                        kept_workspace = None

                return OpenCodeResult(
                    success=True,
                    files=files,
                    opencode_log=log,
                    elapsed_sec=elapsed,
                )

            last_error = _summarize_log(log)
            logger.warning(
                "Beast mode: OpenCode attempt %d failed (%.1fs): %s",
                attempt + 1,
                elapsed,
                log[:500],
            )
            # Recover output even on failure/timeout: opencode may have written a
            # complete package before the process was killed (e.g. timed out
            # during final full-run validation). Prefer that over discarding it;
            # only wipe the workspace when there is nothing useful to keep.
            if workspace and workspace.exists():
                recovered = self._collect_files(workspace)
                prev_error = _summarize_log(log)
                prev_files = recovered
                if "main.py" not in recovered:
                    if self._workspace_cleanup:
                        _rmtree_force(workspace)
                    else:
                        _retire(workspace)
                else:
                    logger.info(
                        "Beast mode: recovered %d file(s) from failed attempt; "
                        "stashing as fallback and continuing to retry",
                        len(recovered),
                    )
                    # BUG-03: stash, do NOT break. The newest salvage wins —
                    # it saw the accumulated retry guidance.
                    salvaged_files = recovered
                    salvaged_error = log
                    salvaged_elapsed = elapsed
                    _retire(workspace)

        if salvaged_files:
            files = self._ensure_main_entry_point(salvaged_files)
            if self._workspace_cleanup and kept_workspace is not None:
                _rmtree_force(kept_workspace)
                kept_workspace = None
            logger.warning(
                "Beast mode: all %d attempt(s) failed; falling back to %d "
                "file(s) salvaged from a failed run — package was never "
                "validated by the agent",
                1 + self._max_retries,
                len(files),
            )
            return OpenCodeResult(
                success=True,
                files=files,
                opencode_log=salvaged_error,
                elapsed_sec=salvaged_elapsed,
                recovered_from_failure=True,
            )

        # All attempts failed. The full transcript used to be dropped here (only
        # the last attempt's final line survived in memory, nothing on disk), so
        # persist the whole log for post-mortem. `last_error` stays the compact
        # ~2000-char summary that fills RETRY.md's {reason} — the in-memory
        # opencode_log field keeps that instead of the full log so a caller's
        # retry prompt does not grow.
        if last_log:
            try:
                (stage_dir / "opencode_log.txt").write_text(
                    last_log or "", encoding="utf-8",
                )
            except OSError as _wexc:
                logger.warning("Beast mode: failed to write log: %s", _wexc)
        return OpenCodeResult(
            success=False,
            opencode_log=last_error,
            error=(
                f"OpenCode failed after {1 + self._max_retries} attempt(s): "
                f"{last_error or 'unknown'}"
            ),
        )


# ---------------------------------------------------------------------------
# Helper: count historical failures
# ---------------------------------------------------------------------------

def count_historical_failures(run_dir: Path, stage_name: str = "stage-10") -> int:
    """Count past Stage 10 failures from stage directories and logs.

    Each stage directory is counted at most once, even if multiple failure
    indicators are present.
    """
    failures = 0
    for d in run_dir.glob(f"{stage_name}*"):
        failed = False
        # Check for beast_mode_log.json
        bm_log = d / "beast_mode_log.json"
        if bm_log.exists():
            try:
                data = json.loads(bm_log.read_text(encoding="utf-8"))
                if not data.get("success", True):
                    failed = True
            except Exception:  # noqa: BLE001
                pass
        # Check for stage health failures
        if not failed:
            health = d / "stage_health.json"
            if health.exists():
                try:
                    data = json.loads(health.read_text(encoding="utf-8"))
                    if data.get("status") == "FAILED":
                        failed = True
                except Exception:  # noqa: BLE001
                    pass
        # Check for validation report with FAILED status
        if not failed:
            vr = d / "validation_report.md"
            if vr.exists():
                try:
                    content = vr.read_text(encoding="utf-8")
                    if "BLOCKED" in content or "FAILED" in content:
                        failed = True
                except Exception:  # noqa: BLE001
                    pass
        if failed:
            failures += 1
    return failures
