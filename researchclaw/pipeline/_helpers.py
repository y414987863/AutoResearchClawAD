"""Shared constants, data classes, and utility functions for the pipeline executor."""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from researchclaw.config import RCConfig
from researchclaw.hardware import HardwareProfile, is_metric_name
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline.stages import (
    NEXT_STAGE,
    Stage,
    StageStatus,
)
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageResult:
    """Outcome of executing a single stage."""

    stage: Stage
    status: StageStatus
    artifacts: tuple[str, ...]
    error: str | None = None
    decision: str = "proceed"
    evidence_refs: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NOTE: ``torch`` is intentionally absent — the deployment server has no GPU,
# so a torch import is a genuine error we want to surface, not auto-install.
_SANDBOX_SAFE_PACKAGES = {
    "numpy", "scipy", "sklearn", "matplotlib",
    "pandas", "seaborn", "tqdm", "gymnasium", "gym",
}

_METACLAW_SKILLS_DIR = str(Path.home() / ".metaclaw" / "skills")

# User-level custom skills directory (cross-project)
_USER_SKILLS_DIR = Path.home() / ".researchclaw" / "skills"

# Lazy-initialized skill registry (singleton for the process)
_skill_registry: object | None = None


def _get_skill_registry(config: object | None = None) -> object:
    """Return the global SkillRegistry, creating it on first call.

    Loads skills from (in priority order):
    1. Built-in skills shipped with the package
    2. User-level ``~/.researchclaw/skills/``
    3. Project-level ``.claude/skills/``
    4. MetaClaw cross-run skills ``~/.metaclaw/skills/``
    5. User-configured ``config.yaml → skills.custom_dirs``
    """
    global _skill_registry  # noqa: PLW0603
    if _skill_registry is not None:
        return _skill_registry
    try:
        from researchclaw.skills.registry import SkillRegistry

        custom_dirs: list[str] = []

        # User-level skills
        if _USER_SKILLS_DIR.is_dir():
            custom_dirs.append(str(_USER_SKILLS_DIR))

        # Project-level .claude/skills/
        project_skills = Path(__file__).resolve().parent.parent.parent / ".claude" / "skills"
        if project_skills.is_dir():
            custom_dirs.append(str(project_skills))

        # MetaClaw skills
        metaclaw = Path(_METACLAW_SKILLS_DIR)
        if metaclaw.is_dir():
            custom_dirs.append(str(metaclaw))

        # Config-specified custom dirs
        if config is not None:
            skills_cfg = getattr(config, "skills", None)
            if skills_cfg:
                for d in getattr(skills_cfg, "custom_dirs", ()):
                    if d:
                        custom_dirs.append(str(d))
                for d in getattr(skills_cfg, "external_dirs", ()):
                    if d:
                        custom_dirs.append(str(d))

        _skill_registry = SkillRegistry(
            custom_dirs=custom_dirs,
            auto_match=True,
            max_skills_per_stage=getattr(
                getattr(config, "skills", None), "max_skills_per_stage", 3
            ) if config else 3,
            fallback_matching=True,
        )
        logger.info(
            "Skill registry initialized: %d skills from %d sources",
            _skill_registry.count(),
            1 + len(custom_dirs),
        )
    except Exception:  # noqa: BLE001
        # Fallback: create empty registry so we never crash
        from researchclaw.skills.registry import SkillRegistry
        _skill_registry = SkillRegistry(builtin_dir="/dev/null")
        logger.debug("Skill registry init failed, using empty registry")
    return _skill_registry

# --- P1-1: Topic keyword extraction for domain pre-filter ---
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "of",
        "for",
        "to",
        "with",
        "by",
        "at",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "under",
        "over",
        "using",
        "based",
        "via",
        "toward",
        "towards",
        "new",
        "novel",
        "approach",
        "method",
        "study",
        "research",
        "paper",
        "work",
        "propose",
        "proposed",
    }
)

# ---------------------------------------------------------------------------
# Timestamp utility
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Fallback query builder
# ---------------------------------------------------------------------------


def _build_fallback_queries(topic: str) -> list[str]:
    """Extract meaningful search queries from a long topic string.

    Instead of using the raw topic as a query (which is often 200+ chars
    and returns garbage from search engines), extract noun phrases and
    domain keywords. Returns 5-10 targeted queries.
    """
    # Split on common delimiters and extract meaningful chunks
    chunks = re.split(r"[,:;()\[\]]+", topic)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 8]
    cleaned_chunks = []
    for c in chunks:
        c = re.sub(
            r"^(and|or|the|a|an|in|of|for|with|across|multiple|three|various)\s+",
            "", c, flags=re.IGNORECASE,
        )
        c = c.strip()
        if len(c) > 8:
            cleaned_chunks.append(c)
    chunks = cleaned_chunks

    # Extract key terms (words that look like domain terms, not stopwords)
    _stop = {
        "the", "and", "for", "with", "from", "that", "this", "into",
        "over", "across", "multiple", "three", "result", "comprehensive",
        "using", "based", "between", "various", "different", "several",
        "parameter", "parameters", "analysis", "approach", "method",
        "framework", "frameworks",
    }
    words = topic.lower().split()
    key_terms = [w for w in words if len(w) > 3 and w not in _stop]

    queries: list[str] = []

    # Strategy 1: Use meaningful chunks (up to 60 chars each)
    for chunk in chunks[:4]:
        if len(chunk) > 60:
            chunk = " ".join(chunk.split()[:6])
        if chunk and chunk not in queries:
            queries.append(chunk)

    # Strategy 2: Bigrams of key terms
    clean_terms = [t for t in key_terms if re.match(r"^[a-z]", t) and ":" not in t]
    for i in range(min(len(clean_terms) - 1, 4)):
        bigram = f"{clean_terms[i]} {clean_terms[i + 1]}"
        if bigram not in queries:
            queries.append(bigram)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q_lower = q.strip().lower()
        if q_lower and q_lower not in seen:
            seen.add(q_lower)
            unique.append(q.strip())

    # Ensure we have at least a few useful queries
    topic_short = topic[:60].strip()
    for suffix in ("survey", "review", "benchmark", "state of the art", "recent advances"):
        if len(unique) >= 5:
            break
        candidate = f"{topic_short} {suffix}".strip()
        if candidate.lower() not in seen:
            seen.add(candidate.lower())
            unique.append(candidate)

    return unique[:10]


# ---------------------------------------------------------------------------
# Stage metadata I/O
# ---------------------------------------------------------------------------


def _write_stage_meta(
    stage_dir: Path, stage: Stage, run_id: str, result: "StageResult"
) -> None:
    if result.status is StageStatus.DONE:
        next_stage = NEXT_STAGE[stage]
    else:
        # Failed / paused / blocked stages should point back to themselves so
        # retry-resume tooling does not imply that the pipeline advanced.
        next_stage = stage
    meta = {
        "stage_id": f"{int(stage):02d}-{stage.name.lower()}",
        "run_id": run_id,
        "status": result.status.value,
        "decision": result.decision,
        "output_artifacts": list(result.artifacts),
        "evidence_refs": list(result.evidence_refs),
        "error": result.error,
        "ts": _utcnow_iso(),
        "next_stage": int(next_stage) if next_stage is not None else None,
    }
    (stage_dir / "decision.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Sandbox dependency helper
# ---------------------------------------------------------------------------


def _ensure_sandbox_deps(code: str, python_path: str) -> list[str]:
    """P7: Scan code imports and auto-install missing common packages.

    Only packages in :data:`_SANDBOX_SAFE_PACKAGES` are considered. A package is
    skipped if it already imports; otherwise we install it and return its name.
    A failed install does **not** block the run — per the deployment decision,
    let the code blow up with the real ``ModuleNotFoundError`` rather than fail
    early on our own install guess. We just stop claiming it was installed.

    Prefers the interpreter's own pip first, then ``uv pip`` (uv-created venvs
    often ship without pip). Both are silent no-ops for already-satisfied
    packages, so re-running is harmless.
    """
    import subprocess as _sp

    imports: set[str] = set()
    for line in code.splitlines():
        m = re.match(r"^(?:from|import)\s+(\w+)", line.strip())
        if m:
            imports.add(m.group(1))

    to_check = imports & _SANDBOX_SAFE_PACKAGES
    if not to_check:
        return []

    py = python_path
    py_path = Path(py)
    if not py_path.is_absolute():
        py_path = Path.cwd() / py_path

    installed: list[str] = []
    for pkg in sorted(to_check):
        try:
            r = _sp.run(
                [str(py_path), "-c", f"import {pkg}"],
                capture_output=True, timeout=10,
                encoding="utf-8", errors="replace",
            )
            if r.returncode == 0:
                continue  # already available
            pip_name = "scikit-learn" if pkg == "sklearn" else pkg
            logger.info("Sandbox: installing missing dependency '%s'", pip_name)
            ok = _pip_install(str(py_path), pip_name)
            if ok:
                installed.append(pip_name)
            else:
                # Not fatal per decision above — just don't claim success. The
                # sandbox will surface the real ModuleNotFoundError if it needs
                # this package.
                logger.warning(
                    "Sandbox: failed to install '%s' via pip and uv. "
                    "Letting the experiment run; it will error on import if "
                    "the dependency is actually required.",
                    pip_name,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sandbox: failed to check/install '%s': %s", pkg, exc)

    if installed:
        logger.info("Sandbox: auto-installed packages: %s", ", ".join(installed))
    return installed


def _pip_install(python: str, pip_name: str) -> bool:
    """Install *pip_name* into the sandbox interpreter or the active uv env.

    Tries ``python -m pip`` first, then ``uv pip`` (which targets the active
    environment, so it works even where pip is missing). Returns True on a
    zero exit code — i.e. the package is now importable.
    """
    import subprocess as _sp

    # 1) Interpreter's own pip.
    try:
        r = _sp.run(
            [python, "-m", "pip", "install", pip_name, "--quiet"],
            capture_output=True, timeout=180,
            encoding="utf-8", errors="replace",
        )
        if r.returncode == 0:
            return True
        if "No module named pip" not in r.stderr:
            logger.warning(
                "Sandbox: pip install of '%s' failed: %s",
                pip_name, (r.stderr or r.stdout).strip()[-300:],
            )
        # else fall through to uv
    except Exception as exc:  # noqa: BLE001
        logger.debug("Sandbox: pip install '%s' raised: %s", pip_name, exc)

    # 2) uv as fallback for pip-less (uv-created) venvs. ``--python`` pins the
    # target interpreter explicitly — a bare ``uv pip install`` from an unset
    # shell targets the *system* Python, not this venv, which would install to
    # the wrong place.
    try:
        r = _sp.run(
            ["uv", "pip", "install", "--python", python, pip_name, "--quiet"],
            capture_output=True, timeout=180,
            encoding="utf-8", errors="replace",
        )
        if r.returncode == 0:
            return True
        logger.warning(
            "Sandbox: uv pip install of '%s' failed: %s",
            pip_name, (r.stderr or r.stdout).strip()[-300:],
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Sandbox: uv pip install '%s' raised: %s", pip_name, exc)

    return False


# ---------------------------------------------------------------------------
# Prior artifact I/O
# ---------------------------------------------------------------------------


def _read_best_analysis(run_dir: Path) -> str:
    """BUG-225: Read analysis.md from the best Stage 14 iteration.

    Prefers ``analysis_best.md`` at run root (written by
    ``_promote_best_stage14``) over ``_read_prior_artifact("analysis.md")``
    which may pick a degenerate non-versioned stage-14 directory.
    """
    best = run_dir / "analysis_best.md"
    if best.exists():
        return best.read_text(encoding="utf-8")
    return _read_prior_artifact(run_dir, "analysis.md") or ""


def _read_prior_artifact(run_dir: Path, filename: str) -> str | None:
    # R14-2: Sort so non-versioned dirs (stage-13) come before versioned (stage-13_v1).
    # Within the same stage number, prefer the latest (non-versioned) copy.
    def _stage_sort_key(p: Path) -> tuple[str, int]:
        name = p.name
        # Extract base stage name and version
        if "_v" in name:
            base, _, ver = name.rpartition("_v")
            try:
                return (base, -int(ver))  # Versioned: lower priority (negative version)
            except ValueError:
                return (name, -999)
        return (name, 0)  # Non-versioned: highest priority

    for stage_subdir in sorted(run_dir.glob("stage-*"), key=_stage_sort_key, reverse=True):
        candidate = stage_subdir / filename
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as exc:
                logger.warning("Cannot read %s: %s — skipping", candidate, exc)
                continue
        if filename.endswith("/") and (stage_subdir / filename.rstrip("/")).is_dir():
            return str(stage_subdir / filename.rstrip("/"))
    return None


# The generated experiment declares which way its primary metric is judged, in
# either the static dict (preferred — readable without running the code) or the
# runtime line it prints. The dict pattern stays inside one brace pair
# (``[^}]*``, which already spans newlines) rather than using a lazy ``.*?``:
# the lazy form walks past a METRIC_DEF that omits ``direction`` and latches
# onto the next unrelated dict that happens to have one.
_METRIC_DIRECTION_DICT_RE = re.compile(
    r'METRIC_DEF\s*=\s*\{[^}]*"direction"\s*:\s*"(maximize|minimize)"',
    re.IGNORECASE,
)
_METRIC_DIRECTION_PRINT_RE = re.compile(
    r"METRIC_DEF\s*:.*?direction\s*=\s*(higher|lower)", re.IGNORECASE
)


def correct_metric_direction(run_dir: Path, config: RCConfig) -> RCConfig:
    """Return ``config`` with the direction the generated experiment declares.

    Returns ``config`` unchanged when no experiment has been generated yet or it
    declares nothing. Called once per stage from the pipeline loop, so every
    stage after code generation reads a corrected value through the ordinary
    ``config.experiment.metric_direction`` and needs no lookup of its own.

    ``metric_direction`` in the config is a template default (``minimize``) that
    nobody judged against the metric the model actually generated. In the
    rc_full3 run that made promote read ``valid_prediction_time`` (``max(0,
    baseline - corrected)``, larger is better) as lower-is-better and ship a
    regression as a -100% "improvement". The code computing the metric is the
    only thing that knows which way is better, so its declaration wins.
    """
    exp_dir = _read_prior_artifact(run_dir, "experiment/")
    if not (exp_dir and Path(exp_dir).is_dir()):
        return config

    declared = ""
    for name in ("evaluator.py", "main.py"):
        try:
            # errors="replace": a model-generated file with one stray byte must
            # not raise UnicodeDecodeError here — this runs before every stage,
            # so an uncaught decode error would kill the whole run.
            text = (Path(exp_dir) / name).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        m = _METRIC_DIRECTION_DICT_RE.search(text)
        if m:
            declared = m.group(1).lower()
            break
        m = _METRIC_DIRECTION_PRINT_RE.search(text)
        if m:
            declared = "maximize" if m.group(1).lower() == "higher" else "minimize"
            break

    current = str(getattr(config.experiment, "metric_direction", "") or "").lower()
    if not declared or declared == current:
        return config

    logger.warning(
        "Metric direction corrected: config says %r but the generated experiment "
        "declares %r for metric %r. Using %r — the code that computes the metric "
        "is authoritative. Set experiment.metric_direction=%r in the config to "
        "silence this.",
        current, declared, config.experiment.metric_key, declared, declared,
    )
    return replace(
        config, experiment=replace(config.experiment, metric_direction=declared)
    )


def _find_prior_file(run_dir: Path, filename: str) -> Path | None:
    """Like ``_read_prior_artifact`` but returns the *Path* instead of content."""
    def _stage_sort_key(p: Path) -> tuple[str, int]:
        name = p.name
        if "_v" in name:
            base, _, ver = name.rpartition("_v")
            try:
                return (base, -int(ver))
            except ValueError:
                return (name, -999)
        return (name, 0)

    for stage_subdir in sorted(run_dir.glob("stage-*"), key=_stage_sort_key, reverse=True):
        candidate = stage_subdir / filename
        if candidate.is_file():
            return candidate
    return None


def _load_hardware_profile(run_dir: Path) -> dict[str, Any] | None:
    """Load hardware_profile.json from a prior stage (usually stage-01)."""
    raw = _read_prior_artifact(run_dir, "hardware_profile.json")
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------


def _extract_yaml_block(text: str) -> str:
    """Extract YAML from text that may contain ACP noise.

    Strips [thinking] blocks, insight blocks, and other ACP artifacts
    before looking for YAML in markdown fences or raw text.
    """
    # Strip ACP noise: [thinking]..., insight blocks, [plan]...
    cleaned = re.sub(
        r"\[thinking\].*?(?=\n```|\n[A-Z]|\Z)",
        "", text, flags=re.DOTALL,
    )
    cleaned = re.sub(r"\[plan\].*?\n\n", "", cleaned, flags=re.DOTALL)

    # Try markdown fences first (most reliable) — on cleaned text
    if "```yaml" in cleaned:
        return cleaned.split("```yaml", 1)[1].split("```", 1)[0].strip()
    if "```yml" in cleaned:
        return cleaned.split("```yml", 1)[1].split("```", 1)[0].strip()
    if "```" in cleaned:
        block = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        if block:
            return block

    # Try the original text too (in case cleaning removed too much)
    if "```yaml" in text:
        return text.split("```yaml", 1)[1].split("```", 1)[0].strip()
    if "```yml" in text:
        return text.split("```yml", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        block = text.split("```", 1)[1].split("```", 1)[0].strip()
        if block:
            return block

    # Last resort: try to find YAML-like content (lines starting with key:)
    yaml_lines: list[str] = []
    in_yaml = False
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not in_yaml and re.match(r"^[a-z_]+:", stripped):
            in_yaml = True
        if in_yaml:
            if stripped and not stripped.startswith("#"):
                yaml_lines.append(line)
            elif not stripped and yaml_lines:
                yaml_lines.append(line)
    if yaml_lines:
        return "\n".join(yaml_lines).strip()

    return text.strip()


_YAML_BLOCK_HEADER_RE = re.compile(r"^[|>][+-]?\d*\s*$")
_YAML_KEY_SEP_RE = re.compile(r":(?:\s|$)")


def _repair_yaml_line(line: str) -> str | None:
    """Quote the scalar value on ``line`` so PyYAML stops rejecting it.

    Models routinely emit values that open with a YAML indicator character —
    ``||x - x*||_2`` (a norm), ``**bold**``, ``*.py``, ``> 0.5`` — or that
    carry a bare ``:`` inside the value.  Wrapping the value in double quotes
    makes it a plain string without changing what it says.

    Returns None when there is nothing safe to rewrite, which tells the caller
    to give up rather than risk corrupting the document.
    """
    stripped = line.lstrip()
    if not stripped or stripped.startswith(("#", "---", "...")):
        return None
    indent = line[: len(line) - len(stripped)]

    prefix = ""
    inner = stripped
    while inner.startswith("- "):
        prefix += "- "
        inner = inner[2:]
    if not inner:
        return None

    if inner[0] in "{[":
        # A flow collection (``- {lr: 0.01}``) is valid structure. Bailing out
        # before the key split matters: the split would otherwise cut at the
        # ``:`` *inside* the braces and quote the fragment, turning
        # ``- {a: 1}`` into ``- {a: "1}"``. The ``value[0] in "{["`` guard
        # further down cannot catch this — after the split the value is ``1}``.
        return None

    match = _YAML_KEY_SEP_RE.search(inner)
    if match and inner[0] not in "'\"":
        key = inner[: match.start() + 1]
        value = inner[match.end():].strip()
    else:
        # A bare list item (``- ||x||``): the whole entry is the value.  A
        # quoted scalar is left alone — any ``:`` inside it is part of the
        # string, not a key separator.
        if not prefix or inner[0] in "'\"":
            return None
        key = ""
        value = inner

    # Leave real block scalars (``key: |``) and flow collections
    # (``[a, b]`` / ``{lr: 0.01}``) alone — those are valid structure.
    if not value or _YAML_BLOCK_HEADER_RE.match(value) or value[0] in "{[":
        return None

    cleaned = value.strip("\"'").replace("\\", "\\\\").replace('"', '\\"')
    if not cleaned:
        return None
    separator = f"{key} " if key else ""
    return f'{indent}{prefix}{separator}"{cleaned}"'


def _describe_yaml_failure(text: str) -> str:
    """Return a one-line diagnosis of why ``text`` did not yield a plan.

    ``_safe_load_yaml`` swallows the parser error so callers can fall back, but
    that leaves the failure log unable to tell a stray colon in one scalar from
    a truncated response — and the document is often 14 KB, so "first 200
    chars" shows nothing about the actual defect. This re-parses on the failure
    path only, and reports the parser's own position plus the offending line.

    Also distinguishes the case where YAML parsed *fine* but produced a list or
    a string, which the caller rejects for a different reason entirely.
    """
    if not text or not text.strip():
        return "empty content"
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None) or getattr(
            exc, "context_mark", None
        )
        problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        if mark is None:
            return str(problem)
        lines = text.split("\n")
        src = (
            lines[mark.line].strip()
            if 0 <= mark.line < len(lines)
            else "<line out of range>"
        )
        return "%s at line %d col %d: %r" % (
            problem, mark.line + 1, mark.column + 1, src[:160],
        )
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return (
        f"valid YAML but top level is {type(parsed).__name__}, not a mapping"
    )


_YAML_NUMBERED_ITEM_RE = re.compile(r"^(\s*)(\d+[.)])(\s+)(\S.*)$")
_YAML_EMPTY_VALUE_KEY_RE = re.compile(
    r"^\s*(?:-\s+)?(?:[^\s:#][^:]*|\"[^\"]*\"|'[^']*'):\s*$"
)
_YAML_BLOCK_SCALAR_RE = re.compile(r":\s*[|>][+-]?\d*\s*$")


def _normalize_markdown_lists(text: str) -> str:
    """Turn markdown numbered lists back into YAML sequences.

    Models routinely answer a key that wants a list with the prose form they
    were trained on::

        algorithm_steps:
          1. Compute sample covariance matrix
          2. Estimate optimal shrinkage intensity
          3. Shrink toward identity: Sigma = d * I + (1-d) * S

    YAML reads that as one plain multi-line scalar, so the ``:`` on the third
    line raises "mapping values are not allowed here". ``_repair_yaml_line``
    cannot help: quoting the value leaves the line a mapping inside a scalar
    and the parser fails at the same spot, so the repair loop spends its budget
    without progress. An observed Stage 9 response carried 20 such lines across
    5 blocks — past ``max_repairs`` even if each one had been fixable.

    Conversion is deliberately narrow. A line is rewritten only when it is
    indented under a key that declared no value, or continues a run already
    being rewritten, and never inside a ``|``/``>`` block scalar where a
    numbered list is intended as literal text. The number is kept in the string
    because the ordering it encodes is part of the content.
    """
    out: list[str] = []
    block_indent: int | None = None  # inside a |/> scalar started at this indent
    key_indent: int | None = None  # nearest key that declared an empty value
    run_indent: int | None = None  # indent of the run currently being rewritten

    for line in text.split("\n"):
        indent = len(line) - len(line.lstrip())
        if block_indent is not None:
            if line.strip() and indent > block_indent:
                out.append(line)
                continue
            block_indent = None
        if _YAML_BLOCK_SCALAR_RE.search(line):
            block_indent, key_indent, run_indent = indent, None, None
            out.append(line)
            continue

        match = _YAML_NUMBERED_ITEM_RE.match(line)
        if match and (
            run_indent == indent
            or (key_indent is not None and indent > key_indent)
        ):
            prefix, number, _, rest = match.groups()
            body = f"{number} {rest}".replace("\\", "\\\\").replace('"', '\\"')
            out.append(f'{prefix}- "{body}"')
            run_indent = indent
            continue

        run_indent = None
        if line.strip():
            key_indent = (
                indent if _YAML_EMPTY_VALUE_KEY_RE.match(line) else None
            )
        out.append(line)
    return "\n".join(out)


def _safe_load_yaml(text: str, max_repairs: int = 12) -> Any:
    """Parse YAML, repairing only the lines the parser actually rejects.

    ``yaml.safe_load`` is all-or-nothing: one bad scalar discards an otherwise
    complete document.  For LLM output that means a well-formed experiment plan
    can be thrown away over a single character, and the caller silently falls
    back to a generic template.

    Repairs are driven by the parser's own error position, so valid content is
    never rewritten — whole-document rewriting is not safe here, because it
    mistakes multi-line quoted scalars for unterminated quotes.  Returns None
    when the document cannot be recovered.
    """
    if not text or not text.strip():
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        pass

    # Tabs are never legal YAML indentation, but models emit them anyway.
    lines = [
        re.sub(r"^\t+", lambda m: "  " * len(m.group()), ln)
        for ln in text.split("\n")
    ]
    # Markdown numbered lists defeat the line-driven repair below, so fold them
    # into sequences first. Failure path only: a document that already parsed
    # returned above and is never touched.
    lines = _normalize_markdown_lists("\n".join(lines)).split("\n")
    repaired: set[int] = set()
    for _ in range(max_repairs):
        try:
            return yaml.safe_load("\n".join(lines))
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None) or getattr(
                exc, "context_mark", None
            )
            if mark is None or mark.line >= len(lines) or mark.line in repaired:
                return None
            repaired.add(mark.line)
            fixed = _repair_yaml_line(lines[mark.line])
            if fixed is None or fixed == lines[mark.line]:
                return None
            lines[mark.line] = fixed
    return None


def _safe_json_loads(text: str, default: Any) -> Any:
    """Parse JSON from text, handling noisy ACP output.

    Tries multiple strategies: direct parse, markdown fence extraction,
    balanced brace matching (largest dict wins), and array brackets.
    """
    if not text or not text.strip():
        return default

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        pass

    # Strategy 2: Find JSON in markdown code fences
    fence_pattern = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
    for match in fence_pattern.finditer(text):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 3: Find outermost balanced braces
    brace_depth = 0
    start = -1
    candidates: list[str] = []
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                candidates.append(text[start : i + 1])
                start = -1

    # Try candidates from largest to smallest
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 4: Same for array [ ]
    bracket_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "[":
            if bracket_depth == 0:
                start = i
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth == 0 and start >= 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass
                start = -1

    return default


def _is_python_info_string(info: str) -> bool:
    """True if a fence info string denotes Python (or is unlabelled).

    Accepts ``""``, ``python``, ``py``, ``python:main.py``, ``python filename:x.py``
    and rejects ``bash``, ``json``, ``yaml``, ``text`` … so a shell/JSON block
    elsewhere in the reply is not mistaken for the code.
    """
    info = info.strip()
    if not info:
        return True
    # First token up to a separator is the language tag.
    tag = re.split(r"[\s:=]", info, maxsplit=1)[0].lower()
    if tag in ("python", "python3", "py", ""):
        return True
    # Unlabelled but path-carrying: ```:main.py or ``` filename:main.py
    return tag.endswith(".py") or "filename" in info.lower()


def _extract_code_block(content: str) -> str:
    # Consume the whole info string on the opening fence (```python:foo.py,
    # ```py title=x, ```python) so it never lands in the code body. Matching
    # only ``` plus an optional `python` left a `:foo.py` remnant as the first
    # line (a SyntaxError). Non-Python fences are skipped, not returned, so a
    # leading ```bash / ```json block isn't mistaken for the code.
    for match in re.finditer(r"```([^\n`]*)\n(.*?)\s*```", content, flags=re.DOTALL):
        if _is_python_info_string(match.group(1)):
            return match.group(2).strip()
    # Fenced snippet with no newline before the code: ```code```
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    # No fence found. Only trust the raw response as code if it actually
    # parses as Python — otherwise a conversational reply (e.g. "Same, no
    # change.") gets accepted verbatim and silently corrupts the file.
    stripped = content.strip()
    if not stripped:
        return ""
    try:
        ast.parse(stripped)
    except SyntaxError:
        return ""
    return stripped


def _extract_multi_file_blocks(content: str) -> dict[str, str]:
    """Parse LLM response containing multiple files with filename markers.

    Expected format::

        ```filename:main.py
        import model
        ...
        ```

        ```filename:model.py
        class MyModel:
        ...
        ```

    Also handles common LLM format variations:
    - ````` ```python:main.py````` (colon directly after the language tag)
    - ````` ```python filename:main.py````` (space before filename)
    - ````` ``` filename:main.py````` (space after backticks)
    - ``filename:main.py`` on next line after backticks
    - ``# FILE: main.py`` comment markers inside code blocks

    Falls back to treating the entire code block as ``main.py`` if no
    ``filename:`` markers are found.

    Returns a dict mapping filename → code content.
    """
    # R13-2: Multiple patterns to handle LLM format variations
    patterns = [
        # ```python:xxx.py or ```:xxx.py — no `filename:` keyword, the path
        # follows the language tag directly. Models emit this often; when it
        # went unrecognized every file collapsed into the single-block
        # fallback below, which yielded one main.py holding another file's
        # body with a stray `:xxx.py` first line.
        re.compile(
            r"```(?:python)?:(\S+\.\w+)[^\S\n]*\n(.*?)```",
            flags=re.DOTALL,
        ),
        # Original: ```filename:xxx.py or ```python filename:xxx.py
        re.compile(
            r"```(?:python\s+)?filename:(\S+)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
        # Variation: ``` filename:xxx.py (space after backticks)
        re.compile(
            r"```\s+filename:(\S+)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
        # Variation: ```python\nfilename:xxx.py (filename on next line)
        re.compile(
            r"```(?:python)?\s*\nfilename:(\S+)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
        # Variation: ```python\n# filename: xxx.py (comment marker)
        re.compile(
            r"```(?:python)?\s*\n#\s*(?:FILE|filename)\s*:\s*(\S+)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
    ]

    matches: list[tuple[str, str]] = []
    for pattern in patterns:
        matches = pattern.findall(content)
        if matches:
            break

    if matches:
        files: dict[str, str] = {}
        for fname, code in matches:
            fname = fname.strip()
            # Security: prevent path traversal
            if ".." in fname or fname.startswith("/"):
                continue
            # Keep relative subdirectory paths (strip leading ./) so nested
            # files like algorithms/<algo>/<algo>.py and data/*.json survive.
            fname = fname.replace("\\", "/")
            fname = fname[2:] if fname.startswith("./") else fname
            if fname:
                files[fname] = code.strip()
        if files:
            # No entry-point promotion here, by design. Every filename in this
            # branch was stated explicitly by the model, so renaming one to
            # main.py discards information it gave us and fabricates an entry
            # point, corrupting both files — real main.py is overwritten by a
            # helper's body, and the helper never lands under its own name (so
            # a repair aimed at it is lost). The forgery also satisfies every
            # caller's `"main.py" in files` guard. A response with no main.py is
            # returned as-is so callers reject it; promotion is only sound in
            # the unnamed single-block fallback, where no filename is lost.
            return files

    # Fallback: single code block → main.py
    code = _extract_code_block(content)
    if code.strip():
        return {"main.py": code}
    return {}


def _parse_jsonl_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _safe_json_loads(line, {})
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_metrics_from_stdout(stdout: str) -> dict[str, Any]:
    """Parse metric lines from experiment stdout.

    Handles multiple formats:
    - ``name: value`` (e.g. ``loss: 0.0042``)
    - ``UCB (Stochastic) cumulative_regret: 361.9233``
    - ``condition=name metric=value`` (per-condition output)
    - ``condition=name/metric_name metric=value``

    Returns a flat dict of metric_name -> value.
    Filters out log/status lines using :func:`is_metric_name`.
    """
    # BUG-173: regex for condition=name metric=value format
    _CONDITION_RE = re.compile(
        r"^condition=(\S+)\s+metric=([0-9eE.+-]+)\s*$"
    )
    metrics: dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        # --- Format 2: condition=xxx metric=yyy ---
        m = _CONDITION_RE.match(line)
        if m:
            cond_name = m.group(1)
            try:
                fval = float(m.group(2))
                metrics[cond_name] = fval
            except (ValueError, TypeError):
                pass
            continue
        # --- Format 1: name: value ---
        if ":" not in line:
            continue
        # Split on the LAST colon to handle names with colons
        parts = line.rsplit(":", 1)
        if len(parts) != 2:
            continue
        name_part = parts[0].strip()
        value_part = parts[1].strip()
        # Filter out log lines that look like status messages
        if not is_metric_name(name_part):
            continue
        try:
            fval = float(value_part)
            # Use the full name (e.g. "UCB (Stochastic) cumulative_regret")
            metrics[name_part] = fval
        except (ValueError, TypeError):
            pass
    return metrics


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _retry_empty_reasoning_response(
    llm: LLMClient,
    resp: Any,
    messages: list[dict[str, str]],
    *,
    system: str,
    json_mode: bool,
    max_tokens: int | None,
    strip_thinking: bool,
    reasoning: bool | None,
) -> Any:
    """Re-ask once when a reasoning model burned its whole budget thinking.

    A reasoning model counts its hidden reasoning against ``max_tokens``. When
    the prompt is long and the budget is tight it can hit the ceiling before
    emitting a single visible token, and the provider returns HTTP 200 with an
    empty body and ``finish_reason="length"`` — not an error, so nothing
    upstream retries. The stage then falls back to a much weaker path: Stage 9
    collapsed a 6767-token design prompt to a 122-token generic one, losing the
    hypotheses, the literature and the real compute budget.

    Retrying the *same* prompt with the reasoning pass disabled and double the
    budget keeps the full context, which is the part that actually determines
    plan quality. Returns None when this does not apply, so the caller keeps
    the original response.
    """
    if (getattr(resp, "content", "") or "").strip():
        return None
    if str(getattr(resp, "finish_reason", "")).lower() != "length":
        return None

    _retry_max = (max_tokens or 4096) * 2
    logger.warning(
        "LLM returned an empty body with finish_reason=length "
        "(completion_tokens=%s, max_tokens=%s) — the reasoning pass consumed "
        "the budget. Re-asking the same prompt with reasoning off and "
        "max_tokens=%d.",
        getattr(resp, "completion_tokens", "?"),
        max_tokens,
        _retry_max,
    )
    try:
        retry = llm.chat(
            messages,
            system=system,
            json_mode=json_mode,
            max_tokens=_retry_max,
            strip_thinking=strip_thinking,
            reasoning=False,
        )
    except Exception as exc:  # noqa: BLE001
        # Best-effort salvage: the caller still has the (empty) original and
        # its own fallbacks. Turning this into a raise would make a recoverable
        # stage fail outright.
        logger.warning("Empty-response retry failed: %s", exc)
        return None
    if not (getattr(retry, "content", "") or "").strip():
        logger.warning(
            "Empty-response retry also returned nothing (finish_reason=%s). "
            "Set llm.reasoning_off_params for this provider — without it the "
            "retry only raises the budget.",
            getattr(retry, "finish_reason", "?"),
        )
        return None
    logger.info("Empty-response retry recovered %d chars.", len(retry.content))
    return retry


def _chat_with_prompt(
    llm: LLMClient,
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    max_tokens: int | None = None,
    retries: int = 0,
    strip_thinking: bool = True,
    reasoning: bool | None = None,
) -> Any:
    """Send a chat request with optional retry on timeout/transient errors.

    Parameters
    ----------
    retries:
        Number of extra attempts after the first failure (0 = no retry).
        Uses exponential backoff: 2s, 4s, 8s, ...
    strip_thinking:
        If True (default for pipeline usage), strip ``<think>`` tags from
        the LLM response.  This prevents chain-of-thought leakage from
        breaking YAML / JSON / LaTeX parsers downstream.
    reasoning:
        ``False`` asks the provider to skip its reasoning pass. Only worth
        setting for fixed-schema output; see ``LlmConfig.reasoning_off_params``.
    """
    import time

    messages = [{"role": "user", "content": user}]
    last_exc: Exception | None = None
    _effective_json_mode = json_mode
    for attempt in range(1 + retries):
        try:
            resp = llm.chat(
                messages,
                system=system,
                json_mode=_effective_json_mode,
                max_tokens=max_tokens,
                strip_thinking=strip_thinking,
                reasoning=reasoning,
            )
            _salvaged = _retry_empty_reasoning_response(
                llm,
                resp,
                messages,
                system=system,
                json_mode=_effective_json_mode,
                max_tokens=max_tokens,
                strip_thinking=strip_thinking,
                reasoning=reasoning,
            )
            return _salvaged if _salvaged is not None else resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            # Auto-disable json_mode on HTTP 400 — likely provider incompatibility
            _err_str = str(exc)
            if _effective_json_mode and "400" in _err_str:
                logger.warning(
                    "HTTP 400 with json_mode=True — disabling json_mode for retry "
                    "(provider may not support response_format)."
                )
                _effective_json_mode = False
            if attempt < retries:
                delay = 2 ** (attempt + 1)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    1 + retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                raise last_exc from None
    raise last_exc  # type: ignore[misc]  # unreachable but satisfies type checker


def _get_evolution_overlay(
    run_dir: Path | None,
    stage_name: str,
    *,
    config: object | None = None,
    topic: str = "",
) -> str:
    """Load evolution lessons + matched skills for prompt injection.

    Combines three sources:
    1. Intra-run lessons (from current run's evolution dir)
    2. Cross-run MetaClaw skills (from ~/.metaclaw/skills/)
    3. Matched skills from the SkillRegistry (builtin + user + external)

    The SkillRegistry automatically matches skills to the current stage
    using trigger keywords and stage applicability metadata.

    Returns empty string if no relevant lessons/skills exist or on any error.
    """
    parts: list[str] = []

    # --- Section 1: Evolution lessons + MetaClaw arc-* skills ---
    if run_dir is not None:
        try:
            from researchclaw.evolution import EvolutionStore

            store = EvolutionStore(run_dir / "evolution")
            evo_overlay = store.build_overlay(
                stage_name, max_lessons=5, skills_dir=_METACLAW_SKILLS_DIR
            )
            if evo_overlay:
                parts.append(evo_overlay)
        except Exception:  # noqa: BLE001
            pass

    # --- Section 2: Matched skills from SkillRegistry ---
    try:
        registry = _get_skill_registry(config)
        context = f"{stage_name} {topic}".strip()
        matched = registry.match(context, stage_name)
        if matched:
            skills_text = registry.export_for_prompt(matched, max_chars=4000)
            if skills_text:
                parts.append(f"\n## Matched Domain Skills\n{skills_text}")
    except Exception:  # noqa: BLE001
        pass

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def _collect_json_context(
    directory: Path,
    *,
    max_files: int = 30,
    max_total_chars: int = 50_000,
) -> str:
    """Collect JSON context from a directory, with size limits.

    Large fields like ``stderr`` and ``stdout`` are stripped to avoid
    exceeding LLM token limits (the raw experiment output can be 5 MB+).
    """
    chunks: list[str] = []
    total = 0
    for file_path in sorted(directory.glob("*.json"))[:max_files]:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # Strip verbose fields that bloat the context
        if isinstance(data, dict):
            for key in ("stderr", "stdout", "raw_output", "traceback"):
                if key in data and isinstance(data[key], str) and len(data[key]) > 500:
                    data[key] = data[key][:500] + f"\n... [truncated, {len(data[key])} chars total]"
        chunk = json.dumps(data, indent=2, ensure_ascii=False)
        if total + len(chunk) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > 200:
                chunks.append(chunk[:remaining] + "\n... [truncated]")
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n\n".join(chunks)


def _collect_experiment_results(
    run_dir: Path,
    metric_key: str = "",
    metric_direction: str = "maximize",
) -> dict[str, Any]:
    """Aggregate experiment metrics from runs/ directory across prior stages.

    Returns a dict with ``runs``, ``metrics_summary``, ``best_run``,
    ``latex_table``, and optionally ``structured_results``.
    """
    runs_data: list[dict[str, Any]] = []
    structured_results: Any = None

    # Scan all stage dirs for runs/ subdirectory
    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        # Check for structured results.json first
        results_json = stage_subdir / "results.json"
        if results_json.exists() and structured_results is None:
            try:
                structured_results = json.loads(
                    results_json.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                pass

        for run_file in sorted(stage_subdir.glob("*.json")):
            if run_file.name == "results.json":
                continue  # Already handled above
            parsed = _safe_json_loads(run_file.read_text(encoding="utf-8"), {})
            if isinstance(parsed, dict) and "metrics" in parsed:
                # Also check for structured_results inside run payload
                if "structured_results" in parsed and structured_results is None:
                    structured_results = parsed["structured_results"]
                runs_data.append(parsed)
            elif isinstance(parsed, dict) and "key_metrics" in parsed:
                # Simulated mode uses key_metrics
                parsed["metrics"] = parsed.pop("key_metrics")
                runs_data.append(parsed)

    if not runs_data:
        result: dict[str, Any] = {"runs": [], "metrics_summary": {}, "best_run": None, "latex_table": ""}
        if structured_results is not None:
            result["structured_results"] = structured_results
        return result

    # Aggregate metrics across runs
    all_metric_keys: set[str] = set()
    for r in runs_data:
        m = r.get("metrics") or {}
        if isinstance(m, dict):
            all_metric_keys.update(m.keys())

    metrics_summary: dict[str, dict[str, float | None]] = {}
    for key in sorted(all_metric_keys):
        values = []
        for r in runs_data:
            m = r.get("metrics") or {}
            if isinstance(m, dict) and key in m:
                try:
                    _fv = float(m[key])
                    if _fv == _fv and abs(_fv) != float("inf"):  # filter NaN/Inf
                        values.append(_fv)
                except (ValueError, TypeError):
                    pass
        if values:
            metrics_summary[key] = {
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "mean": round(sum(values) / len(values), 6),
                "count": len(values),
            }

    # Find best run using metric_key and metric_direction
    best_run: dict[str, Any] | None = None
    if runs_data:

        def _primary_metric(r: dict[str, Any]) -> float:
            m = r.get("metrics") or {}
            if isinstance(m, dict):
                # Try specific metric_key first
                if metric_key and metric_key in m:
                    try:
                        return float(m[metric_key])
                    except (ValueError, TypeError):
                        pass
                # Fallback to first metric
                for v in m.values():
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        pass
            return 0.0

        _cmp = min if metric_direction == "minimize" else max
        best_run = _cmp(runs_data, key=_primary_metric)

    # Build LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Experiment Results}",
    ]
    if metrics_summary:
        cols = sorted(metrics_summary.keys())
        header = "Metric & Min & Max & Mean & N \\\\"
        latex_lines.append(r"\begin{tabular}{l" + "r" * 4 + "}")
        latex_lines.append(r"\hline")
        latex_lines.append(header)
        latex_lines.append(r"\hline")
        for col in cols:
            s = metrics_summary[col]
            row = f"{col} & {s['min']:.4f} & {s['max']:.4f} & {s['mean']:.4f} & {s['count']} \\\\"
            latex_lines.append(row)
        latex_lines.append(r"\hline")
        latex_lines.append(r"\end{tabular}")
    else:
        latex_lines.append(r"\begin{tabular}{l}")
        latex_lines.append("No experiment data available \\\\")
        latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    # R18-1: Extract paired statistical comparisons from stdout
    from researchclaw.experiment.sandbox import extract_paired_comparisons

    paired_comparisons: list[dict[str, object]] = []
    for r in runs_data:
        stdout = r.get("stdout", "")
        if stdout:
            paired_comparisons.extend(extract_paired_comparisons(stdout))

    collected: dict[str, Any] = {
        "runs": runs_data,
        "metrics_summary": metrics_summary,
        "best_run": best_run,
        "latex_table": "\n".join(latex_lines),
    }
    if paired_comparisons:
        collected["paired_comparisons"] = paired_comparisons
    if structured_results is not None:
        collected["structured_results"] = structured_results
    return collected


def _build_context_preamble(
    config: RCConfig,
    run_dir: Path,
    *,
    include_goal: bool = False,
    include_hypotheses: bool = False,
    include_synthesis: bool = False,
    include_exp_plan: bool = False,
    include_analysis: bool = False,
    include_decision: bool = False,
    include_experiment_data: bool = False,
) -> str:
    parts = [
        "## Research Context",
        f"**Topic**: {config.research.topic}",
        f"**Domains**: {', '.join(config.research.domains) if config.research.domains else 'general'}",
    ]
    if include_goal:
        goal = _read_prior_artifact(run_dir, "goal.md")
        if goal:
            parts.append(f"\n### Goal\n{goal[:2200]}")
    if include_hypotheses:
        hyp = _read_prior_artifact(run_dir, "hypotheses.md")
        if hyp:
            parts.append(f"\n### Hypotheses\n{hyp[:2200]}")
    if include_synthesis:
        synthesis = _read_prior_artifact(run_dir, "synthesis.md")
        if synthesis:
            parts.append(f"\n### Synthesis\n{synthesis[:2200]}")
    if include_exp_plan:
        plan = _read_prior_artifact(run_dir, "exp_plan.yaml")
        if plan:
            parts.append(f"\n### Experiment Plan\n{plan[:2000]}")
    if include_analysis:
        analysis = _read_best_analysis(run_dir)
        if analysis:
            parts.append(f"\n### Result Analysis\n{analysis[:2500]}")
    if include_decision:
        decision = _read_prior_artifact(run_dir, "decision.md")
        if decision:
            parts.append(f"\n### Research Decision\n{decision[:1500]}")
    if include_experiment_data:
        hw_profile = _load_hardware_profile(run_dir)
        if hw_profile:
            hw_lines = ["### Hardware Environment"]
            for hk, hv in hw_profile.items():
                hw_lines.append(f"- **{hk}**: {hv}")
            parts.append("\n" + "\n".join(hw_lines))
        exp_summary = _read_prior_artifact(run_dir, "experiment_summary.json")
        if exp_summary:
            summary = _safe_json_loads(exp_summary, {})
            if isinstance(summary, dict) and summary.get("metrics_summary"):
                parts.append("\n### Experiment Results (Quantitative)")
                ms = summary["metrics_summary"]
                for mk, mv in ms.items():
                    if isinstance(mv, dict):
                        parts.append(
                            f"- **{mk}**: mean={mv.get('mean', '?')}, "
                            f"min={mv.get('min', '?')}, max={mv.get('max', '?')}, n={mv.get('count', '?')}"
                        )
                if summary.get("latex_table"):
                    parts.append(
                        f"\n### LaTeX Table\n```latex\n{summary['latex_table']}\n```"
                    )
    # --- HITL guidance injection ---
    for stage_dir in sorted(run_dir.glob("stage-*/hitl_guidance.md")):
        try:
            guidance = stage_dir.read_text(encoding="utf-8").strip()
            if guidance:
                stage_name = stage_dir.parent.name
                parts.append(
                    f"\n### Human Guidance ({stage_name})\n{guidance[:1000]}"
                )
        except (OSError, UnicodeDecodeError):
            pass
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Topic keywords and constraints
# ---------------------------------------------------------------------------


def _extract_topic_keywords(
    topic: str, domains: tuple[str, ...] | list[str] = ()
) -> list[str]:
    """Extract meaningful keywords from the research topic + domain list.

    Returns lowercased keyword list (2+ chars, no stop words).
    Used by the domain pre-filter to drop obviously irrelevant papers.
    """
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", topic.lower())
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) >= 3]
    # Add domain names as keywords
    for d in domains:
        for part in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", d.lower()):
            if part not in _STOP_WORDS and len(part) >= 2:
                keywords.append(part)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


# --- P1-2: Topic constraint block for paper generation stages ---
def _topic_constraint_block(topic: str) -> str:
    """Return a hard constraint instruction that anchors paper content to the topic.

    Prevents the common LLM failure mode of drifting off-topic or
    presenting environmental/infrastructure issues as research contributions.
    """
    return (
        "\n\n=== HARD TOPIC CONSTRAINT ===\n"
        f"The paper MUST be about: {topic}\n"
        "PROHIBITED content (unless user explicitly specifies case-study mode):\n"
        "- Do NOT treat environment setup, dependency installation, or infrastructure "
        "failures as a research contribution.\n"
        "- Do NOT present debugging logs, system errors, or configuration issues "
        "as experimental findings.\n"
        "- Do NOT drift to tangential topics not directly related to the stated topic.\n"
        "- Every section MUST connect back to the core research question.\n"
        "- The Abstract and Introduction MUST clearly state the research problem "
        f"derived from: {topic}\n"
        "- The Method section MUST describe a technical approach, not a workflow.\n"
        "- The Results section MUST report quantitative outcomes of experiments, "
        "not environment status.\n"
        "=== END CONSTRAINT ===\n"
    )


# ---------------------------------------------------------------------------
# Runtime issue detection
# ---------------------------------------------------------------------------


def _detect_runtime_issues(sandbox_result: Any) -> str:
    """Detect NaN/Inf in metrics and extract stderr warnings from sandbox run.

    Returns a formatted string describing all runtime issues, or empty string
    if no issues are found.
    """
    issues: list[str] = []

    # Check metrics for NaN/Inf
    metrics = getattr(sandbox_result, "metrics", {}) or {}
    for key, val in metrics.items():
        try:
            fval = float(val)
            if math.isnan(fval):
                issues.append(f"METRIC NaN: '{key}' returned NaN — likely a division by zero or invalid computation in code")
            elif math.isinf(fval):
                issues.append(f"METRIC Inf: '{key}' returned Infinity — likely overflow or unbounded computation")
        except (TypeError, ValueError):
            pass

    # Check stdout for NaN values (word boundary to avoid matching "Nanotechnology" etc.)
    stdout = getattr(sandbox_result, "stdout", "") or ""
    _nan_re = re.compile(r"\bnan\b", re.IGNORECASE)
    if _nan_re.search(stdout):
        nan_lines = [
            line.strip()
            for line in stdout.splitlines()
            if _nan_re.search(line)
        ]
        if nan_lines:
            issues.append(
                f"NaN values detected in output:\n" + "\n".join(nan_lines[:10])
            )

    # Extract meaningful warnings from stderr
    stderr = getattr(sandbox_result, "stderr", "") or ""
    if stderr.strip():
        warning_lines = []
        for line in stderr.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Keep RuntimeWarning, ValueError, ZeroDivisionError, etc.
            if any(
                kw in line_stripped
                for kw in (
                    "Warning",
                    "Error",
                    "Traceback",
                    "Exception",
                    "divide",
                    "overflow",
                    "invalid value",
                    "NaN",
                    "inf",
                )
            ):
                warning_lines.append(line_stripped)
        if warning_lines:
            issues.append(
                "Runtime warnings/errors from stderr:\n"
                + "\n".join(warning_lines[:15])
            )

    # Check for identical metric values across all entries in stdout
    # (e.g., all algorithms reporting convergence_rate=1.0)
    stdout = getattr(sandbox_result, "stdout", "") or ""
    if stdout:
        from collections import Counter

        metric_values_by_name: dict[str, list[float]] = {}
        for line in stdout.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.rsplit(":", 1)
            if len(parts) != 2:
                continue
            try:
                fval = float(parts[1].strip())
            except (ValueError, TypeError):
                continue
            # Extract metric suffix (e.g. "convergence_rate" from "UCB (Stochastic) convergence_rate")
            name = parts[0].strip()
            metric_suffix = name.split()[-1] if name.split() else name
            metric_values_by_name.setdefault(metric_suffix, []).append(fval)

        for metric_name, vals in metric_values_by_name.items():
            if len(vals) >= 3:
                unique = set(vals)
                if len(unique) <= 2:
                    issues.append(
                        f"DUMMY METRIC: '{metric_name}' has only {len(unique)} unique value(s) "
                        f"across {len(vals)} entries ({unique}) — likely a placeholder. "
                        f"Implement real measurement logic (e.g., track iterations to convergence)."
                    )

    # R5-3: Check for diverging loss values (fast-fail indicator)
    for key, val in metrics.items():
        try:
            fval = float(val)
            if "loss" in key.lower() and fval > 100:
                issues.append(
                    f"DIVERGING LOSS: '{key}' = {fval} (>100) — the optimization is "
                    f"diverging. Reduce learning rate, check gradient computation, "
                    f"or add gradient clipping."
                )
        except (TypeError, ValueError):
            pass

    if not issues:
        return ""

    return (
        "## Runtime Issues Detected\n\n"
        "The experiment code ran but produced problematic results. "
        "Fix the ROOT CAUSE of these issues in the code:\n\n"
        + "\n\n".join(f"- {issue}" for issue in issues)
    )


# ---------------------------------------------------------------------------
# NeurIPS checklist
# ---------------------------------------------------------------------------


def _generate_neurips_checklist(
    has_experiments: bool = True,
    has_theory: bool = False,
    has_code: bool = True,
) -> str:
    """Generate a NeurIPS-style paper checklist appendix in markdown.

    This checklist is based on the NeurIPS 2025 submission requirements.
    It is appended to the paper before LaTeX conversion.
    """
    items = [
        ("Claims", "Do the main claims accurately reflect the paper's contributions and scope?", "Yes"),
        ("Limitations", "Does the paper discuss limitations of the work?", "Yes"),
    ]
    if has_theory:
        items.append(
            ("Theory", "Are all assumptions stated and proofs included?", "Yes")
        )
    items.extend([
        ("Experiments reproducibility", "Does the paper fully disclose experimental settings?", "Yes" if has_experiments else "NA"),
        ("Code and data", "Is code or data provided for reproducibility?", "Yes" if has_code else "No"),
        ("Experimental details", "Are training details and hyperparameters specified?", "Yes" if has_experiments else "NA"),
        ("Error bars", "Are error bars or confidence intervals reported?", "Yes" if has_experiments else "NA"),
        ("Compute resources", "Are compute requirements documented?", "Yes" if has_experiments else "NA"),
        ("Code of ethics", "Does the work comply with the code of ethics?", "Yes"),
        ("Broader impacts", "Are potential negative societal impacts discussed?", "Yes"),
        ("Licenses", "Are licenses for used assets respected?", "Yes"),
        ("New assets", "Are newly released assets documented?", "NA"),
        ("Human subjects", "Were IRB approvals obtained if applicable?", "NA"),
    ])

    lines = [
        "## NeurIPS Paper Checklist",
        "",
    ]
    for label, question, answer in items:
        lines.append(f"**{label}**: {question}")
        lines.append(f"Answer: [{answer}]")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper title extraction
# ---------------------------------------------------------------------------


def _extract_paper_title(md_text: str) -> str:
    """Extract paper title from markdown text for LaTeX generation.

    Prioritises H1 headings that appear *before* the abstract section and
    look like real titles (>= 4 words, starts with uppercase).  This avoids
    picking up pseudocode comments or algorithm step labels.

    Also handles the common LLM pattern where a ``# Title`` heading is
    followed by the actual title as a plain text line (possibly bold):

        # Title

        NORM-PPO: Observation Normalization and Reward Scaling Effects
    """
    import re as _re

    # Strip outer markdown fence (LLMs sometimes wrap entire paper)
    _text = md_text
    _fence_m = _re.match(r"^\s*```(?:markdown|md|latex|tex)?\s*\n", _text)
    if _fence_m:
        _text = _text[_fence_m.end():]
        # Also strip trailing fence
        _text = _re.sub(r"\n\s*```\s*$", "", _text)

    # Limit search to content before Abstract heading
    abstract_pos = _re.search(
        r"^#{1,2}\s+(Abstract|ABSTRACT)", _text, _re.MULTILINE
    )
    search_region = _text[: abstract_pos.start()] if abstract_pos else _text[:3000]

    _SKIP = {"title", "abstract", "references", "appendix"}
    candidates: list[str] = []
    _saw_title_heading = False

    lines = search_region.splitlines()
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()

        # BUG-171: When we see a "# Title" or "## Title" heading, the actual
        # title is often on the next non-empty line as plain text or bold text.
        if _saw_title_heading and line:
            # Strip bold markers: **Title Text** → Title Text
            candidate = _re.sub(r"\*\*(.+?)\*\*", r"\1", line).strip()
            # Make sure it's not another heading or a skip heading
            if not line.startswith("#") and candidate:
                candidates.insert(0, candidate)  # highest priority
            _saw_title_heading = False

        # Match H1 or H2 headings
        hm = _re.match(r"^(#{1,2})\s+(.+)$", line)
        if hm:
            heading = hm.group(2).strip()
            heading_lower = heading.lower()
            # Handle "## Title Actual Paper Title" pattern
            if heading_lower.startswith("title ") and len(heading) > 6:
                heading = heading[6:].strip()
                heading_lower = heading.lower()
            if heading_lower in _SKIP:
                # Mark that we saw a "# Title" heading — next non-empty line
                # is likely the actual title text
                if heading_lower == "title":
                    _saw_title_heading = True
                continue
            candidates.append(heading)
            continue
        # Bold title line (e.g. **My Paper Title**)
        m = _re.match(r"\*\*(.+?)\*\*$", line)
        if m and len(m.group(1).split()) >= 3:
            candidates.append(m.group(1))

    # Prefer candidates that look like real titles (>= 4 words, capitalised)
    for c in candidates:
        words = c.split()
        if len(words) >= 4 and c[0].isupper():
            return c

    # Fallback: any candidate
    if candidates:
        return candidates[0]

    return "Untitled Paper"


# ---------------------------------------------------------------------------
# Framework diagram prompt
# ---------------------------------------------------------------------------


def _generate_framework_diagram_prompt(
    paper_text: str,
    config: "RCConfig",
    *,
    llm: "LLMClient | None" = None,
) -> str:
    """Generate a text-to-image prompt for a methodology framework diagram.

    Reads the paper's method section and produces a detailed prompt suitable
    for AI image generators (DALL-E, Midjourney, etc.).  The prompt describes
    an academic-style architecture/framework overview figure.

    Returns the prompt as a Markdown string, or empty string on failure.
    """
    import re as _re

    # Extract method/approach section from paper
    _method_section = ""
    _method_patterns = [
        r"(?:^#{1,3}\s+(?:Method(?:ology)?|Approach|Proposed\s+(?:Method|Framework|Approach)|Our\s+Method|Technical\s+Approach|Model\s+Architecture).*?)(?=^#{1,3}\s+|\Z)",
    ]
    for _pat in _method_patterns:
        _match = _re.search(_pat, paper_text, _re.MULTILINE | _re.DOTALL | _re.IGNORECASE)
        if _match:
            _method_section = _match.group(0)[:3000]
            break

    if not _method_section:
        # Fallback: use abstract + first 1500 chars
        _abs_match = _re.search(
            r"(?:^#{1,2}\s+Abstract\s*\n)(.*?)(?=^#{1,2}\s+|\Z)",
            paper_text, _re.MULTILINE | _re.DOTALL | _re.IGNORECASE,
        )
        _method_section = (_abs_match.group(1)[:1500] if _abs_match else paper_text[:2000])

    title = _extract_paper_title(paper_text)
    topic = config.research.topic

    # Use LLM to generate the prompt if available
    if llm is not None:
        _system = (
            "You are an expert academic figure designer. Generate a detailed text-to-image "
            "prompt for creating a methodology framework/architecture overview diagram.\n\n"
            "Requirements:\n"
            "- Academic style: clean, professional, suitable for a top-tier ML conference paper\n"
            "- Color palette: sophisticated and harmonious (suggest specific hex colors, "
            "prefer muted blues #4477AA, teals #44AA99, warm accents #CCBB44, soft purples #AA3377)\n"
            "- Layout: left-to-right or top-to-bottom data flow, with clearly labeled components\n"
            "- Components: boxes/modules with rounded corners, directional arrows, clear labels\n"
            "- Information density: high but not cluttered — each box should have a short label\n"
            "- Text on figure: minimal, only component names and key annotations\n"
            "- Background: white or very light grey\n"
            "- Style: vector-art look, flat design with subtle shadows, NO photorealism\n\n"
            "Output ONLY the prompt text (no markdown headers, no explanations). "
            "The prompt should be 150-300 words, highly specific and actionable."
        )
        _user = (
            f"Paper title: {title}\n"
            f"Research topic: {topic}\n\n"
            f"Method section excerpt:\n{_method_section}\n\n"
            "Generate a detailed text-to-image prompt for the methodology framework diagram."
        )
        try:
            resp = _chat_with_prompt(llm, _system, _user, max_tokens=1024)
            _llm_prompt = resp.content.strip()
            if len(_llm_prompt) > 50:
                return (
                    f"# Framework Diagram Prompt\n\n"
                    f"**Paper**: {title}\n\n"
                    f"## Image Generation Prompt\n\n"
                    f"{_llm_prompt}\n\n"
                    f"## Usage Instructions\n\n"
                    f"1. Copy the prompt above into an AI image generator "
                    f"(DALL-E 3, Midjourney, Ideogram, etc.)\n"
                    f"2. Generate the image at high resolution (2048x1024 or similar landscape)\n"
                    f"3. Save as `framework_diagram.png` in the same `charts/` folder\n"
                    f"4. Insert into the paper's Method section using:\n"
                    f"   - LaTeX: `\\includegraphics[width=\\textwidth]{{charts/framework_diagram.png}}`\n"
                    f"   - Markdown: `![Framework Overview](charts/framework_diagram.png)`\n"
                )
        except Exception:
            logger.debug("Framework prompt LLM generation failed, using template")

    # Fallback: template-based prompt without LLM
    _components = []
    _component_patterns = [
        (r"(?:encoder|decoder|transformer|attention|convolution|MLP|GNN|ResNet|ViT)", "Neural Network Module"),
        (r"(?:loss|objective|criterion|training|optimization)", "Training/Optimization"),
        (r"(?:data|dataset|input|preprocessing|augmentation)", "Data Pipeline"),
        (r"(?:output|prediction|inference|evaluation)", "Output/Evaluation"),
    ]
    _method_lower = _method_section.lower()
    for pat, label in _component_patterns:
        if _re.search(pat, _method_lower):
            _components.append(label)

    if not _components:
        _components = ["Input Processing", "Core Model", "Training Loop", "Evaluation"]

    return (
        f"# Framework Diagram Prompt\n\n"
        f"**Paper**: {title}\n\n"
        f"## Image Generation Prompt\n\n"
        f"Create a clean, academic-style methodology framework diagram for a research paper "
        f"titled \"{title}\". "
        f"The diagram should show a left-to-right data flow pipeline with these main components: "
        f"{', '.join(_components)}. "
        f"Use a professional color palette with muted blues (#4477AA), teals (#44AA99), "
        f"warm yellows (#CCBB44), and soft purples (#AA3377) on a white background. "
        f"Each component should be a rounded rectangle with a short label inside. "
        f"Connect components with clean directional arrows. "
        f"Add subtle shadows for depth. Flat vector-art style, no photorealism. "
        f"High information density but visually clean. "
        f"Suitable for a top-tier machine learning conference paper (ICML/NeurIPS/ICLR). "
        f"Landscape orientation, 2048x1024 resolution.\n\n"
        f"## Usage Instructions\n\n"
        f"1. Copy the prompt above into an AI image generator "
        f"(DALL-E 3, Midjourney, Ideogram, etc.)\n"
        f"2. Generate the image at high resolution (2048x1024 or similar landscape)\n"
        f"3. Save as `framework_diagram.png` in the same `charts/` folder\n"
        f"4. Insert into the paper's Method section using:\n"
        f"   - LaTeX: `\\includegraphics[width=\\textwidth]{{charts/framework_diagram.png}}`\n"
        f"   - Markdown: `![Framework Overview](charts/framework_diagram.png)`\n"
    )


# ---------------------------------------------------------------------------
# Filename and data helpers
# ---------------------------------------------------------------------------


def _safe_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
    return name[:100] or "unnamed"


# ---------------------------------------------------------------------------
# Default fallbacks
# ---------------------------------------------------------------------------


def _default_hypotheses(topic: str) -> str:
    return f"""# Hypotheses

## H1
Increasing protocol control for {topic} improves metric stability across random seeds.

## H2
Adding robustness-aware objectives for {topic} improves out-of-domain performance without major in-domain regression.

## H3
The combined approach outperforms either component under fixed compute budget.

## Generated
{_utcnow_iso()}
"""


def _default_paper_outline(topic: str) -> str:
    return f"""# Paper Outline

## 1. Title
Focused title on {topic}

## 2. Abstract
- Problem framing
- Method overview
- Key quantitative result

## 3. Introduction
- Motivation
- Gap statement
- Contributions

## 4. Related Work
- Method families
- Evaluation practices

## 5. Method
- Problem setup
- Model/algorithm
- Complexity and constraints

## 6. Experiments
- Datasets and metrics
- Baselines and ablations
- Reproducibility protocol

## 7. Results
- Main table
- Robustness analysis
- Failure cases

## 8. Discussion
- Practical implications
- Limitations

## 9. Conclusion
- Findings and next steps

Generated: {_utcnow_iso()}
"""


def _default_quality_report(threshold: float) -> dict[str, Any]:
    # When LLM fails, return below-threshold score to force revision
    score = max(1.0, float(threshold) - 2.0) if threshold > 0 else 5.0
    score = max(1.0, min(10.0, score))
    verdict = "revise"
    return {
        "score_1_to_10": round(score, 2),
        "verdict": verdict,
        "criteria": {
            "novelty": round(min(10.0, score + 0.3), 2),
            "methodological_rigor": round(score, 2),
            "clarity": round(max(1.0, score - 0.2), 2),
            "reproducibility": round(min(10.0, score + 0.1), 2),
        },
        "strengths": [
            "Stage-by-stage evidence chain preserved",
            "Experiment artifacts are generated and archived",
        ],
        "weaknesses": [
            "Statistical significance may need stronger reporting",
            "Broader external validity remains partially evaluated",
        ],
        "required_actions": [
            "Report confidence intervals and seed variance",
            "Include at least one stronger external baseline",
        ],
        "generated": _utcnow_iso(),
    }


# ---------------------------------------------------------------------------
# Multi-perspective generation
# ---------------------------------------------------------------------------


def _multi_perspective_generate(
    llm: LLMClient,
    roles: dict[str, dict[str, str]],
    variables: dict[str, str],
    perspectives_dir: Path,
) -> dict[str, str]:
    """Generate outputs from multiple debate perspectives.

    Each role has its own system/user prompt. Outputs are saved to
    *perspectives_dir* and returned as ``{role_name: response_text}``.
    """
    import os as _os  # noqa: PLC0415
    from researchclaw.prompts import _render  # noqa: PLC0415

    # Ablation hook: ARC_ABL_DISABLE_DEBATE=1 collapses the debate to a
    # single role so the multi-perspective synthesizer degenerates into a
    # plain prompt. Used by experiments/component_ablation only.
    if _os.environ.get("ARC_ABL_DISABLE_DEBATE", "").strip() == "1" and roles:
        first_key = next(iter(roles))
        roles = {first_key: roles[first_key]}
        logger.info("ARC_ABL_DISABLE_DEBATE=1 — debate collapsed to role %s", first_key)

    perspectives_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    for role_name, role_prompts in roles.items():
        try:
            system = _render(role_prompts["system"], variables)
            user = _render(role_prompts["user"], variables)
            resp = llm.chat(
                [{"role": "user", "content": user}],
                system=system,
            )
            results[role_name] = resp.content
            (perspectives_dir / f"{role_name}.md").write_text(
                resp.content, encoding="utf-8"
            )
            logger.info("Debate perspective '%s' generated (%d chars)", role_name, len(resp.content))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Debate perspective '%s' failed: %s", role_name, exc)
    if len(results) < 2:
        logger.error("Multi-perspective debate: only %d/%d roles succeeded", len(results), len(roles))
    return results


def _synthesize_perspectives(
    llm: LLMClient,
    perspectives: dict[str, str],
    sub_prompt_name: str,
    prompts: PromptManager,
) -> str:
    """Synthesize multiple perspective outputs into a unified result."""
    parts = []
    for role_name, text in perspectives.items():
        parts.append(f"### Perspective: {role_name}\n{text}")
    combined = "\n\n---\n\n".join(parts)
    sp = prompts.sub_prompt(sub_prompt_name, perspectives=combined)
    resp = llm.chat(
        [{"role": "user", "content": sp.user}],
        system=sp.system,
        max_tokens=sp.max_tokens,
    )
    return resp.content


def reconcile_figure_refs(
    tex_path: Path,
    charts_dir: Path,
) -> dict[str, str]:
    """Fix ``\\includegraphics`` paths in *tex_path* that don't match files in *charts_dir*.

    Three-tier matching strategy:
      1. **Exact stem** — e.g. ``accuracy_plot`` matches ``accuracy_plot.png``
      2. **Normalized keyword overlap** — tokenize on ``[-_]``, apply singular/plural
         normalization, require Jaccard similarity >= 0.4
      3. **Substring containment** — one stem is a substring of the other

    Returns a ``{old_path: new_path}`` dict of fixes applied (empty if none needed).
    """
    if not tex_path.exists():
        return {}

    tex_text = tex_path.read_text(encoding="utf-8")
    fig_refs = re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_text
    )
    if not fig_refs:
        return {}

    # Build map of actual chart files: lowered-stem -> charts/filename
    actual_files: dict[str, str] = {}
    if charts_dir.is_dir():
        for af in charts_dir.iterdir():
            if af.is_file() and af.suffix.lower() in (
                ".png", ".jpg", ".jpeg", ".pdf", ".svg",
            ):
                actual_files[af.stem.lower()] = f"charts/{af.name}"

    if not actual_files:
        return {}

    def _singularize(word: str) -> str:
        """Cheap singular/plural normalization."""
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("ses") and len(word) > 4:
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
            return word[:-1]
        return word

    def _tokenize(stem: str) -> set[str]:
        return {_singularize(w) for w in stem.replace("-", "_").split("_") if w}

    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    fixes: dict[str, str] = {}
    for ref in fig_refs:
        ref_resolved = tex_path.parent / ref
        if ref_resolved.exists():
            continue

        ref_stem = Path(ref).stem.lower()

        # Tier 1: exact stem match
        if ref_stem in actual_files:
            fixes[ref] = actual_files[ref_stem]
            continue

        # Tier 2: keyword overlap with Jaccard >= 0.4
        ref_tokens = _tokenize(ref_stem)
        best_match, best_score = "", 0.0
        for stem, apath in actual_files.items():
            score = _jaccard(ref_tokens, _tokenize(stem))
            if score > best_score:
                best_score = score
                best_match = apath
        if best_score >= 0.4 and best_match:
            fixes[ref] = best_match
            continue

        # Tier 3: substring containment
        for stem, apath in actual_files.items():
            if ref_stem in stem or stem in ref_stem:
                fixes[ref] = apath
                break

    if fixes:
        for old_path, new_path in fixes.items():
            tex_text = tex_text.replace(f"{{{old_path}}}", f"{{{new_path}}}")
        tex_path.write_text(tex_text, encoding="utf-8")
        logger.warning(
            "reconcile_figure_refs: Fixed %d figure path mismatch(es): %s",
            len(fixes),
            ", ".join(f"{k} → {v}" for k, v in fixes.items()),
        )

    return fixes
