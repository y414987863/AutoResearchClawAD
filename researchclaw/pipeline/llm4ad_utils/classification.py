"""Algorithm role classification and ``evolve_scope`` resolution.

The job is small: every directory under ``experiment/algorithms/`` plays one of
three roles in the comparison — the method this paper proposes, a baseline it is
compared against, or an ablation of the proposal. ``evolve_scope:
{categories: [proposed]}`` means "evolve the proposals, leave the baselines
alone", so something has to decide which directory is which.

Where the decision comes from
-----------------------------
The stage-9 experiment plan says what the experiment is; the directory names say
what was actually generated. Neither is sufficient alone — plans describe methods
in prose ("a CMA-ES variant with a covariance floor"), directories are terse
identifiers (``cmaes_floor``). So stage 10 hands both to the model in one call
and asks for the mapping directly. That is the whole classifier
(:func:`classify_algorithms`): no name-matching heuristics, no plan parsing, no
partial-credit scoring. The model reads the plan the way a person would.

Because the answer is a model's, it is validated rather than trusted: one label
per discovered directory, drawn from the three canonical roles. A reply that
fails validation is retried once; a directory the model leaves out defaults to
:data:`PROPOSED`, which is the safe default — evolving a proposed method is
always allowed, while evolving a baseline silently destroys the comparison.

Canonical on-disk form::

    {"generated": "...", "source": "...", "signature": "...",
     "classification": {"<algo_dir_name>": "proposed"|"baseline"|"ablation"}}

Only ``classification`` is trusted on read; the rest is provenance. The file must
always exist and always cover every directory, because
:func:`read_classification` returning ``None`` makes :func:`filter_by_scope` fail
closed — a scope that matches nothing evolves nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from researchclaw.pipeline._helpers import _chat_with_prompt, _safe_json_loads

logger = logging.getLogger(__name__)

#: The three roles the scope filter understands. Every discovered algorithm gets
#: exactly one of these; the classifier has no "unknown" outcome.
PROPOSED = "proposed"
BASELINE = "baseline"
ABLATION = "ablation"
CATEGORIES: frozenset[str] = frozenset({PROPOSED, BASELINE, ABLATION})

#: Ordered for prompt use; `proposed` first because it is the common case.
CATEGORY_LIST: tuple[str, ...] = (PROPOSED, BASELINE, ABLATION)

#: Default for a directory the model did not classify. `proposed` is the safe
#: side: it can only cause evolution to run on the method the paper is about,
#: never on a baseline whose immobility the comparison depends on.
DEFAULT_CATEGORY = PROPOSED

#: The file lives inside the experiment directory, whose name is derived per run
#: and therefore never known in advance — always located relative to ``exp_dir``.
CLASSIFICATION_FILENAME = "algorithms_classification.json"


# --------------------------------------------------------------------------
# Canonical labels
# --------------------------------------------------------------------------

def normalize_category(value: Any) -> str | None:
    """Map a role label onto one of :data:`CATEGORIES`, else ``None``.

    Models drift into synonyms (``"ours"``, ``"novel"``, ``"primary"``) and into
    prose (``"proposed method"``). Only the three canonical roles are usable
    downstream, so a synonym is resolved when it maps unambiguously and rejected
    otherwise — never guessed.
    """
    if not isinstance(value, str):
        return None
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not v:
        return None
    if v in CATEGORIES:
        return v
    # Plural / prose-wrapped spellings of a canonical role.
    for cat in CATEGORY_LIST:
        if v in (f"{cat}s", f"{cat}_method", f"{cat}_methods"):
            return cat
    if v in ("reference", "references"):
        # Not an invention: ConditionRole.REFERENCE is the schema's own name for
        # this role (domains/experiment_schema.py), so an S9 plan or a
        # hand-written file may spell it this way.
        return BASELINE
    if v in ("ours", "our_method", "our_methods", "main", "main_method",
             "contribution", "primary", "novel", "proposed_method"):
        return PROPOSED
    if v in ("variant", "variants"):
        return ABLATION
    # Everything else is left unmapped on purpose: "sota", "state_of_the_art",
    # "custom", "competitor" *suggest* a role, but guessing puts an algorithm in
    # the evolution scope on a hunch. That is the failure that matters — a
    # baseline mislabelled "proposed" gets evolved, and the comparison the paper
    # rests on stops being a comparison.
    return None


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def discover_algorithms(exp_dir: Path) -> list[tuple[str, Path]]:
    """Return ``(algo_name, source_file)`` for each ``algorithms/<algo>/<algo>.py``.

    The **directory name is the algorithm identity** everywhere downstream: it
    keys the classification, it is what ``evolve_scope: {names: [...]}`` matches,
    and it is what task packages are named after. A directory without a matching
    ``<algo>.py`` is skipped, because that is what the package builder iterates.
    """
    algo_root = exp_dir / "algorithms"
    if not algo_root.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for sub in sorted(algo_root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        src = sub / f"{sub.name}.py"
        if src.is_file():
            found.append((sub.name, src))
    return found


def discover_algorithm_names(exp_dir: Path) -> list[str]:
    """Directory names under ``algorithms/``, sorted."""
    return [name for name, _ in discover_algorithms(exp_dir)]


def _signature(names: list[str]) -> str:
    """Stable hash of the classified name set, so a re-run can be skipped."""
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()[:16]


def missing_from(mapping: dict[str, str], exp_dir: Path) -> list[str]:
    """Algorithm directories present on disk but absent from ``mapping``."""
    return sorted(set(discover_algorithm_names(exp_dir)) - set(mapping))


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def read_classification(exp_dir: Path) -> dict[str, str] | None:
    """Return ``{algo_name: category}`` from the canonical file, else ``None``.

    ``None`` means "no usable role information" and covers every way that can
    happen: the file is absent, unparsable, not a dict, has no ``classification``
    mapping, or holds nothing but non-canonical labels. The caller must treat it
    as a hard absence (fail closed), never as "everything is fine".

    A bare ``{algo: role}`` top level is also accepted, for files written by hand
    or by an older revision; a dict-of-lists / dict-of-dicts
    (``{"algorithm_types": {...}}``, ``{"esn": {...}}``) is **not**, because its
    values name a family, not a role — matching a requested category against them
    classifies nothing correctly.
    """
    path = exp_dir / CLASSIFICATION_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    mapping: Any = data.get("classification")
    if not isinstance(mapping, dict):
        # Flat top level, but only when every value is a scalar role. A
        # dict-of-lists grouping has non-string values and is rejected here.
        if data and all(isinstance(v, str) for v in data.values()):
            mapping = data
        else:
            return None

    cleaned: dict[str, str] = {}
    for algo, raw in mapping.items():
        cat = normalize_category(raw)
        if cat is not None:
            cleaned[str(algo)] = cat
    if not cleaned:
        return None
    return cleaned


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def write_classification(
    exp_dir: Path,
    mapping: dict[str, str],
    *,
    source: str,
    signature: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the canonical classification file, returning its path.

    Written temp-then-replace so a crash mid-write cannot leave a truncated file
    that :func:`read_classification` rejects as unparsable — which the scope
    filter then reads as "no classification" and drops the whole scope.
    """
    from datetime import datetime, timezone

    payload: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "classification": dict(sorted(mapping.items())),
    }
    if signature is not None:
        payload["signature"] = signature
    if extra:
        payload.update(extra)
    path = exp_dir / CLASSIFICATION_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def _read_signature(exp_dir: Path) -> str | None:
    path = exp_dir / CLASSIFICATION_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    sig = data.get("signature")
    return sig if isinstance(sig, str) and sig else None


# --------------------------------------------------------------------------
# Classifying
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You label machine-learning / optimization experiment algorithms by their "
    "role in a comparison. You always answer with a single JSON object and "
    "nothing else."
)


def _build_prompt(names: list[str], exp_plan: Any) -> str:
    plan_text = exp_plan if isinstance(exp_plan, str) else (
        json.dumps(exp_plan, indent=2, ensure_ascii=False, default=str)
        if exp_plan is not None else ""
    )
    plan_text = plan_text.strip()
    if len(plan_text) > 12000:
        plan_text = plan_text[:12000] + "\n... [plan truncated]"
    if not plan_text:
        plan_text = "(the experiment plan is unavailable — classify from the names alone)"
    return (
        "Below is the experiment plan from the design stage, followed by the "
        "algorithm directories that were actually generated.\n"
        "These directories are the complete and authoritative list: classify "
        "every one of them and invent nothing.\n\n"
        "<experiment_plan>\n" + plan_text + "\n</experiment_plan>\n\n"
        "ALGORITHM DIRECTORIES:\n"
        + "\n".join(f"- {n}" for n in names)
        + "\n\nAssign each directory exactly one role:\n"
        f'- "{PROPOSED}": the method this experiment proposes and is evaluated on '
        "(the paper's contribution).\n"
        f'- "{BASELINE}": an existing/reference method kept fixed for comparison.\n'
        f'- "{ABLATION}": a deliberate variant of the proposed method that removes '
        "or alters one of its components.\n\n"
        "Find each role from the plan's prose, never from its condition list:\n"
        "- `research_question` and `hypotheses` name the contribution. A hypothesis "
        f"of the form X beats Y on <metric> makes X the {PROPOSED} method and Y a "
        f"{BASELINE}.\n"
        "- Whatever the plan lists under `baselines`/`ablations`, or calls a "
        f"reference, comparison point, or sanity check, takes that role; a "
        f"deliberately naive method (random or uniform search) is a {BASELINE}.\n"
        "- `conditions:` lists what will RUN, not what each entry IS, so it is not "
        "evidence of a role. Do not read every entry as a baseline merely because "
        "the plan lists them together.\n"
        f"- A comparison study argues for something, so {PROPOSED} is normally "
        f"non-empty. Labelling every directory {BASELINE} claims the plan proposes "
        "nothing, which is a misreading of the prose. Re-read it before answering "
        "that.\n"
        "Match each directory by the algorithm it denotes rather than by the plan's "
        "wording; a grid or parameter sweep takes the role of the method it sweeps. "
        "If the plan is genuinely ambiguous, judge from what the algorithm is.\n\n"
        "Reply with JSON only, in exactly this shape:\n"
        '{"classification": {' + ", ".join(f'"{n}": "<role>"' for n in names) + "}}\n"
        f"where each <role> is exactly one of {list(CATEGORY_LIST)} (lowercase, no "
        "other spelling). Include every directory listed above, one entry each."
    )


def _validate_reply(
    raw: Any,
    names: list[str],
) -> tuple[dict[str, str], list[str]] | None:
    """Extract a usable classification, or ``None`` if the reply is unusable.

    Returns ``(mapping, dropped)`` where ``mapping`` covers every requested name
    (defaulting omissions to :data:`DEFAULT_CATEGORY`) and ``dropped`` lists the
    names the reply invented. ``None`` means the reply could not be read at all,
    which is the caller's cue to retry.
    """
    if not isinstance(raw, dict):
        return None
    mapping = raw.get("classification")
    if isinstance(mapping, dict) and mapping:
        pass
    elif raw and all(isinstance(v, str) for v in raw.values()):
        # Tolerate a bare {"<algo>": "<role>"} reply.
        mapping = raw
    else:
        return None

    wanted = set(names)
    result: dict[str, str] = {}
    dropped: list[str] = []
    for name, label in mapping.items():
        key = str(name).strip()
        if key not in wanted:
            dropped.append(key)
            continue
        cat = normalize_category(label)
        if cat is None:
            logger.warning(
                "llm4ad: classifier returned role %r for '%s', which is not one of "
                "%s — using '%s'.", label, key, sorted(CATEGORIES), DEFAULT_CATEGORY,
            )
            cat = DEFAULT_CATEGORY
        result[key] = cat

    if not result:
        return None
    return result, dropped


def classify_algorithms(
    exp_dir: Path,
    exp_plan: Any,
    llm: Any = None,
    *,
    max_attempts: int = 2,
    reuse_existing: bool = False,
) -> dict[str, str]:
    """Classify every algorithm under ``exp_dir/algorithms`` into a role.

    One LLM call with the plan and the directory names; the reply is validated to
    cover exactly the discovered directories with canonical labels, retried once
    if unusable, and anything still unclassified falls back to
    :data:`DEFAULT_CATEGORY`. Always returns one entry per discovered algorithm,
    or ``{}`` when there is no algorithm tree. Never raises.

    With ``llm=None`` every algorithm is :data:`DEFAULT_CATEGORY` — still a
    complete mapping, so callers never have to special-case a missing model.
    """
    names = discover_algorithm_names(exp_dir)
    if not names:
        return {}
    signature = _signature(names)

    if reuse_existing and _read_signature(exp_dir) == signature:
        existing = read_classification(exp_dir)
        if existing and set(existing) == set(names):
            logger.info("llm4ad: reusing %s (unchanged algorithm set)", CLASSIFICATION_FILENAME)
            return existing

    if llm is None:
        logger.info(
            "llm4ad: no LLM available; all %d algorithm(s) default to '%s'",
            len(names), DEFAULT_CATEGORY,
        )
        return {n: DEFAULT_CATEGORY for n in names}

    prompt = _build_prompt(names, exp_plan)
    result: dict[str, str] | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _chat_with_prompt(
                llm, _SYSTEM_PROMPT, prompt, json_mode=True, max_tokens=2048,
            )
            raw = (getattr(resp, "content", "") or "").strip()
            parsed = _safe_json_loads(raw, None)
            validated = _validate_reply(parsed, names)
            if validated is not None:
                result, dropped = validated
                if dropped:
                    logger.warning(
                        "llm4ad: classifier returned %d name(s) that are not algorithm "
                        "directories and were ignored: %s", len(dropped), ", ".join(dropped),
                    )
                break
            logger.warning(
                "llm4ad: classification attempt %d/%d returned no usable mapping (got %r)",
                attempt, max_attempts, raw[:300],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "llm4ad: classification attempt %d/%d raised %s",
                attempt, max_attempts, exc,
            )
        if attempt < max_attempts:
            import time as _time

            _time.sleep(2 * attempt)

    if result is None:
        logger.warning(
            "llm4ad: classifier produced no usable mapping after %d attempt(s); "
            "all %d algorithm(s) default to '%s'.",
            max_attempts, len(names), DEFAULT_CATEGORY,
        )
        result = {}

    missing = [n for n in names if n not in result]
    if missing:
        logger.warning(
            "llm4ad: %d algorithm(s) were not classified (%s) — defaulting them to '%s'.",
            len(missing), ", ".join(missing), DEFAULT_CATEGORY,
        )
        for name in missing:
            result[name] = DEFAULT_CATEGORY

    ordered = {name: result[name] for name in names}
    logger.info("llm4ad: classified %d algorithm(s): %s", len(ordered), ordered)
    write_classification(
        exp_dir, ordered, source="llm", signature=signature,
        extra={"classified": len(ordered)},
    )
    return ordered


# --------------------------------------------------------------------------
# Scoping
# --------------------------------------------------------------------------

def scope_categories(evolve_scope: dict[str, Any] | None) -> set[str]:
    """Canonical categories requested by ``evolve_scope`` (invalid ones dropped)."""
    if not isinstance(evolve_scope, dict):
        return set()
    out: set[str] = set()
    for raw in evolve_scope.get("categories", []) or []:
        cat = normalize_category(raw)
        if cat is not None:
            out.add(cat)
        elif str(raw).strip():
            logger.warning(
                "llm4ad: evolve_scope category %r is not one of %s and is ignored",
                raw, sorted(CATEGORIES),
            )
    return out


def scope_names(evolve_scope: dict[str, Any] | None) -> set[str]:
    """Algorithm directory names requested by ``evolve_scope``."""
    if not isinstance(evolve_scope, dict):
        return set()
    return {str(n).strip() for n in evolve_scope.get("names", []) or [] if str(n).strip()}


def scope_is_selective(evolve_scope: dict[str, Any] | None) -> bool:
    """True when the scope selects a subset (so a classification is required).

    An absent/empty scope evolves everything and needs no classification; a scope
    whose ``categories`` are all unrecognised is *not* treated as empty (that
    would evolve the baselines, the one outcome scoping exists to prevent).
    """
    if not isinstance(evolve_scope, dict):
        return False
    raw_cats = [str(c).strip() for c in (evolve_scope.get("categories") or []) if str(c).strip()]
    return bool(raw_cats) or bool(scope_names(evolve_scope))


def filter_by_scope(
    algorithms: list[tuple[str, Path]],
    exp_dir: Path,
    evolve_scope: dict[str, Any] | None,
) -> list[tuple[str, Path]]:
    """Keep only the algorithms the scope selects; fail closed on no signal.

    A scope that selects nothing returns an empty list — evolution is skipped.
    It must NOT fall back to the full list: with ``{"categories": ["proposed"]}``
    that fallback would evolve the baselines too and silently destroy the
    comparison the paper rests on. Skipping evolution is visible; evolving the
    baselines is not. So an unreadable classification (missing, unparsable, or
    written in a non-canonical shape) yields ``[]`` plus a warning, not a default.
    """
    if not scope_is_selective(evolve_scope):
        return algorithms

    wanted_categories = scope_categories(evolve_scope)
    wanted_names = scope_names(evolve_scope)
    classification = read_classification(exp_dir)

    if wanted_categories and classification is None:
        logger.warning(
            "llm4ad: evolve_scope %s selects by category, but no readable %s is "
            "under %s (absent, unparsable, or not in the canonical "
            '{"classification": {algo: role}} form). Nothing will be evolved this '
            "run.",
            evolve_scope, CLASSIFICATION_FILENAME, exp_dir,
        )
        return []

    if wanted_categories and classification is not None:
        unclassified = [a for a, _ in algorithms if a not in classification]
        if unclassified:
            logger.warning(
                "llm4ad: %d algorithm(s) under %s carry no role: %s — left out of "
                "evolution.",
                len(unclassified), exp_dir, ", ".join(sorted(unclassified)),
            )

    filtered: list[tuple[str, Path]] = []
    for algo, src in algorithms:
        cat = classification.get(algo) if classification else None
        if algo in wanted_names or (cat is not None and cat in wanted_categories):
            filtered.append((algo, src))
        else:
            logger.info(
                "llm4ad: algorithm '%s' excluded by evolve_scope %s (role=%s)",
                algo, evolve_scope, cat or "none",
            )
    if not filtered:
        logger.warning(
            "llm4ad: evolve_scope %s matched none of the %d algorithm(s) under %s; "
            "skipping evolution. Check the scope against %s.",
            evolve_scope, len(algorithms), exp_dir, CLASSIFICATION_FILENAME,
        )
    return filtered
