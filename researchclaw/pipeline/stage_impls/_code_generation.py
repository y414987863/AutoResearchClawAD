"""Stage 10: Code generation."""

from __future__ import annotations

import ast
import json
import logging
import re
import sys
import time
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
    _ensure_sandbox_deps,
    _extract_code_block,
    _extract_multi_file_blocks,
    _extract_yaml_block,
    _get_evolution_overlay,
    _load_hardware_profile,
    _read_prior_artifact,
    _safe_json_loads,
    _utcnow_iso,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)

# Improvement G: Continuous-action environments that are incompatible with DQN
_CONTINUOUS_ENVS = {
    "pendulum", "halfcheetah", "hopper", "walker2d", "ant", "humanoid",
    "swimmer", "reacher", "invertedpendulum", "inverteddoublependulum",
    "mountaincarcontinuous", "lunarlander-continuous",
}


def _execute_collider_plan_generation(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """Stage 10 (collider_agent mode): generate a ColliderAgent physics prompt.

    Reads the experiment design plan from Stage 9 and uses the LLM to
    translate it into a detailed ColliderAgent-compatible Markdown prompt
    (similar to ``paper-reproduction/*/prompt_figure_N.md``).

    The generated prompt is saved as ``collider_plan.md`` in the stage
    directory.  Stage 12 reads this file and invokes Claude Code with the
    ColliderAgent skills to execute the full physics pipeline.
    """
    exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    hypothesis = _read_prior_artifact(run_dir, "hypotheses.json") or ""
    topic = config.research.topic

    # System prompt: instruct LLM to produce a ColliderAgent-style prompt
    system_prompt = (
        "You are a particle physics expert generating a detailed execution plan for "
        "the ColliderAgent framework. ColliderAgent uses Claude Code to orchestrate "
        "the full collider phenomenology pipeline:\n"
        "  1. FeynRules model generation from a Lagrangian\n"
        "  2. UFO export for MadGraph5\n"
        "  3. MadGraph5 event generation with Pythia8/Delphes\n"
        "  4. MadAnalysis5 analysis\n"
        "  5. Numerical post-processing and figure generation\n\n"
        "The execution plan must follow this Markdown structure:\n"
        "  # 1. Target\n"
        "  (what figure/result to produce)\n"
        "  # 2. Model\n"
        "  ## 2.1 Lagrangian\n"
        "  ## 2.2 Parameters\n"
        "  ## 2.3 Particles\n"
        "  # 3. Collider Process\n"
        "  ## 3.1 Signal Process\n"
        "  ## 3.2 Background Process (if any)\n"
        "  # 4. Numerical Analysis\n"
        "  (step-by-step procedure)\n\n"
        "Be as precise as possible with formulas, parameter values, and analysis steps. "
        "If the topic does not have a defined Lagrangian or specific HEP process, "
        "generate an equivalent phenomenological study appropriate for the topic. "
        "If MadGraph/Monte Carlo is not needed (pure numerical analysis), skip those steps "
        "and describe only the post-processing steps."
    )
    user_prompt = (
        f"Research topic: {topic}\n\n"
        f"Experiment design plan:\n{exp_plan}\n\n"
        f"Hypotheses:\n{hypothesis}\n\n"
        "Generate a detailed ColliderAgent execution plan as a Markdown document."
    )

    collider_plan: str
    if llm is not None:
        try:
            resp = _chat_with_prompt(llm, system_prompt, user_prompt, max_tokens=4096)
            collider_plan = resp.content.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 10 (collider_agent): LLM call failed (%s) — using fallback plan", exc)
            collider_plan = _fallback_collider_plan(topic, exp_plan)
    else:
        collider_plan = _fallback_collider_plan(topic, exp_plan)

    # Write the plan
    plan_path = stage_dir / "collider_plan.md"
    plan_path.write_text(collider_plan, encoding="utf-8")
    logger.info("Stage 10 (collider_agent): wrote physics prompt to %s", plan_path)

    # Also write a metadata file
    import json as _json
    meta = {
        "generated": _utcnow_iso(),
        "mode": "collider_agent",
        "topic": topic,
        "plan_file": "collider_plan.md",
        "plan_length_chars": len(collider_plan),
    }
    (stage_dir / "collider_meta.json").write_text(
        _json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Satisfy Stage 10 contract (output_files requires "experiment/" and
    # "experiment_spec.md").  In collider_agent mode there is no Python
    # experiment; instead we place the ColliderAgent prompt inside the
    # experiment/ directory so downstream contract validation passes.
    exp_dir = stage_dir / "experiment"
    exp_dir.mkdir(exist_ok=True)
    (exp_dir / "collider_plan.md").write_text(collider_plan, encoding="utf-8")

    spec_md = (
        f"# Experiment Specification (collider_agent mode)\n\n"
        f"**Topic:** {topic}\n\n"
        f"**Backend:** ColliderAgent — full HEP pipeline via Claude Code\n\n"
        f"**Physics plan:** `collider_plan.md`\n\n"
        f"Stage 12 will invoke the Claude Code CLI with the ColliderAgent skills\n"
        f"to execute the Lagrangian → FeynRules → UFO → MadGraph5 → Delphes →\n"
        f"MadAnalysis5 pipeline and produce publication-quality figures.\n"
    )
    (stage_dir / "experiment_spec.md").write_text(spec_md, encoding="utf-8")

    return StageResult(
        stage=Stage.CODE_GENERATION,
        status=StageStatus.DONE,
        artifacts=("collider_plan.md", "collider_meta.json", "experiment/", "experiment_spec.md"),
        evidence_refs=("stage-10/collider_plan.md",),
    )


def _fallback_collider_plan(topic: str, exp_plan: str) -> str:
    """Generate a minimal fallback ColliderAgent prompt when LLM is unavailable."""
    return f"""# 1. Target

Investigate the following physics topic using the ColliderAgent pipeline:
**{topic}**

{exp_plan or "Execute the relevant collider phenomenology analysis and generate exclusion contours or kinematic distributions as appropriate."}

---

# 2. Model

## 2.1 Lagrangian

Use the Standard Model as baseline. For beyond-SM contributions,
refer to the experiment design plan above.

## 2.2 Parameters

Use SM parameters. Scan over new-physics parameters as described in the plan.

## 2.3 Particles

Use standard SM particles.

---

# 3. Collider Process

## 3.1 Signal Process

Run the relevant signal processes at the LHC (√s = 13 TeV).

---

# 4. Numerical Analysis

## Step 1: Execute the phenomenology pipeline
Follow the experiment design plan to produce the required figures and results.

## Step 2: Generate output figures
Save all figures to output/figures/ in PDF and PNG format.
"""


def _check_rl_compatibility(code: str) -> list[str]:
    """Detect DQN + continuous-action environment mismatches.

    Returns a list of error strings if incompatible combinations are found.
    """
    errors: list[str] = []
    code_lower = code.lower()
    has_dqn = "dqn" in code_lower
    if not has_dqn:
        return errors

    for env_name in _CONTINUOUS_ENVS:
        if env_name in code_lower:
            errors.append(
                f"RL COMPATIBILITY ERROR: DQN is used with continuous-action "
                f"environment '{env_name}'. DQN only works with DISCRETE action "
                f"spaces. Use SAC, TD3, or PPO instead."
            )
    return errors


def _hard_indexed_instance_keys(code: str) -> set[str]:
    """Keys an algorithm reads as ``instance["k"]`` with no fallback.

    Derived from the source rather than from a fixed list, so this stays valid for
    any problem family: whatever the generated code decides an instance looks
    like, that is what the data files must provide.

    Tracks the first parameter of ``optimize`` plus locals aliased directly from
    it (``inst = dict(instance)``), which is how generated algorithms normally
    copy before mutating. Keys the code also probes defensively
    (``instance.get("k", ...)`` or ``"k" in instance``) are treated as optional
    and excluded — a guarded read cannot raise ``KeyError``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "optimize":
            # Positional-only params come FIRST in the signature, so they must be
            # concatenated, not used as a fallback: `def optimize(instance, /, seed)`
            # has posonlyargs=[instance] and args=[seed], and picking `args` there
            # tracked `seed` and silently disabled this whole check.
            args = node.args.posonlyargs + node.args.args or node.args.kwonlyargs
            if args:
                aliases.add(args[0].arg)
    if not aliases:
        return set()

    # `inst = instance` / `inst = dict(instance)` / `inst = {**instance}`.
    for _ in range(3):  # transitive, but generated code never nests deeply
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            src = node.value
            if isinstance(src, ast.Call) and src.args:
                src = src.args[0]
            elif isinstance(src, ast.Dict) and len(src.keys) == 1 and src.keys[0] is None:
                src = src.values[0]
            if isinstance(src, ast.Name) and src.id in aliases:
                aliases.add(target.id)

    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Load):
            continue
        if not (isinstance(node.value, ast.Name) and node.value.id in aliases):
            continue
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            keys.add(sl.value)

    optional = set()
    for key in keys:
        if f'.get("{key}"' in code or f".get('{key}'" in code:
            optional.add(key)
        elif f'"{key}" in ' in code or f"'{key}' in " in code:
            optional.add(key)
    return keys - optional


def _injected_instance_keys(code: str) -> set[str]:
    """Keys the evaluator supplies to the algorithm on top of the instance file.

    Recognises the three ways generated evaluators build an augmented instance:

    - a dict literal, usually ``{**instance, "x0": ..., "_objective": ...}``
      passed straight into the call. This is the idiomatic form because it does
      not mutate the caller's dict, and it is exactly what the deficiency
      message recommends, so missing it flagged correct code as broken.
    - ``dict(instance, x0=...)`` keyword form.
    - ``<name>["k"] = ...`` subscript assignment.
    - ``<name>.setdefault("k", ...)``, which injects a key without overwriting.

    Deliberately generous: the point is to avoid false positives in the
    data-field check, and an over-broad exemption only weakens that check rather
    than blocking a valid experiment.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    keys: set[str] = set()
    for node in ast.walk(tree):
        # {"k": v} / {**instance, "k": v}
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
        # dict(instance, k=v)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
        ):
            keys.update(kw.arg for kw in node.keywords if kw.arg)
        # d.setdefault("k", v)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
        # d["k"] = v
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant):
                if isinstance(t.slice.value, str):
                    keys.add(t.slice.value)
    return keys


def _evolve_block_problems(path: str, code: str) -> list[str]:
    """Check that the EVOLVE region actually contains the algorithm.

    A block that merely constructs a module-level helper class and calls one of
    its methods leaves the real algorithm outside the evolvable surface, so
    evolution can do nothing but retune constructor arguments. Both checks are
    purely structural — they say nothing about what the algorithm computes.
    """
    lines = code.splitlines()
    starts = [i + 1 for i, ln in enumerate(lines) if "EVOLVE_START" in ln]
    ends = [i + 1 for i, ln in enumerate(lines) if "EVOLVE_END" in ln]
    if not starts or not ends:
        return []  # already reported by the marker check
    start_line, end_line = starts[0], ends[-1]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # already reported by the syntax check

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "optimize"),
        None,
    )
    if fn is None:
        return []  # already reported by the `def optimize(` check

    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # a leading docstring may sit above EVOLVE_START

    problems: list[str] = []

    outside = [
        s for s in body
        if s.lineno < start_line or (s.end_lineno or s.lineno) > end_line
    ]
    if outside:
        problems.append(
            f"LLM4AD_STRUCTURE: `{path}` — the EVOLVE markers cover only part of "
            f"`optimize` ({len(outside)} of {len(body)} statements are outside, "
            f"first at line {outside[0].lineno}). Move EVOLVE_START to the top of "
            "the body (below the docstring) and EVOLVE_END below the final "
            "`return` so the whole algorithm is evolvable."
        )

    helpers = {
        n.name for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n.name != "optimize"
    }
    if helpers:
        # A name bound inside `optimize` is a LOCAL that shadows the module-level
        # helper, not a call to it (`tour_length = 0.0` beside `def tour_length`
        # is not delegation). Reading the local afterwards is also a Load, so
        # every locally-bound name is excluded outright. ast.arg covers nested
        # closure params too. Erring generous only weakens this advisory check;
        # a false positive would accuse correct code of hiding the algorithm.
        local_names = {
            a.arg for a in ast.walk(fn) if isinstance(a, ast.arg)
        } | {
            n.id for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        used = sorted(
            n.id for n in ast.walk(fn)
            if isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and n.id in helpers
            and n.id not in local_names
            and start_line <= n.lineno <= end_line
        )
        if used:
            problems.append(
                f"LLM4AD_STRUCTURE: `{path}` — the EVOLVE block delegates to "
                f"{', '.join(f'`{u}`' for u in dict.fromkeys(used))} defined in the "
                "same file outside the markers, so the algorithm itself is not "
                "evolvable. Inline that logic into `optimize` between the markers; "
                "if it is genuinely shared across algorithms, move it to a module at "
                "the experiment root and import it."
            )

    return problems


def _check_llm4ad_structure(files: dict[str, str]) -> list[str]:
    """Validate that generated files satisfy the LLM4AD task-package layout.

    Returns deficiency strings; empty means the structure is package-ready (one
    evolvable module per algorithm, EVOLVE markers, static data/, and a main.py
    supporting the ``--algorithm`` / importlib / primary-metric contract).
    """
    problems: list[str] = []

    algo_files = [
        k for k in files
        if k.startswith("algorithms/")
        and k.endswith(".py")
        # `__init__.py` / `__main__.py` are package machinery, not algorithms —
        # the packager only picks `algorithms/<algo>/<algo>.py`, and the nested
        # layout needs an `__init__.py` in every dir, so this fired on every run.
        # A non-dunder module with a mismatched name is still reported below.
        and not Path(k).name.startswith("__")
    ]
    if not algo_files:
        problems.append(
            "LLM4AD_STRUCTURE: no `algorithms/<algo>/<algo>.py` found — each "
            "evolvable algorithm must live in its own subdirectory."
        )

    for k in algo_files:
        parts = k.split("/")
        if len(parts) < 3 or parts[-2] != parts[-1][: -len(".py")]:
            problems.append(
                f"LLM4AD_STRUCTURE: `{k}` — file name must match its directory "
                "name (algorithms/<algo>/<algo>.py)."
            )
        code = files[k]
        if "EVOLVE_START" not in code or "EVOLVE_END" not in code:
            problems.append(
                f"LLM4AD_STRUCTURE: `{k}` missing EVOLVE_START/EVOLVE_END markers."
            )
        if "def optimize(" not in code:
            problems.append(
                f"LLM4AD_STRUCTURE: `{k}` missing `def optimize(instance, seed)`."
            )
        problems.extend(_evolve_block_problems(k, code))

    # Any extension counts. Instance formats in this domain are not all JSON
    # (TSPLIB, .mps, .npz, .csv), and the runner delegates parsing to
    # `evaluator.load_instance`, so demanding `.json` here would reject valid
    # experiments. What matters is that instances are shipped as static files.
    _instance_names = [
        k for k in files
        if k.startswith("data/") and "/" not in k[len("data/"):]
    ]
    if not _instance_names:
        problems.append(
            "LLM4AD_STRUCTURE: no instance files under `data/` — static "
            "instances must be generated once and shipped with the code."
        )
    else:
        # A non-JSON instance is fine, but only if the experiment says how to
        # read it: run_single.py falls back to json.load and raises otherwise.
        _non_json = [k for k in _instance_names if not k.endswith(".json")]
        if _non_json and "def load_instance(" not in files.get("evaluator.py", ""):
            problems.append(
                f"LLM4AD_STRUCTURE: `{_non_json[0]}` is not JSON, but evaluator.py "
                "defines no `load_instance(path)`. Add it so the package runner "
                "can read this format (it only falls back to json.load)."
            )

    main_code = files.get("main.py", "")
    if not main_code:
        problems.append(
            "LLM4AD_STRUCTURE: missing `main.py` entry point."
        )
    else:
        for needle, label in (
            ("--algorithm", "`--algorithm` CLI argument"),
            ("importlib", "dynamic `importlib` loading"),
            ("primary_metric", "`primary_metric` output"),
            ('if __name__ == "__main__":', "`if __name__ == \"__main__\":` entry guard"),
        ):
            if needle not in main_code:
                problems.append(
                    f"LLM4AD_STRUCTURE: `main.py` missing {label}."
                )

    # Stage-13's package runner imports these two symbols BY NAME and nothing
    # else; a missing export fails every algorithm at package-build time (after
    # Stage 10). Check the code text rather than execute it — importing would run
    # arbitrary generated code.
    #
    # Deliberately only `evaluator.py`: everything domain-specific (seeds,
    # budgets, bounds, how to read the algorithm's return value) is that module's
    # private business. Requiring named helpers in `benchmarks.py`/`stats_utils.py`
    # would hard-wire one problem family — continuous box-constrained
    # optimisation — into a profile that must cover the whole domain.
    for mod, needles, label in (
        ("evaluator.py", ("PRIMARY_METRIC", "evaluate_instance"),
         "`PRIMARY_METRIC`, `evaluate_instance(instance, solve)`"),
    ):
        code = files.get(mod, "")
        # ``needles`` holds only the symbols/text that must appear in the module;
        # ``label`` is the human-readable name used in the message.
        missing = [n for n in needles if n not in code]
        if missing:
            problems.append(
                f"LLM4AD_STRUCTURE: `{mod}` missing {label} — "
                f"the task-package runner imports it by name."
            )

    # DIRECTION CONTRACT: evaluator.py must declare, statically and
    # unambiguously, which way its primary metric is judged. Downstream stages
    # (13's promote/evolve, 14's analysis, 17's paper tables) read this ONE
    # declaration as the authoritative direction, instead of each independently
    # guessing from the metric name (which mis-read ``valid_prediction_time`` as
    # lower-is-better and promoted a regression as an "improvement"). Without a
    # declaration the pipeline falls back to config, which is hard-coded
    # ``minimize`` and openly conflicts with a maximise metric.
    #
    # Require the static dict form — a runtime ``print`` line is parseable from
    # stdout but the *evaluator* often never runs at structure-check time, and a
    # bare MetricType import couples the module to a class it may not need.
    _eval_dir_code = files.get("evaluator.py", "")
    if "evaluate_instance" in _eval_dir_code and "PRIMARY_METRIC" in _eval_dir_code:
        if not re.search(
            r'METRIC_DEF\s*=\s*\{[^}]*"direction"\s*:\s*"(minimize|maximize)"',
            _eval_dir_code,
        ):
            problems.append(
                "LLM4AD_STRUCTURE: `evaluator.py` must define a static direction "
                "declaration so downstream stages agree on which way is better. "
                "Add a module-level constant, e.g.\n"
                '    METRIC_DEF = {"primary_metric": PRIMARY_METRIC, '
                '"direction": "maximize"}   # "maximize" when larger is better, '
                '"minimize" when smaller is better\n'
                "and set \"direction\" to match how PRIMARY_METRIC is actually "
                "computed (read the comment describing the metric — e.g. "
                "\"larger = better correction\" means \"maximize\")."
            )

    # A name check cannot tell the required `evaluate_instance(instance, solve)`
    # apart from an aggregate-only helper such as `evaluate_instance(per_seed)`,
    # and the difference is fatal: the runner passes the algorithm callable in.
    _eval_code = files.get("evaluator.py", "")
    if "evaluate_instance" in _eval_code:
        try:
            _tree = ast.parse(_eval_code)
        except SyntaxError:
            _tree = None
        if _tree is not None:
            for node in ast.walk(_tree):
                if not isinstance(node, ast.FunctionDef) or node.name != "evaluate_instance":
                    continue
                n_pos = len(node.args.posonlyargs) + len(node.args.args)
                if n_pos < 2:
                    problems.append(
                        "LLM4AD_STRUCTURE: `evaluator.py` — `evaluate_instance` takes "
                        f"{n_pos} positional argument(s); the runner calls "
                        "`evaluate_instance(instance, solve)` where `solve` is the "
                        "algorithm's `optimize`. Move the seeds/repeats loop inside "
                        "this function so the runner stays problem-agnostic."
                    )
                break

    # Every key an algorithm indexes unconditionally must exist in every instance
    # file, or that algorithm dies with a KeyError the moment LLM4AD evaluates it.
    # Stage 12 can mask this (main.py may inject values at runtime), so it only
    # surfaces during evolution, where every individual scores -inf and the run
    # looks like a modelling failure rather than a contract violation.
    #
    # The required keys come from the source, not from a hardcoded list, so this
    # check makes no assumption about the problem family.
    instance_files = {
        k: v for k, v in files.items()
        if k.startswith("data/") and k.endswith(".json")
    }
    if instance_files and algo_files:
        injected = _injected_instance_keys(files.get("evaluator.py", ""))
        parsed: dict[str, set[str]] = {}
        for name, raw in instance_files.items():
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                problems.append(
                    f"LLM4AD_STRUCTURE: `{name}` is not valid JSON — instance "
                    "files must be machine-readable."
                )
                continue
            if isinstance(obj, dict):
                parsed[name] = set(obj)
        for k in sorted(algo_files):
            required = _hard_indexed_instance_keys(files[k]) - injected
            for key in sorted(required):
                lacking = sorted(n for n, present in parsed.items() if key not in present)
                if not lacking:
                    continue
                shown = ", ".join(lacking[:3])
                more = f" (+{len(lacking) - 3} more)" if len(lacking) > 3 else ""
                problems.append(
                    f"LLM4AD_STRUCTURE: `{k}` reads `instance[\"{key}\"]` "
                    f"unconditionally, but {shown}{more} lack that key. Add it to "
                    "every instance file, inject it in `evaluate_instance` before "
                    "calling the algorithm, or read it with a default via "
                    f"`instance.get(\"{key}\", ...)`."
                )

    return problems


def _repair_llm4ad_structure(
    files: dict[str, str],
    problems: list[str],
    *,
    llm: Any,
    _pm: Any,
    max_repair: int,
) -> tuple[dict[str, str], list[str], int]:
    """Round-trip LLM4AD structure deficiencies through the audit repair loop.

    ``_check_llm4ad_structure`` flags defects (partial EVOLVE coverage, an
    algorithm delegating to a module-level helper, a missing marker, etc.) but
    those run in a separate channel from ``validate_code``, so the stage
    repaired syntax/import errors and then reported the structure issues as
    "FAILED after 0 total repair attempt(s)" — a false negative that read as a
    generation failure and leaked into the paper. This routine re-uses the same
    single-file ``code_repair`` prompt the syntax loop does, so the LLM gets ONE
    consistent shot at each flagged file, and the structure check is re-run
    afterwards so a fix actually has to make the check pass.

    Only issues bound to a file that already exists in ``files`` are repaired.
    Global deficiencies ("no ``data/``", "missing ``main.py``") cannot be fixed
    by editing an existing file, and handing them to a single-file repair prompt
    invites the LLM to invent unrelated code, so they are left for the caller to
    report.

    Returns ``(new_files, remaining_problems, repair_attempts)``.
    """
    import re as _re

    # Group the flagged issues by the file they name. A problem string carries
    # its path inside backticks (e.g. ``algorithms/x/x.py``); problems without
    # a resolvable, existing file are held back.
    targeted: dict[str, list[str]] = {}
    leftover: list[str] = []
    for prob in problems:
        _m = _re.search(r"`([^`]+\.py)`", prob)
        if _m is None:
            leftover.append(prob)
            continue
        path = _m.group(1)
        if path not in files:
            leftover.append(prob)
            continue
        targeted.setdefault(path, []).append(prob)

    rp_attempts = 0
    for fname, file_problems in targeted.items():
        issues_text = "\n".join(f"- {p}" for p in file_problems)
        # Scope the context to the file being repaired and its imports instead
        # of every generated file — the full dump pushed single requests past
        # the model's tolerance and produced 504/429 failures that aborted the
        # stage after Beast mode had already succeeded.
        all_files_ctx = _scoped_files_ctx(files, fname)
        rp = _pm.sub_prompt(
            "code_repair",
            fname=fname,
            issues_text=issues_text,
            all_files_ctx=all_files_ctx,
        )
        resp = _chat_with_prompt(llm, rp.system, rp.user, max_tokens=rp.max_tokens)
        _fixed = _extract_code_block(resp.content)
        rp_attempts += 1
        if not _fixed.strip():
            logger.warning(
                "LLM4AD structure repair for %s returned empty code, keeping original",
                fname,
            )
            continue
        if validate_code(_fixed).ok:
            files[fname] = _fixed
            logger.info(
                "LLM4AD structure repair fixed %s (%d issue(s))",
                fname, len(file_problems),
            )
        else:
            logger.warning(
                "LLM4AD structure repair for %s produced code that fails "
                "validation; keeping original (%d issue(s) remain)",
                fname, len(file_problems),
            )

    remaining = _check_llm4ad_structure(files)
    return files, remaining, rp_attempts


def _scoped_files_ctx(files: dict[str, str], target: str) -> str:
    """Build a code_repair context scoped to ``target`` and its local imports.

    The old ``all_files_ctx`` dumped EVERY file the experiment generated into
    each repair prompt. A stage-10 codebase has a dozen+ modules and several of
    them are large (a neural-net algorithm, an evaluator, a data generator), so
    the single ``code_repair`` request routinely ran into the hundreds of KB of
    input, and the upstream model answered with 504 Gateway Time-out / 429 —
    the repair loop then burned all retries and failed the STAGE even though the
    code had already been generated (Beast mode had succeeded).

    For fixing ONE file we only need that file plus the sibling modules it
    imports. Sending a data generator and every algorithm alongside an ESN is
    noise that inflates the request for no benefit. Top-level imports are read
    statically (AST); transitive closures are deliberately NOT followed, because
    a single repair should be minimal and the re-validation after the prompt
    re-checks the whole project anyway.
    """
    deps: set[str] = set()
    content = files.get(target, "")
    try:
        tree = ast.parse(content)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if f"{name}.py" in files:
                    deps.add(f"{name}.py")
    ordered = [target] + sorted(deps)
    return "\n\n".join(
        f"```filename:{f}\n{files[f]}\n```" for f in ordered if f in files
    )


def _summarize_files_ctx(
    files: dict[str, str],
    entry_hint: str = "main.py",
    full_files: tuple[str, ...] = (),
) -> str:
    """Render the project as a bounded context for CROSS-FILE repairs.

    ``_scoped_files_ctx`` is for fixing ONE file in isolation. But ``deep
    repair`` and ``ablation repair`` ask the model to touch MULTIPLE files
    (rewriting an ablation class, renaming a config that shadows a stdlib
    package), so the model must see the whole project — scoping it to one file
    would make it rewrite a class without knowing the siblings it must stay
    consistent with, and degrade the result.

    The problem is the same as ``code_repair`` had: dumping every file verbatim
    into a single request can reach hundreds of KB and 504/429. The fix used by
    the alignment check is better than a flat truncation: keep the ENTRY point
    in full (capped), and summarize the rest as imports + signatures. That keeps
    the cross-file structure visible while bounding the request. Truncating
    blindly was tried before (BUG-171) and produced false "incomplete" reads,
    hence the summary, not a ``[:N]`` cap.

    ``full_files`` is the exception to the summary: a repair that must see a
    method BODY (deep validation reports "identical AST" / "copy-paste
    ablation", which cannot be judged from a signature alone) passes the files
    it needs verbatim so the model can actually fix them. Everything else stays
    summarized to keep the request bounded.
    """
    entry = entry_hint if entry_hint in files else "main.py"
    # Prefer the real entry point: a file with a __main__ guard, main.py first.
    if entry == "main.py" and "main.py" in files:
        for _fn, _cd in files.items():
            if _fn != "main.py" and 'if __name__' in _cd and '__main__' in _cd:
                if _cd.count("\n") > files["main.py"].count("\n") * 1.5:
                    entry = _fn
                    break

    def _sig_preview(fname: str, code: str) -> str:
        sig_lines = [
            l for l in code.split("\n")
            if l.strip().startswith(("def ", "class ", "async def ", "import ", "from "))
        ]
        if sig_lines:
            return f"# --- {fname} (imports + signatures) ---\n" + "\n".join(sig_lines)
        prev = code[:800]
        if len(code) > 800:
            prev += f"\n... [{len(code) - 800} more chars]"
        return f"# --- {fname} (preview) ---\n{prev}"

    parts: list[str] = []
    for fname, code in sorted(files.items()):
        if fname in full_files:
            block = f"# --- {fname} (FULL) ---\n{code}"
            if len(block) > 20000:
                block = block[:20000] + "\n... [truncated at 20000 chars]"
            parts.append(block)
        elif fname == entry:
            block = f"# --- {fname} (FULL entry point) ---\n{code}"
            if len(block) > 12000:
                block = block[:12000] + "\n... [entry truncated at 12000 chars]"
            parts.append(block)
        else:
            parts.append(_sig_preview(fname, code))
    out = "\n\n".join(parts)
    if len(out) > 30000:
        out = out[:30000] + "\n... [project truncated]"
    return out


def _dangling_local_imports(files: dict[str, str]) -> list[tuple[str, str]]:
    """Report imports nothing can satisfy — informational only, never a gate.

    A real judgment on whether the package runs belongs to the smoke run; this
    only surfaces candidates when a generated file imports a name that is
    neither a generated module nor importable here.
    """
    import importlib.util as _ilu

    known = {
        Path(f).stem for f in files if f.endswith(".py")
    } | {
        f[: -len(".py")].replace("/", ".") for f in files if f.endswith(".py")
    }

    _installed: dict[str, bool] = {}

    def _is_installed(mod: str) -> bool:
        if mod not in _installed:
            try:
                _installed[mod] = _ilu.find_spec(mod) is not None
            except (ImportError, ValueError, AttributeError):
                # A parent package that refuses to load tells us nothing about
                # whether the name resolves at runtime; assume it does rather
                # than raise a false alarm.
                _installed[mod] = True
        return _installed[mod]

    out: list[tuple[str, str]] = []
    for fname, code in files.items():
        if not fname.endswith(".py"):
            continue
        for mod in re.findall(
            r"^\s*(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            code, re.MULTILINE,
        ):
            if mod in known or mod.startswith("_"):
                continue
            if not _is_installed(mod):
                out.append((fname, mod))
    return out


def _merge_repaired_files(
    files: dict[str, str], repaired: dict[str, str] | None, *, label: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Merge a repair reply into ``files``, accepting all new files.

    A repair reply contains only the files it changed, so requiring ``main.py``
    would discard targeted fixes. New files are accepted wholesale — a rename
    arrives as a NEW module plus edits to its importers. Whether the merge is
    sane is the smoke run's job; guessing here mis-classifies.
    """
    repaired = repaired or {}
    known = {k: v for k, v in repaired.items() if k in files}
    new = {k: v for k, v in repaired.items() if k not in files}
    if new:
        logger.info("Stage 10: %s added new file(s) %s", label, ", ".join(sorted(new)))
    applied = {**known, **new}
    return {**files, **applied}, applied


def _try_smoke_run(exp_dir: Path, config: Any) -> tuple[int, str] | None:
    """Run the DEFAULT entry point once; None on success.

    Stage 12 runs ``main.py`` with no flags, so a crash on that path (an
    ablation assert, a full-comparison check) must be caught here — running
    ``--algorithm`` would bypass it.
    """
    import subprocess as _sp

    py = getattr(getattr(config, "experiment", None), "sandbox", None)
    py_path = getattr(py, "python_path", "") or ""

    # Resolve to absolute paths. run_dir (and thus exp_dir/main_py) is built
    # from the user's `--output`, which may be relative — passing a relative
    # main_py with cwd=exp_dir makes Python resolve the path against cwd again,
    # producing ".../experiment/artifacts/.../experiment/main.py" (duplicated)
    # and "can't open file" even though main.py exists.
    exp_dir = Path(exp_dir).resolve()
    main_py = exp_dir / "main.py"
    if not main_py.is_file():
        # No entry point — the stage's own main.py check reports that.
        return None

    try:
        proc = _sp.run(
            [py_path or sys.executable, str(main_py)],
            cwd=str(exp_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except _sp.TimeoutExpired:
        # A slow-but-legal run is not a launch failure: it started and is
        # simply past the budget. OpenCode is expected to ship a seconds-scale
        # demo config; if it did not, that is a warning, not a reason to kill
        # the stage. Return None (success) and let the run proceed.
        logger.warning(
            "Stage 10: smoke run exceeded 120s — experiment started but is slow; "
            "not treating as a launch failure"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        return (1, f"failed to launch: {exc}")
    if proc.returncode == 0:
        return None
    tail = (proc.stderr or "")[-1500:] or (proc.stdout or "")[-1500:]
    return (proc.returncode, tail)


def _normalize_plan_items(value: Any) -> list[str]:
    """Flatten a plan field (list[str] / list[dict] / dict[str, str]) to names.

    S9 plans do not share a fixed schema for how methods/baselines/ablations are
    declared: a value can be a list of strings, a list of dicts (each with
    ``name``/``class_name``/``citation``/``description``), a dict keyed by names,
    or a prose string. We only need a bag of name candidates, so any string leaf
    is kept and the original structure is discarded deliberately (classification
    is per-algorithm, not per-plan-row).
    """
    out: list[str] = []
    # English prose stopwords: a description like "pointwise_linear is the
    # primary baseline" would otherwise leak "is"/"the"/"primary" into the name
    # bag and the LLM prompt. Only applied to the prose branch below, never to a
    # name in a list/dict key or value, so real algorithm names are never
    # filtered out.
    _STOPWORDS = {
        "a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "is",
        "it", "of", "on", "or", "that", "the", "this", "to", "with", "we",
        "our", "method", "methods", "baseline", "baselines", "primary",
        "main", "proposed", "ablation", "ablations",
    }
    if value is None:
        return out
    if isinstance(value, str):
        # A description phrase ("pointwise_linear is the primary baseline") is
        # not a name; keep only plausible identifier-like tokens so the LLM and
        # the fallback matcher are not flooded with prose.
        for tok in re.split(r"[\s,;:()\"']+", value):
            tok = tok.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.\-]*", tok) and tok.lower() not in _STOPWORDS:
                out.append(tok)
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(_normalize_plan_items(item))
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            out.append(str(k))
            out.extend(_normalize_plan_items(v))
        return out
    out.append(str(value))
    return out


def _plan_field_category(key: str) -> str | None:
    """Classify a plan key to {proposed,baseline,ablation} by semantic prefix.

    The stage-9 generator does not emit a fixed schema: the split may be
    ``proposed_methods`` / ``baselines`` / ``ablations``, or ``conditions``,
    ``method``, ``baseline_1`` / ``baseline_2``, or a topic-specific spelling.
    Rather than enumerate keys, we match on a stable prefix so any spelling a
    future prompt produces is still sorted into the right bag. The priority
    (baseline over ablation over proposed) is applied later in the classifier;
    here we only decide which bag a key feeds.
    """
    k = (key or "").strip().lower()
    if not k:
        return None
    if k.startswith("baseline") or k.startswith("base_"):
        return "baseline"
    if k.startswith("ablat"):
        return "ablation"
    if (
        k.startswith("propos")
        or k.startswith("method")
        or k.startswith("condition")
        or k in ("ours", "main")
    ):
        return "proposed"
    return None


def _is_algorithm_row(value: Any) -> bool:
    """True when a dict is an experiment row, not a nested config/prose group.

    The S9 generator models each method/baseline/ablation as a dict carrying a
    ``name`` (or ``class_name``) plus prose fields like ``description`` /
    ``implementation_spec`` / ``expected_effect``. That dict is a *row*, and only
    its ``name``/``class_name`` are algorithm names worth extracting; recursing
    into the prose would flood the name bag with natural-language tokens. A dict
    without either field (e.g. ``compute_budget``, ``metrics``, a keyed-by-name
    map) is not a row and is flattened normally.
    """
    if not isinstance(value, dict):
        return False
    return "name" in value or "class_name" in value


def _collect_plan_names(value: Any) -> list[str]:
    """Flatten a plan field value into a list of name-candidate strings.

    ``value`` may be a list[str], list[dict], dict[str, ...], str, None, or a
    nested mix of these. Recursion flattens whatever the plan puts under a given
    key, except that a dict *row* (a method/baseline/ablation declaration) only
    contributes its ``name``/``class_name`` — its prose fields are not names.
    Callers decide which bag the result feeds by the key it was found under.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return _normalize_plan_items(value)
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_collect_plan_names(item))
        return out
    if isinstance(value, dict):
        if _is_algorithm_row(value):
            row = [value.get("name"), value.get("class_name")]
            return [n for n in row if isinstance(n, str) and n.strip()]
        out = []
        for k, v in value.items():
            # For a row {'name': 'cma_es_default'} the key is a field name, not an
            # algorithm name — do not add it. A dict keyed BY algorithm name
            # ({'cma_es_default': {...}}) should contribute that key; we cannot
            # tell the two apart without knowing the schema, and adding an extra
            # candidate is harmless because the classifier only keeps names that
            # match a real algorithm dir.
            out.append(str(k))
            out.extend(_collect_plan_names(v))
        return out
    return [str(value)]


def _extract_plan_names(exp_plan: Any) -> tuple[list[str], list[str], list[str]]:
    """Return (proposed, baselines, ablations) name bags from the S9 plan.

    ``exp_plan`` is the parsed YAML dict (or str). We do NOT assume any fixed
    set of keys: the whole plan tree is walked, and every key is routed into a
    bag by the semantic prefix of its name (see :func:`_plan_field_category`).
    Anything under a key that looks like a baseline is a baseline, etc. This
    makes the classifier tolerant of every schema the generator has produced
    (proposed_methods / conditions / method / baseline_1 ...) and of future
    paraphrases, instead of failing when it meets a key it never saw.
    """
    if isinstance(exp_plan, str):
        if not exp_plan.strip():
            return [], [], []
        parsed = None
        try:
            import yaml as _yaml
            parsed = _yaml.safe_load(exp_plan)
        except Exception:  # noqa: BLE001 - prose plan, not YAML
            parsed = None
        if not isinstance(parsed, dict):
            return [], [], []
    elif isinstance(exp_plan, dict):
        parsed = exp_plan
    else:
        return [], [], []

    proposed: list[str] = []
    baselines: list[str] = []
    ablations: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                cat = _plan_field_category(key)
                if cat is None:
                    # Unclassified key (e.g. metadata, datasets, metrics): drill
                    # into it so any nested algorithm names still surface, but
                    # contribute nothing itself.
                    _walk(val)
                    continue
                # The top-level key is authoritative for its value's category,
                # e.g. ``proposed_methods:`` makes every row below it proposed.
                # Do not recurse: the value's inner field keys (``name``,
                # ``description`` ...) are not themselves category keys, and
                # recursing would re-read their values into the wrong bag.
                bag = proposed if cat == "proposed" else baselines if cat == "baseline" else ablations
                bag.extend(_collect_plan_names(val))
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(parsed)
    return proposed, baselines, ablations


def _classify_algorithms_with_fallback(
    algo_names: list[str],
    exp_plan: Any,
) -> dict[str, str]:
    """Deterministic classification used when the LLM is unavailable or fails.

    Exact name match is authoritative (a name listed under ``proposed_methods``
    is proposed). Substring match (an ``esn_N200`` grid entry inside ``esn``)
    falls back to that name's category, with ties resolved by a stable
    ``proposed > baseline > ablation`` priority so the outcome does not depend
    on dict iteration order.

    An algorithm that matches no declared baseline or ablation is **proposed**,
    not "unknown": the stage-9 plan models the experiment as ``conditions:``
    (the methods being tested) plus separately-declared baselines, so anything
    that is not a baseline is by construction the method under test. This is
    what keeps a ``{categories: [proposed]}`` scope from silently shrinking to
    zero packages on a topic that never declared ``proposed_methods``.
    """
    if not algo_names:
        return {}
    proposed, baselines, ablations = _extract_plan_names(exp_plan)

    def _cat_for(name: str) -> str | None:
        # baselines/ablations are the stricter label: a method can appear under
        # ``conditions:`` (which we fold into proposed) AND be named as a
        # baseline in prose ("pointwise_linear is the primary baseline"). A name
        # explicitly declared a baseline must stay a baseline, so check those
        # before the proposed bag.
        if name in baselines:
            return "baseline"
        if name in ablations:
            return "ablation"
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.\-]*", name or "") and name in proposed:
            return "proposed"
        return None

    result: dict[str, str] = {}
    for name in algo_names:
        cat = _cat_for(name)
        if cat is None:
            # Substring candidates: match the algorithm's stem against any plan
            # name. ``esn`` matches ``esn_N200``; stem is the directory name.
            stem = re.sub(r"[\d_.\-]+$", "", name).strip("_")
            cand_names = [n for n in proposed + baselines + ablations if stem and stem in n]
            if cand_names:
                # Same baseline > ablation > proposed priority as exact match.
                if any(n in baselines for n in cand_names):
                    cat = "baseline"
                elif any(n in ablations for n in cand_names):
                    cat = "ablation"
                elif any(n in proposed for n in cand_names):
                    cat = "proposed"
            if cat is None:
                cat = "proposed"
        result[name] = cat
    return result


def _classify_algorithms(
    exp_dir: Path,
    exp_plan: Any,
    llm: LLMClient | None,
    *,
    max_attempts: int = 2,
) -> dict[str, str]:
    """Classify the generated algorithm tree as proposed/baseline/ablation.

    The plan's split keys and the directory names do not correspond one-to-one
    (a topic may list ``conditions:`` or grid entries, or omit a split entirely)
    so the classifier operates on *actual* algorithm names and asks the LLM to
    decide, then falls back to a deterministic matcher when the LLM call fails
    or returns nothing usable. Never raises: classification is advisory and a
    missing/partial result only degrades evolve_scope filtering.
    """
    try:
        from researchclaw.pipeline.llm4ad_task_packages import _discover_algorithms
        discovered = _discover_algorithms(exp_dir)
        algo_names = [a for a, _ in discovered]
    except Exception:  # noqa: BLE001
        algo_names = []
    if not algo_names:
        return {}

    fallback = _classify_algorithms_with_fallback(algo_names, exp_plan)

    if llm is None:
        logger.info("Stage 10: no LLM — classifying algorithms by plan-name fallback")
        return fallback

    # Build the prompt from plan names and the *actual* tree, so we never ask
    # the model to imagine algorithms that are not in the directory.
    proposed, baselines, ablations = _extract_plan_names(exp_plan)
    _plan_section = (
        f"proposed_methods: {proposed or '(<not declared>)'}\n"
        f"baselines: {baselines or '(<not declared> or free-form prose in plan>)'}\n"
        f"ablations: {ablations or '(<not declared>)'}\n"
    )
    _algo_section = "\n".join(f"- {a}" for a in algo_names)

    user_prompt = (
        "The following experimental algorithms were generated in the pipeline.\n\n"
        "PLAN (from the experiment design stage):\n"
        f"{_plan_section}\n\n"
        "ACTUAL ALGORITHM DIRECTORIES (names are authoritative):\n"
        f"{_algo_section}\n\n"
        "For EACH actual algorithm, decide whether it is the PROPOSED method, a "
        "BASELINE, or an ABLATION, using your judgment and the plan above. The plan "
        "may not declare all categories or the names may differ from the directory "
        "names; use code/libary knowledge to decide. Do NOT invent algorithms. If a "
        "category is not declared, still classify by the nature of the algorithm.\n"
        "Return ONLY JSON of shape "
        '{"classification": {"<algo_name>": "proposed"|"baseline"|"ablation"}}, '
        "with every actual algorithm from the list above and no others.\n"
    )
    system_prompt = (
        "You classify machine-learning / optimization experiment algorithms."
    )

    for attempt in range(max_attempts):
        try:
            resp = _chat_with_prompt(llm, system_prompt, user_prompt, max_tokens=2048)
            raw = resp.content or ""
            parsed = None
            _yaml_block = _extract_yaml_block(raw)
            for candidate in (raw, _yaml_block):
                if not candidate or not candidate.strip():
                    continue
                try:
                    candidate_parsed = _safe_json_loads(candidate, None)
                except Exception:  # noqa: BLE001
                    candidate_parsed = None
                if isinstance(candidate_parsed, dict):
                    parsed = candidate_parsed
                    break
            mapping = parsed.get("classification", {}) if isinstance(parsed, dict) else {}
            if isinstance(mapping, dict) and mapping:
                result = {
                    a: str(mapping.get(a, fallback.get(a, "proposed")))
                    for a in algo_names
                }
                # Keep any algorithm the model skipped as the fallback guess;
                # an algorithm the model labels "unknown" is kept as-is.
                for a in algo_names:
                    if a not in result:
                        result[a] = fallback.get(a, "proposed")
                logger.info(
                    "Stage 10: LLM classified %d algorithm(s): %s",
                    len(algo_names), result,
                )
                return result
            logger.warning(
                "Stage 10: algorithm classification attempt %d returned no usable "
                "JSON (got %r) — %s", attempt + 1, raw[:200], "retrying" if attempt + 1 < max_attempts else "falling back",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Stage 10: algorithm classification call %d raised %s — %s",
                attempt + 1, exc,
                "retrying" if attempt + 1 < max_attempts else "falling back",
            )
        if attempt + 1 < max_attempts:
            time.sleep(2 * (attempt + 1))
    return fallback


def _maybe_classify_algorithms(
    exp_dir: Path,
    exp_plan_text: str,
    config: Any,
    llm: LLMClient | None,
) -> None:
    """Best-effort write of ``experiment/algorithms_classification.json``.

    Runs only when LLM4AD boost is enabled — the file is only consumed by the
    evolution scoping filter, so without boost it would be dead output. Does
    not raise; an absent file means the stage-13 filter treats evolution scope
    as "everything".
    """
    _l4b = getattr(getattr(config, "experiment", None), "llm4ad_boost", None)
    if _l4b is None or not getattr(_l4b, "enabled", False):
        return
    _evo = getattr(_l4b, "evolution", None)
    _scope = getattr(_evo, "evolve_scope", None) if _evo is not None else None
    if not _scope or not isinstance(_scope, dict):
        # No category filtering requested — no need to burn an LLM call.
        return

    try:
        if exp_plan_text and not exp_plan_text.strip():
            plan = None
        elif exp_plan_text:
            try:
                import yaml as _yaml
                plan = _yaml.safe_load(exp_plan_text)
            except Exception:  # noqa: BLE001
                plan = exp_plan_text
        else:
            plan = None
        result = _classify_algorithms(exp_dir, plan, llm)
        payload = {
            "generated": _utcnow_iso(),
            "source": "stage-10 llm" if llm is not None else "stage-10 plan fallback",
            "classification": result,
        }
        (exp_dir / "algorithms_classification.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Stage 10: wrote algorithms_classification.json (n=%d)", len(result))
    except Exception as exc:  # noqa: BLE001 - advisory, never fail the stage
        logger.warning(
            "Stage 10: algorithm classification skipped (non-fatal): %s", exc,
        )


def _execute_code_generation(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    # ── ColliderAgent mode: generate a physics prompt instead of Python code ─
    if config.experiment.mode == "collider_agent":
        return _execute_collider_plan_generation(
            stage_dir, run_dir, config, adapters, llm=llm, prompts=prompts
        )
    # ── End ColliderAgent bypass ──────────────────────────────────────────────

    exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    metric = config.experiment.metric_key
    max_repair = 5  # BUG-14: Increased from 3 to give more chances for critical bugs
    files: dict[str, str] = {}
    validation_log: list[str] = []

    # --- Detect available packages for sandbox ---
    _pm = prompts or PromptManager()

    # --- Hardware-aware package hint ---
    hw_profile = _load_hardware_profile(run_dir)
    if config.experiment.mode in ("sandbox", "docker"):
        if config.experiment.mode == "docker":
            pkg_prefix = "docker mode"
            _net_policy = config.experiment.docker.network_policy
            _base_pkgs = (
                ", torchvision, torchaudio, matplotlib, seaborn, scipy, "
                "tqdm, torchdiffeq, gymnasium, networkx, PyYAML, Pillow, "
                "transformers, datasets, accelerate, peft, bitsandbytes, "
                "timm, einops, torchmetrics, h5py"
            )
            if _net_policy == "none":
                pkg_extras = _base_pkgs + " (ONLY pre-installed packages — NO pip install available)"
            elif _net_policy in ("setup_only", "pip_only"):
                pkg_extras = _base_pkgs + ", and additional pip-installable packages via requirements.txt"
            else:
                pkg_extras = _base_pkgs + ", and additional pip-installable packages (auto-detected from imports)"
        else:
            pkg_prefix = "sandbox mode"
            pkg_extras = ""
        if hw_profile and hw_profile.get("has_gpu"):
            gpu_type = hw_profile.get("gpu_type", "cuda")
            gpu_name = hw_profile.get("gpu_name", "GPU")
            tier = hw_profile.get("tier", "limited")
            if tier == "high":
                device_hint = f"torch.device('{gpu_type}')"
                pkg_hint = (
                    f"\nAVAILABLE PACKAGES ({pkg_prefix}): Python stdlib, numpy, torch, sklearn, scipy, pandas{pkg_extras}.\n"
                    f"GPU: {gpu_name} ({gpu_type}). You MAY use PyTorch with GPU acceleration.\n"
                    f"Use `device = {device_hint}` for tensor operations.\n"
                )
            else:  # limited (low VRAM NVIDIA or MPS)
                device_hint = f"torch.device('{gpu_type}')"
                pkg_hint = (
                    f"\nAVAILABLE PACKAGES ({pkg_prefix}): Python stdlib, numpy, torch, sklearn, scipy, pandas{pkg_extras}.\n"
                    f"GPU: {gpu_name} ({gpu_type}) — LIMITED performance.\n"
                    f"Use `device = {device_hint}` but design LIGHTWEIGHT experiments:\n"
                    f"- Small models (<1M parameters)\n"
                    f"- Few epochs (<=20)\n"
                    f"- Small datasets (<=10K samples)\n"
                    f"- Avoid large batch sizes\n"
                )
        else:
            pkg_hint = _pm.block("pkg_hint_sandbox")
    else:
        pkg_hint = ""

    # --- Compute budget hint ---
    time_budget_sec = config.experiment.time_budget_sec
    try:
        compute_budget = _pm.block("compute_budget").replace(
            "{time_budget_sec}", str(time_budget_sec)
        )
    except Exception:  # noqa: BLE001
        compute_budget = (
            f"\n## Compute Budget Constraint\n"
            f"- Total execution time limit: {time_budget_sec} seconds\n"
            f"- Design experiments that complete within this budget\n"
            f"- Implement a time guard: stop gracefully at 80% of budget\n"
        )

    # --- Dataset guidance + setup script + HP reporting (docker/sandbox modes) ---
    extra_guidance = ""
    _net_policy = getattr(getattr(config, "docker", None), "network_policy", "setup_only")
    if config.experiment.mode in ("sandbox", "docker"):
        _net_policy = (
            config.experiment.docker.network_policy
            if config.experiment.mode == "docker"
            else "none"  # sandbox mode has no network
        )
        if _net_policy == "none":
            # Network disabled: inject strict offline-only guidance
            try:
                extra_guidance += _pm.block("network_disabled_guidance")
            except Exception:  # noqa: BLE001
                pass
        elif _net_policy == "full":
            try:
                extra_guidance += _pm.block("dataset_guidance")
                extra_guidance += _pm.block("network_full_guidance")
            except Exception:  # noqa: BLE001
                pass
        else:
            # setup_only or pip_only — existing behavior
            try:
                extra_guidance += _pm.block("dataset_guidance")
            except Exception:  # noqa: BLE001
                pass
            if config.experiment.mode == "docker":
                try:
                    extra_guidance += _pm.block("setup_script_guidance")
                except Exception:  # noqa: BLE001
                    pass
        try:
            extra_guidance += _pm.block("hp_reporting")
        except Exception:  # noqa: BLE001
            pass
        # I-06: Multi-seed enforcement for all experiments
        try:
            extra_guidance += _pm.block("multi_seed_enforcement")
        except Exception:  # noqa: BLE001
            pass

    # --- BA: Inject BenchmarkAgent plan from Stage 9 ---
    _bp_path = None
    for _s9_dir in sorted(run_dir.glob("stage-09*"), reverse=True):
        _candidate = _s9_dir / "benchmark_plan.json"
        if _candidate.exists():
            _bp_path = _candidate
            break
    if _bp_path is not None:
        try:
            import json as _json_bp
            _bp_data = _json_bp.loads(_bp_path.read_text(encoding="utf-8"))
            # Reconstruct the prompt block
            from researchclaw.agents.benchmark_agent.orchestrator import BenchmarkPlan
            _bp = BenchmarkPlan(
                selected_benchmarks=_bp_data.get("selected_benchmarks", []),
                selected_baselines=_bp_data.get("selected_baselines", []),
                data_loader_code=_bp_data.get("data_loader_code", ""),
                baseline_code=_bp_data.get("baseline_code", ""),
                experiment_notes=_bp_data.get("experiment_notes", ""),
            )
            _bp_block = _bp.to_prompt_block()
            if _bp_block:
                extra_guidance += (
                    "\n\n## BenchmarkAgent Selections (USE THESE)\n"
                    "The following datasets, baselines, and code snippets were "
                    "automatically selected and validated by the BenchmarkAgent. "
                    "You MUST use these selections in your experiment code.\n\n"
                    + _bp_block
                )
                logger.info(
                    "BA: Injected benchmark plan (%d benchmarks, %d baselines)",
                    len(_bp.selected_benchmarks), len(_bp.selected_baselines),
                )
        except Exception as _bp_exc:
            logger.debug("BA: Failed to load benchmark plan: %s", _bp_exc)

    # --- P2.2+P2.3: LLM training topic detection and guidance ---
    _llm_keywords = (
        "language model", "llm", "fine-tun", "lora", "qlora", "peft",
        "instruction tun", "rlhf", "dpo", "sft", "alignment",
        "transformer train", "causal lm", "chat model", "qwen", "llama",
        "mistral", "phi-", "gemma", "pretraining", "tokeniz",
    )
    topic_lower = config.research.topic.lower()
    is_llm_topic = any(kw in topic_lower for kw in _llm_keywords)

    # --- I-08: RL topic detection and step guidance ---
    _rl_keywords = (
        "reinforcement learning", "policy gradient", "ppo", "sac", "td3",
        "ddpg", "dqn", "a2c", "a3c", "mujoco", "locomotion", "continuous control",
        "reward shaping", "exploration", "multi-agent rl", "marl", "curriculum rl",
        "imitation learning", "inverse rl", "offline rl", "model-based rl",
        "actor-critic", "reinforce", "gym", "gymnasium",
    )
    is_rl_topic = any(kw in topic_lower for kw in _rl_keywords)
    if is_rl_topic:
        try:
            extra_guidance += _pm.block("rl_step_guidance")
        except Exception:  # noqa: BLE001
            pass

    # --- F-01: Framework API doc injection (auto-detected) ---
    try:
        from researchclaw.data import detect_frameworks, load_framework_docs
        _hypothesis_text = _read_prior_artifact(run_dir, "hypotheses.md") or ""
        _fw_ids = detect_frameworks(
            config.research.topic, _hypothesis_text, exp_plan or ""
        )
        if _fw_ids:
            _fw_docs = load_framework_docs(_fw_ids, max_chars=8000)
            if _fw_docs:
                extra_guidance += _fw_docs
                logger.info("F-01: Injected framework docs for: %s", _fw_ids)
    except Exception:  # noqa: BLE001
        logger.debug("F-01: Framework doc injection skipped", exc_info=True)

    if is_llm_topic and config.experiment.mode == "docker":
        try:
            extra_guidance += _pm.block("llm_training_guidance")
        except Exception:  # noqa: BLE001
            pass
        try:
            extra_guidance += _pm.block("llm_eval_guidance")
        except Exception:  # noqa: BLE001
            pass
        # P2.3: Warn if time budget is too short for LLM training
        if time_budget_sec < 3600:
            extra_guidance += (
                "\n## COMPUTE BUDGET WARNING\n"
                f"Current time_budget_sec={time_budget_sec} is likely TOO SHORT "
                f"for LLM fine-tuning. Typical LoRA training needs 1-4 hours. "
                f"Design a LIGHTWEIGHT experiment:\n"
                f"- Use a small dataset (<=5000 samples)\n"
                f"- Train for 1-3 epochs only\n"
                f"- Use small batch size (1-2) with gradient accumulation\n"
                f"- Use 4-bit quantization (QLoRA) to minimize memory\n"
                f"- Limit max_seq_length to 512-1024\n"
                f"- If possible, use a smaller model (<=7B parameters)\n"
            )

    # --- Domain-specific guidance injection for non-ML domains ---
    try:
        from researchclaw.domains.detector import detect_domain as _dd_s10, is_ml_domain as _is_ml_s10
        _dp = _dd_s10(topic=config.research.topic)
        if not _is_ml_s10(_dp):
            from researchclaw.domains.prompt_adapter import get_adapter as _ga
            _adapter = _ga(_dp)
            _blocks = _adapter.get_code_generation_blocks({})
            if _blocks.compute_budget:
                compute_budget = _blocks.compute_budget
            if _blocks.dataset_guidance:
                extra_guidance = _blocks.dataset_guidance + "\n" + extra_guidance
            if _blocks.code_generation_hints:
                extra_guidance += "\n" + _blocks.code_generation_hints
            if _blocks.output_format_guidance:
                extra_guidance += "\n" + _blocks.output_format_guidance
            logger.info("Injected domain-specific guidance for %s", _dp.domain_id)
    except Exception:  # noqa: BLE001
        logger.debug("Domain guidance injection skipped", exc_info=True)

    # BUG-R6-01: Add explicit implementation constraints to prevent LLM
    # from substituting unrelated DL models for lightweight algorithms.
    extra_guidance += (
        "\n\nIMPLEMENTATION CONSTRAINTS (MUST FOLLOW):\n"
        "- Implement EXACTLY the algorithm/method described in the topic.\n"
        "- Do NOT replace the stated method with a deep-learning proxy "
        "(e.g. ResNet, BERT, GPT, Gymnasium+SB3) unless the topic "
        "EXPLICITLY requires deep learning.\n"
        "- Prefer lightweight CPU-friendly libraries (numpy, scipy, "
        "sklearn, pandas) unless deep learning is inherent to the topic.\n"
        "- The experiment MUST be self-contained and runnable without GPU.\n"
    )

    # --- LLM4AD task-package structure guidance (all channels) -----------
    # Injected before the Beast/CodeAgent/Legacy dispatch so every channel
    # that consumes extra_guidance carries the LLM4AD structure requirement.
    _l4b = getattr(config.experiment, "llm4ad_boost", None)
    if (
        _l4b is not None
        and getattr(_l4b, "enabled", False)
        and config.experiment.mode in ("sandbox", "docker")
    ):
        try:
            extra_guidance += _pm.block("llm4ad_task_package_guidance")
        except Exception:  # noqa: BLE001
            logger.debug(
                "llm4ad_task_package_guidance block unavailable; skipping",
                exc_info=True,
            )

    # --- Code generation: Beast Mode → CodeAgent → Legacy single-shot ---
    _code_agent_active = False
    _beast_mode_used = False
    _code_max_tokens = 8192

    # ── Beast Mode: OpenCode external agent (optional) ─────────────────
    _oc_cfg = config.experiment.opencode
    if _oc_cfg.enabled:
        from researchclaw.pipeline.opencode_bridge import (
            OpenCodeBridge,
            OpenCodeResult,
            count_historical_failures,
            score_complexity,
        )

        _hist_failures = count_historical_failures(run_dir)
        _cplx = score_complexity(
            exp_plan=exp_plan,
            topic=config.research.topic,
            historical_failures=_hist_failures,
            threshold=_oc_cfg.complexity_threshold,
        )

        # Persist complexity analysis
        (stage_dir / "complexity_analysis.json").write_text(
            json.dumps(
                {
                    "score": _cplx.score,
                    "signals": _cplx.signals,
                    "recommendation": _cplx.recommendation,
                    "reason": _cplx.reason,
                    "threshold": _oc_cfg.complexity_threshold,
                    "historical_failures": _hist_failures,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if _cplx.recommendation == "beast_mode":
            _proceed = _oc_cfg.auto
            if not _proceed:
                # Non-auto mode: check for HITL adapter
                if adapters.hitl is not None:
                    try:
                        _proceed = adapters.hitl.confirm(
                            f"Beast Mode: complexity={_cplx.score:.2f} "
                            f"(threshold={_oc_cfg.complexity_threshold}). "
                            f"Route to OpenCode?"
                        )
                    except Exception:  # noqa: BLE001
                        logger.info(
                            "Beast mode: HITL adapter unavailable, skipping "
                            "(set opencode.auto=true for non-interactive runs)"
                        )
                else:
                    logger.info(
                        "Beast mode: no HITL adapter, skipping "
                        "(set opencode.auto=true for non-interactive runs)"
                    )

            if _proceed:
                _oc_model = _oc_cfg.model or config.llm.primary_model
                _bridge = OpenCodeBridge(
                    model=_oc_model,
                    llm_base_url=config.llm.base_url,
                    api_key=config.llm.api_key,
                    api_key_env=config.llm.api_key_env,
                    llm_provider=config.llm.provider,
                    timeout_sec=_oc_cfg.timeout_sec,
                    max_retries=_oc_cfg.max_retries,
                    workspace_cleanup=_oc_cfg.workspace_cleanup,
                    # Verify with the interpreter the sandbox will actually use,
                    # so "it ran for the agent" means it runs in Stage 12 too.
                    python_path=config.experiment.sandbox.python_path,
                )

                logger.info(
                    "Beast mode: ENGAGED (complexity=%.2f, model=%s)",
                    _cplx.score,
                    _oc_model,
                )

                _oc_result: OpenCodeResult = _bridge.generate(
                    stage_dir=stage_dir,
                    topic=config.research.topic,
                    exp_plan=exp_plan,
                    metric=metric,
                    pkg_hint=pkg_hint + "\n" + compute_budget,
                    extra_guidance=extra_guidance,
                    time_budget_sec=config.experiment.time_budget_sec,
                )

                # Persist beast mode log
                (stage_dir / "beast_mode_log.json").write_text(
                    json.dumps(
                        {
                            "success": _oc_result.success,
                            "recovered_from_failure": (
                                _oc_result.recovered_from_failure
                            ),
                            "elapsed_sec": _oc_result.elapsed_sec,
                            "files": list(_oc_result.files.keys()),
                            "error": _oc_result.error,
                            "complexity_score": _cplx.score,
                            "model": _oc_model,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                if _oc_result.success and _oc_result.files:
                    files = _oc_result.files
                    _beast_mode_used = True
                    _code_agent_active = True  # skip legacy path
                    if _oc_result.recovered_from_failure:
                        # BUG-03: every OpenCode attempt failed; these files
                        # were salvaged from a killed/timed-out run and the
                        # agent never validated them. They still go through the
                        # validation + repair loop below, but flag it loudly so
                        # a downstream failure is attributable.
                        logger.warning(
                            "Beast mode: SALVAGED — %d files recovered from a "
                            "FAILED OpenCode run (never agent-validated); "
                            "relying on stage-10 validation/repair",
                            len(files),
                        )
                    else:
                        logger.info(
                            "Beast mode: SUCCESS — %d files in %.1fs",
                            len(files),
                            _oc_result.elapsed_sec,
                        )
                else:
                    if _oc_cfg.require_success:
                        # OpenCode is the intended sole generator and it
                        # failed. A fallback artifact would be a different
                        # (usually weaker) channel — degrade loudly instead of
                        # silently blending quality. Mark beast mode as "used"
                        # so the Legacy fallback below is not reached.
                        logger.error(
                            "Beast mode: FAILED (%s) and "
                            "opencode.require_success=true — failing Stage 10 "
                            "instead of falling back",
                            _oc_result.error or "unknown error",
                        )
                        _beast_mode_used = True
                        return StageResult(
                            stage=Stage.CODE_GENERATION,
                            status=StageStatus.FAILED,
                            artifacts=(),
                            evidence_refs=(),
                        )
                    logger.warning(
                        "Beast mode: FAILED (%s) — falling back to CodeAgent",
                        _oc_result.error or "unknown error",
                    )
        else:
            logger.info(
                "Beast mode: complexity=%.2f (threshold=%.2f), not triggered",
                _cplx.score,
                _oc_cfg.complexity_threshold,
            )

    if not _beast_mode_used and config.experiment.code_agent.enabled and llm is not None:
        # ── F-02: Advanced Code Agent path ────────────────────────────────
        from researchclaw.pipeline.code_agent import CodeAgent as _CodeAgent

        _ca_cfg = config.experiment.code_agent
        # Ensure we have a proper config object
        if not hasattr(_ca_cfg, "enabled"):
            from researchclaw.pipeline.code_agent import (
                CodeAgentConfig as _CAConfig,
            )
            _ca_cfg = _CAConfig()

        # Sandbox factory (only for sandbox/docker modes)
        _sandbox_factory = None
        if config.experiment.mode in ("sandbox", "docker"):
            from researchclaw.experiment.factory import (
                create_sandbox as _csb,
            )
            _sandbox_factory = _csb

        if any(
            config.llm.primary_model.startswith(p)
            for p in ("gpt-5", "o3", "o4")
        ):
            _code_max_tokens = 16384

        # ── Domain detection + Code Search for non-ML domains ──────────
        _domain_profile = None
        _code_search_result = None
        try:
            from researchclaw.domains.detector import detect_domain as _dd
            from researchclaw.domains.detector import is_ml_domain as _is_ml
            _domain_profile = _dd(topic=config.research.topic)
            logger.info(
                "CodeAgent: domain=%s (%s)",
                _domain_profile.display_name,
                _domain_profile.domain_id,
            )
            # Run code search for non-ML domains (ML has enough built-in knowledge)
            if not _is_ml(_domain_profile):
                try:
                    from researchclaw.agents.code_searcher import CodeSearchAgent
                    _cs_agent = CodeSearchAgent(llm=llm)
                    _code_search_result = _cs_agent.search(
                        topic=config.research.topic,
                        domain=_domain_profile,
                    )
                    if _code_search_result and _code_search_result.patterns.has_content:
                        logger.info(
                            "Code search: %d patterns, %d repos found",
                            len(_code_search_result.patterns.api_patterns),
                            len(_code_search_result.repos_found),
                        )
                except Exception:  # noqa: BLE001
                    logger.debug("Code search unavailable", exc_info=True)
        except Exception:  # noqa: BLE001
            logger.debug("Domain detection unavailable", exc_info=True)

        _agent = _CodeAgent(
            llm=llm,
            prompts=_pm,
            config=_ca_cfg,
            stage_dir=stage_dir,
            sandbox_factory=_sandbox_factory,
            experiment_config=config.experiment,
            domain_profile=_domain_profile,
            code_search_result=_code_search_result,
        )
        _agent_result = _agent.generate(
            topic=config.research.topic,
            exp_plan=exp_plan,
            metric=metric,
            pkg_hint=pkg_hint + "\n" + compute_budget + "\n" + extra_guidance,
            max_tokens=_code_max_tokens,
        )
        files = _agent_result.files
        _code_agent_active = True

        # Write agent artifacts
        (stage_dir / "code_agent_log.json").write_text(
            json.dumps(
                {
                    "log": _agent_result.validation_log,
                    "llm_calls": _agent_result.total_llm_calls,
                    "sandbox_runs": _agent_result.total_sandbox_runs,
                    "best_score": _agent_result.best_score,
                    "tree_nodes_explored": _agent_result.tree_nodes_explored,
                    "review_rounds": _agent_result.review_rounds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if _agent_result.architecture_spec:
            (stage_dir / "architecture_spec.yaml").write_text(
                _agent_result.architecture_spec, encoding="utf-8",
            )
        logger.info(
            "CodeAgent: %d LLM calls, %d sandbox runs, score=%.2f",
            _agent_result.total_llm_calls,
            _agent_result.total_sandbox_runs,
            _agent_result.best_score,
        )
    elif not _beast_mode_used and llm is not None:
        # ── Legacy single-shot generation ─────────────────────────────────
        topic = config.research.topic
        _md = config.experiment.metric_direction
        _md_hint = (
            f"`{_md}` — use direction={'lower' if _md == 'minimize' else 'higher'} "
            f"in METRIC_DEF. You MUST NOT use the opposite direction."
        )
        _overlay = _get_evolution_overlay(run_dir, "code_generation")
        sp = _pm.for_stage(
            "code_generation",
            evolution_overlay=_overlay,
            topic=topic,
            metric=metric,
            pkg_hint=pkg_hint + "\n" + compute_budget + "\n" + extra_guidance,
            exp_plan=exp_plan,
            metric_direction_hint=_md_hint,
        )
        # R13-3: Use higher max_tokens for reasoning models (they consume tokens
        # for internal chain-of-thought). Retry once with even higher limit on empty.
        _code_max_tokens = sp.max_tokens or 8192
        if any(config.llm.primary_model.startswith(p) for p in ("gpt-5", "o3", "o4")):
            _code_max_tokens = max(_code_max_tokens, 16384)

        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=_code_max_tokens,
        )
        files = _extract_multi_file_blocks(resp.content)
        if not files and not resp.content.strip():
            # Empty response — retry with higher token limit
            logger.warning(
                "R13-3: Empty LLM response for code_generation (len=%d, "
                "finish_reason=%s, tokens=%d). Retrying with 32768 tokens.",
                len(resp.content),
                resp.finish_reason,
                resp.total_tokens,
            )
            resp = _chat_with_prompt(
                llm,
                sp.system,
                sp.user,
                json_mode=sp.json_mode,
                max_tokens=32768,
            )
            files = _extract_multi_file_blocks(resp.content)
        if files and "main.py" not in files:
            # _extract_multi_file_blocks no longer fabricates a main.py by
            # renaming an arbitrary file (that corrupted two files at once and
            # satisfied every `"main.py" in files` guard downstream). Instead,
            # re-ask with an explicit correction — a named-but-entry-point-less
            # response is a formatting failure, not a content failure.
            logger.warning(
                "Stage 10: LLM returned %d file(s) but no main.py (%s). "
                "Retrying with an explicit entry-point requirement.",
                len(files),
                ", ".join(sorted(files)),
            )
            resp = _chat_with_prompt(
                llm,
                sp.system,
                sp.user
                + (
                    "\n\nIMPORTANT — your previous reply was REJECTED: it "
                    f"defined {', '.join(sorted(files))} but no `main.py`.\n"
                    "Re-emit ALL files, and include `main.py` as the runnable "
                    "entry point with an `if __name__ == \"__main__\":` block. "
                    "Label every block as ```python filename:<path>. Do NOT "
                    "rename an existing file to main.py — write a real entry "
                    "point that imports and drives the others."
                ),
                json_mode=sp.json_mode,
                max_tokens=_code_max_tokens,
            )
            _retry_files = _extract_multi_file_blocks(resp.content)
            if _retry_files and "main.py" in _retry_files:
                files = _retry_files
            else:
                # Keep the original files rather than the entry-point-less
                # retry; the fallback experiment below supplies a main.py only
                # when `files` is empty, so log loudly that this will fail.
                logger.error(
                    "Stage 10: retry still produced no main.py — the generated "
                    "package has no entry point and will fail at execution"
                )
        if not files:
            logger.warning(
                "R13-2: _extract_multi_file_blocks returned empty. "
                "LLM response length=%d, first 300 chars: %s",
                len(resp.content),
                resp.content[:300],
            )

    # --- Fallback: generic numerical experiment ---
    if not files:
        files = {
            "main.py": (
                "import numpy as np\n"
                "\n"
                "np.random.seed(42)\n"
                "\n"
                "# Fallback experiment: parameter sweep on a synthetic objective\n"
                "# This runs when LLM code generation fails to produce valid code.\n"
                "dim = 10\n"
                "n_conditions = 3\n"
                "results = {}\n"
                "\n"
                "for cond_idx in range(n_conditions):\n"
                "    cond_name = f'condition_{cond_idx}'\n"
                "    scores = []\n"
                "    for seed in range(3):\n"
                "        rng = np.random.RandomState(seed + cond_idx * 100)\n"
                "        x = rng.randn(dim)\n"
                "        score = float(1.0 / (1.0 + np.sum(x ** 2)))\n"
                "        scores.append(score)\n"
                "    mean_score = float(np.mean(scores))\n"
                "    results[cond_name] = mean_score\n"
                f"    print(f'condition={{cond_name}} {metric}: {{mean_score:.6f}}')\n"
                "\n"
                "best = max(results, key=results.get)\n"
                f"print(f'{metric}: {{results[best]:.6f}}')\n"
            )
        }

    # --- Validate each file + auto-repair loop ---
    all_valid = True
    attempt = 0

    # --- LLM4AD task-package structure validation (all channels) ---
    # Structure defects (partial EVOLVE coverage, an algorithm calling a module-
    # level helper) used to flip `all_valid` without being fed back to the LLM,
    # so a syntactically-fine but un-evolvable package died with "FAILED after 0
    # repair attempt(s)". Route them through the same repair channel as syntax
    # defects, re-check after each pass, and only give up at `max_repair`.
    _l4b_v = getattr(config.experiment, "llm4ad_boost", None)
    if _l4b_v is not None and getattr(_l4b_v, "enabled", False):
        _l4b_problems = _check_llm4ad_structure(files)
        for _prob in _l4b_problems:
            logger.warning("Stage 10: %s", _prob)
            validation_log.append(_prob)
        if _l4b_problems and llm is not None:
            _l4b_round = 0
            while _l4b_problems and _l4b_round < max_repair:
                _l4b_round += 1
                logger.info(
                    "LLM4AD structure repair round %d/%d (%d issue(s))",
                    _l4b_round, max_repair, len(_l4b_problems),
                )
                files, _l4b_problems, _l4b_attempts = _repair_llm4ad_structure(
                    files, _l4b_problems, llm=llm, _pm=_pm, max_repair=max_repair,
                )
                attempt += _l4b_attempts
                for _prob in _l4b_problems:
                    logger.warning("Stage 10: %s", _prob)
                    validation_log.append(_prob)
        if _l4b_problems:
            all_valid = False

    for fname, code in list(files.items()):
        # Skip non-Python files (requirements.txt, setup.py, etc.)
        if not fname.endswith(".py"):
            continue
        validation = validate_code(code)
        repair_attempt = 0
        while not validation.ok and llm is not None and repair_attempt < max_repair:
            repair_attempt += 1
            attempt += 1
            # Only send errors to the LLM — warnings don't block validation
            # and confuse the LLM into over-correcting (e.g. removing runtime imports)
            errors_only = type(validation)(
                issues=[i for i in validation.issues if i.severity == "error"]
            )
            issues_text = format_issues_for_llm(errors_only)
            validation_log.append(
                f"File {fname} attempt {repair_attempt}: {validation.summary()}"
            )
            logger.info(
                "Code validation failed for %s (attempt %d/%d): %s",
                fname,
                repair_attempt,
                max_repair,
                validation.summary(),
            )
            all_files_ctx = _scoped_files_ctx(files, fname)
            rp = _pm.sub_prompt(
                "code_repair",
                fname=fname,
                issues_text=issues_text,
                all_files_ctx=all_files_ctx,
            )
            resp = _chat_with_prompt(llm, rp.system, rp.user, max_tokens=rp.max_tokens)
            _repaired = _extract_code_block(resp.content)
            if _repaired.strip():
                files[fname] = _repaired
            else:
                logger.warning("Repair attempt returned empty code, keeping original")
            validation = validate_code(files[fname])
        if not validation.ok:
            all_valid = False
            # BUG-14: Log remaining issues prominently
            logger.warning(
                "Code validation FAILED for %s after %d repair attempts: %s",
                fname, max_repair, validation.summary(),
            )

    # Improvement G: RL algorithm-environment compatibility check
    for fname, code in list(files.items()):
        if not fname.endswith(".py"):
            continue
        _rl_errors = _check_rl_compatibility(code)
        if _rl_errors:
            for _rl_err in _rl_errors:
                logger.error("Stage 10: %s (in %s)", _rl_err, fname)
                validation_log.append(f"RL_COMPAT: {fname}: {_rl_err}")
            all_valid = False

    # BUG-14: Block on critical validation failures (syntax/import errors)
    if not all_valid:
        _has_critical = False
        for fname, code in files.items():
            # Only Python files. `validate_code` parses its input as Python, so
            # a README.md reports a "syntax" error and blocks the whole stage —
            # the per-file loop above skips non-.py for exactly this reason, and
            # this loop must agree. Any check that flips `all_valid` (including
            # the LLM4AD structure checks) otherwise fails the stage on prose.
            if not fname.endswith(".py"):
                continue
            _v = validate_code(code)
            if not _v.ok:
                for issue in _v.issues:
                    if issue.severity == "error" and issue.category in (
                        "syntax", "import",
                    ):
                        _has_critical = True
        if _has_critical:
            logger.error(
                "Stage 10: CRITICAL validation issues remain after %d repair "
                "attempts. Blocking stage.", max_repair,
            )
            (stage_dir / "validation_report.md").write_text(
                "# Code Validation Report\n\n"
                f"**Status**: BLOCKED — critical issues remain after {max_repair} repairs\n\n"
                + "\n".join(f"- {e}" for e in validation_log),
                encoding="utf-8",
            )
            return StageResult(
                stage=Stage.CODE_GENERATION,
                status=StageStatus.FAILED,
                artifacts=("validation_report.md",),
                evidence_refs=(),
            )

    # --- BUG-184: Cross-import validation — warn if a .py file imports a
    # local module that doesn't exist in the files dict.  This catches the
    # case where Beast Mode/CodeAgent produced an intermediate file that
    # got lost during repair iterations.
    for _f, _m in _dangling_local_imports(files):
        logger.warning(
            "BUG-184: %s imports '%s' which is not in generated "
            "files — experiment may crash on import",
            _f, _m,
        )

    # --- Write experiment directory ---
    exp_dir = stage_dir / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for fname, code in files.items():
        _wp = exp_dir / fname
        _wp.parent.mkdir(parents=True, exist_ok=True)
        _wp.write_text(code, encoding="utf-8")

    # --- Write validation report ---
    if validation_log or not all_valid:
        report_lines = ["# Code Validation Report\n"]
        if all_valid:
            report_lines.append(f"**Status**: PASSED after {attempt} total repair(s)\n")
        else:
            report_lines.append(
                f"**Status**: FAILED after {attempt} total repair attempt(s)\n"
            )
        for entry in validation_log:
            report_lines.append(f"- {entry}")
        (stage_dir / "validation_report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

    # --- R10-Fix6: Code complexity and quality check ---
    from researchclaw.experiment.validator import (
        auto_fix_unbound_locals,
        check_code_complexity,
        deep_validate_files,
    )

    # --- BUG-3 fix: Programmatic auto-fix for UnboundLocalError patterns ---
    _total_ub_fixes = 0
    for fname, code in list(files.items()):
        if fname.endswith(".py"):
            fixed_code, n_fixes = auto_fix_unbound_locals(code)
            if n_fixes > 0:
                files[fname] = fixed_code
                (exp_dir / fname).write_text(fixed_code, encoding="utf-8")
                _total_ub_fixes += n_fixes
                logger.info(
                    "Stage 10: auto-fixed %d UnboundLocalError risk(s) in %s",
                    n_fixes, fname,
                )
    if _total_ub_fixes:
        logger.info(
            "Stage 10: auto-fixed %d total UnboundLocalError risks", _total_ub_fixes
        )

    complexity_warnings: list[str] = []
    for fname, code in files.items():
        if fname.endswith(".py"):
            cw = check_code_complexity(code)
            for w in cw:
                complexity_warnings.append(f"[{fname}] {w}")
                logger.warning("Stage 10 code quality: [%s] %s", fname, w)

    # --- P1.1+P1.2: Deep quality analysis (class quality, scoping, API) ---
    deep_warnings = deep_validate_files(files)
    for w in deep_warnings:
        logger.warning("Stage 10 deep quality: %s", w)
    complexity_warnings.extend(deep_warnings)

    # --- P1.2: If critical deep issues found, attempt one repair cycle ---
    critical_deep = [w for w in deep_warnings if any(
        kw in w for kw in ("UnboundLocalError", "unregistered", "does not exist",
                           "empty or trivial subclass", "does NOT override",
                           "Import-usage mismatch", "NameError",
                           "was removed", "ptp()",
                           "copy-paste", "identical method signatures",
                           "identical AST", "NOT a real ablation",
                           "shadows stdlib/pip")
    )]
    if critical_deep and llm is not None:
        logger.info(
            "Stage 10: %d critical code issues found — triggering repair cycle",
            len(critical_deep),
        )
        repair_issues = "\n".join(f"- {w}" for w in critical_deep)
        # A deep issue (identical AST, copy-paste ablation, shadows stdlib) can
        # only be fixed by reading the METHOD BODY, not a signature — so pass
        # the files those warnings name verbatim via ``full_files``, and
        # summarize the rest. Names carry the file as ``[<file>[:line]]``.
        _crit_files = sorted({
            m.group(1) for w in critical_deep
            for m in [re.match(r"\[([^:\]]+)", w)]
            if m and m.group(1) in files
        })
        # Cross-file repair (it may rewrite several classes/imports), so the
        # model must see the whole project — but summarized, not dumped in full.
        all_code_ctx = _summarize_files_ctx(files, full_files=tuple(_crit_files))
        repair_prompt = (
            f"CRITICAL CODE QUALITY ISSUES FOUND:\n{repair_issues}\n\n"
            f"Fix ALL these issues in the code below. Return the complete "
            f"corrected files using ```filename:xxx.py format.\n\n"
            f"RULES:\n"
            f"- nn.Linear/nn.Conv must be created in __init__(), not forward()\n"
            f"- Variables used after if/else must be defined before the branch\n"
            f"- Use scipy.special.erf, not np.erf\n"
            f"- Ablation/variant classes must have genuinely different logic\n"
            f"- Every class must have a real implementation, not just `pass`\n"
            f"- Ablation classes MUST override the parent method that implements "
            f"the component being ablated (e.g., if ablating attention, override "
            f"the attention method with a simpler alternative like mean pooling)\n"
            f"- IMPORT CONSISTENCY: if you write `from X import Y`, call `Y()` "
            f"directly — NOT `X.Y()`. Mixing styles causes NameError.\n"
            f"- NumPy 2.0: ndarray.ptp() was removed — use arr.max()-arr.min()\n"
            f"- NumPy 2.0: np.bool/np.int/np.float removed — use builtins\n"
            f"- Pretrained models (EfficientNet, ResNet, ViT) expect 224×224 input "
            f"— add `transforms.Resize(224)` when using CIFAR (32×32) or similar\n"
            f"- Copy-paste ablation: if two classes have identical bodies, REWRITE "
            f"the ablation to genuinely remove/reduce a component (e.g., zero out "
            f"attention weights, halve hidden dimensions, remove a loss term)\n"
            f"- KD: teacher must be frozen, add projection layers if teacher_dim != "
            f"student_dim, use temperature T=4 for soft targets\n"
            f"- FILENAME COLLISIONS: If a file like config.py shadows a pip/stdlib "
            f"package, rename it (e.g., config.py → experiment_config.py) and update "
            f"ALL imports referencing it\n\n"
            f"Current code:\n{all_code_ctx}\n"
        )
        try:
            repair_resp = _chat_with_prompt(
                llm,
                _pm.system("code_generation"),
                repair_prompt,
                max_tokens=_code_max_tokens,
            )
            repaired = _extract_multi_file_blocks(repair_resp.content)
            files, _known = _merge_repaired_files(
                files, repaired, label="deep repair"
            )
            if _known:
                for fname, code in _known.items():
                    _wp = exp_dir / fname
                    _wp.parent.mkdir(parents=True, exist_ok=True)
                    _wp.write_text(code, encoding="utf-8")
                # Re-check after repair
                deep_warnings_after = deep_validate_files(files)
                fixed = len(critical_deep) - len([
                    w for w in deep_warnings_after
                    if any(kw in w for kw in (
                        "UnboundLocalError", "unregistered", "does not exist",
                        "empty or trivial subclass", "does NOT override",
                        "Import-usage mismatch", "NameError",
                        "was removed", "ptp()",
                        "copy-paste", "identical method signatures",
                        "identical AST", "NOT a real ablation",
                        "shadows stdlib/pip",
                    ))
                ])
                logger.info(
                    "Stage 10: Deep repair fixed %d/%d critical issues",
                    fixed, len(critical_deep),
                )
                complexity_warnings.append(
                    f"[REPAIR] Deep repair fixed {fixed}/{len(critical_deep)} "
                    f"critical issues"
                )
        except Exception as exc:
            logger.debug("Deep repair failed: %s", exc)

    if complexity_warnings:
        health: dict[str, Any] = {}
        health["code_complexity_warnings"] = complexity_warnings
        (stage_dir / "code_complexity.json").write_text(
            json.dumps(health, indent=2), encoding="utf-8"
        )

    # --- P1.4: LLM Code Review (Stage 10.5) ---
    # Skip when CodeAgent is active — Phase 4 review already covers this.
    if llm is not None and not _code_agent_active:
        all_code_review = "\n\n".join(
            f"# --- {fname} ---\n{code}" for fname, code in files.items()
        )
        if len(all_code_review) > 12000:
            all_code_review = all_code_review[:12000] + "\n... [truncated]"
        review_prompt = (
            f"You are a senior researcher reviewing experiment code for a "
            f"research submission.\n\n"
            f"TOPIC: {config.research.topic}\n"
            f"EXPERIMENT PLAN:\n{exp_plan[:3000]}\n\n"
            f"CODE:\n```python\n{all_code_review}\n```\n\n"
            f"Review the code and return JSON with this EXACT structure:\n"
            f'{{"score": <1-10>, "issues": ['
            f'{{"severity": "critical|major|minor", '
            f'"description": "...", "fix": "..."}}], '
            f'"verdict": "pass|needs_fix"}}\n\n'
            f"Check specifically:\n"
            f"1. Does each algorithm/method have a DISTINCT implementation? "
            f"(Not just renamed copies)\n"
            f"2. Are ablation conditions genuinely different from the main method?\n"
            f"3. Are loss functions / training loops mathematically correct?\n"
            f"4. Will the code actually run without errors? Check variable scoping, "
            f"API usage, tensor shape compatibility.\n"
            f"5. Is the code complex enough for a research paper? (Not trivial)\n"
            f"6. Are experimental conditions fairly compared (same seeds, data)?\n"
            f"7. If using pretrained models (EfficientNet, ResNet, ViT), are input "
            f"images resized to the model's expected size (e.g., 224x224)? CIFAR "
            f"images are 32x32 and MUST be resized for pretrained models.\n"
            f"8. Are imports consistent? `from X import Y` must use `Y()`, not `X.Y()`.\n"
        )
        try:
            review_resp = llm.chat(
                [{"role": "user", "content": review_prompt}],
                system="You are a meticulous ML code reviewer. Be strict.",
                max_tokens=2048,
            )
            # Extract JSON from LLM response (may be wrapped in markdown fences)
            _review_text = review_resp.content if hasattr(review_resp, "content") else str(review_resp)
            # Strip markdown JSON fences if present
            _review_text = _review_text.strip()
            if _review_text.startswith("```"):
                _lines = _review_text.splitlines()
                _start = 1 if _lines[0].strip().startswith("```") else 0
                _end = len(_lines) - 1 if _lines[-1].strip() == "```" else len(_lines)
                _review_text = "\n".join(_lines[_start:_end])
            review_data = _safe_json_loads(_review_text, {})
            if isinstance(review_data, dict):
                review_score = review_data.get("score", 0)
                review_verdict = review_data.get("verdict", "unknown")
                review_issues = review_data.get("issues", [])

                # Write review report
                review_report = {
                    "score": review_score,
                    "verdict": review_verdict,
                    "issues": review_issues,
                    "timestamp": _utcnow_iso(),
                }
                (stage_dir / "code_review.json").write_text(
                    json.dumps(review_report, indent=2), encoding="utf-8"
                )

                # If critical issues found and score low, attempt fix
                critical_issues = [
                    i for i in review_issues
                    if isinstance(i, dict)
                    and i.get("severity") == "critical"
                ]
                if critical_issues and review_score <= 4:
                    logger.warning(
                        "Stage 10 code review: score=%d, %d critical issues — "
                        "attempting fix",
                        review_score, len(critical_issues),
                    )
                    fix_descriptions = "\n".join(
                        f"- [{i.get('severity', '?')}] {i.get('description', '?')}: "
                        f"{i.get('fix', 'no fix suggested')}"
                        for i in critical_issues
                    )
                    fix_prompt = (
                        f"Code review found {len(critical_issues)} CRITICAL issues "
                        f"(score: {review_score}/10):\n{fix_descriptions}\n\n"
                        f"Fix ALL critical issues. Return complete corrected files "
                        f"using ```filename:xxx.py format.\n\n"
                        f"Current code:\n"
                        + "\n\n".join(
                            f"```filename:{f}\n{c}\n```" for f, c in files.items()
                        )
                    )
                    try:
                        fix_resp = _chat_with_prompt(
                            llm,
                            _pm.system("code_generation"),
                            fix_prompt,
                            max_tokens=_code_max_tokens,
                        )
                        fixed_files = _extract_multi_file_blocks(fix_resp.content)
                        # Partial reply is normal — see deep-repair note above.
                        files, _fx = _merge_repaired_files(
                            files, fixed_files, label="review-fix"
                        )
                        if _fx:
                            for fname, code in _fx.items():
                                _wp = exp_dir / fname
                                _wp.parent.mkdir(parents=True, exist_ok=True)
                                _wp.write_text(code, encoding="utf-8")
                            logger.info(
                                "Stage 10: Code fixed after review "
                                "(was %d/10, %d critical issues)",
                                review_score, len(critical_issues),
                            )
                    except Exception as exc:
                        logger.debug("Review-fix failed: %s", exc)
        except Exception as exc:
            logger.debug("Code review failed: %s", exc)

    # --- FIX-3: Topic-experiment alignment check ---
    # BUG-171: Previous 8000-char truncation caused false-positive misalignment
    # for multi-file experiments (30-90K chars). LLM saw "[truncated]" and
    # concluded code was incomplete. Fix: build a structured summary that
    # includes file inventory + full main.py + per-file function/class headers.
    alignment_ok = True
    alignment_note = ""
    if llm is not None:
        # Build structured code summary for alignment check
        _file_inventory = []
        for _fn, _cd in files.items():
            _lines = _cd.count("\n") + 1
            _file_inventory.append(f"  {_fn}: {_lines} lines, {len(_cd)} chars")
        _inventory_block = "FILES GENERATED:\n" + "\n".join(_file_inventory)

        # BUG-179: Beast Mode may use a different entry point (e.g.
        # run_experiment.py).  Detect the actual entry point by scanning
        # for ``if __name__ == "__main__"`` in all files, preferring main.py.
        _entry_file = "main.py"
        if "main.py" not in files or not files.get("main.py", "").strip():
            for _fn, _cd in files.items():
                if 'if __name__' in _cd and '__main__' in _cd:
                    _entry_file = _fn
                    break
        elif files.get("main.py", ""):
            # main.py exists but may be a stub — if another file has the
            # real orchestration (more lines + __main__ guard), prefer it
            _main_lines = files["main.py"].count("\n")
            for _fn, _cd in files.items():
                if _fn == "main.py":
                    continue
                if ('if __name__' in _cd and '__main__' in _cd
                        and _cd.count("\n") > _main_lines * 1.5):
                    _entry_file = _fn
                    break

        _main_code = files.get(_entry_file, files.get("main.py", ""))
        _main_block = f"# --- {_entry_file} (FULL — entry point) ---\n{_main_code}"
        # Cap main.py at 12000 chars to stay within token budget
        if len(_main_block) > 12000:
            _main_block = _main_block[:12000] + "\n... [main.py truncated at 12000 chars]"

        # For other files, include imports + function/class signatures
        _other_summaries = []
        for _fn, _cd in files.items():
            if _fn == _entry_file:
                continue
            _sig_lines = []
            for _line in _cd.split("\n"):
                _stripped = _line.strip()
                if (_stripped.startswith("def ") or _stripped.startswith("class ")
                        or _stripped.startswith("async def ")
                        # BUG-209: Include import lines — they reveal which
                        # techniques/libraries are used (e.g. CosineAnnealingLR)
                        or _stripped.startswith("import ")
                        or _stripped.startswith("from ")):
                    _sig_lines.append(_line)
            if _sig_lines:
                _other_summaries.append(
                    f"# --- {_fn} (imports + signatures) ---\n"
                    + "\n".join(_sig_lines)
                )
            else:
                # Small file — include first 800 chars
                _preview = _cd[:800]
                if len(_cd) > 800:
                    _preview += f"\n... [{len(_cd) - 800} more chars]"
                _other_summaries.append(f"# --- {_fn} (preview) ---\n{_preview}")
        _other_block = "\n\n".join(_other_summaries)
        # Cap other summaries
        if len(_other_block) > 6000:
            _other_block = _other_block[:6000] + "\n... [other files truncated]"

        all_code_for_check = (
            f"{_inventory_block}\n\n{_main_block}\n\n{_other_block}"
        )
        align_prompt = (
            f"Research topic: {config.research.topic}\n\n"
            f"Experiment code:\n```python\n{all_code_for_check}\n```\n\n"
            "TASK: Evaluate whether this experiment code actually tests the "
            "stated research topic. Answer with JSON:\n"
            '{"aligned": true/false, "reason": "...", "suggestions": "..."}\n\n'
            "IMPORTANT: The code spans MULTIPLE files. The file inventory above "
            "shows ALL generated files. Only main.py is shown in full; other "
            "files show function/class signatures. Do NOT mark as misaligned "
            "just because helper files are summarized — they contain full "
            "implementations.\n\n"
            "Check specifically:\n"
            "- Does main.py orchestrate an experiment matching the topic?\n"
            "- Do the helper file signatures indicate relevant models/methods?\n"
            "- If the topic mentions a specific technique, is there evidence of "
            "its implementation (function names, class names, imports)?\n"
            "- Are the experimental conditions meaningfully different from each other?\n"
        )
        try:
            align_resp = llm.chat(
                [{"role": "user", "content": align_prompt}],
                system="You are a scientific code reviewer checking topic-experiment alignment.",
                max_tokens=1024,
            )
            align_data = _safe_json_loads(align_resp.content, {})
            if isinstance(align_data, dict) and not align_data.get("aligned", True):
                alignment_ok = False
                alignment_note = align_data.get("reason", "Misaligned")
                suggestions = align_data.get("suggestions", "")
                logger.warning(
                    "Stage 10: Topic-experiment MISALIGNMENT detected: %s",
                    alignment_note,
                )
                # BUG-R6-01: Allow up to 2 regeneration attempts with re-check.
                _max_regen = 2
                for _regen_attempt in range(1, _max_regen + 1):
                    logger.info(
                        "Stage 10: Alignment regen attempt %d/%d",
                        _regen_attempt, _max_regen,
                    )
                    regen_prompt = (
                        f"The experiment code you previously generated does NOT align "
                        f"with the research topic.\n\n"
                        f"TOPIC: {config.research.topic}\n"
                        f"MISALIGNMENT: {alignment_note}\n"
                        f"SUGGESTIONS: {suggestions}\n\n"
                        f"REGENERATE the experiment code to DIRECTLY test the stated "
                        f"topic. The code MUST implement the core technique described "
                        f"in the topic, not a generic proxy.\n\n"
                        f"CRITICAL CONSTRAINTS:\n"
                        f"- You MUST implement the EXACT algorithm/method from the topic.\n"
                        f"- Do NOT substitute a deep-learning proxy (ResNet, BERT, etc.) "
                        f"when the topic describes a tabular, bandit, or game-theoretic method.\n"
                        f"- Use ONLY lightweight CPU-friendly libraries (numpy, scipy, "
                        f"sklearn) unless the topic EXPLICITLY requires deep learning.\n"
                        f"- The experiment must be self-contained and runnable without GPU.\n\n"
                        f"{pkg_hint}\n{compute_budget}\n"
                        f"PLAN:\n{exp_plan}\n\n"
                        f"Return multiple files using ```filename:xxx.py format."
                    )
                    regen_resp = _chat_with_prompt(
                        llm,
                        system=_pm.system("code_generation"),
                        user=regen_prompt,
                        max_tokens=_code_max_tokens,
                    )
                    regen_files = _extract_multi_file_blocks(regen_resp.content)
                    if not regen_files or "main.py" not in regen_files:
                        logger.warning(
                            "Stage 10: Regen attempt %d produced no main.py",
                            _regen_attempt,
                        )
                        continue
                    files = regen_files
                    for fname, code in files.items():
                        _wp = exp_dir / fname
                        _wp.parent.mkdir(parents=True, exist_ok=True)
                        _wp.write_text(code, encoding="utf-8")
                    # Re-check alignment on regenerated code (BUG-171 fix)
                    _rc_inv = []
                    for _fn, _cd in files.items():
                        _rc_inv.append(f"  {_fn}: {_cd.count(chr(10))+1} lines")
                    _rc_main = files.get("main.py", "")
                    if len(_rc_main) > 12000:
                        _rc_main = _rc_main[:12000] + "\n... [truncated]"
                    _rc_sigs = []
                    for _fn, _cd in files.items():
                        if _fn == "main.py":
                            continue
                        # BUG-209: Include imports alongside signatures
                        _slines = [l for l in _cd.split("\n")
                                   if l.strip().startswith((
                                       "def ", "class ", "async def ",
                                       "import ", "from ",
                                   ))]
                        if _slines:
                            _rc_sigs.append(f"# {_fn} imports+signatures:\n" + "\n".join(_slines))
                    recheck_code = (
                        "FILES:\n" + "\n".join(_rc_inv) + "\n\n"
                        f"# main.py (FULL):\n{_rc_main}\n\n"
                        + "\n".join(_rc_sigs)
                    )
                    recheck_resp = llm.chat(
                        [{"role": "user", "content": (
                            f"Research topic: {config.research.topic}\n\n"
                            f"Experiment code:\n```python\n{recheck_code}\n```\n\n"
                            "TASK: Evaluate whether this experiment code actually tests "
                            "the stated research topic. Only main.py is shown in full; "
                            "other files show signatures only. Answer with JSON:\n"
                            '{"aligned": true/false, "reason": "...", "suggestions": "..."}\n'
                        )}],
                        system="You are a scientific code reviewer checking topic-experiment alignment.",
                        max_tokens=1024,
                    )
                    recheck_data = _safe_json_loads(recheck_resp.content, {})
                    if isinstance(recheck_data, dict) and recheck_data.get("aligned", False):
                        alignment_ok = True
                        alignment_note = f"Regenerated after alignment check (attempt {_regen_attempt})"
                        logger.info(
                            "Stage 10: Code aligned after regen attempt %d",
                            _regen_attempt,
                        )
                        break
                    else:
                        alignment_note = recheck_data.get("reason", alignment_note)
                        suggestions = recheck_data.get("suggestions", suggestions)
                        logger.warning(
                            "Stage 10: Regen attempt %d still misaligned: %s",
                            _regen_attempt, alignment_note,
                        )
        except Exception as exc:
            logger.debug("Alignment check failed: %s", exc)

    # --- FIX-7: Ablation distinctness check ---
    main_code = files.get("main.py", "")
    if llm is not None and main_code and "condition" in main_code.lower():
        try:
            ablation_prompt = (
                f"Examine this experiment code:\n```python\n{main_code[:6000]}\n```\n\n"
                "Check if any experimental conditions (methods/ablations) have "
                "IDENTICAL configurations (same hyperparameters, same code paths). "
                "Answer JSON: "
                '{"has_duplicates": true/false, "details": "which conditions are identical"}'
            )
            abl_resp = llm.chat(
                [{"role": "user", "content": ablation_prompt}],
                system="You are a code reviewer checking experimental conditions.",
                max_tokens=512,
            )
            abl_data = _safe_json_loads(abl_resp.content, {})
            if isinstance(abl_data, dict) and abl_data.get("has_duplicates"):
                logger.warning(
                    "Stage 10: Duplicate ablation conditions detected: %s",
                    abl_data.get("details", ""),
                )
                (stage_dir / "ablation_warning.json").write_text(
                    json.dumps(abl_data, indent=2), encoding="utf-8"
                )
                # --- Attempt ablation repair ---
                # Rewrites condition classes project-wide, so keep the whole
                # project in view — summarized, not full-dumped.
                all_code_ctx = _summarize_files_ctx(files)
                dup_details = abl_data.get("details", "unknown")
                abl_repair_prompt = (
                    f"ABLATION REPAIR REQUIRED — duplicate conditions detected:\n"
                    f"{dup_details}\n\n"
                    f"Rewrite the ablation/variant conditions so each one is "
                    f"GENUINELY DIFFERENT. Concrete strategies:\n"
                    f"- 'no_<component>': REMOVE the component entirely "
                    f"(e.g., replace attention with mean pooling, remove a loss term)\n"
                    f"- 'reduced_capacity': HALVE hidden dimensions or layers\n"
                    f"- Conditions should differ in CODE, not merely in a label.\n"
                    f"Do NOT add a startup assertion that raises when two conditions "
                    f"produce the same numeric output: stochastic optimizers can "
                    f"coincidentally agree on one input, and a raise at runtime then "
                    f"kills the whole experiment at Stage 12. Instead, if you want a "
                    f"self-check, print a diagnostic line (e.g. "
                    f"\"ABLATION_CHECK: <cond1> vs <cond2> outputs_differ=True/False\") "
                    f"without ever raising.\n\n"
                    f"Return ALL files using ```filename:xxx.py format.\n\n"
                    f"Current code:\n{all_code_ctx}\n"
                )
                try:
                    abl_repair_resp = _chat_with_prompt(
                        llm,
                        _pm.system("code_generation"),
                        abl_repair_prompt,
                        max_tokens=_code_max_tokens,
                    )
                    repaired_files = _extract_multi_file_blocks(
                        abl_repair_resp.content
                    )
                    if repaired_files and "main.py" in repaired_files:
                        files = repaired_files
                        for fname, code in files.items():
                            _wp = exp_dir / fname
                            _wp.parent.mkdir(parents=True, exist_ok=True)
                            _wp.write_text(code, encoding="utf-8")
                        logger.info(
                            "Stage 10: Ablation repair applied — "
                            "rewrote duplicate conditions"
                        )
                except Exception as exc:
                    logger.debug("Ablation repair failed: %s", exc)
        except Exception as exc:
            logger.debug("Ablation validation skipped: %s", exc)

    # --- Write spec ---
    file_list = ", ".join(f"`{f}`" for f in sorted(files.keys()))
    main_validation = validate_code(files.get("main.py", ""))
    _align_status = "ALIGNED" if alignment_ok else f"MISALIGNED: {alignment_note}"
    spec = f"""# Experiment Specification

## Topic
{config.research.topic}

## Project Structure
Multi-file experiment project with {len(files)} file(s): {file_list}

## Entry Point
`main.py` \u2014 executed directly via sandbox

## Outputs
- `main.py` emits metric lines in `name: value` format
- Primary metric key: `{metric}`

## Topic-Experiment Alignment
{_align_status}

## Constraints
- Time budget per run: {config.experiment.time_budget_sec}s
- Max iterations: {config.experiment.max_iterations}
- Self-contained execution (no external data, no network)
- Validated: {main_validation.summary()}

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "experiment_spec.md").write_text(spec, encoding="utf-8")

    artifacts = ["experiment/", "experiment_spec.md"]
    if (stage_dir / "validation_report.md").exists():
        artifacts.append("validation_report.md")

    # BUG-R6-01: Fail stage if alignment check detected persistent mismatch
    # after all regen attempts, instead of silently proceeding.
    if not alignment_ok:
        logger.error(
            "Stage 10: Persistent topic-experiment misalignment after all "
            "regen attempts. Failing stage. Reason: %s",
            alignment_note,
        )
        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.FAILED,
            artifacts=tuple(artifacts),
            evidence_refs=tuple(f"stage-10/{a}" for a in artifacts),
            error=f"Topic-experiment misalignment: {alignment_note}",
        )

    # --- Real smoke run: the last thing that can go wrong is a runtime crash
    # the static gates cannot see (e.g. a self-check assert that only raises at
    # run time). Stage 12 has no retry, so run main.py with the exact sandbox
    # interpreter and feed any traceback back to the LLM — a loop, not a FAIL.
    _smoke_failures = 0
    _smoke_max = 3
    while True:
        _smoke = _try_smoke_run(exp_dir, config)
        if _smoke is None:
            break
        _rc, _tail = _smoke
        _smoke_failures += 1
        logger.error(
            "Stage 10: generated experiment failed to run (attempt %d/%d, exit=%s) — "
            "feeding traceback back to fix.\n%s",
            _smoke_failures, _smoke_max, _rc, _tail,
        )
        if llm is None or _smoke_failures >= _smoke_max:
            break

        # Ask the LLM to fix the files the runtime crash points at.
        _ctx = "\n\n".join(
            f"```filename:{f}\n{c}\n```" for f, c in files.items()
        )
        _fix_prompt = (
            f"The generated experiment crashed on the DEFAULT entry point "
            f"(what Stage 12 runs). Fix the runtime error.\n\n"
            f"RUNTIME ERROR (exit {_rc}):\n```\n{_tail}\n```\n\n"
            f"Current code:\n{_ctx}\n\n"
            "Return the FIXED files. Every import the code references MUST resolve "
            "(the default run loads ALL algorithms; a missing module or a stray "
            "entry in the algorithm list is a runtime failure). Keep the result "
            "runnable in a few seconds. Do NOT remove, rename or reselect any "
            "algorithm or ablation condition that EXPERIMENT_PLAN.yaml lists — your "
            "task is to make the existing design run, not to shrink it."
        )
        try:
            _fix_resp = _chat_with_prompt(
                llm, _pm.system("code_generation"), _fix_prompt,
                max_tokens=_code_max_tokens,
            )
            _fixed = _extract_multi_file_blocks(_fix_resp.content)
            files, _applied = _merge_repaired_files(files, _fixed, label="smoke fix")
            for _fn, _code in _applied.items():
                _wp = exp_dir / _fn
                _wp.parent.mkdir(parents=True, exist_ok=True)
                _wp.write_text(_code, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 10: smoke-fix repair failed: %s", exc)
            break

    if _smoke is not None:
        _rc, _tail = _smoke
        logger.error(
            "Stage 10: generated experiment still does not run (exit=%s) after "
            "%d fix attempt(s) — refusing to ship code that cannot run.\n%s",
            _rc, _smoke_failures, _tail,
        )
        (stage_dir / "validation_report.md").write_text(
            "# Code Validation Report\n\n"
            "**Status**: BLOCKED — experiment failed to run\n\n"
            f"exit code: {_rc}\n\n```\n{_tail}\n```",
            encoding="utf-8",
        )
        if "validation_report.md" not in artifacts:
            artifacts.append("validation_report.md")
        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.FAILED,
            artifacts=tuple(artifacts),
            evidence_refs=tuple(f"stage-10/{a}" for a in artifacts),
            error=f"Generated experiment did not run (exit={_rc}): {_tail[:300]}",
        )

    # --- LLM4AD evolution scope: classify the final algorithm tree ---
    # The plan's proposed/baseline/ablation split is authoritative only at S9,
    # but the stage-10 generator names algorithm directories freely, so the two
    # drift apart. Classify here (both the plan and the final tree are in scope,
    # after all fix/regeneration loops have settled) and freeze the result to
    # algorithms_classification.json, which stage-13 reads to apply
    # evolution.evolve_scope. Advisory: on failure the file simply is not
    # written and evolution falls back to evolving everything.
    _maybe_classify_algorithms(exp_dir, exp_plan, config, llm)
    if (exp_dir / "algorithms_classification.json").is_file():
        artifacts.append("experiment/algorithms_classification.json")

    return StageResult(
        stage=Stage.CODE_GENERATION,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-10/{a}" for a in artifacts),
    )

