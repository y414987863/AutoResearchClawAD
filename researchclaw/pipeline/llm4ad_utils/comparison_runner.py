"""Score one algorithm file across all experiment instances (subprocess entry).

The stage-13 promotion step decides whether an evolved algorithm is actually
an improvement.  To decide that it needs a protocol-identical number for both
the clean baseline and the evolved candidate.  Reading ``results.json`` is not
generic — the generated experiment owns its own schema — so instead we load the
experiment's own ``evaluator.py`` and call ``evaluate_instance`` for every
instance, exactly as ``main.py`` and ``run_single.py`` do.  The result is the
same primary metric under the same protocol as the paper pipeline, for any
generated project.

The contract used here is exactly the one Stage 10 enforces via
``_check_llm4ad_structure`` — ``evaluator.PRIMARY_METRIC`` and
``evaluator.evaluate_instance``, plus the optional ``load_instance`` hook — and
nothing else.  Reaching for any other named helper hard-wires whichever
experiment happened to define it: an earlier version called
``evaluator.list_instance_paths``, which only one topic's evaluator had, so
scoring raised ``AttributeError`` and silently disabled promotion everywhere
else.  ``run_single.py`` in the task package resolves instances the same way.

This module is meant to run as a subprocess (``python comparison_runner.py
<experiment_dir> <algo_name> <algo_file>``): the experiment's ``evaluator.py``
imports siblings like ``experiment_config`` and ``objectives`` by bare module
name, so loading it in the parent process would collide with shared module names
across experiments.  A subprocess keeps ``sys.path``/``sys.modules`` clean.

Instance paths arrive as a JSON list on stdin so that enumeration has a single
implementation (``llm4ad_task_packages._discover_instances``) shared with the
task packager.  With no stdin the same rule is applied locally, which keeps the
script runnable by hand for debugging.

Output is per-instance rather than a mean: the caller intersects the baseline's
and the candidate's instance sets before averaging, because a mean taken over
different instances is not a comparison.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path


def _discover_instances_locally(exp_dir: Path) -> list[Path]:
    """Fallback enumeration mirroring ``_discover_instances``.

    Every file directly under ``data/``, whatever its extension: instances are
    TSPLIB ``.tsp``, ``.mps``, ``.npz`` or ``.csv`` as often as they are JSON,
    and narrowing to ``*.json`` scores zero instances for those experiments.
    """
    data_dir = exp_dir / "data"
    if not data_dir.is_dir():
        return []
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )


def _make_instance_reader(evaluator):
    """Return ``read(path) -> instance``, honouring the optional hook.

    Parsing an instance file is problem-specific, so the experiment owns it via
    ``evaluator.load_instance``.  JSON is the default only because it is the
    common case, not because it is required — same rule as ``run_single.py``.
    """
    hook = getattr(evaluator, "load_instance", None)
    if callable(hook):
        return hook

    def _read(path):
        p = Path(path)
        if p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        raise RuntimeError(
            f"{p.name} is not JSON and evaluator.py defines no "
            "load_instance(path); add it so the runner can read this format"
        )

    return _read


def _load_optimize(algo_file: Path, algo_name: str):
    spec = importlib.util.spec_from_file_location(f"_cmp_{algo_name}", str(algo_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load algorithm from {algo_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "optimize", None)
    if fn is None:
        raise RuntimeError(f"{algo_file.name} has no optimize(instance, seed) function")
    return fn


def _emit(payload: dict) -> int:
    print(json.dumps(payload))
    return 0


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: python comparison_runner.py <experiment_dir> <algo_name> <algo_file>",
              file=sys.stderr)
        return 2

    experiment_dir = Path(sys.argv[1]).resolve()
    algo_name = sys.argv[2]
    # Resolved before the chdir below, so a relative path still points at the
    # file the caller meant.
    algo_file = Path(sys.argv[3]).resolve()

    # Instance paths from stdin keep enumeration single-sourced; absent stdin
    # (hand invocation) falls back to the identical rule.
    instance_paths: list[Path] = []
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                return _emit({"error": f"unparsable instance list on stdin: {exc}"})
            if not isinstance(parsed, list):
                return _emit({"error": "instance list on stdin must be a JSON array"})
            instance_paths = [Path(p).resolve() for p in parsed]
    if not instance_paths:
        instance_paths = [p.resolve() for p in _discover_instances_locally(experiment_dir)]

    if not instance_paths:
        return _emit({"error": "no instance files under data/", "values": {},
                      "n_instances": 0})

    # The experiment's evaluator imports its siblings (experiment_config,
    # objectives, utils) by bare name, so the package root must be on sys.path.
    sys.path.insert(0, str(experiment_dir))
    # main.py runs from the experiment root, so an evaluator may legitimately
    # open a path relative to it.  Match that working directory.
    try:
        os.chdir(experiment_dir)
    except OSError as exc:
        return _emit({"error": f"cannot enter experiment dir: {exc}"})

    try:
        import evaluator  # noqa: E402  (imported after sys.path is set)
    except Exception as exc:  # noqa: BLE001 — generated code, any failure is data
        return _emit({"error": f"cannot import evaluator.py: {exc!r}"})

    try:
        pm = evaluator.PRIMARY_METRIC
        evaluate_instance = evaluator.evaluate_instance
    except AttributeError as exc:
        return _emit({"error": f"evaluator.py does not satisfy the Stage-10 "
                               f"contract: {exc}"})

    try:
        solve = _load_optimize(algo_file, algo_name)
    except Exception as exc:  # noqa: BLE001
        return _emit({"error": f"cannot load algorithm: {exc!r}", "primary_metric": pm})

    read_instance = _make_instance_reader(evaluator)

    values: dict[str, float] = {}
    failures: dict[str, str] = {}
    for path in instance_paths:
        # Keyed by name, not full path: the baseline and the evolved candidate
        # are scored from different directories, and the caller intersects these
        # keys to compare like with like.
        key = path.name
        # One bad instance must not void the whole algorithm — an evolved
        # candidate that crashes on a single instance is still comparable on
        # the rest, and which instances it lost is exactly what we report.
        try:
            instance = read_instance(str(path))
            metrics = evaluate_instance(instance, solve)
        except Exception as exc:  # noqa: BLE001 — generated code, any failure is data
            failures[key] = repr(exc)
            continue
        if not isinstance(metrics, dict):
            failures[key] = f"evaluate_instance returned {type(metrics).__name__}, not dict"
            continue
        # Accept either spelling of the primary metric: evaluate_instance()
        # may aggregate over seeds and emit ``<PRIMARY_METRIC>_mean`` rather
        # than the bare name. The task-package evaluator (_write_evaluator)
        # already tolerates both; this is the path promotion reads, so it must
        # agree — otherwise a genuinely improved candidate is silently scored
        # as a failure and never promoted.
        try:
            val = float(metrics[pm])
        except (KeyError, TypeError, ValueError):
            try:
                val = float(metrics[pm + "_mean"])
            except (KeyError, TypeError, ValueError) as exc:
                failures[key] = f"primary metric {pm!r} unusable: {exc!r}"
                continue
        # A non-finite score is not a win.  Under `minimize`, -inf compares
        # better than every real baseline, so a degenerate candidate would be
        # promoted on the strength of a numerical blow-up.
        if not math.isfinite(val):
            failures[key] = f"primary metric {pm!r} is not finite ({val})"
            continue
        values[key] = val

    return _emit({
        "primary_metric": pm,
        "values": values,
        "failures": failures,
        "n_instances": len(instance_paths),
    })


if __name__ == "__main__":
    raise SystemExit(main())
