"""Pre-evolution fitness sanity gate (Stage 13).

LLM4AD is only useful when the primary metric is a well-posed scalar that the
evolved algorithm can actually move. Three degeneracies appear in products when
that is not true, and each one wastes a run before it is noticed:

- **constant** — the metric does not depend on the algorithm at all.  The ml23
  runs reported baseline == evolved == 2005.6570... for *every* algorithm
  because ``valid_prediction_time`` was ``1 / (workload * d)``: a model-invariant
  constant.  Evolution ran, "succeeded" 7/7, and improved nothing.
- **wall-clock** — the metric is ``time.perf_counter()`` around ``solve``.  It is
  nondeterministic and trivially gameable: an algorithm that returns a result
  without doing the work scores fastest, so the ml03 runs "improved" by 87–93%
  and promoted code that solved nothing.
- **non-finite everywhere** — the TSP runs scored every algorithm ``inf`` because
  the runner fed ``optimize`` a raw instance while the evaluator built the cost
  matrix internally.  Every baseline and every candidate scored ``inf``, so
  promotion had nothing to compare and reported 0 promoted for the wrong reason.

This module runs *before* any task package is built.  It scores the clean
baselines through the exact same subprocess path the promotion step uses
(``comparison_runner``), so it sees what promotion will see, and rejects the
metric when any degeneracy is detected.  The result is a hard failure at
evolution start with a precise diagnosis instead of an empty-looking run.

It is intentionally *not* a quality check: it does not judge whether the metric
matches the research goal (that is the experiment designer's call) or whether an
evolved candidate is good (that is ``_promote_llm4ad_to_experiment_final``).  It
only asks one question — *is this a metric that evolution can optimize at all?*
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_RUNNER = Path(__file__).resolve().parent / "comparison_runner.py"

# A metric is considered "constant" when every baseline score lies within this
# relative window of one another.  ``valid_prediction_time`` for ml23 was
# bit-identical; a (necessarily small) tolerance avoids flagging a genuinely
# near-flat but real signal, while still catching the model-invariant case.
_CONSTANT_REL_TOL = 1e-6


def _parse_result(stdout: str) -> dict[str, Any] | None:
    """Pull the ``@@LLM4AD_RESULT@@`` payload out of a noisy subprocess stdout."""
    marker = "@@LLM4AD_RESULT@@"
    for line in (stdout or "").splitlines():
        idx = line.find(marker)
        if idx == -1:
            continue
        payload = line[idx + len(marker):].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


def _score_one(exp_dir: Path, algo_name: str, algo_file: Path,
               instance_argv: str, timeout_sec: int) -> dict[str, Any]:
    """Score one algorithm via the promotion subprocess; never raises."""
    try:
        proc = subprocess.run(
            [sys.executable, str(_RUNNER), str(exp_dir), algo_name, str(algo_file)],
            input=instance_argv,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_sec,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"algo": algo_name, "values": {}, "detail": f"scoring failed to run: {exc}",
                "num_instances": 0}
    if proc.returncode != 0:
        return {"algo": algo_name, "values": {}, "detail": f"scoring exited {proc.returncode}",
                "num_instances": 0}
    payload = _parse_result(proc.stdout)
    if not isinstance(payload, dict):
        return {"algo": algo_name, "values": {}, "detail": "unparsable stdout", "num_instances": 0}
    values = payload.get("values")
    if not isinstance(values, dict) or not values:
        failed = payload.get("failures") or {}
        first = next(iter(failed.values()), "")
        return {
            "algo": algo_name, "values": {},
            "detail": (f"no finite primary metric on any instance; e.g. {first}"
                       if isinstance(first, str) and first
                       else "no finite primary metric on any instance"),
            "num_instances": int(payload.get("n_instances") or 0),
        }
    try:
        values = {str(k): float(v) for k, v in values.items()}
    except (TypeError, ValueError):
        return {"algo": algo_name, "values": {}, "detail": "non-numeric values", "num_instances": 0}
    return {"algo": algo_name, "values": values, "detail": "",
            "num_instances": int(payload.get("n_instances") or 0)}


def _non_finite(entry: dict[str, Any]) -> bool:
    """An algorithm that exposes no finite score on any instance."""
    return not entry["values"]


def _constant(entries: list[dict[str, Any]]) -> bool:
    """True when every algorithm scores the SAME value on each instance.

    Constantness is a property *per instance*: a metric that changes with the
    instance but not with the algorithm (ml23's ``valid_prediction_time``, a
    model-invariant ``1/(total_docs*d)``) is just as unevolvable as one that is
    constant everywhere, and comparing a global max/min over all instances would
    mask it.  So group the finite scores by instance name and require that, for
    EVERY instance, the scores across all algorithms fall in one relative band.
    """
    if not entries:
        return False
    # ``instance -> [algo1, algo2, ...]``. Only instances present on every
    # algorithm count (a missing one is already reported as non-finite).
    per_instance: dict[str, list[float]] = {}
    for e in entries:
        for inst, val in e["values"].items():
            per_instance.setdefault(inst, []).append(val)
    if not per_instance:
        return False
    for vals in per_instance.values():
        if len(vals) < 2:
            # One algorithm is not a "this algorithm changes nothing" signal.
            # If the single algorithm's metric still varies across instances it
            # is a real, instance-sensitive objective — flagging it as constant
            # would wrongly block evolution of a perfectly evolvable metric.
            continue
        lo, hi = min(vals), max(vals)
        if hi != lo and (hi - lo) > abs(hi) * _CONSTANT_REL_TOL:
            return False
    # Guard the all-single-algorithm case directly: with fewer than two
    # algorithms there is nothing to compare, so "constant" is unknowable and
    # must not be asserted.
    if len(entries) < 2:
        return False
    return True


def score_algorithms(
    exp_dir: Path,
    algo_files: list[tuple[str, Path]],
    *,
    timeout_sec: int = 300,
) -> list[dict[str, Any]]:
    """Score each ``(algo_name, algo_file)`` on every instance.

    Each entry is ``{"algo", "values", "detail", "num_instances"}`` where
    ``values`` maps ``instance_name -> metric`` for instances that produced a
    finite primary metric, ``detail`` carries the failure reason when scoring did
    not (or ``""`` on success) and ``num_instances`` is how many instances the
    experiment exposes.  Failed packages are returned (not raised) so a single
    broken algorithm does not abort the gate — it is itself a useful signal.
    """
    from researchclaw.pipeline.llm4ad_task_packages import _discover_instances

    instance_argv = json.dumps([str(p) for p in _discover_instances(exp_dir)])
    return [
        _score_one(exp_dir, algo_name, algo_file, instance_argv, timeout_sec)
        for algo_name, algo_file in algo_files
    ]


def fitness_sanity_gate(
    exp_dir: Path,
    algo_files: list[tuple[str, Path]],
    *,
    metric_direction: str = "minimize",
    timeout_sec: int = 300,
) -> list[str]:
    """Return a list of violations; empty means the metric is evolvable.

    ``algo_files`` are the algorithms the run is about to evolve (post
    ``evolve_scope``).  Checked, in order:

    1. every selected algorithm produce a finite primary metric on at least one
       instance (else scoring is broken — see the TSP case);
    2. the metric is not a model-invariant constant across algorithms (ml23);
    3. the metric is deterministic — re-scoring the first algorithm reproduces
       the same value, else it is wall-clock/stateful and gameable (ml03).
    """
    violations: list[str] = []
    entries = score_algorithms(exp_dir, algo_files, timeout_sec=timeout_sec)

    broken = [e for e in entries if _non_finite(e)]
    if not entries:
        violations.append(f"no algorithm could be scored under {exp_dir}; nothing to evolve")
    elif broken:
        names = ", ".join(e["algo"] for e in broken)
        detail = next((e["detail"] for e in broken if e["detail"]), "")
        violations.append(
            f"{len(broken)}/{len(entries)} selected algorithm(s) produced no "
            f"finite primary metric on any instance ({names}); first failure: {detail}"
        )

    if entries and _constant(entries):
        violations.append(
            "primary metric is constant across the selected algorithms "
            f"(min==max within {_CONSTANT_REL_TOL:g}); it does not depend on the "
            "algorithm, so evolution cannot improve it. Pick a metric that "
            "differentially scores the algorithms."
        )

    # Determinism, on one algorithm only (enough to catch a timing metric).
    first = next((e for e in entries if e["values"]), None)
    if first is not None:
        first_file = next((f for name, f in algo_files if name == first["algo"]), None)
        if first_file is not None:
            from researchclaw.pipeline.llm4ad_task_packages import _discover_instances

            instance_argv = json.dumps([str(p) for p in _discover_instances(exp_dir)])
            again = _score_one(exp_dir, first["algo"], first_file, instance_argv, timeout_sec)
            if not _non_finite(again):
                drifted = [
                    k for k, v in again["values"].items()
                    if k in first["values"] and v != first["values"][k]
                ]
                if drifted:
                    violations.append(
                        f"metric is nondeterministic (wall-clock or stateful) for "
                        f"'{first['algo']}': {len(drifted)} instance(s) changed on "
                        f"re-score, e.g. {drifted[0]}. A time-based primary metric "
                        "violates the LLM4AD determinism requirement and is gameable."
                    )

    return violations
