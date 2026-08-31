"""Score one algorithm file across all experiment instances (subprocess entry).

The stage-13 promotion step decides whether an evolved algorithm is actually
an improvement.  To decide that it needs a protocol-identical number for both
the clean baseline and the evolved candidate.  Reading ``results.json`` is not
generic — the generated experiment owns its own schema — so instead we load the
experiment's own ``evaluator.py`` and call ``evaluate_instance`` for every
instance, exactly as ``main.py`` and ``run_single.py`` do.  The result is the
same primary metric under the same protocol as the paper pipeline, for any
generated project.

This module is meant to run as a subprocess (``python comparison_runner.py
<experiment_dir> <algo_name> <algo_file>``): the experiment's ``evaluator.py``
imports siblings like ``experiment_config`` and ``objectives`` by bare module
name, so loading it in the parent process would collide with shared module names
across experiments.  A subprocess keeps ``sys.path``/``sys.modules`` clean.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


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


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: python comparison_runner.py <experiment_dir> <algo_name> <algo_file>",
              file=sys.stderr)
        return 2

    experiment_dir = Path(sys.argv[1]).resolve()
    algo_name = sys.argv[2]
    algo_file = Path(sys.argv[3]).resolve()

    # The experiment's evaluator imports its siblings (experiment_config,
    # objectives, utils) by bare name, so the package root must be on sys.path.
    sys.path.insert(0, str(experiment_dir))
    import evaluator  # noqa: E402  (imported after sys.path is set)

    data_dir = experiment_dir / "data"
    instance_paths = evaluator.list_instance_paths(str(data_dir))
    if not instance_paths:
        print(json.dumps({"error": "no instances found", "primary_metric": None,
                          "mean": None, "n_instances": 0}))
        return 0

    solve = _load_optimize(algo_file, algo_name)
    pm = evaluator.PRIMARY_METRIC

    per_instance: list[float] = []
    for p in instance_paths:
        inst = evaluator.load_instance(p)
        metrics = evaluator.evaluate_instance(inst, solve)
        try:
            per_instance.append(float(metrics[pm]))
        except (KeyError, TypeError, ValueError):
            # Non-finite or missing primary metric disqualifies this instance.
            continue

    if not per_instance:
        print(json.dumps({"error": "no finite primary metric", "primary_metric": pm,
                          "mean": None, "n_instances": len(instance_paths)}))
        return 0

    payload = {
        "primary_metric": pm,
        "mean": sum(per_instance) / len(per_instance),
        "n_instances": len(instance_paths),
        "scored_instances": len(per_instance),
        "per_instance": per_instance,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
