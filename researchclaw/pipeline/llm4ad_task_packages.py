"""Stage-13 LLM4AD task-package generation (deterministic, no LLM/bytecode).

Turns a stage-10 ``experiment/`` directory into one self-contained LLM4AD
task package per algorithm. Each package copies the fixed evaluation modules,
flattens the evolvable algorithm module to the package root, and ships a
custom evaluator + ``config.yaml`` wired to LLM4AD's ``local_path="."``.

Design decisions (verified against LLM4AD_Next/src/llm4ad):
- ``local_path`` must be the package root (the algorithm module imports the
  shared modules at the same level; a subdir-only worktree would break imports).
- ``main.py`` hard-codes one smoke instance, so it is NOT used. Each package
  gets ``run_single.py`` that reads any instance file from ``EvalContext.data_path``.
- The evaluator reports per-instance metrics; LLM4AD aggregates across instances.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PackageManifest:
    algo: str
    path: str
    files: list[str] = field(default_factory=list)
    primary_metric: str = "mean_best_objective_value"
    metric_direction: str = "MINIMIZE"
    n_instances: int = 0


# Shared module basenames that are fixed evaluation logic (copied verbatim).
_SHARED_MODULES = ("benchmarks.py", "evaluator.py", "stats_utils.py")

# Metric-name fragments that decide the optimisation direction. LLM4AD needs a
# MetricType per metric; getting it wrong silently evolves the algorithm in the
# WRONG direction (no error, plausible-looking numbers), so this is inferred
# from the metric name rather than hard-coded to MINIMIZE.
_MINIMIZE_HINTS = (
    "loss", "error", "err", "cost", "objective", "regret", "rmse", "mae",
    "mse", "nll", "perplexity", "ppl", "distance", "dist", "gap", "latency",
    "runtime", "time", "penalty", "violation", "fid",
)
_MAXIMIZE_HINTS = (
    "acc", "accuracy", "reward", "return", "f1", "auc", "auroc", "precision",
    "recall", "iou", "dice", "bleu", "rouge", "success", "win", "map",
    "ndcg", "psnr", "ssim", "r2", "coverage", "throughput", "speedup",
    "fitness", "utility", "profit", "score",
)


def infer_metric_direction(metric: str) -> str:
    """Return ``"MINIMIZE"`` or ``"MAXIMIZE"`` for a primary-metric name.

    Checked in minimize-first order so ``mean_best_objective_value`` resolves to
    MINIMIZE, and so compounds like ``val_loss`` are not captured by the
    "score"/"fitness" family. Unknown names fall back to MINIMIZE, matching the
    default ``mean_best_objective_value`` benchmark task.
    """
    name = (metric or "").lower()
    for hint in _MINIMIZE_HINTS:
        if hint in name:
            return "MINIMIZE"
    for hint in _MAXIMIZE_HINTS:
        if hint in name:
            return "MAXIMIZE"
    return "MINIMIZE"


def build_providers_yaml(llm_config: dict[str, Any] | None) -> str:
    """Build the LLM4AD ``providers`` YAML block from a ResearchClaw LLM config.

    ``llm_config`` may carry any of: base_url, api_key, model, provider,
    temperature, max_tokens, timeout. Missing keys fall back to the literal
    environment placeholders LLM4AD resolves at run time. The api_key is
    written verbatim (no redaction) per project requirement.
    """
    import yaml as _yaml

    cfg = llm_config or {}
    provider = {
        "name": "default",
        "type": "openai_compatible",
        "base_url": str(cfg.get("base_url") or "${LLM_BASE_URL}"),
        "api_key": str(cfg.get("api_key") or "${LLM_API_KEY}"),
        "model": str(cfg.get("model") or "${LLM_MODEL}"),
    }
    if cfg.get("temperature") is not None:
        provider["temperature"] = float(cfg["temperature"])
    if cfg.get("max_tokens") is not None:
        provider["max_tokens"] = int(cfg["max_tokens"])
    if cfg.get("timeout") is not None:
        provider["timeout"] = float(cfg["timeout"])
    # yaml.safe_dump sorts allow_unicode=False; block style keeps it readable.
    body = _yaml.safe_dump(
        [provider], sort_keys=False, default_flow_style=False, width=120
    )
    # Indent the sequence under "providers:".
    return "".join("  " + line + "\n" for line in body.splitlines())


def _read_primary_metric(exp_dir: Path) -> str:
    """Best-effort read of the primary metric name from the generated main.py."""
    main_py = exp_dir / "main.py"
    try:
        text = main_py.read_text(encoding="utf-8")
    except OSError:
        return "mean_best_objective_value"
    # Look for `primary_metric = "..."` / `"primary_metric": "..."` (either case).
    import re as _re

    for pat in (
        r'primary_metric\s*=\s*"([a-zA-Z0-9_]+)"',
        r'PRIMARY_METRIC\s*=\s*"([a-zA-Z0-9_]+)"',
        r'"primary_metric"\s*:\s*"([a-zA-Z0-9_]+)"',
        r"'primary_metric'\s*:\s*'([a-zA-Z0-9_]+)'",
    ):
        m = _re.search(pat, text)
        if m:
            return m.group(1)
    # Fall back to HYPERPARAMETERS-like key if present; otherwise the default.
    return "mean_best_objective_value"


def _discover_algorithms(exp_dir: Path) -> list[tuple[str, Path]]:
    """Return (algo_name, source_file) for each algorithms/<algo>/<algo>.py."""
    algo_root = exp_dir / "algorithms"
    found: list[tuple[str, Path]] = []
    if not algo_root.is_dir():
        return found
    for sub in sorted(algo_root.iterdir()):
        if not sub.is_dir():
            continue
        src = sub / (sub.name + ".py")
        if src.is_file():
            found.append((sub.name, src))
    return found


def _discover_instances(exp_dir: Path) -> list[Path]:
    data_dir = exp_dir / "data"
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob("*.json"))


def _algo_class_name(algo: str) -> str:
    """PascalCase class name for the evaluator (e.g. nelder_mead -> NelderMead)."""
    return "".join(p.capitalize() for p in algo.split("_")) + "Evaluator"


def _write_run_single(algo: str, package: Path, primary_metric: str) -> None:
    """Write the single-instance entry point `run_single.py`.

    It loads the evolvable algorithm module, realises the seeds/starts loop, and
    delegates the instance-level metric to the shared ``evaluator.py`` (via
    ``prepare_start``/``evaluate_instance``/``PRIMARY_METRIC``). Because the
    package runner and ``main.py`` call the SAME evaluator helpers, the number
    LLM4AD optimises equals the number the paper reports.
    """
    text = (
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
        "\n"
        "# Fixed evaluation data/logic (never evolved). These are the same helpers\n"
        "# stage-12 uses, so the arithmetic here is bit-for-bit what the paper\n"
        "# reports. A missing module means the package was built on an experiment\n"
        "# that did not follow the task-package layout, which is a build error.\n"
        "from benchmarks import get_bounds\n"
        "from stats_utils import SEEDS, mean_std, rng_from_instance\n"
        "from evaluator import evaluate_instance, PRIMARY_METRIC, prepare_start\n"
        "\n"
        f'ALGO = "{algo}"\n'
        "\n"
        "\n"
        "def main():\n"
        "    if len(sys.argv) < 2:\n"
        '        print("usage: python run_single.py <instance.json>", file=sys.stderr)\n'
        "        sys.exit(1)\n"
        "    inst_path = Path(sys.argv[1])\n"
        "    if not inst_path.is_absolute():\n"
        "        inst_path = (Path.cwd() / inst_path).resolve()\n"
        "    with open(inst_path, \"r\", encoding=\"utf-8\") as f:\n"
        "        instance = json.load(f)\n"
        "\n"
        "    import importlib\n"
        "    alg_mod = importlib.import_module(ALGO)\n"
        "    optimize = getattr(alg_mod, \"optimize\", None)\n"
        "    if optimize is None:\n"
        '        raise RuntimeError(f"{ALGO}.py has no optimize(instance, seed) function")\n'
        "\n"
        "    n_starts = int(instance.get('n_starts', 1))\n"
        "    per_seed = {}\n"
        "    for seed in SEEDS:\n"
        "        values = []\n"
        "        for sidx in range(n_starts):\n"
        "            inst = prepare_start(instance, seed, sidx)\n"
        "            result = optimize(inst, seed=seed)\n"
        "            best_f = float(result.get('best_f', float('inf')))\n"
        "            if not np.isfinite(best_f):\n"
        "                best_f = float(1e12)\n"
        "            values.append(best_f)\n"
        "        if values:\n"
        "            per_seed[seed] = float(np.mean(values))\n"
        "\n"
        "    metric = evaluate_instance(per_seed)\n"
        "    payload = {\"primary_metric\": PRIMARY_METRIC, \"metrics\": metric}\n"
        "    print(json.dumps(payload))\n"
        "\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    main()\n"
    )
    (package / "run_single.py").write_text(text, encoding="utf-8")


def _write_evaluator(
    algo: str, package: Path, primary_metric: str, metric_direction: str = "MINIMIZE"
) -> None:
    """Write a custom LLM4AD BaseEvaluator that runs run_single.py per instance."""
    cls = _algo_class_name(algo)
    _better = "lower" if metric_direction == "MINIMIZE" else "higher"
    text = (
        "import asyncio\n"
        "import json\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "\n"
        "from llm4ad.evaluator.base import (\n"
        "    BaseEvaluator,\n"
        "    EvaluationResult,\n"
        "    Metric,\n"
        "    MetricType,\n"
        ")\n"
        "\n"
        "\n"
        "\n"
        f'PRIMARY_METRIC = "{primary_metric}"\n'
        "\n"
        "\n"
        f'@BaseEvaluator.register("{algo}_evaluator")\n'
        f"class {cls}(BaseEvaluator):\n"
        f'    """Evaluate the "{algo}" algorithm on one instance file."""\n'
        "\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self._metrics = [\n"
        "            Metric(\n"
        f'                name=PRIMARY_METRIC,\n'
        f'                type=MetricType.{metric_direction},\n'
        f'                weight=1.0,\n'
        f'                description="Instance-level {primary_metric} ({_better} is better)",\n'
        "            ),\n"
        "        ]\n"
        "\n"
        "    @property\n"
        "    def name(self):\n"
        f'        return "{algo}_evaluator"\n'
        "\n"
        "    @property\n"
        "    def metrics(self):\n"
        "        return self._metrics\n"
        "\n"
        "    async def evaluate(self, cfg):\n"
        "        start = time.time()\n"
        "        project_root = Path(cfg.project_root)\n"
        "        data_path = Path(cfg.data_path) if cfg.data_path else None\n"
        "        if data_path is None or not data_path.is_file():\n"
        "            return EvaluationResult(\n"
        "                score=0.0, metrics={}, success=False,\n"
        f'                error_message=f"Data file not found: {{data_path}}",\n'
        "                duration_ms=(time.time() - start) * 1000,\n"
        "            )\n"
        "        run_script = project_root / \"run_single.py\"\n"
        "        if not run_script.is_file():\n"
        "            return EvaluationResult(\n"
        "                score=0.0, metrics={}, success=False,\n"
        '                error_message="run_single.py not found in project_root",\n'
        "                duration_ms=(time.time() - start) * 1000,\n"
        "            )\n"
        "        run_start = time.time()\n"
        "        try:\n"
        "            proc = await asyncio.create_subprocess_exec(\n"
        "                sys.executable, str(run_script), str(data_path),\n"
        "                cwd=str(project_root),\n"
        "                stdout=asyncio.subprocess.PIPE,\n"
        "                stderr=asyncio.subprocess.PIPE,\n"
        "            )\n"
        "            stdout_bytes, stderr_bytes = await asyncio.wait_for(\n"
        "                proc.communicate(), timeout=cfg.timeout or 60.0)\n"
        "        except asyncio.TimeoutError:\n"
        "            proc.kill()\n"
        "            await proc.communicate()\n"
        "            return EvaluationResult(\n"
        "                score=0.0, metrics={}, success=False,\n"
        f'                error_message=f"Evaluation timed out after {{cfg.timeout}}s",\n'
        "                duration_ms=(cfg.timeout or 60.0) * 1000,\n"
        "            )\n"
        "        duration_ms = (time.time() - run_start) * 1000\n"
        "        stdout = stdout_bytes.decode(\"utf-8\", errors=\"replace\")\n"
        "        stderr = stderr_bytes.decode(\"utf-8\", errors=\"replace\")\n"
        "        if proc.returncode != 0:\n"
        "            return EvaluationResult(\n"
        "                score=0.0, metrics={}, success=False,\n"
        f'                error_message=f"run_single failed: {{stderr.strip()}}",\n'
        "                duration_ms=duration_ms,\n"
        "            )\n"
        "        try:\n"
        "            payload = json.loads(stdout.strip())\n"
        "        except json.JSONDecodeError:\n"
        "            return EvaluationResult(\n"
        "                score=0.0, metrics={}, success=False,\n"
        f'                error_message=f"Invalid JSON from run_single: {{stdout[:200]}}",\n'
        "                duration_ms=duration_ms,\n"
        "            )\n"
        "        metrics = payload.get(\"metrics\", {})\n"
        "        import math\n"
        "        # EvaluationResult.metrics is a dict[str, float]; keep only finite scalars.\n"
        "        scalar = {}\n"
        "        for k, v in metrics.items():\n"
        "            try:\n"
        "                fv = float(v)\n"
        "            except (TypeError, ValueError):\n"
        "                continue\n"
        "            if math.isfinite(fv):\n"
        "                scalar[k] = fv\n"
        "        value = scalar.get(PRIMARY_METRIC)\n"
        "        if value is None:\n"
        "            # Two distinct causes, so name them separately: the key was\n"
        "            # never emitted (evaluator/run_single contract mismatch), or\n"
        "            # it was emitted as inf/nan (a diverged run).\n"
        "            if PRIMARY_METRIC in metrics:\n"
        "                reason = \"non-finite value %r\" % (metrics[PRIMARY_METRIC],)\n"
        "            else:\n"
        "                reason = \"key absent; run_single emitted %s\" % (\n"
        "                    sorted(metrics) or \"no metrics\",\n"
        "                )\n"
        "            return EvaluationResult(\n"
        "                score=0.0, metrics={}, success=False,\n"
        '                error_message="Primary metric %s unusable: %s" % (\n'
        "                    PRIMARY_METRIC, reason),\n"
        "                duration_ms=duration_ms,\n"
        "            )\n"
        "        score = self.compute_score(scalar)\n"
        "        return EvaluationResult(\n"
        "            score=score, metrics=scalar, success=True,\n"
        "            duration_ms=duration_ms,\n"
        '            metadata={"dataset": str(data_path)},\n'
        "        )\n"
    )
    (package / f"{algo}_evaluator.py").write_text(text, encoding="utf-8")


def _yaml_float(value: float) -> str:
    """Render a float so YAML 1.1 parses it as a float, not a string.

    ``str(1e-06)`` is ``"1e-06"``, which YAML 1.1 reads as a *string* because
    the mantissa has no decimal point and the exponent no sign. ``1.0e-06`` is
    unambiguous.
    """
    text = repr(float(value))
    if "e" in text or "E" in text:
        mantissa, _, exponent = text.partition("e")
        if "." not in mantissa:
            mantissa += ".0"
        if not exponent.startswith(("+", "-")):
            exponent = "+" + exponent
        return f"{mantissa}e{exponent}"
    return text


def _write_config(
    algo: str,
    package: Path,
    primary_metric: str,
    providers_yaml: str,
    evolution: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    description: str = "",
    background: str = "",
    max_evals: int = 0,
) -> None:
    """Write the LLM4AD config.yaml for a self-contained task package.

    ``evolution`` and ``resources`` come from ``config.experiment.llm4ad_boost``
    so the user-edited yaml actually drives the run (previously hard-coded).
    """
    import yaml as _yaml

    cls = _algo_class_name(algo)
    evo = dict(evolution or {})
    res = dict(resources or {})
    island = dict(evo.get("island") or {})

    eval_timeout = float(res.get("eval_timeout_sec") or 60.0)
    workers = int(res.get("parallel_workers") or 1)
    # LLM4AD's `parallel` is a bool; `parallel_workers <= 1` means serial.
    parallel = "true" if workers > 1 else "false"

    evo_lines = [
        "evolution:\n",
        '  type: "{}"\n'.format(evo.get("method") or "island_ga"),
        "  max_generations: {}\n".format(int(evo.get("max_generations") or 2)),
        "  elite_ratio: {}\n".format(_yaml_float(evo.get("elite_ratio") or 0.2)),
        "  mutation_rate: {}\n".format(_yaml_float(evo.get("mutation_rate") or 0.6)),
        "  crossover_rate: {}\n".format(_yaml_float(evo.get("crossover_rate") or 0.3)),
        "  early_stop_patience: {}\n".format(int(island.get("early_stop_patience") or 5)),
        "  early_stop_threshold: {}\n".format(
            _yaml_float(island.get("early_stop_threshold") or 1e-06)
        ),
        "  num_islands: {}\n".format(int(island.get("num_islands") or 1)),
        "  island_population_size: {}\n".format(
            int(island.get("island_population_size") or 2)
        ),
    ]
    evolution_yaml = "".join(evo_lines)

    desc_en = description or (
        f"Evolve the {algo} implementation on the generated benchmark instances."
    )

    # The live research topic becomes the `background` LLM4AD feeds the sampler;
    # fall back to a one-liner so the field is always present.
    bg = background.strip() or desc_en

    # Render as a YAML block scalar so newlines/colons in the topic survive.
    bg_block = _yaml.safe_dump(
        {"background": bg}, sort_keys=False, default_flow_style=False
    ).rstrip("\n")

    config = (
        "# LLM4AD task config - auto-generated for the {algo} benchmark\n"
        "project_name: \"{algo}_task\"\n"
        "\n"
        "description_en: \"{desc_en}\"\n"
        "{bg_block}\n"
        "base_dir: \"./runs\"\n"
        "random_seed: 42\n"
        "\n"
        "providers:\n"
        "{providers_yaml}"
        "\n"
        "evaluator:\n"
        "  type: \"custom\"\n"
        "  module: \"{algo}_evaluator.py:{cls}\"\n"
        "  timeout: {eval_timeout}\n"
        "  max_retries: 2\n"
        "  parallel: {parallel}\n"
        "  batch_size: 1\n"
        "  dataset:\n"
        "    mode: \"directory\"\n"
        "    path: \"data\"\n"
        "    recursive: false\n"
        "  metrics: [\"{primary_metric}\"]\n"
        "\n"
        "version_control:\n"
        "  enabled: true\n"
        "  type: \"git_worktree\"\n"
        "  local_path: \".\"\n"
        "  auto_initialize: true\n"
        "  auto_cleanup: true\n"
        "\n"
        "repo_analyzer:\n"
        "  type: \"evolve_detector\"\n"
        "  include: [\"*.py\"]\n"
        # runs/ and best/ hold copies of earlier candidates that still carry
        # EVOLVE markers; without these excludes a second run would treat its
        # own history as additional evolvable targets.
        "  exclude: [\".git/**\", \"__pycache__/**\", \"*.pyc\", \"runs/**\", \"best/**\", \"data/**\"]\n"
        "\n"
        "planner:\n"
        "  type: \"llm_evolution\"\n"
        "  selection_strategy: \"weighted\"\n"
        "  parent_selection_strategy: \"tournament\"\n"
        "  samplers:\n"
        "    - name: \"init_sampler\"\n"
        "    - name: \"mutation_sampler\"\n"
        "    - name: \"crossover_sampler\"\n"
        "\n"
        "coder:\n"
        "  type: \"custom\"\n"
        "  provider: \"default\"\n"
        "  prompt_template: |\n"
        "    Rewrite ONLY the function body of `optimize(instance, seed)` in\n"
        "    {algo}.py between the `# EVOLVE_START` and `# EVOLVE_END` markers.\n"
        "    The instance dict carries function/dimension/seed/lb/ub/max_evals/x0.\n{max_evals_line}"
        "    Return a dict with best_f (finite) and n_evals (<= max_evals).\n"
        "\n"
        "    OUTPUT FORMAT (STRICT): return EXACTLY ONE fenced code block with a\n"
        "    `python:{algo}.py` header, containing the FULL {algo}.py file. The\n"
        "    block header and closing fence must be the first and last lines.\n"
        "    Keep imports and module-level constants outside the EVOLVE markers.\n"
        "    Output NOTHING outside the fence - no prose, no explanation, no\n"
        "    other markdown blocks, no sample outputs. In particular do not\n"
        "    output a JSON block or a `python` block without the `:{algo}.py`\n"
        "    suffix, or the response will be rejected.\n"
        "\n"
        "    ```python:{algo}.py\n"
        "    <full file: imports, constants, then optimize() with the evolved\n"
        "     body between # EVOLVE_START and # EVOLVE_END>\n"
        "    ```\n"
        "\n"
        "{evolution_yaml}"
        "\n"
        "logging:\n"
        "  level: \"INFO\"\n"
        "  console: true\n"
    ).format(
        algo=algo,
        cls=cls,
        primary_metric=primary_metric,
        providers_yaml=providers_yaml,
        evolution_yaml=evolution_yaml,
        eval_timeout=eval_timeout,
        parallel=parallel,
        desc_en=desc_en,
        bg_block=bg_block,
        max_evals_line=(
            f"    Evaluation budget: {int(max_evals)} = max_evals. Use it to\n"
            "    make real progress; do not stop early after a handful of evals.\n"
            if max_evals > 0 else ""
        ),
    )
    (package / "config.yaml").write_text(config, encoding="utf-8")


def _rewrite_max_evals(inst_file: Path, dst: Path, max_evals: int) -> None:
    """Copy *inst_file* to *dst*, overriding its ``max_evals`` if present.

    The per-instance eval budget is baked into the JSON at stage-10 generation
    (typically 200). A tight budget leaves an already-saturated 2-D benchmark no
    headroom for LLM4AD to improve. This lets the task run under a larger budget
    so evolution has room; the caller must ensure the baseline is re-run under
    the SAME budget for a fair comparison.
    """
    try:
        data = json.loads(inst_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Not a JSON instance — copy verbatim; non-JSON files aren't ours.
        shutil.copy2(inst_file, dst)
        return
    if isinstance(data, dict) and "max_evals" in data:
        data["max_evals"] = int(max_evals)
        dst.write_text(json.dumps(data, indent=1), encoding="utf-8")
    else:
        shutil.copy2(inst_file, dst)


def generate_task_packages(
    exp_dir: Path,
    out_dir: Path,
    llm_config: dict[str, Any] | None = None,
    evolution: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    *,
    background: str = "",
    metric_direction: str = "",
    max_evals: int = 0,
) -> list[PackageManifest]:
    """Generate one self-contained LLM4AD task package per algorithm.

    Args:
        exp_dir: The generated experiment directory (stage-10/experiment).
        out_dir: Destination for task packages (e.g. stage-13/task_packages).
        llm_config: LLM connection settings (base_url, api_key, model, provider,
            timeout, …) injected into each package's ``config.yaml`` providers
            section. If omitted, the config uses ``${LLM_BASE_URL}`` /
            ``${LLM_API_KEY}`` / ``${LLM_MODEL}`` placeholders.
        evolution: Evolution hyperparameters (method, max_generations,
            elite_ratio, mutation_rate, crossover_rate, island) written into
            each ``config.yaml``.
        resources: Resource budget (eval_timeout_sec, parallel_workers) written
            into each ``config.yaml``.
        background: The research topic, written as the LLM4AD ``background``
            field the sampler feeds to the LLM. Empty falls back to the
            per-package description.
        metric_direction: Explicit optimisation direction ("minimize"/"maximize",
            case-insensitive) from ``config.experiment.metric_direction``. Takes
            precedence over name-based inference; empty uses
            :func:`infer_metric_direction`.
        max_evals: Per-instance evaluation budget override. When > 0 it replaces
            ``max_evals`` in each copied data/*.json and the coder prompt so the
            evolved candidate keeps the same budget. 0 keeps the stage-10 value.

    Returns:
        A list of PackageManifest describing each generated package.
    """
    exp_dir = Path(exp_dir)
    out_dir = Path(out_dir)
    primary_metric = _read_primary_metric(exp_dir)
    if metric_direction:
        metric_direction = metric_direction.strip().upper()
        if metric_direction not in ("MINIMIZE", "MAXIMIZE"):
            metric_direction = infer_metric_direction(primary_metric)
    else:
        metric_direction = infer_metric_direction(primary_metric)
    providers_yaml = build_providers_yaml(llm_config)
    algorithms = _discover_algorithms(exp_dir)
    instances = _discover_instances(exp_dir)

    manifests: list[PackageManifest] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for algo, src in algorithms:
        package = out_dir / algo

        # Rebuild from scratch every run: llm4ad's git_worktree branches new
        # candidates from HEAD, so a package left over from a previous
        # generation would resurrect its old snapshot. Deleting the dir (and its
        # .git) makes llm4ad re-initialize to the files we just wrote.
        if package.exists():
            shutil.rmtree(package, ignore_errors=True)
        package.mkdir(parents=True, exist_ok=True)

        # 1. Copy shared fixed evaluation modules.
        for mod in _SHARED_MODULES:
            shared_src = exp_dir / mod
            if shared_src.is_file():
                shutil.copy2(shared_src, package / mod)

        # 2. Copy instance data (optionally overriding the eval budget).
        if instances:
            (package / "data").mkdir(parents=True, exist_ok=True)
        for inst in instances:
            if max_evals > 0:
                _rewrite_max_evals(inst, package / "data" / inst.name, max_evals)
            else:
                shutil.copy2(inst, package / "data" / inst.name)

        # 3. Flatten the evolvable algorithm to the package root.
        shutil.copy2(src, package / (algo + ".py"))

        # 4. Write the single-instance entry point.
        _write_run_single(algo, package, primary_metric)

        # 5. Write the custom evaluator.
        _write_evaluator(algo, package, primary_metric, metric_direction)

        # 6. Write config.yaml.
        _write_config(
            algo, package, primary_metric, providers_yaml, evolution, resources,
            description=background,
            background=background,
            max_evals=max_evals,
        )

        files = sorted(p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file())
        manifests.append(
            PackageManifest(
                algo=algo,
                path=str(package),
                files=files,
                primary_metric=primary_metric,
                metric_direction=metric_direction,
                n_instances=len(instances),
            )
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([m.__dict__ for m in manifests], indent=2),
        encoding="utf-8",
    )

    return manifests


@dataclass
class EvolutionResult:
    """Outcome of running LLM4AD evolution on a single task package."""

    algo: str
    success: bool = False
    best_score: float | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)
    best_code_dir: str | None = None
    run_id: str | None = None
    error_message: str | None = None
    log_tail: str = ""


_INSTALL_HINT = (
    "Use the venv that has llm4ad installed: "
    "set llm4ad_cmd to '<venv>/Scripts/llm4ad.exe' or 'llm4ad' on PATH."
)


def _read_best_metadata(best_dir: Path) -> tuple[float | None, dict[str, float]]:
    """Read ``score``/``metrics`` from an LLM4AD ``best/metadata.json``.

    LLM4AD's ``infra.best_exporter`` writes ``{run}/best/metadata.json`` with an
    ``evaluation`` block carrying the winning individual's score and metrics.
    Without this the pipeline recorded ``best_score: null`` for every run, i.e.
    no measurable evidence that evolution improved anything.
    """
    meta = best_dir / "metadata.json"
    if not meta.is_file():
        return None, {}
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {}
    evaluation = data.get("evaluation") or {}
    score = evaluation.get("score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    metrics: dict[str, float] = {}
    for k, v in (evaluation.get("metrics") or {}).items():
        try:
            metrics[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return score, metrics


def _resolve_run_best(
    package_dir: Path, algo: str = ""
) -> tuple[str | None, str | None, float | None, dict[str, float]]:
    """Locate the best/ snapshot or the evolved algorithm code from a run.

    Returns ``(best_code_dir, run_id, best_score, best_metrics)``. Priority:
      1. a non-empty ``best/`` dir (export_best snapshot),
      2. the deepest directory (under runs/) containing a .py file whose name
         matches the algo and carries an EVOLVE block (the evolved candidate),
      3. the deepest dir under runs/ containing any .py.

    ``run_id`` is the run directory name (the child of ``runs/`` on the path to
    the chosen directory); score/metrics are only available via route 1.
    """
    runs_root = package_dir / "runs"
    if not runs_root.is_dir():
        return None, None, None, {}

    def _run_id_of(p: Path) -> str | None:
        try:
            rel = p.relative_to(runs_root)
        except ValueError:
            return None
        return rel.parts[0] if rel.parts else None

    def _has_evolve(p: Path) -> bool:
        for f in p.rglob("*.py"):
            try:
                if "EVOLVE_START" in f.read_text(encoding="utf-8"):
                    return True
            except OSError:
                continue
        return False

    # 1. Non-empty best/ snapshot.
    best_dirs = [p for p in runs_root.rglob("best") if p.is_dir()]
    if best_dirs:
        for _b in sorted(best_dirs, key=lambda p: len(p.parts), reverse=True):
            if any(f.suffix == ".py" for f in _b.rglob("*.py")):
                score, metrics = _read_best_metadata(_b)
                # LLM4AD's export_best writes the winning worktree under
                # best/code/ and the score under best/metadata.json. Returning
                # best/ (not best/code/) made the downstream consumer copy a
                # directory whose <algo>.py sits one level deep, so the promote
                # step could never find evolution_results/<algo>/<algo>.py and
                # the "evolved" result came back as the pristine stage-10 code.
                code_dir = _b / "code"
                if not code_dir.is_dir():
                    code_dir = _b
                return str(code_dir), _run_id_of(_b), score, metrics
    # 2. Deepest dir containing an evolvable algo .py.
    evolve_dirs = [p for p in runs_root.rglob("*") if p.is_dir() and _has_evolve(p)]
    if evolve_dirs:
        chosen = max(evolve_dirs, key=lambda p: len(p.parts))
        return str(chosen), _run_id_of(chosen), None, {}
    # 3. Deepest dir containing any .py.
    code_dirs = [
        p for p in runs_root.rglob("*")
        if p.is_dir() and any(f.suffix == ".py" for f in p.rglob("*.py"))
    ]
    if code_dirs:
        chosen = max(code_dirs, key=lambda p: len(p.parts))
        return str(chosen), _run_id_of(chosen), None, {}
    return None, None, None, {}


def run_evolution_on_packages(
    packages_dir: Path,
    llm4ad_cmd: str = "llm4ad",
    timeout_sec: int = 1800,
    env: dict[str, Any] | None = None,
) -> list[EvolutionResult]:
    """Run ``llm4ad run config.yaml`` in each task package.

    Each package is evolved serially (no parallel worktrees). A package that
    fails is recorded non-fatally so the caller can degrade gracefully.

    Args:
        packages_dir: Directory containing one ``config.yaml`` per algorithm
            (e.g. stage-13/task_packages).
        llm4ad_cmd: Command name or path to the llm4ad executable.
        timeout_sec: Per-package wall-clock timeout.
        env: Extra environment variables (e.g. provider keys). Merged over
            ``os.environ``.

    Returns:
        Per-algorithm EvolutionResult list (one per package with config.yaml).
    """
    import os as _os
    import subprocess as _sp

    base_env = dict(_os.environ)
    if env:
        base_env.update({str(k): str(v) for k, v in env.items()})

    results: list[EvolutionResult] = []
    packages_dir = Path(packages_dir)
    if not packages_dir.is_dir():
        return results

    configs = sorted(packages_dir.glob("*/config.yaml"))
    for cfg_path in configs:
        package_dir = cfg_path.parent
        algo = package_dir.name
        try:
            proc = _sp.run(
                [llm4ad_cmd, "run", "config.yaml"],
                cwd=str(package_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
                env=base_env,
            )
            log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            tail = log[-2000:]
            if proc.returncode != 0:
                results.append(
                    EvolutionResult(
                        algo=algo,
                        success=False,
                        error_message=(
                            f"llm4ad run exited {proc.returncode}: "
                            f"{tail[-300:]}"
                        ),
                        log_tail=tail,
                    )
                )
                continue
            best_dir, run_id, best_score, best_metrics = _resolve_run_best(
                package_dir, algo
            )
            results.append(
                EvolutionResult(
                    algo=algo,
                    success=True,
                    best_score=best_score,
                    best_metrics=best_metrics,
                    best_code_dir=best_dir,
                    run_id=run_id,
                    log_tail=tail,
                )
            )
        except _sp.TimeoutExpired:
            results.append(
                EvolutionResult(
                    algo=algo,
                    success=False,
                    error_message=f"llm4ad run timed out after {timeout_sec}s",
                )
            )
        except FileNotFoundError:
            results.append(
                EvolutionResult(
                    algo=algo,
                    success=False,
                    error_message=(
                        f"llm4ad executable '{llm4ad_cmd}' not found. {_INSTALL_HINT}"
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                EvolutionResult(
                    algo=algo,
                    success=False,
                    error_message=f"evolution error: {exc}",
                )
            )

    return results
