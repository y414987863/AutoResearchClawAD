import json, tempfile, asyncio, importlib.util
from pathlib import Path
import pytest
import researchclaw.pipeline.llm4ad_task_packages as lp
from llm4ad.config.schema import EvalContext


def _build_exp(algo_src, eval_src, data_files, primary='m'):
    tmp = Path(tempfile.mkdtemp())
    exp = tmp / 'e'
    (exp / 'algorithms' / 'nm').mkdir(parents=True)
    (exp / 'algorithms' / 'nm' / 'nm.py').write_text(algo_src)
    (exp / 'evaluator.py').write_text(eval_src)
    (exp / 'main.py').write_text('x=1\n')
    (exp / 'data').mkdir()
    for name, content in data_files.items():
        (exp / 'data' / name).write_text(content)
    return exp


_ALGO = 'def optimize(i,s):\n    # EVOLVE_START\n    return {"f":float(len(i["coords"]))}\n    # EVOLVE_END\n'


def _evaluate_pkg(exp, algname, datafile):
    out = Path(tempfile.mkdtemp()) / 'o'
    lp.generate_task_packages(exp, out, None, None, None, background='t', metric_direction='minimize')
    pkg = out / 'nm'
    spec = importlib.util.spec_from_file_location("ev", str(pkg / 'nm_evaluator.py'))
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    return asyncio.run(getattr(ev, lp._algo_class_name('nm'))().evaluate(EvalContext(
        data_path=str(pkg / 'data' / datafile), project_root=str(pkg / 'algorithms' / 'nm'), timeout=60)))


def test_nonjson_instance_with_load_instance():
    ev = '''
PRIMARY_METRIC="cost"
def load_instance(path):
    pts=[]
    for line in open(path, encoding="utf-8"):
        p=line.split()
        if len(p)==3 and p[0].isdigit(): pts.append((float(p[1]),float(p[2])))
    return {"coords": pts}
def evaluate_instance(instance, solve):
    return {"cost": float(solve(instance,0)["f"])}
'''
    exp = _build_exp(_ALGO, ev, {"b.tsp": "NODE_COORD_SECTION\n1 0.0 0.0\n2 3.0 4.0\nEOF\n"})
    r = _evaluate_pkg(exp, 'nm', 'b.tsp')
    assert r.success and r.score == -2.0


def test_unicode_stdout_pollution():
    ev = '''
PRIMARY_METRIC="m"
def evaluate_instance(instance, solve):
    for i in range(3):
        print(f"iter {i} \\u2713 \\u2022 \\u5b8c\\u6210")
    out = solve(instance, 0)
    return {"m": float(out["f"])}
'''
    exp = _build_exp(_ALGO, ev, {"i.json": '{"coords":[[0,0],[1,1]]}'})
    r = _evaluate_pkg(exp, 'nm', 'i.json')
    assert r.success and r.score == -2.0


def test_json_instance_default():
    ev = '''
PRIMARY_METRIC="m"
def evaluate_instance(instance, solve):
    return {"m": float(solve(instance,0)["f"])}
'''
    exp = _build_exp(_ALGO, ev, {"i.json": '{"coords":[[0,0],[1,1]]}'})
    r = _evaluate_pkg(exp, 'nm', 'i.json')
    assert r.success and r.score == -2.0


def test_generated_evaluator_uses_timeout_constant():
    """The evaluator must honour the configured eval_timeout_sec.

    llm4ad does not thread a timeout into a custom evaluator's EvalContext (it
    defaults to 60s), so the generated evaluator carries its own ``_EVAL_TIMEOUT``
    constant sourced from resources.eval_timeout_sec and must NOT fall back to
    ``cfg.timeout``. A stray ``cfg.timeout or 60.0`` would silently reset every
    run's timeout to 60s regardless of config.
    """
    exp = _build_exp(_ALGO, 'PRIMARY_METRIC="m"\ndef evaluate_instance(i,s): return {"m":1.0}\n', {"i.json": "{}"})
    import tempfile as _tf
    out = Path(_tf.mkdtemp()) / 'o'
    # 30s configured — inject into resources so _write_config and the evaluator agree.
    lp.generate_task_packages(exp, out, None, None, {"eval_timeout_sec": 30.0}, background='t', metric_direction='minimize')
    pkg = out / 'nm'
    ev_text = (pkg / 'nm_evaluator.py').read_text(encoding='utf-8')
    assert '_EVAL_TIMEOUT = 30.0' in ev_text
    # The executable path must use the constant, not cfg.timeout: the comment
    # explaining *why* may mention cfg.timeout, but the running code may not.
    code_only = "\n".join(
        l for l in ev_text.splitlines() if not l.strip().startswith("#")
    )
    assert 'cfg.timeout' not in code_only


def test_algorithm_module_name_matches_its_directory():
    """`solve.__module__` must report the algorithm's name, not a private alias.

    A generated evaluator may identify the algorithm it is scoring from the
    callable it was handed:

        f"condition={solve.__module__.split('.')[-1]}"

    Loading the file under a private alias (``_evolved_<algo>`` in run_single,
    ``_cmp_<algo>`` in the promotion runner) made that expression yield the
    alias, so the evaluator looked it up in its own table, raised
    ``ValueError('Unknown algorithm: _cmp_listmle_lite')``, and every candidate
    scored nothing — the fitness gate then skipped evolution entirely.
    """
    ev = '''
PRIMARY_METRIC="m"
def evaluate_instance(instance, solve):
    # A real experiment does exactly this to label its output.
    name = solve.__module__.split(".")[-1]
    return {"m": float(len(instance["coords"])) if name == "nm" else -1.0}
'''
    exp = _build_exp(_ALGO, ev, {"i.json": '{"coords":[[0,0],[1,1]]}'})
    r = _evaluate_pkg(exp, 'nm', 'i.json')
    assert r.success, r
    # -2.0 is the "name matched" branch; the old alias produced -1.0.
    assert r.score == -2.0, f"__module__ did not report 'nm' (score={r.score})"


def test_comparison_runner_loads_under_the_real_name():
    """The promotion scorer must agree with run_single about the module name."""
    import importlib.util as _ilu

    from researchclaw.pipeline.llm4ad_utils.comparison_runner import _load_optimize

    tmp = Path(tempfile.mkdtemp())
    algo = tmp / 'my_algo.py'
    algo.write_text('def optimize(i, s):\n    return {"v": 1.0}\n', encoding='utf-8')
    try:
        fn = _load_optimize(algo, 'my_algo')
        assert fn.__module__ == 'my_algo', fn.__module__
    finally:
        import sys as _sys
        _sys.modules.pop('my_algo', None)
