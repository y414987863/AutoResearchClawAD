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
