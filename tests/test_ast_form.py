import json
from researchclaw.pipeline.stage_impls._code_generation import (
    _hard_indexed_instance_keys as H, _injected_instance_keys as I,
    _evolve_block_problems as E, _check_llm4ad_structure as C,
    _dangling_local_imports as D, _merge_repaired_files as M)


def test_posonly_and_kwonly():
    assert sorted(H('def optimize(instance, /, seed):\n    return instance["dim"]\n')) == ['dim']
    assert sorted(H('def optimize(*, instance, seed):\n    return instance["dim"]\n')) == ['dim']


def test_injection_forms():
    forms = {
        "dict_literal": 'def evaluate_instance(i,s):\n    return s({**i,"k":1},0)\n',
        "subscript":    'def evaluate_instance(i,s):\n    d=dict(i)\n    d["k"]=1\n    return s(d,0)\n',
        "dict_kw":      'def evaluate_instance(i,s):\n    return s(dict(i,k=1),0)\n',
        "update":       'def evaluate_instance(i,s):\n    d=dict(i)\n    d.update({"k":1})\n    return s(d,0)\n',
        "setdefault":   'def evaluate_instance(i,s):\n    d=dict(i)\n    d.setdefault("k",1)\n    return s(d,0)\n',
        "dict_union":   'def evaluate_instance(i,s):\n    return s({**i} | {"k":1},0)\n',
    }
    for n, c in forms.items():
        assert 'k' in I(c), f"missing injection form {n}"


def test_shadowing_not_delegation():
    def mk(b): return ('def helper(x):\n    return x\n\ndef optimize(instance, seed):\n    # EVOLVE_START\n' + b + '    return {"a":1}\n    # EVOLVE_END\n')
    assert not E('a.py', mk('    helper = 1\n    y=helper\n'))
    assert not E('a.py', mk('    def inner(helper):\n        return helper\n    inner(1)\n'))
    assert E('a.py', mk('    return {"a": helper(1)}\n'))


def test_init_excluded_and_nonjson():
    base = {"main.py": 'import importlib\nprimary_metric="x"\nif __name__ == "__main__":\n    pass\n# --algorithm\n',
            "evaluator.py": 'PRIMARY_METRIC="m"\ndef evaluate_instance(instance, solve):\n    return {"m":1.0}\n'}
    files = dict(base); files["algorithms/a/__init__.py"] = ""
    files["algorithms/a/a.py"] = 'def optimize(i,s):\n    # EVOLVE_START\n    return {"z":i["d"]}\n    # EVOLVE_END\n'
    files["data/x.json"] = json.dumps({"d":2})
    probs = [p for p in C(files) if 'no `algorithms' not in p and '__init__.py' not in p]
    assert not probs, probs

    files2 = dict(base); files2["algorithms/a/a.py"] = 'def optimize(i,s):\n    # EVOLVE_START\n    return {"z":1}\n    # EVOLVE_END\n'
    files2["data/b.tsp"] = "NODE_COORD_SECTION\n"
    assert any('is not JSON' in p for p in C(files2))


def test_dangling_warns_only_and_no_stdlib_false_positive():
    base = {"main.py": "", "evaluator.py": "", "config.py": "x=1\n"}
    assert not D({**base, "main.py": "import importlib\nimport hashlib\n"})  # stdlib, no misfire
    assert D({**base, "main.py": "import cvxpy\n"})  # not importable here -> candidate


def test_merge_accepts_renames():
    files = {"main.py": "import config\n", "config.py": "x=1\n"}
    repaired = {"experiment_config.py": "y=1\n", "main.py": "import experiment_config as config\n", "notes_random.py": "z=1\n"}
    merged, _ = M(files, repaired, label="deep repair")
    assert "experiment_config.py" in merged  # rename kept
    assert "notes_random.py" in merged       # all-new accepted (smoke judges)
