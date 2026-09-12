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


def test_helper_defined_inside_the_markers_is_not_delegation():
    """Only a helper OUTSIDE the markers hides the algorithm.

    A module-level class/function that sits inside the marked region is part of
    the evolvable unit: LLM4AD rewrites that whole region, so calling it is not
    delegation. The reference task packages are laid out exactly this way — a
    marker block containing a helper class *and* the function that uses it. The
    old rule keyed on "is a module-level definition", which flagged a correct
    `class _OptimizationTimeoutError(Exception)` inside the block and sent the
    repair loop after working code.
    """
    inside = (
        "# EVOLVE_START\n"
        "class _Timeout(Exception):\n"
        "    pass\n"
        "\n"
        "\n"
        "def optimize(instance, seed):\n"
        "    raise _Timeout()\n"
        "# EVOLVE_END\n"
    )
    assert not E('a.py', inside)

    # Same file, but the helper now sits ABOVE the marker: frozen, unreachable,
    # and genuinely a defect.
    outside = (
        "class _Timeout(Exception):\n"
        "    pass\n"
        "\n"
        "\n"
        "# EVOLVE_START\n"
        "def optimize(instance, seed):\n"
        "    raise _Timeout()\n"
        "# EVOLVE_END\n"
    )
    assert E('a.py', outside)


def test_helper_inside_but_partially_outside_is_still_flagged():
    """A helper straddling the boundary is only partly evolvable."""
    straddling = (
        "# EVOLVE_START\n"
        "def optimize(instance, seed):\n"
        "    return {'a': helper(1)}\n"
        "# EVOLVE_END\n"
        "\n"
        "\n"
        "def helper(x):\n"
        "    return x\n"
    )
    assert E('a.py', straddling)


def test_init_excluded_and_nonjson():
    # The evaluator carries METRIC_DEF because the structure check requires a
    # static direction declaration; this test is about `__init__.py` exclusion
    # and non-JSON instances, so the fixture must otherwise be contract-clean.
    base = {"main.py": 'import importlib\nprimary_metric="x"\nif __name__ == "__main__":\n    pass\n# --algorithm\n',
            "evaluator.py": 'PRIMARY_METRIC="m"\nMETRIC_DEF={"primary_metric":PRIMARY_METRIC,"direction":"minimize"}\ndef evaluate_instance(instance, solve):\n    return {"m":1.0}\n'}
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


# ---------------------------------------------------------------------------
# `instance` alias tracking must not invent aliases
# ---------------------------------------------------------------------------

def test_call_result_is_not_an_instance_alias():
    """`m = score_model(instance, ...)` does NOT make `m` an instance.

    The old rule took any call's first argument as the alias source, so a
    result dict indexed by a metric name — `metrics["ndcg_at_10"]` — was read as
    `instance["ndcg_at_10"]` and reported as a missing data field. That sent the
    repair loop after correct code.
    """
    code = (
        "def optimize(instance, seed):\n"
        "    candidate = {'w': 1}\n"
        "    validation_metrics = score_model(instance, candidate, split='validation')\n"
        "    return float(validation_metrics['ndcg_at_10'])\n"
    )
    assert H(code) == set()


def test_constructor_result_is_not_an_instance_alias():
    code = (
        "def optimize(instance, seed):\n"
        "    model = Wrapper(instance)\n"
        "    return model['weights']\n"
    )
    assert H(code) == set()


def test_value_preserving_calls_still_alias():
    """The idiomatic instance copies must keep working."""
    for assign in ("dict(instance)", "copy(instance)", "deepcopy(instance)",
                   "instance", "{**instance}"):
        code = (
            "def optimize(instance, seed):\n"
            f"    inst = {assign}\n"
            "    return inst['x0']\n"
        )
        assert H(code) == {"x0"}, f"alias lost for `inst = {assign}`"


def test_unguarded_real_read_is_still_reported():
    assert H("def optimize(instance, seed):\n    return instance['missing']\n") == {"missing"}


# ---------------------------------------------------------------------------
# main.py's metric output check must accept the experiment's real metric
# ---------------------------------------------------------------------------

def _valid_project(**overrides) -> dict:
    """A minimal project that passes the structure check, minus overrides."""
    files = {
        "main.py": (
            "import importlib\n"
            "PRIMARY_METRIC = 'ndcg_at_10'\n"
            "print(f'ndcg_at_10: {1.0}')\n"
            "if __name__ == '__main__':\n"
            "    pass\n"
            "# --algorithm\n"
        ),
        "evaluator.py": (
            "PRIMARY_METRIC = 'ndcg_at_10'\n"
            'METRIC_DEF = {"primary_metric": PRIMARY_METRIC, "direction": "maximize"}\n'
            "def evaluate_instance(instance, solve):\n"
            "    return {'ndcg_at_10': 1.0}\n"
        ),
        "algorithms/a/a.py": (
            "def optimize(instance, seed):\n"
            "    # EVOLVE_START\n"
            "    return {'v': instance['d']}\n"
            "    # EVOLVE_END\n"
        ),
        "data/x.json": json.dumps({"d": 1}),
    }
    files.update(overrides)
    return files


def test_metric_output_accepts_the_declared_metric_name():
    """`print(f'ndcg_at_10: {v}')` IS the metric output.

    The check searched for the literal `primary_metric`, so any experiment whose
    metric is named after its own measure looked like it printed nothing.
    """
    probs = C(_valid_project(), metric_key="ndcg_at_10")
    assert not any("metric" in p.lower() and "output" in p.lower() for p in probs), probs


def test_metric_output_accepts_config_metric_key():
    """The config's metric_key is equally authoritative."""
    files = _valid_project()
    files["main.py"] = files["main.py"].replace("ndcg_at_10", "accuracy")
    files["evaluator.py"] = files["evaluator.py"].replace("ndcg_at_10", "accuracy")
    probs = C(files, metric_key="accuracy")
    assert not any("output" in p.lower() for p in probs), probs


def test_missing_metric_output_is_still_reported():
    """A main.py that prints nothing recognisable must still fail."""
    files = _valid_project()
    files["main.py"] = (
        "import importlib\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
        "# --algorithm\n"
    )
    probs = C(files, metric_key="ndcg_at_10")
    assert any("output" in p.lower() for p in probs), probs
