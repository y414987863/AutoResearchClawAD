import json, tempfile, yaml
from pathlib import Path
import researchclaw.pipeline.llm4ad_task_packages as lp


def _exp():
    tmp = Path(tempfile.mkdtemp()); exp = tmp / 'e'
    (exp / 'algorithms' / 'nm').mkdir(parents=True)
    (exp / 'algorithms' / 'nm' / 'nm.py').write_text('def optimize(i,s):\n    # EVOLVE_START\n    return {"f":1.0}\n    # EVOLVE_END\n')
    (exp / 'evaluator.py').write_text('PRIMARY_METRIC="m"\ndef evaluate_instance(i,s):\n    return {"m":1.0}\n')
    (exp / 'main.py').write_text('x=1\n'); (exp / 'data').mkdir(); (exp / 'data' / 'i.json').write_text('{}')
    return exp


def test_token_is_per_call_and_scopes_both_gen_and_collect():
    """Each package-generation call gets a FRESH token, and that same token
    appears in the config base_dir that collection reads back."""
    exp = _exp()
    tmp = Path(tempfile.mkdtemp())

    # Call 1
    out1 = tmp / 'p1'
    tok1 = 'tok_one'
    lp.generate_task_packages(exp, out1, None, None, None, background='t',
                              metric_direction='minimize',
                              runs_base_dir=Path(tempfile.gettempdir())/"rc_llm4ad"/"ab"/f"run_{tok1}",
                              run_id=tok1)
    # Call 2 with a DIFFERENT token
    out2 = tmp / 'p2'
    tok2 = 'tok_two'
    lp.generate_task_packages(exp, out2, None, None, None, background='t',
                              metric_direction='minimize',
                              runs_base_dir=Path(tempfile.gettempdir())/"rc_llm4ad"/"ab"/f"run_{tok2}",
                              run_id=tok2)

    # Config base_dir carries the token -> collection reads it back
    cfg1 = yaml.safe_load((out1/'nm'/'config.yaml').read_text())
    cfg2 = yaml.safe_load((out2/'nm'/'config.yaml').read_text())
    assert 'run_tok_one' in cfg1['base_dir'] and cfg1['run_id'] == 'tok_one'
    assert 'run_tok_two' in cfg2['base_dir'] and cfg2['run_id'] == 'tok_two'
    assert 'run_tok_one' in cfg1['base_dir'] and 'run_tok_one' not in cfg2['base_dir']


def test_two_calls_isolated_end_to_end():
    """Two tokens -> two workspaces; resolve reads only its own."""
    exp = _exp()
    tmp = Path(tempfile.mkdtemp())

    def write_best(base, algo, runid, score):
        w = base / algo / f"{algo}_task" / runid / 'best' / 'code'
        w.mkdir(parents=True, exist_ok=True)
        (w / f"{algo}.py").write_text('# EVOLVE_START\n# EVOLVE_END\n')
        (w.parent / 'metadata.json').write_text(json.dumps({'evaluation': {'score': score}}))

    baseA = Path(tempfile.gettempdir())/"rc_llm4ad"/"ab"/"run_AAA"
    outA = tmp/'a'; lp.generate_task_packages(exp, outA, None, None, None, background='t',
        metric_direction='minimize', runs_base_dir=baseA, run_id='AAA')
    write_best(baseA, 'nm', 'AAA', -99.0)

    baseB = Path(tempfile.gettempdir())/"rc_llm4ad"/"ab"/"run_BBB"
    outB = tmp/'b'; lp.generate_task_packages(exp, outB, None, None, None, background='t',
        metric_direction='minimize', runs_base_dir=baseB, run_id='BBB')
    write_best(baseB, 'nm', 'BBB', -5.0)

    cfgB = yaml.safe_load((outB/'nm'/'config.yaml').read_text())
    rootB = Path(cfgB['base_dir'])
    _, _, sc, _ = lp._resolve_run_best(outB/'nm', 'nm', runs_root=rootB)
    assert sc == -5.0  # not -99 from call A
