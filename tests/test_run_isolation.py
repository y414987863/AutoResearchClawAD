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


def _write_best(base, algo, run_id, score):
    w = base / algo / f"{algo}_task" / run_id / 'best' / 'code'
    w.mkdir(parents=True, exist_ok=True)
    (w / f"{algo}.py").write_text('# EVOLVE_START\n# EVOLVE_END\n')
    (w.parent / 'metadata.json').write_text(json.dumps({'evaluation': {'score': score}}))


def test_run_id_pinned_in_config():
    exp = _exp(); out = Path(tempfile.mkdtemp()) / 'o'
    base = Path(tempfile.gettempdir()) / "rc_llm4ad" / "ab-proj" / "run_TOK"
    lp.generate_task_packages(exp, out, None, None, None, background='t', metric_direction='minimize',
                              runs_base_dir=base, run_id='TOK')
    cfg = yaml.safe_load((out / 'nm' / 'config.yaml').read_text())
    assert cfg['run_id'] == 'TOK'
    assert 'run_TOK' in cfg['base_dir']


def test_resolve_isolated_per_run():
    # RUN1 token=AAAA leaves a stale -99; RUN2 token=BBBB writes -5; resolve RUN2 reads -5
    exp = _exp()
    baseA = Path(tempfile.gettempdir()) / "rc_llm4ad" / "ab-proj" / "run_AAAA"
    out1 = Path(tempfile.mkdtemp()) / 'p1'
    lp.generate_task_packages(exp, out1, None, None, None, background='t', metric_direction='minimize',
                              runs_base_dir=baseA, run_id='AAAA')
    _write_best(baseA, 'nm', 'AAAA', -99.0)

    baseB = Path(tempfile.gettempdir()) / "rc_llm4ad" / "ab-proj" / "run_BBBB"
    out2 = Path(tempfile.mkdtemp()) / 'p2'
    lp.generate_task_packages(exp, out2, None, None, None, background='t', metric_direction='minimize',
                              runs_base_dir=baseB, run_id='BBBB')
    _write_best(baseB, 'nm', 'BBBB', -5.0)

    cfg = yaml.safe_load((out2 / 'nm' / 'config.yaml').read_text())
    root = Path(cfg['base_dir'])
    _, _, sc, _ = lp._resolve_run_best(out2 / 'nm', 'nm', runs_root=root)
    assert sc == -5.0  # reads THIS run, not stale -99
