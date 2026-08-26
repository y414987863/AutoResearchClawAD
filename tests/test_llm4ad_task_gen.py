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


# ---------------------------------------------------------------------------
# Streaming subprocess output
# ---------------------------------------------------------------------------

def test_streaming_delivers_lines_while_the_child_runs(tmp_path):
    """Output must arrive line by line, not all at the end.

    `subprocess.run(capture_output=True)` buffers everything until the process
    exits, so a 40-minute LLM4AD evolution is a black box until it finishes. The
    streaming runner exists so the operator can watch it live; this pins that it
    really is incremental, which is the property that regressed silently when
    the reader was replaced by a buffered one.
    """
    import sys
    import time as _t

    from researchclaw.pipeline.llm4ad_task_packages import _run_streaming

    child = tmp_path / "child.py"
    child.write_text(
        "import sys, time\n"
        "for i in range(4):\n"
        "    print(f'out{i}', flush=True)\n"
        "    print(f'err{i}', file=sys.stderr, flush=True)\n"
        "    time.sleep(0.25)\n",
        encoding="utf-8",
    )

    arrivals: list[tuple[float, str]] = []
    t0 = _t.monotonic()
    log_path = tmp_path / "runs" / "stream.log"
    rc, out, timed_out = _run_streaming(
        [sys.executable, "-u", str(child)],
        cwd=tmp_path,
        env=None,
        timeout_sec=30,
        stream_path=log_path,
        on_line=lambda line: arrivals.append((_t.monotonic() - t0, line.strip())),
    )

    assert rc == 0 and not timed_out
    # 8 lines: stdout and stderr are merged into one stream.
    assert [l for _, l in arrivals] == ["out0", "err0", "out1", "err1",
                                        "out2", "err2", "out3", "err3"]
    # Incremental: the last line arrived well after the first, which cannot
    # happen if everything were returned in one batch at exit.
    assert arrivals[-1][0] - arrivals[0][0] > 0.3, arrivals
    # And the same text was teed to disk as it arrived.
    assert log_path.read_text(encoding="utf-8").splitlines() == [l for _, l in arrivals]
    assert "out0" in out and "err3" in out  # merged capture


def test_streaming_kills_a_hung_child_at_the_deadline(tmp_path):
    """A child that never exits must not park the reader forever.

    Reading the pipe blocks until the child closes it, so enforcing the timeout
    with `Popen.wait` after the read loop would never be reached — the first
    version of this helper ran a 60s child to completion under a 3s budget. The
    watchdog kills the tree at the deadline instead, which closes the pipe.
    """
    import sys
    import time as _t

    from researchclaw.pipeline.llm4ad_task_packages import _run_streaming

    child = tmp_path / "hang.py"
    child.write_text(
        "import time\nprint('started', flush=True)\ntime.sleep(120)\nprint('never')\n",
        encoding="utf-8",
    )

    t0 = _t.monotonic()
    rc, out, timed_out = _run_streaming(
        [sys.executable, "-u", str(child)],
        cwd=tmp_path,
        env=None,
        timeout_sec=3,
        stream_path=tmp_path / "s.log",
    )
    elapsed = _t.monotonic() - t0

    assert timed_out is True
    assert elapsed < 30, f"timeout not enforced (took {elapsed:.1f}s)"
    # Output produced before the kill is still returned, not discarded.
    assert "started" in out
    assert "never" not in out


def test_streaming_reports_a_missing_executable_as_an_exception(tmp_path):
    """A bad command must raise OSError, which the caller reports as FileNotFound."""
    import pytest

    from researchclaw.pipeline.llm4ad_task_packages import _run_streaming

    with pytest.raises(OSError):
        _run_streaming(
            ["definitely-not-a-real-command-xyz"],
            cwd=tmp_path, env=None, timeout_sec=5,
        )


def test_llm4ad_output_is_forwarded_to_the_pipeline_log(tmp_path, caplog):
    """Every line must reach the logger, or the stage looks hung.

    `_run_streaming` already took an `on_line` callback, but the call site never
    passed one — output went only to the tee file. An operator watching a live
    run saw "running LLM4AD evolution ..." and then nothing for the whole
    duration, which reads as a hang, while llm4ad was working normally and its
    artifacts were appearing on disk.
    """
    import logging
    import sys

    from researchclaw.pipeline.llm4ad_task_packages import (
        _forward_llm4ad_line,
        _run_streaming,
    )

    child = tmp_path / "chatty.py"
    child.write_text(
        "import sys, time\n"
        "for i in range(3):\n"
        "    print(f'generation {i} best=0.9{i}', flush=True)\n"
        "    time.sleep(0.2)\n"
        "print()\n"          # blank: dropped, not forwarded as noise
        "print('Y' * 900)\n"  # huge: truncated, not dropped
        "time.sleep(0.2)\n",
        encoding="utf-8",
    )

    caplog.set_level(logging.INFO, logger="researchclaw.pipeline.llm4ad_task_packages")
    with caplog.at_level(logging.INFO, logger="researchclaw.pipeline.llm4ad_task_packages"):
        rc, _out, _to = _run_streaming(
            [sys.executable, "-u", str(child)],
            cwd=tmp_path, env=None, timeout_sec=30,
            stream_path=tmp_path / "s.log",
            on_line=_forward_llm4ad_line,
        )

    assert rc == 0
    forwarded = [r.getMessage() for r in caplog.records
                 if r.name == "researchclaw.pipeline.llm4ad_task_packages"]
    assert any("[llm4ad] generation 0 best=0.90" in m for m in forwarded), forwarded
    assert any("generation 2 best=0.92" in m for m in forwarded), forwarded
    # The blank line produced no record.
    assert not any(m.strip() == "[llm4ad]" for m in forwarded), forwarded
    # The oversized line is bounded rather than dumped whole.
    big = [m for m in forwarded if "Y" * 100 in m]
    assert big and len(big[0]) < 700, [len(m) for m in big]


def test_forwarder_bounds_a_huge_line():
    """A 6.5 KB prompt echo must not swamp the pipeline log."""
    import logging

    from researchclaw.pipeline.llm4ad_task_packages import _forward_llm4ad_line

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    lgr = logging.getLogger("researchclaw.pipeline.llm4ad_task_packages")
    h = _Capture()
    _prev_level = lgr.level
    lgr.setLevel(logging.INFO)   # the forwarder logs at INFO
    lgr.addHandler(h)
    try:
        _forward_llm4ad_line("Z" * 6551 + "\n")
    finally:
        lgr.removeHandler(h)
        lgr.setLevel(_prev_level)

    assert len(records) == 1
    rendered = records[0].getMessage()
    assert len(rendered) < 600, len(rendered)
    assert "more chars" in rendered


def test_forwarder_ignores_blank_lines():
    import logging

    from researchclaw.pipeline.llm4ad_task_packages import _forward_llm4ad_line

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    lgr = logging.getLogger("researchclaw.pipeline.llm4ad_task_packages")
    h = _Capture()
    _prev_level = lgr.level
    lgr.setLevel(logging.INFO)   # the forwarder logs at INFO
    lgr.addHandler(h)
    try:
        _forward_llm4ad_line("\n")
        _forward_llm4ad_line("   \n")
        _forward_llm4ad_line("\ttab indented but empty\n")
    finally:
        lgr.removeHandler(h)
        lgr.setLevel(_prev_level)

    assert len(records) == 1  # only the last one has content
