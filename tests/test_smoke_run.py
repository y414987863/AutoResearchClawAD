import tempfile, sys
from pathlib import Path
from unittest.mock import patch
import researchclaw.pipeline.stage_impls._code_generation as cg


class _S:
    python_path = str(Path(sys.executable))
class _E:
    sandbox = _S()
class _Cfg:
    experiment = _E()
class _ES:
    python_path = ""
class _EE:
    sandbox = _ES()
class _CfgEmpty:
    experiment = _EE()


def _make(code):
    d = Path(tempfile.mkdtemp()) / 'e'; d.mkdir(parents=True)
    (d / 'main.py').write_text(code)
    return d


def test_smoke_healthy():
    assert cg._try_smoke_run(_make('print("primary_metric: 1.0")\n'), _Cfg()) is None


def test_smoke_crash_blocks():
    r = cg._try_smoke_run(_make('raise AssertionError("boom")\n'), _Cfg())
    assert r is not None and 'AssertionError' in r[1]


def test_smoke_timeout_passes():
    def to(args, **kw):
        raise __import__('subprocess').TimeoutExpired(args, 120)
    with patch('subprocess.run', to):
        assert cg._try_smoke_run(_make('x=1\n'), _Cfg()) is None


def test_smoke_empty_python_path():
    assert cg._try_smoke_run(_make('print(1)\n'), _CfgEmpty()) is None
