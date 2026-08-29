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


class _R:
    returncode = 0
    stdout = ""
    stderr = ""


def test_smoke_single_call_blocks_on_crash():
    d = Path(tempfile.mkdtemp()) / 'e'; d.mkdir(parents=True)
    (d / 'main.py').write_text('raise AssertionError("boom")\n')
    with patch('subprocess.run', return_value=_R()):
        pass  # returncode=0 -> None, gives false pass; so use real subprocess for crash
    r = cg._try_smoke_run(d, _Cfg())
    assert r is not None and 'AssertionError' in r[1]


def test_smoke_reaches_retry_cap():
    # The loop in _execute_code_generation retries up to _smoke_max; here we only
    # assert _try_smoke_run returns a failure (so the caller will loop) rather than
    # silently passing.
    d = Path(tempfile.mkdtemp()) / 'e'; d.mkdir(parents=True)
    (d / 'main.py').write_text('import nonexistent_mod_xyz\n')
    r = cg._try_smoke_run(d, _Cfg())
    assert r is not None and 'ModuleNotFoundError' in r[1]
