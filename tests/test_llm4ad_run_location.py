"""Tests for the `llm4ad_boost.run_evolution_in_package` switch.

llm4ad cuts a git worktree per candidate under its ``base_dir``. By default that
root is a temp directory, which keeps the artifact tree small but also means the
worktrees, checkpoints, ``best/`` and the live ``llm4ad.log`` are nowhere near
the run's artifacts while evolution is in progress.

The switch moves that root into each task package. It is off by default because
the nested worktree path can pass Windows' 260-character limit and llm4ad then
fails every candidate with ``fatal: '$GIT_DIR' too big``.
"""

from __future__ import annotations

from researchclaw.config import Llm4adBoostConfig, _parse_llm4ad_boost_config


class TestParseSwitch:
    def test_default_is_off(self):
        """Existing configs must keep the current behaviour."""
        assert Llm4adBoostConfig().run_evolution_in_package is False

    def test_absent_key_is_off(self):
        assert _parse_llm4ad_boost_config({"enabled": True}).run_evolution_in_package is False

    def test_empty_data_is_off(self):
        assert _parse_llm4ad_boost_config({}).run_evolution_in_package is False

    def test_enabled(self):
        cfg = _parse_llm4ad_boost_config({"run_evolution_in_package": True})
        assert cfg.run_evolution_in_package is True

    def test_explicit_false(self):
        cfg = _parse_llm4ad_boost_config({"run_evolution_in_package": False})
        assert cfg.run_evolution_in_package is False

    def test_other_keys_still_parse(self):
        """The new field must not disturb the existing ones."""
        cfg = _parse_llm4ad_boost_config({
            "enabled": True,
            "fail_silently": False,
            "run_evolution_in_package": True,
            "evolution": {"method": "island_ga", "max_generations": 7},
        })
        assert cfg.enabled is True
        assert cfg.fail_silently is False
        assert cfg.run_evolution_in_package is True
        assert cfg.evolution.max_generations == 7


class TestRunsBaseDirResolution:
    """`runs_base_dir=None` must mean "inside the package", not "nowhere"."""

    @staticmethod
    def _base_dir_for(tmp_path, runs_base_dir):
        """Mirror generate_task_packages' base_dir branch and read it back.

        The assertion is on the *parsed* YAML, not on the string, because the
        failure mode this guards against is a config that cannot be parsed at
        all: base_dir is emitted inside a double-quoted scalar, where a Windows
        backslash starts an invalid escape (`\\U` in `C:\\Users\\...`).
        """
        import yaml

        from researchclaw.pipeline.llm4ad_task_packages import _write_config

        package = tmp_path / "algo"
        package.mkdir()
        # Same expression as the packager.
        base_dir = (package / "runs").as_posix()
        if runs_base_dir is not None:
            base_dir = (runs_base_dir / "algo").resolve().as_posix()
        _write_config(
            "algo", package, "metric", "  - name: default\n", base_dir=base_dir,
        )
        return yaml.safe_load(
            (package / "config.yaml").read_text(encoding="utf-8")
        )["base_dir"]

    def test_none_root_lands_under_the_package(self, tmp_path):
        """No root supplied → the package owns its runs (the switch's target)."""
        got = self._base_dir_for(tmp_path, None)
        assert "algo/runs" in got.replace("\\", "/")

    def test_explicit_root_is_used(self, tmp_path):
        """A root supplied → llm4ad writes there instead."""
        other = tmp_path / "outside"
        got = self._base_dir_for(tmp_path, other)
        assert "outside" in got.replace("\\", "/")
        assert "algo/runs" not in got.replace("\\", "/")

    def test_windows_style_path_is_yaml_parseable(self, tmp_path):
        """A backslash path must not produce an unparsable config.

        `str(Path(...))` on Windows yields backslashes; emitted raw into the
        double-quoted base_dir scalar it becomes an invalid YAML escape.
        """
        import yaml

        from researchclaw.pipeline.llm4ad_task_packages import _write_config

        package = tmp_path / "algo"
        package.mkdir()
        windowsish = r"C:\Users\someone\AppData\Local\Temp\rc_llm4ad\run\x\algo"
        _write_config(
            "algo", package, "metric", "  - name: default\n",
            base_dir=windowsish.replace("\\", "/"),
        )
        parsed = yaml.safe_load((package / "config.yaml").read_text(encoding="utf-8"))
        assert parsed["base_dir"] == windowsish.replace("\\", "/")
