from __future__ import annotations

import os
import pathlib
import tempfile
import unittest
from unittest import mock

from nomadic import cli, quantize, runtime


class RuntimePythonTests(unittest.TestCase):
    def test_source_install_uses_current_interpreter(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOMADIC_PYTHON", None)
            self.assertEqual(cli.runtime_python(), cli.sys.executable)

    def test_env_override_wins(self):
        with mock.patch.dict(os.environ, {"NOMADIC_PYTHON": "/bin/sh"}):
            self.assertEqual(cli.runtime_python("mlx_lm"), "/bin/sh")
        with mock.patch.dict(os.environ, {"NOMADIC_PYTHON": "/opt/missing/py"}):
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                cli.runtime_python("mlx_lm")

    def test_frozen_without_runtime_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as home, \
             mock.patch.dict(os.environ, {"NOMADIC_HOME": home, "XDG_DATA_HOME": home, "PATH": home}), \
             mock.patch.object(cli.sys, "frozen", True, create=True):
            os.environ.pop("NOMADIC_PYTHON", None)
            os.environ.pop("UV_TOOL_DIR", None)
            with self.assertRaisesRegex(RuntimeError, "uv tool install mlx-lm"):
                cli.runtime_python("mlx_lm")

    def test_convert_command_is_separate_argv_tokens(self):
        cfg = cli.ServerConfig()
        with mock.patch.object(cli, "is_text_only_model", return_value=True), \
             mock.patch.object(cli, "runtime_python", return_value="/py"):
            self.assertEqual(cfg.server_argv[:4], ["/py", "-m", "mlx_lm", "server"])

    def test_version_is_resolvable(self):
        self.assertRegex(cli.get_version(), r"^\d+\.\d+\.\d+")


def _fake_venv(root: pathlib.Path, module: str) -> pathlib.Path:
    py = root / "bin" / "python"
    (root / "bin").mkdir(parents=True)
    py.write_text("")
    (root / "lib" / "python3.14" / "site-packages" / module).mkdir(parents=True)
    return py


class RuntimeResolutionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        env = {"NOMADIC_HOME": str(self.root / "nh"), "XDG_DATA_HOME": str(self.root / "data"),
               "UV_TOOL_DIR": str(self.root / "tools"), "PATH": str(self.root / "bin")}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        os.environ.pop("NOMADIC_PYTHON", None)
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        frozen = mock.patch.object(runtime.sys, "frozen", True, create=True)
        frozen.start()
        self.addCleanup(frozen.stop)

    def test_finds_uv_tool_environment(self):
        py = _fake_venv(self.root / "tools" / "mlx-lm", "mlx_lm")
        found = runtime.resolve("mlx_lm")
        self.assertEqual((found.python, found.source), (str(py), "uv-tool"))
        with self.assertRaises(runtime.RuntimeNotFound):
            runtime.resolve("mlx_vlm")

    def test_with_extra_package_shares_one_tool_env(self):
        py = _fake_venv(self.root / "tools" / "mlx-lm", "mlx_lm")
        (self.root / "tools" / "mlx-lm" / "lib" / "python3.14" / "site-packages" / "mlx_vlm").mkdir()
        self.assertEqual(runtime.resolve("mlx_vlm").python, str(py))

    def test_finds_environment_via_entry_point_shebang(self):
        py = _fake_venv(self.root / "pipx" / "mlx-lm", "mlx_lm")
        script = self.root / "bin" / "mlx_lm.server"
        script.parent.mkdir()
        script.write_text(f"#!{py}\nprint('x')\n")
        script.chmod(0o755)
        found = runtime.resolve("mlx_lm")
        self.assertEqual((found.python, found.source), (str(py), "path-script"))

    def test_pin_overrides_discovery_and_can_be_cleared(self):
        _fake_venv(self.root / "tools" / "mlx-lm", "mlx_lm")
        pinned = _fake_venv(self.root / "custom", "mlx_lm")
        runtime.pin_python(str(pinned))
        self.assertEqual(runtime.resolve("mlx_lm").source, "pinned")
        runtime.pin_python(None)
        self.assertEqual(runtime.resolve("mlx_lm").source, "uv-tool")

    def test_pinned_venv_without_module_falls_through(self):
        uv_py = _fake_venv(self.root / "tools" / "mlx-vlm", "mlx_vlm")
        runtime.pin_python(str(_fake_venv(self.root / "custom", "mlx_lm")))
        self.assertEqual(runtime.resolve("mlx_vlm").python, str(uv_py))

    def test_runtime_command_reports_and_pins(self):
        py = _fake_venv(self.root / "tools" / "mlx-lm", "mlx_lm")
        self.assertEqual(cli.cmd_runtime("show", None), 0)
        self.assertEqual(cli.cmd_runtime("set", str(py)), 0)
        self.assertEqual(runtime.pinned_python(), str(py))
        self.assertEqual(cli.cmd_runtime("set", "/nope/python"), 2)
        self.assertEqual(cli.cmd_runtime("unset", None), 0)
        self.assertIsNone(runtime.pinned_python())


class ModelConfigLookupTests(unittest.TestCase):
    def test_reads_hf_cache_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = pathlib.Path(tmp) / "models--acme--tiny" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "config.json").write_text('{"hidden_size": 8}')
            quantize._CONFIG_CACHE.clear()
            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": tmp}), \
                 mock.patch.object(quantize, "_hub_download_config", side_effect=AssertionError("network")):
                self.assertEqual(quantize.load_model_config("acme/tiny"), {"hidden_size": 8})
                self.assertTrue(quantize.is_text_only_model("acme/tiny"))

    def test_falls_back_to_hub_http_fetch(self):
        quantize._CONFIG_CACHE.clear()
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.dict(os.environ, {"HF_HUB_CACHE": tmp}), \
             mock.patch.object(quantize, "_hub_download_config", return_value={"vision_config": {}}):
            self.assertFalse(quantize.is_text_only_model("acme/vlm"))


if __name__ == "__main__":
    unittest.main()
