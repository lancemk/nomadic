from __future__ import annotations

import os
import unittest
from unittest import mock

from nomadic import cli


class RuntimePythonTests(unittest.TestCase):
    def test_source_install_uses_current_interpreter(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOMADIC_PYTHON", None)
            self.assertEqual(cli.runtime_python(), cli.sys.executable)

    def test_env_override_wins(self):
        with mock.patch.dict(os.environ, {"NOMADIC_PYTHON": "/opt/py"}):
            self.assertEqual(cli.runtime_python(), "/opt/py")

    def test_frozen_without_venv_raises_actionable_error(self):
        with mock.patch.dict(os.environ, {"NOMADIC_HOME": "/nonexistent"}), \
             mock.patch.object(cli.sys, "frozen", True, create=True):
            os.environ.pop("NOMADIC_PYTHON", None)
            with self.assertRaisesRegex(RuntimeError, "install.sh --with-mlx"):
                cli.runtime_python()

    def test_convert_command_is_separate_argv_tokens(self):
        cfg = cli.ServerConfig()
        self.assertNotIn(" ", "".join(a for a in cfg.server_argv[1:3]))

    def test_version_is_resolvable(self):
        self.assertRegex(cli.get_version(), r"^\d+\.\d+\.\d+")


if __name__ == "__main__":
    unittest.main()
