from __future__ import annotations

import os
import pathlib
import tempfile
import unittest
from unittest import mock

from nomadic import paths


def _clean_env(**extra):
    keep = {k: v for k, v in os.environ.items()
            if k not in {"NOMADIC_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"}}
    keep.update(extra)
    return mock.patch.dict(os.environ, keep, clear=True)


class LayoutTests(unittest.TestCase):
    def test_defaults_under_home(self):
        with _clean_env(), mock.patch.object(pathlib.Path, "home", return_value=pathlib.Path("/h")):
            self.assertEqual(paths.config_dir(), pathlib.Path("/h/.config/nomadic"))
            self.assertEqual(paths.data_dir(), pathlib.Path("/h/.local/share/nomadic"))
            self.assertEqual(paths.cache_dir(), pathlib.Path("/h/.cache/nomadic"))
            self.assertEqual(paths.state_dir(), pathlib.Path("/h/.local/state/nomadic"))

    def test_xdg_overrides(self):
        with _clean_env(XDG_CACHE_HOME="/x/cache"):
            self.assertEqual(paths.cache_dir(), pathlib.Path("/x/cache/nomadic"))

    def test_nomadic_home_relocates_everything(self):
        with _clean_env(NOMADIC_HOME="/r", XDG_CACHE_HOME="/ignored"):
            self.assertEqual(paths.config_dir(), pathlib.Path("/r/config"))
            self.assertEqual(paths.data_dir(), pathlib.Path("/r/data"))
            self.assertEqual(paths.cache_dir(), pathlib.Path("/r/cache"))
            self.assertEqual(paths.state_dir(), pathlib.Path("/r/state"))

    def test_explicit_home_arg_keeps_legacy_config_layout(self):
        self.assertEqual(paths.config_dir("/u"), pathlib.Path("/u/.config/nomadic"))

    def test_default_db_is_absolute_and_in_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp, _clean_env(NOMADIC_HOME=tmp):
            cwd = os.getcwd()
            os.chdir(tmp)  # no legacy kernel.sqlite3 here
            try:
                db = paths.default_db_path()
            finally:
                os.chdir(cwd)
            self.assertTrue(db.is_absolute())
            self.assertEqual(db, pathlib.Path(tmp) / "data" / "nomadic.sqlite3")

    def test_legacy_cwd_database_still_honoured(self):
        with tempfile.TemporaryDirectory() as tmp, _clean_env(NOMADIC_HOME=tmp):
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                pathlib.Path("kernel.sqlite3").write_bytes(b"")
                self.assertEqual(paths.default_db_path(), pathlib.Path("kernel.sqlite3"))
            finally:
                os.chdir(cwd)

    def test_quantized_models_go_to_cache_not_config(self):
        with tempfile.TemporaryDirectory() as tmp, _clean_env(NOMADIC_HOME=tmp):
            p = paths.quantized_cache_dir("org/model", 4)
            self.assertEqual(p, pathlib.Path(tmp) / "cache" / "quantized" / "org--model" / "4bit")

    def test_legacy_quantized_copy_reused(self):
        with tempfile.TemporaryDirectory() as tmp, _clean_env(NOMADIC_HOME=tmp):
            legacy = pathlib.Path(tmp) / "config" / "quantized" / "org--model" / "4bit"
            legacy.mkdir(parents=True)
            (legacy / "config.json").write_text("{}")
            self.assertEqual(paths.quantized_cache_dir("org/model", 4), legacy)

    def test_history_migrates_from_config(self):
        with tempfile.TemporaryDirectory() as tmp, _clean_env(NOMADIC_HOME=tmp):
            old = pathlib.Path(tmp) / "config" / "history.txt"
            old.parent.mkdir(parents=True)
            old.write_text("hello\n")
            new = paths.history_path()
            self.assertEqual(new, pathlib.Path(tmp) / "state" / "history.txt")
            self.assertEqual(new.read_text(), "hello\n")


if __name__ == "__main__":
    unittest.main()
