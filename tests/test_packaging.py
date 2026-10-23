from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run_python(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


class PackageImportTests(unittest.TestCase):
    def test_package_imports_resolve(self):
        package = importlib.import_module("nomadic")
        db_module = importlib.import_module("nomadic.storage.db")
        agent_module = importlib.import_module("nomadic.agent")

        self.assertTrue(hasattr(package, "parse_kernel_input"))
        self.assertTrue(hasattr(db_module, "connect"))
        self.assertTrue(hasattr(agent_module, "build_default_registry"))


class CLICompatibilityTests(unittest.TestCase):
    def test_python_module_help(self):
        result = _run_python("-m", "nomadic", "--help")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("usage:", result.stdout.lower())

    def test_root_nomadic_wrapper_help(self):
        result = _run_python("nomadic.py", "--help")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("usage:", result.stdout.lower())

    def test_root_db_wrapper_help(self):
        result = _run_python("db.py", "init", "--help")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("usage:", result.stdout.lower())


class SchemaParityTests(unittest.TestCase):
    def test_container_postgres_schema_matches_package_schema(self):
        package_schema = (REPO_ROOT / "nomadic" / "storage" / "sql" / "postgres_schema.sql").read_text(encoding="utf-8")
        container_schema = (
            REPO_ROOT / "containers" / "postgres" / "init" / "001_nomadic_schema.sql"
        ).read_text(encoding="utf-8")

        self.assertEqual(_normalize(container_schema), _normalize(package_schema))


if __name__ == "__main__":
    unittest.main()
