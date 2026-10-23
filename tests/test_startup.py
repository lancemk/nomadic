"""Startup-cost guards (see specs-public/PGO.md): keep heavy imports off the fast paths."""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HEAVY = ("requests", "prompt_toolkit", "rich", "psycopg", "pygments", "markdown_it", "huggingface_hub")


def _loaded_after(code: str) -> set[str]:
    """Run *code* in a fresh interpreter and report which HEAVY packages ended up imported."""
    probe = f"import sys\n{code}\nprint('HEAVY:' + ','.join(m for m in {HEAVY!r} if m in sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    marker = next(line for line in out.splitlines() if line.startswith("HEAVY:"))  # skip the code's own output
    return set(filter(None, marker[len("HEAVY:"):].split(",")))


def test_version_fast_path_imports_nothing_heavy():
    assert _loaded_after("from nomadic.entry import main; main(['--version'])") == set()


def test_package_import_is_lazy():
    assert _loaded_after("import nomadic") == set()


def test_cli_import_defers_http_prompt_and_postgres_stacks():
    loaded = _loaded_after("import nomadic.cli")
    assert not loaded & {"requests", "prompt_toolkit", "psycopg", "pygments", "markdown_it"}, loaded


def test_lazy_package_exports_still_resolve():
    import nomadic

    assert callable(nomadic.parse_kernel_input)


def test_version_flag_output_matches_cli_parser():
    fast = subprocess.run(
        [sys.executable, "-m", "nomadic", "--version"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    slow = subprocess.run(
        [sys.executable, "-c", "import sys; from nomadic.cli import main; sys.argv=['nomadic','--version']; main()"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert fast.returncode == 0 and fast.stdout == slow.stdout


def test_nomadic_db_dispatch_by_program_name(tmp_path):
    link = tmp_path / "nomadic-db"
    link.write_text(
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); sys.argv = ['nomadic-db', '--help']\n"
        "from nomadic.entry import main; main()\n"
    )
    result = subprocess.run([sys.executable, str(link)], capture_output=True, text=True)
    assert "usage: nomadic-db" in result.stdout


def test_check_startup_script_enforces_budget(tmp_path):
    fast = tmp_path / "fast"
    fast.write_text("#!/bin/sh\necho ok\n")
    fast.chmod(0o755)
    script = REPO_ROOT / "scripts" / "check_startup.py"
    ok = subprocess.run([sys.executable, str(script), str(fast), "--runs", "2"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    slow = tmp_path / "slow"
    slow.write_text("#!/bin/sh\nsleep 0.3\n")
    slow.chmod(0o755)
    env = {"NOMADIC_STARTUP_VERSION_BUDGET": "0.05", "NOMADIC_STARTUP_HELP_BUDGET": "0.05", "PATH": "/usr/bin:/bin"}
    over = subprocess.run([sys.executable, str(script), str(slow), "--runs", "2"], capture_output=True, text=True, env=env)
    assert over.returncode == 1 and "OVER BUDGET" in over.stdout
