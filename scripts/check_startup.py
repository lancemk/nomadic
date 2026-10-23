#!/usr/bin/env python3
"""Startup-time budget for a built nomadic executable (see specs-public/PGO.md).

Usage: check_startup.py <executable> [--runs N]

Runs ``--version`` and ``--help`` N times each, prints min/median, and exits
non-zero when a median exceeds its budget.  Budgets are generous (CI runners are
noisy) but far below the 0.7-1.3 s a onefile build costs, so a regression to
per-launch extraction or an eager heavy import is caught.

Environment overrides: NOMADIC_STARTUP_VERSION_BUDGET (default 0.75 s),
NOMADIC_STARTUP_HELP_BUDGET (default 1.5 s), NOMADIC_SKIP_STARTUP_CHECK=1.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time


def measure(cmd: list[str], runs: int) -> list[float]:
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True)
        samples.append(time.perf_counter() - start)
        if result.returncode != 0:
            raise SystemExit(f"{' '.join(cmd)} exited {result.returncode}: {result.stderr.decode()[:300]}")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("executable")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if os.environ.get("NOMADIC_SKIP_STARTUP_CHECK") == "1":
        print("startup check skipped")
        return 0

    budgets = {
        "--version": float(os.environ.get("NOMADIC_STARTUP_VERSION_BUDGET", "0.75")),
        "--help": float(os.environ.get("NOMADIC_STARTUP_HELP_BUDGET", "1.5")),
    }
    failed = False
    measure([args.executable, "--version"], 1)  # warm the page cache; first run is not representative
    for flag, budget in budgets.items():
        samples = measure([args.executable, flag], args.runs)
        median = statistics.median(samples)
        verdict = "ok" if median <= budget else "OVER BUDGET"
        print(f"startup {flag:<10} min={min(samples):.2f}s median={median:.2f}s budget={budget:.2f}s {verdict}")
        failed |= median > budget
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
