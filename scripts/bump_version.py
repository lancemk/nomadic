#!/usr/bin/env python3
"""Bump the root package version using semantic version rules."""

from __future__ import annotations

import argparse
import pathlib
import re
import tomllib

VERSION_RE = re.compile(r'^(version\s*=\s*")(?P<version>\d+\.\d+\.\d+)(")$', re.MULTILINE)


def parse_version(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def bump_version_string(version: str, bump: str) -> str:
    major, minor, patch = parse_version(version)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unsupported bump kind: {bump}")


def read_project_version(pyproject_path: pathlib.Path) -> str:
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return str(payload["project"]["version"])


def write_project_version(pyproject_path: pathlib.Path, next_version: str) -> None:
    text = pyproject_path.read_text(encoding="utf-8")
    updated, count = VERSION_RE.subn(rf'\g<1>{next_version}\g<3>', text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not update version in {pyproject_path}")
    pyproject_path.write_text(updated, encoding="utf-8")


def bump_project_version(pyproject_path: pathlib.Path, bump: str) -> tuple[str, str]:
    current = read_project_version(pyproject_path)
    next_version = bump_version_string(current, bump)
    write_project_version(pyproject_path, next_version)
    return current, next_version


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump the root pyproject.toml semantic version.")
    parser.add_argument("--pyproject", default="pyproject.toml", help="Path to root pyproject.toml")
    parser.add_argument("--bump", required=True, choices=["major", "minor", "patch"], help="Semantic bump kind")
    args = parser.parse_args()

    pyproject_path = pathlib.Path(args.pyproject).resolve()
    _, next_version = bump_project_version(pyproject_path, args.bump)
    print(next_version)


if __name__ == "__main__":
    main()
