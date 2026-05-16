#!/usr/bin/env python3
"""Append merged-PR changelog entries to CHANGELOG.md."""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import date

from collect_commits import MARKER_RE

ANCHOR = "<!-- nomadic:changelog-anchor -->"


def render_section(
    payload: dict[str, object],
    *,
    version: str,
    generated_on: str | None = None,
) -> str:
    base_branch = str(payload["base_branch"])
    merge_sha = str(payload["merge_sha"])
    pr_number = int(payload["pr_number"])
    pr_title = str(payload["pr_title"])
    commits = list(payload.get("commits", []))
    generated_on = generated_on or date.today().isoformat()

    marker = (
        f"<!-- nomadic:changelog pr={pr_number} merge={merge_sha} "
        f"version={version} generated={generated_on} -->"
    )
    lines = [
        f"## [{version}] - {generated_on}",
        marker,
        "",
        f"Target Branch: `{base_branch}`",
        f"PR: `#{pr_number}` {pr_title}",
        "",
        "### Commits",
    ]
    for entry in commits:
        lines.append(f"- `{entry['short_sha']}` {entry['subject']}")
    lines.append("")
    return "\n".join(lines)


def has_marker(changelog_text: str, pr_number: int, merge_sha: str) -> bool:
    for match in MARKER_RE.finditer(changelog_text):
        if int(match.group("pr")) == pr_number and match.group("merge") == merge_sha:
            return True
    return False


def ensure_anchor(changelog_text: str) -> str:
    if ANCHOR in changelog_text:
        return changelog_text

    intro = (
        "## Automated Release Notes\n\n"
        "Entries for merged PRs into `main` and `develop/*` are appended below by GitHub Actions.\n"
        "Each section is keyed by PR number and merge commit SHA for idempotent reruns.\n"
        f"{ANCHOR}\n\n"
    )
    if "\n## " in changelog_text:
        return changelog_text.replace("\n## ", f"\n{intro}## ", 1)
    return changelog_text.rstrip() + "\n\n" + intro


def append_section(changelog_text: str, section_text: str) -> str:
    changelog_text = ensure_anchor(changelog_text)
    anchor_block = f"{ANCHOR}\n"
    return changelog_text.replace(anchor_block, f"{ANCHOR}\n\n{section_text}\n", 1)


def update_changelog(
    *,
    changelog_path: pathlib.Path,
    payload: dict[str, object],
    version: str,
    generated_on: str | None = None,
) -> bool:
    changelog_text = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else "# Changelog\n\n"
    pr_number = int(payload["pr_number"])
    merge_sha = str(payload["merge_sha"])
    commits = list(payload.get("commits", []))

    if has_marker(changelog_text, pr_number, merge_sha) or not commits:
        return False

    section = render_section(payload, version=version, generated_on=generated_on)
    updated = append_section(changelog_text, section)
    changelog_path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Append merged-PR release notes to CHANGELOG.md.")
    parser.add_argument("--changelog", default="CHANGELOG.md", help="Path to changelog file")
    parser.add_argument("--commits-json", required=True, help="Path to commit payload JSON")
    parser.add_argument("--version", required=True, help="Version string to associate with this entry")
    parser.add_argument("--date", default=None, help="Optional YYYY-MM-DD override")
    args = parser.parse_args()

    changelog_path = pathlib.Path(args.changelog).resolve()
    payload = json.loads(pathlib.Path(args.commits_json).read_text(encoding="utf-8"))
    changed = update_changelog(
        changelog_path=changelog_path,
        payload=payload,
        version=args.version,
        generated_on=args.date,
    )
    print("updated" if changed else "no-op")


if __name__ == "__main__":
    main()
