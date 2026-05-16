#!/usr/bin/env python3
"""Collect merged PR commits and derive semantic version bump metadata."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Iterable

MARKER_RE = re.compile(
    r"<!--\s*nomadic:changelog\s+pr=(?P<pr>\d+)\s+merge=(?P<merge>[0-9a-f]{7,40})\s+version=(?P<version>[0-9]+\.[0-9]+\.[0-9]+)\s+generated=(?P<generated>\d{4}-\d{2}-\d{2})\s*-->"
)

CONVENTIONAL_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\([^)]+\))?(?P<breaking>!)?:\s+(?P<subject>.+)$",
    re.IGNORECASE,
)

ALLOWED_BASE_BRANCHES = ("main",)
DEVELOP_PREFIX = "develop/"


@dataclass(frozen=True)
class CommitEntry:
    sha: str
    short_sha: str
    subject: str
    commit_type: str | None
    breaking: bool


def branch_class(branch_name: str) -> str:
    if branch_name == "main":
        return "main"
    if branch_name.startswith(DEVELOP_PREFIX):
        return "develop"
    return "other"


def is_supported_base_branch(branch_name: str) -> bool:
    return branch_class(branch_name) in {"main", "develop"}


def find_existing_marker(changelog_text: str, pr_number: int, merge_sha: str) -> bool:
    for match in MARKER_RE.finditer(changelog_text):
        if int(match.group("pr")) == pr_number and match.group("merge") == merge_sha:
            return True
    return False


def git(repo_root: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_conventional_commit(subject: str) -> tuple[str | None, bool]:
    match = CONVENTIONAL_RE.match(subject.strip())
    if not match:
        if "BREAKING CHANGE" in subject:
            return None, True
        return None, False
    commit_type = match.group("type").lower()
    breaking = bool(match.group("breaking")) or "BREAKING CHANGE" in subject
    return commit_type, breaking


def should_skip_commit(subject: str) -> bool:
    return "[skip ci]" in subject or subject.startswith("chore(changelog):") or subject.startswith("chore(release):")


def classify_bump(commits: Iterable[dict[str, object]]) -> str:
    highest = "patch"
    for commit in commits:
        if bool(commit.get("breaking")):
            return "major"
        commit_type = str(commit.get("commit_type") or "").lower()
        if commit_type == "feat":
            highest = "minor"
    return highest


def _normalize_commit_subjects(explicit_commits: list[str]) -> list[CommitEntry]:
    rows: list[CommitEntry] = []
    for item in explicit_commits:
        if "|" in item:
            sha, subject = item.split("|", 1)
        else:
            sha = "manual"
            subject = item
        if should_skip_commit(subject):
            continue
        commit_type, breaking = parse_conventional_commit(subject)
        rows.append(
            CommitEntry(
                sha=sha,
                short_sha=sha[:7],
                subject=subject,
                commit_type=commit_type,
                breaking=breaking,
            )
        )
    return rows


def _iter_git_log_rows(repo_root: pathlib.Path, rev_range: str) -> list[CommitEntry]:
    raw = git(repo_root, "log", "--reverse", "--pretty=format:%H%x1f%s", rev_range)
    if not raw:
        return []

    rows: list[CommitEntry] = []
    for line in raw.splitlines():
        sha, subject = line.split("\x1f", 1)
        if should_skip_commit(subject):
            continue
        commit_type, breaking = parse_conventional_commit(subject)
        rows.append(
            CommitEntry(
                sha=sha,
                short_sha=sha[:7],
                subject=subject,
                commit_type=commit_type,
                breaking=breaking,
            )
        )
    return rows


def collect_pr_commits(
    *,
    repo_root: pathlib.Path,
    base_branch: str,
    merge_sha: str,
    pr_number: int,
    pr_title: str,
    changelog_path: pathlib.Path,
    commit_items: list[str] | None = None,
) -> dict[str, object]:
    changelog_text = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
    commits = (
        _normalize_commit_subjects(commit_items)
        if commit_items
        else _iter_git_log_rows(repo_root, f"{merge_sha}^1..{merge_sha}")
    )
    commit_payload = [asdict(entry) for entry in commits]
    return {
        "base_branch": base_branch,
        "branch_class": branch_class(base_branch),
        "supported_branch": is_supported_base_branch(base_branch),
        "merge_sha": merge_sha,
        "pr_number": pr_number,
        "pr_title": pr_title,
        "commit_count": len(commit_payload),
        "commits": commit_payload,
        "bump": classify_bump(commit_payload),
        "already_recorded": find_existing_marker(changelog_text, pr_number, merge_sha),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect merged PR commits for automated changelog/version updates.")
    parser.add_argument("--base-branch", required=True, help="PR base branch name, e.g. main or develop/feature-x")
    parser.add_argument("--merge-sha", required=True, help="Merge commit SHA")
    parser.add_argument("--pr-number", required=True, type=int, help="Merged PR number")
    parser.add_argument("--pr-title", required=True, help="Merged PR title")
    parser.add_argument("--repo-root", default=".", help="Repository root to inspect")
    parser.add_argument("--changelog", default="CHANGELOG.md", help="Path to changelog file")
    parser.add_argument(
        "--commit",
        action="append",
        default=[],
        help="Explicit commit in 'sha|subject' form; repeatable. If omitted, derive from merge commit range.",
    )
    parser.add_argument(
        "--commits-file",
        default=None,
        help="Optional JSON file with commit objects containing `sha` and `subject` keys.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    changelog_path = pathlib.Path(args.changelog).resolve()
    commit_items = list(args.commit or [])
    if args.commits_file:
        raw_items = json.loads(pathlib.Path(args.commits_file).read_text(encoding="utf-8"))
        for item in raw_items:
            commit_items.append(f"{item['sha']}|{item['subject']}")
    payload = collect_pr_commits(
        repo_root=repo_root,
        base_branch=args.base_branch,
        merge_sha=args.merge_sha,
        pr_number=args.pr_number,
        pr_title=args.pr_title,
        changelog_path=changelog_path,
        commit_items=commit_items or None,
    )

    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        pathlib.Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
