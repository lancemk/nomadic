from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(SCRIPTS_DIR))
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


collect_commits = _load_module("collect_commits_script", "scripts/collect_commits.py")
append_changelog = _load_module("append_changelog_script", "scripts/append_changelog.py")
bump_version = _load_module("bump_version_script", "scripts/bump_version.py")


class PRBranchPolicyTests(unittest.TestCase):
    def test_base_branch_classification(self):
        self.assertEqual(collect_commits.branch_class("main"), "main")
        self.assertEqual(collect_commits.branch_class("develop/ci"), "develop")
        self.assertEqual(collect_commits.branch_class("feature/x"), "other")

    def test_supported_base_branch_gating(self):
        self.assertTrue(collect_commits.is_supported_base_branch("main"))
        self.assertTrue(collect_commits.is_supported_base_branch("develop/release"))
        self.assertFalse(collect_commits.is_supported_base_branch("feature/release"))


class ConventionalCommitTests(unittest.TestCase):
    def test_parse_conventional_commit(self):
        self.assertEqual(collect_commits.parse_conventional_commit("feat: add cli flow"), ("feat", False))
        self.assertEqual(collect_commits.parse_conventional_commit("fix(db): patch sqlite path"), ("fix", False))
        self.assertEqual(collect_commits.parse_conventional_commit("feat!: break api"), ("feat", True))

    def test_skip_ci_commits_are_filtered(self):
        self.assertTrue(collect_commits.should_skip_commit("chore(release): bump version after PR #1 [skip ci]"))
        self.assertFalse(collect_commits.should_skip_commit("feat: add release automation"))

    def test_bump_classification(self):
        self.assertEqual(
            collect_commits.classify_bump(
                [{"commit_type": "fix", "breaking": False}, {"commit_type": "docs", "breaking": False}]
            ),
            "patch",
        )
        self.assertEqual(
            collect_commits.classify_bump(
                [{"commit_type": "feat", "breaking": False}, {"commit_type": "fix", "breaking": False}]
            ),
            "minor",
        )
        self.assertEqual(
            collect_commits.classify_bump(
                [{"commit_type": "feat", "breaking": True}, {"commit_type": "fix", "breaking": False}]
            ),
            "major",
        )


class VersionBumpTests(unittest.TestCase):
    def test_semantic_version_bumps(self):
        self.assertEqual(bump_version.bump_version_string("0.1.0", "patch"), "0.1.1")
        self.assertEqual(bump_version.bump_version_string("0.1.0", "minor"), "0.2.0")
        self.assertEqual(bump_version.bump_version_string("0.1.0", "major"), "1.0.0")

    def test_project_version_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            pyproject_path = pathlib.Path(tmp) / "pyproject.toml"
            pyproject_path.write_text(
                '[project]\nname = "nomadic"\nversion = "1.2.3"\n',
                encoding="utf-8",
            )

            current, next_version = bump_version.bump_project_version(pyproject_path, "minor")

            self.assertEqual(current, "1.2.3")
            self.assertEqual(next_version, "1.3.0")
            self.assertIn('version = "1.3.0"', pyproject_path.read_text(encoding="utf-8"))


class ChangelogHelpersTests(unittest.TestCase):
    def test_append_changelog_is_idempotent_for_same_pr_and_merge(self):
        payload = {
            "base_branch": "main",
            "merge_sha": "abcdef1234567890",
            "pr_number": 42,
            "pr_title": "feat: add release automation",
            "commits": [{"short_sha": "abcdef1", "subject": "feat: add release automation"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            changelog_path = pathlib.Path(tmp) / "CHANGELOG.md"
            changelog_path.write_text(
                "# Changelog\n\nAll notable changes to this project will be documented in this file.\n",
                encoding="utf-8",
            )

            first = append_changelog.update_changelog(
                changelog_path=changelog_path,
                payload=payload,
                version="0.2.0",
                generated_on="2026-05-16",
            )
            second = append_changelog.update_changelog(
                changelog_path=changelog_path,
                payload=payload,
                version="0.2.0",
                generated_on="2026-05-16",
            )
            final_text = changelog_path.read_text(encoding="utf-8")

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertIn("PR: `#42` feat: add release automation", final_text)
        self.assertEqual(final_text.count("nomadic:changelog pr=42 merge=abcdef1234567890"), 1)

    def test_render_section_contains_version_branch_and_commits(self):
        payload = {
            "base_branch": "develop/release",
            "merge_sha": "abcdef1234567890",
            "pr_number": 7,
            "pr_title": "fix: improve packaging",
            "commits": [
                {"short_sha": "abc1234", "subject": "fix: improve packaging"},
                {"short_sha": "def5678", "subject": "chore: adjust docs"},
            ],
        }

        rendered = append_changelog.render_section(payload, version="0.1.2", generated_on="2026-05-16")

        self.assertIn("## [0.1.2] - 2026-05-16", rendered)
        self.assertIn("Target Branch: `develop/release`", rendered)
        self.assertIn("PR: `#7` fix: improve packaging", rendered)
        self.assertIn("- `abc1234` fix: improve packaging", rendered)


class CommitCollectionTests(unittest.TestCase):
    def _git(self, repo_root: pathlib.Path, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def test_collect_pr_commits_respects_existing_marker_and_filters_bot_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            self._git(repo_root, "init", "-b", "main")
            self._git(repo_root, "config", "user.name", "Test User")
            self._git(repo_root, "config", "user.email", "test@example.com")

            tracked = repo_root / "tracked.txt"
            tracked.write_text("a\n", encoding="utf-8")
            self._git(repo_root, "add", "tracked.txt")
            self._git(repo_root, "commit", "-m", "feat: add base release flow")
            merge_sha = self._git(repo_root, "rev-parse", "HEAD")

            (repo_root / "CHANGELOG.md").write_text(
                "# Changelog\n\n"
                "## Automated Release Notes\n\n"
                "Entries for merged PRs into `main` and `develop/*` are appended below by GitHub\n"
                "Actions after successful post-merge validation and semantic version bumping.\n"
                "These sections use stable markers keyed by PR number and merge commit SHA so\n"
                "reruns remain idempotent.\n"
                "<!-- nomadic:changelog-anchor -->\n\n"
                "## [0.1.0] - 2026-05-16\n"
                f"<!-- nomadic:changelog pr=1 merge={merge_sha} version=0.1.0 generated=2026-05-16 -->\n",
                encoding="utf-8",
            )

            payload = collect_commits.collect_pr_commits(
                repo_root=repo_root,
                base_branch="main",
                merge_sha=merge_sha,
                pr_number=1,
                pr_title="feat: add base release flow",
                changelog_path=repo_root / "CHANGELOG.md",
                commit_items=[
                    f"{merge_sha}|feat: add base release flow",
                    "botsha123|chore(release): bump version after PR #1 [skip ci]",
                ],
            )

        self.assertTrue(payload["supported_branch"])
        self.assertTrue(payload["already_recorded"])
        self.assertEqual(payload["bump"], "minor")
        self.assertEqual(payload["commit_count"], 1)


class BundleScriptTests(unittest.TestCase):
    def test_build_bundle_contains_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = pathlib.Path(tmp)
            dist_dir = temp_root / "dist"
            artifacts_dir = temp_root / "artifacts"
            bundle_tmp_dir = temp_root / "bundle-tmp"
            dist_dir.mkdir()
            artifacts_dir.mkdir()
            bundle_tmp_dir.mkdir()

            (dist_dir / "nomadic-0.1.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
            (dist_dir / "nomadic-0.1.0.tar.gz").write_text("sdist", encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "NOMADIC_BUILD_SKIP_PYTHON_BUILD": "1",
                    "NOMADIC_DIST_DIR": str(dist_dir),
                    "NOMADIC_ARTIFACT_DIR": str(artifacts_dir),
                    "NOMADIC_BUNDLE_TMP_DIR": str(bundle_tmp_dir),
                    "NOMADIC_PLATFORM_ID_OVERRIDE": "linux-x86_64",
                }
            )

            subprocess.run(
                ["bash", "scripts/build_bundle.sh"],
                cwd=REPO_ROOT,
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

            archives = sorted(artifacts_dir.glob("nomadic-*.tar.gz"))
            self.assertEqual(len(archives), 1)

            with tarfile.open(archives[0], "r:gz") as tar:
                names = tar.getnames()

        self.assertTrue(any(name.endswith("/README.md") for name in names))
        self.assertTrue(any(name.endswith("/CHANGELOG.md") for name in names))
        self.assertTrue(any(name.endswith("/LICENSE.md") for name in names))
        self.assertTrue(any(name.endswith("/dist/nomadic-0.1.0-py3-none-any.whl") for name in names))
        self.assertTrue(any(name.endswith("/sql/postgres_schema.sql") for name in names))


if __name__ == "__main__":
    unittest.main()
