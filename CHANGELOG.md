# Changelog

All notable changes to this project will be documented in this file.

## Automated Release Notes

Entries for merged PRs into `main` and `develop/*` are appended below by GitHub
Actions after successful post-merge validation and semantic version bumping.
These sections use stable markers keyed by PR number and merge commit SHA so
reruns remain idempotent.
<!-- nomadic:changelog-anchor -->

## [0.3.0] - 2026-10-07
<!-- nomadic:changelog pr=2 merge=ebbda985f62addb4ec8f092bded7684a3ba66d00 version=0.3.0 generated=2026-10-07 -->

Target Branch: `main`
PR: `#2` feat: standalone CD/CI release pipeline, install.sh, and client file layout (0.2.0)

### Commits
- `e69c2ec` Add multi-agent orchestration, HITL safety review, and task registry
- `f92ee75` Ignore local build scratch and session dump files
- `90fc6cf` Document orchestration, HITL, task registry, and browser tooling
- `8da666e` Fix crash on null model response; add /model selection menu
- `8ef53f0` feat: standalone CD/CI release pipeline, install.sh, and client file layout
- `43d2b0c` Merge branch 'main' into develop


## [0.2.0] - 2026-10-07

Project re-vitalization (minor bump; pre-1.0, so the default database location change ships here with upgrade notes in the README): feature log compacted by capability area, release pipeline reconciled end to end.

### Added
- **Standalone distribution**: tag-driven GitHub Release publishes `nomadic` and `nomadic-db` PyInstaller binaries for `macos-arm64`, `linux-x86_64`, `linux-arm64`, plus `SHA256SUMS` and `install.sh`.
- **`install.sh`**: one-line installer (`curl … | bash`) that resolves the release, verifies the SHA-256, installs to `~/.local/bin`, and optionally (`--with-mlx`) builds a managed MLX runtime venv. Supports `--version`, `--prefix`, `--uninstall`, `--dry-run`, and local-bundle mode.
- **Client-side file layout** (`nomadic/paths.py`, `nomadic paths`): XDG-style config/data/cache/state dirs with a single `NOMADIC_HOME` override; documented in the README.
- **`nomadic --version`**; binaries are version-stamped at build time.
- **CI**: pytest matrix (Ubuntu + macOS), shellcheck, and a binary-smoke job that builds the bundle and installs it via `install.sh`.
- **`pyproject.toml` extras**: `mlx` (Apple Silicon only), `browser`, `dev`; MLX is no longer a hard dependency, so CI and Linux installs work.

### Changed
- **Agent layer (compacted)**: multi-agent `Orchestrator` (`/goal`, `/go`, `/agents`), human-in-the-loop `HITLManager` with safety profiles (`/review`, `/safety`), and a persistent DB-backed `TaskRegistry` (`/tasks`, `/task`).
- **Sessions (compacted)**: `/fork`, `/rewind`/`/undo`, `/rename`, `/export`, `/usage`, `/context`; persistent bounded input history.
- **Workspace & tools (compacted)**: headless Playwright browser tools (`/browser`), `/add-dir`, `/diff`, `/artifacts`, `/btw`; `/model` selection menu and Hugging Face-style `hf://` / `models/` identifiers.
- **Runtime (compacted)**: automatic `mlx_lm` vs `mlx_vlm` server selection for text-only models, unified-memory audit with automatic runtime quantization, inline `key=value` CLI parameters, slash-command autocomplete, and `StatusEvent` progress spinners.
- **Release flow**: merged PRs to `main` now bump the version, append the changelog, create the `v*` tag and dispatch the release build in one chain (previously no tag was ever created, so nothing published). `build.yml` + `release.yml` merged into a single `release.yml`.

### Fixed
- Default SQLite database was `./kernel.sqlite3` relative to the working directory, scattering databases wherever the binary ran; it is now `~/.local/share/nomadic/nomadic.sqlite3` (existing `./kernel.sqlite3` and configured paths still honored).
- Multi-GB quantized models were cached under `~/.config/`, and browser screenshots in the config dir; they now live in the cache and data dirs (legacy copies reused).
- Release matrix used the non-existent `ubuntu-24.04-arm64` runner label (now `ubuntu-24.04-arm`).
- Post-merge workflow interpolated the PR title directly into a shell command (script injection); now passed via environment.
- `mlx_lm convert` / `mlx_vlm convert` was passed as a single argv token and could never run.
- Frozen binaries re-invoked themselves for `serve`; the MLX runtime is now resolved from the managed venv or `$NOMADIC_PYTHON`, with an actionable error when absent.
- Crash on null model response (`'NoneType' object has no attribute 'strip'`) replaced by a clear error; `parse_tool_call` tolerates non-string input.

## [0.1.0] - 2026-05-16

Foundation release (milestones 0-8 baseline):

- **Kernel**: interactive agent REPL, slash commands, multiline input, safe interrupt handling, next-step menu, local `mlx_vlm.server` launcher, one-shot `SKILL.md` extraction.
- **Persistence**: SQLite and Postgres backends, config-driven init, resume by UUID, auto-titling, `/save`, `/chat list`, `/resume`, PWD-aware session filtering.
- **Tools**: `read_skill`, `exec_shell`, `metadata_search`, dynamic tool registry, user tool packs in `~/.config/nomadic/tools`.
- **Context ops**: history compression with VLM-generated memory, `/temp` and `/tokens`, `/preset`, `/profile`, `/template`, image attachment.
- **Hardening**: unit tests, release automation, Apple Silicon baseline.
## [0.1.1] - 2026-05-16
<!-- nomadic:changelog pr=1 merge=ee842d5b9df82f842666b4e2b8d2a02a2ef30796 version=0.1.1 generated=2026-05-16 -->

Target Branch: `main`
PR: `#1` Modify into nomadic/ and Add Postgres Container Middleware; GitHub CI…

### Commits
- `7346b81` Modify into nomadic/ and Add Postgres Container Middleware; GitHub CI/CD and Branch-Aware Release Automation; Refactor GH Workflow Triggers and Post-Merge Versioning


## [0.1.0.0] - 2026-05-16
