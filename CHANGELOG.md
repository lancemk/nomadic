# Changelog

All notable changes to this project will be documented in this file.

## Automated Release Notes

Entries for merged PRs into `main` and `develop/*` are appended below by GitHub
Actions after successful post-merge validation and semantic version bumping.
These sections use stable markers keyed by PR number and merge commit SHA so
reruns remain idempotent.
<!-- nomadic:changelog-anchor -->

## [Unreleased]

### Added
- **Multi-Agent Orchestration**: New `Orchestrator`/`AgentInstance` model (`nomadic/orchestrator.py`) for spawning, tracking, and cancelling parallel background agents from within a session, surfaced via `/goal`, `/go`, `/agents`, and `/tasks`.
- **Human-in-the-Loop Safety Review**: New `HITLManager` (`nomadic/hitl.py`) with configurable `SafetyProfile`s and per-tool `ReviewPolicy` matching, so risky tool calls can require explicit approval before executing. Exposed via `/review` and `/safety`.
- **Persistent Task Registry**: New DB-backed `TaskRegistry`/`Task` model (`nomadic/tasks.py`, `nomadic/sql/tasks.sql`) tracking objective status, assigned agent, and generated artifacts across a session, inspectable via `/tasks` and `/task`.
- **Session Time-Travel**: `/fork` duplicates the active session under a new ID; `/rewind` / `/undo` revert recent history from the database.
- **Headless Browser Tool**: `/browser on` launches a headless Playwright Chromium instance and registers `browser_navigate`, `browser_screenshot`, and `browser_extract_text` as agent tools (requires the optional `playwright` package).
- **Workspace & Developer Ergonomics**: `/add-dir` expands the agent's allowed workspace boundary at runtime; `/diff` renders a unified git diff of the active workspace; `/artifacts` inspects files agents have generated; `/context` shows a token-budget breakdown; `/usage` and `/export` surface session stats and serialize sessions out.
- **Standalone CLI Binaries**: `scripts/build_bundle.sh` now compiles `nomadic` and `nomadic-db` into onefile PyInstaller executables (new `nomadic_bin.py` entrypoint) when PyInstaller is available, with `install.sh` installing the binaries directly instead of requiring a Python environment. `Build Bundles` CI now also targets `linux-arm64`.
- **Dynamic Runtime Quantization & Resource Auditor**: Integrated macOS unified memory auditing (`sysctl hw.memsize`) and model parameter estimation (from configuration metadata or repository name pattern matching) to dynamically calculate inference RAM limits. Auto-decides the optimal bit-width (8, 4, 3, 2-bit) and invokes `mlx_lm convert`/`mlx_vlm.convert` on the fly to quantize dense mid-sized models (e.g., 36B) on constrained edge devices like 48GB unified memory MacBooks.
- **Inline Key-Value CLI Parameters**: Added support for passing parameters in `key=value` format directly to the nomadic commands at invocation (e.g. `model=mlx-community/Qwen2.5-32B-Instruct`), bypassing standard argparse restrictions while performing strict attribute validation.
- **Hugging Face CLI Compatible Model Identifiers**: Enabled support for model identifiers containing `hf://` or `models/` prefixes (matching `huggingface-cli` or `hf` formats), automatically normalizing them to standard repo IDs for local caching and server/inference routines.
- **Interactive Slash Command Autocomplete**: Implemented dynamic, real-time dropdown menu in the REPL for slash commands using `prompt_toolkit`'s `WordCompleter` and float containers.
- **Persistent Input History**: Added `BoundedFileHistory` allowing users to navigate previous inputs across sessions using the up-arrow key. Automatically truncates to 1000 items (configurable via `NOMADIC_HISTORY_SIZE`), saving to `~/.config/nomadic/history.txt`.
- **Status Feedback Layer**: Introduced intermediate state indicators (e.g., *Thinking...*, *Reading codebase...*, *Running tests...*). Emits granular `StatusEvent`s from `AgentRuntime` that render as transient UI spinners via `rich.status` without polluting the persistent chat log.
- **`/model` Selection Menu**: `/model` with no arguments (or `/model select`) now lists known model candidates with their caveats and the currently active one, and accepts a list number as a shorthand for `/model <repo-id>`.

### Fixed
- **Crash on Null Model Response**: `/model` only updates client-side config — the running `serve` process keeps whatever model it was started with until restarted. Sending a request against a mismatched model (or one that returns `tool_calls` instead of text content) previously crashed the kernel with `VLM error: 'NoneType' object has no attribute 'strip'` deep inside tool-call parsing. `vlm_complete` now detects a null/missing `content` field and raises a clear, actionable error instead, and `agent.parse_tool_call` no longer crashes if it's ever handed a non-string reply.
- **Dynamic MLX Server Selection**: Resolved startup crash when attempting to serve text-only language models that have vestigial vision configs (like `Qwen-AgentWorld-35B-A3B-bf16`). The launcher now automatically detects whether a model is text-only (using its config's `language_model_only` attribute or lack of vision configs) and runs `mlx_lm` instead of `mlx_vlm`.

## [0.1.0.0] - 2026-05-16

### Implemented Features (Transferred from README Roadmap)

#### Milestone 0: Foundation
- Core Python CLI structure
- Local `mlx_vlm.server` launcher
- Interactive kernel command
- One-shot `SKILL.md` extraction flow
- Basic local tool-calling loop

#### Milestone 1: Agent REPL Baseline
- REPL-style terminal interaction
- Built-in slash commands
- Multiline input support when `prompt_toolkit` is installed
- Interrupt handling with safe kernel return
- Suggested next-step menu after responses

#### Milestone 2: Session Persistence
- Persistent chat session model
- Session/message/event schema
- Resume by session UUID
- Auto-title from first meaningful user turn
- SQLite backend
- Postgres backend
- Config-driven database initialization
- **Advanced Session Management**:
    - DB-backed `/save <name>` to name/rename sessions in database
    - `/chat list` to inventory sessions associated with the current working directory
    - `/chat resume <id|name>` and `/resume <id|name>` for in-kernel session switching
    - PWD tracking in session metadata for project-aware session filtering

#### Milestone 3: Local Tooling Layer
- `read_skill`
- `exec_shell`
- `metadata_search`
- richer tool registry management (dynamic registration/unregistration)
- user-defined local tool packs (load tools from `~/.config/nomadic/tools`)

#### Milestone 4: Prompting and Context Ops
- history compression
- runtime prompt controls for temperature and token budget
- reusable prompt presets via `/preset` command
- named workspaces and profiles via `/profile` command and `--profile` flag
- prompt templates for common agent tasks via `/template` command
- better session summarization and memory shaping (VLM-driven semantic memory generation during history compression)

#### Milestone 5: Personal Agent Workspace
- project-aware working directories (via PWD metadata tracking and `/chat list`)

#### Milestone 6: Multimodal and Retrieval Expansion
- image attachment to user turns

#### Milestone 7: OSS Hardening
- basic test coverage
- release automation

#### Milestone 8: Platform Expansion
- Apple Silicon baseline
