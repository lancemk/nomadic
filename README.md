# NOMADIC

**Agent First REPL for personal agentic prompting, local tool use, and session management.**

![alt text](banner.png)

`nomadic` is an all-in-one CLI layer for running a local multimodal agent loop on Apple Silicon. It is built around an interactive REPL-style kernel where the agent is the primary interface, not an afterthought bolted onto a script runner.

The project combines:

- an interactive agent-first kernel
- local multimodal prompting through `mlx_vlm.server`
- structured tool-calling for file reading, shell execution, and metadata search
- persistent session storage in SQLite or Postgres
- lightweight prompt, context, and conversation management in one place

This repo is still early, but the direction is clear: `nomadic` aims to become a practical personal working layer for local-first agent workflows.

**Current platform scope:** `nomadic` currently supports **Apple Silicon only**. Other platforms are part of the roadmap, but they are not supported targets today.

## What Nomadic Is

`nomadic` treats the terminal like a personal agent workspace.

Instead of juggling separate scripts, temporary prompts, shell snippets, and half-saved notes, you enter a persistent REPL where you can:

- prompt the agent directly
- load `SKILL.md` files into context
- attach images to the next turn
- run local commands and feed results back into the conversation
- keep sessions resumable and queryable
- manage context growth without losing the thread

The result is a CLI that feels closer to a personal agent operating console than a plain chatbot wrapper.

## Project Positioning

`nomadic` is designed as an **Agent First** CLI:

- the conversation is the main primitive
- tools are native to the loop
- prompts are stateful and resumable
- sessions are durable
- configuration is local and operator-friendly
- the terminal remains fast, inspectable, and scriptable

If you want a local, personal, transparent layer for agentic work, this is the kind of tool `nomadic` is trying to become.

## Current Capabilities

Today, the project already includes:

- `serve`: run `mlx_vlm.server` locally
- `kernel`: launch the interactive agent REPL
- `extract`: summarize a `SKILL.md` into structured context
- `initialize`: configure SQLite or Postgres session storage
- `list-skills`: inventory `SKILL.md` files under a directory

Inside `kernel`, you currently get:

- `/skill <path>` to load a `SKILL.md`
- `/image <path>` to attach an image to the next turn
- `/exec <cmd>` to run a shell command and return output to context
- `/history` to inspect prior turns
- `/config` to inspect current runtime settings
- `/session` to inspect persisted session metadata
- `/model <name>` to switch model config
- `/temp <0.0-1.0>` to change temperature
- `/tokens <n>` to change response budget
- `/clear` to clear the active history
- `/save <file>` to export the conversation as JSON

The default structured tool registry currently includes:

- `read_skill`
- `exec_shell`
- `metadata_search`

## Why This Exists

The gap `nomadic` is trying to close is simple:

Most local LLM tooling still makes you do too much glue work yourself.

You end up with:

- one script for inference
- another for retrieval
- raw shell commands for context gathering
- ad hoc prompt files
- no durable session model
- no honest agent loop in the terminal

`nomadic` tries to collapse that into one coherent surface.

## Requirements

Current practical target environment:

- macOS on Apple Silicon
- Python 3.14+
- `mlx-vlm`
- `requests`
- local model artifacts compatible with the selected MLX VLM server flow

The project metadata is still evolving, so for now the safest install path is to install runtime dependencies explicitly.

## Platform Support

Current supported platform:

- macOS on Apple Silicon

Current non-goals for the present release line:

- Windows support
- Intel macOS support
- Linux support

Those platforms may be supported later, but the current project should be evaluated and used as an Apple Silicon-first tool.

## Install

Basic local setup:

```bash
pip install -e .
pip install mlx-vlm requests
```

If you want richer terminal UX:

```bash
pip install rich prompt-toolkit
```

## Quick Start

Start the local server in one terminal:

```bash
python -m nomadic serve
```

Initialize persistent storage once:

```bash
python -m nomadic initialize --db-kind sqlite3 --db-path kernel.sqlite3
```

Then launch the agent REPL:

```bash
python -m nomadic kernel
```

Typical one-off overrides:

```bash
python -m nomadic kernel --model mlx-community/Pixtral-12B-2409-4bit
python -m nomadic kernel --db-kind postgres --postgres-dsn postgresql://user:pass@host/db
python -m nomadic kernel --max-context-tokens 12000 --context-recompress-to 8000
```

Resume an existing session:

```bash
python -m nomadic kernel --session-id <uuidv4>
```

The legacy root scripts `nomadic.py` and `db.py` are still kept as compatibility wrappers, but the package-first entrypoints above are now the preferred interface.

## Command Surface

### `serve`

Runs the bundled `mlx_vlm.server` flow with inline config.

```bash
python -m nomadic serve
```

### `kernel`

Launches the interactive Agent First REPL.

```bash
python -m nomadic kernel
```

### `extract`

One-shot extraction for a `SKILL.md` file.

```bash
python -m nomadic extract /path/to/SKILL.md
python -m nomadic extract /path/to/SKILL.md --image /path/to/diagram.png
```

### `initialize`

Writes `~/.config/nomadic/config.jsonc` and validates the selected database.

```bash
python -m nomadic initialize --db-kind sqlite3 --db-path kernel.sqlite3
python -m nomadic initialize --db-kind postgres --postgres-dsn postgresql://user:pass@host/db
```

### `list-skills`

Scans a root directory for `SKILL.md` files.

```bash
python -m nomadic list-skills --root /mnt/skills
```

## Session Storage

`nomadic` supports both:

- SQLite for simple local-first session persistence
- Postgres for more durable or shared session backends

For local Postgres middleware, the repo now includes a dedicated Compose service and init SQL under `containers/postgres/`:

```bash
docker compose up -d postgres
python -m nomadic initialize --db-kind postgres --postgres-dsn postgresql://nomadic:nomadic@127.0.0.1:5432/nomadic
```

The config-backed default lives at:

```text
~/.config/nomadic/config.jsonc
```

You can still override the backend per run with CLI flags.

## CI/CD and Releases

The repository now includes a GitHub Actions release layer built around four workflows:

- `CI`: validate imports, unit tests, and package buildability for PRs into `main` and `develop/*`
- `Build Bundles`: create macOS and Linux release archives as GitHub Actions artifacts for tags
- `Post-Merge Version and Changelog`: bump semantic version and append changelog entry after merged PRs into `main` and `develop/*`
- `Publish Release`: attach built archives to a GitHub Release when a version tag is pushed

Branch behavior:

- PRs targeting `main` and `develop/*` run validation without mutating tracked files
- only merged PRs into `main` and `develop/*` can mutate `pyproject.toml` and `CHANGELOG.md`
- tags matching `v*` trigger release publication

Release bundle behavior:

- artifacts are produced for `macos-arm64` and `linux-x86_64`
- the bundle name format is `nomadic-<version>-<platform>.tar.gz`
- each bundle includes the installable package artifacts, docs, license file, changelog, and SQL assets

Linux artifacts are currently provided as CI/release deliverables for packaging and roadmap preparation. They do not change the current product support boundary: Apple Silicon remains the only supported runtime target today.

## Context Management

The REPL includes automatic context compression.

That means:

- the system prompt is preserved
- recent turns stay verbatim as long as possible
- older turns are compressed into a memory summary when the budget is exceeded

This is intentionally approximate and tokenizer-light, which keeps the loop practical for a local terminal workflow.

## Roadmap

The scope of `nomadic` is broader than the current implementation. Below is the working milestone map, with a clear split between what exists now and what comes next.

### Milestone 0: Foundation

- [x] Core Python CLI structure
- [x] Local `mlx_vlm.server` launcher
- [x] Interactive kernel command
- [x] One-shot `SKILL.md` extraction flow
- [x] Basic local tool-calling loop

### Milestone 1: Agent REPL Baseline

- [x] REPL-style terminal interaction
- [x] Built-in slash commands
- [x] Multiline input support when `prompt_toolkit` is installed
- [x] Interrupt handling with safe kernel return
- [x] Suggested next-step menu after responses

### Milestone 2: Session Persistence

- [x] Persistent chat session model
- [x] Session/message/event schema
- [x] Resume by session UUID
- [x] Auto-title from first meaningful user turn
- [x] SQLite backend
- [x] Postgres backend
- [x] Config-driven database initialization

### Milestone 3: Local Tooling Layer

- [x] `read_skill`
- [x] `exec_shell`
- [x] `metadata_search`
- [ ] richer tool registry management
- [ ] tool permissions and safety profiles
- [ ] user-defined local tool packs

### Milestone 4: Prompting and Context Ops

- [x] history compression
- [x] runtime prompt controls for temperature and token budget
- [ ] reusable prompt presets
- [ ] named workspaces and profiles
- [ ] prompt templates for common agent tasks
- [ ] better session summarization and memory shaping

### Milestone 5: Personal Agent Workspace

- [ ] project-aware working directories
- [ ] persistent task boards / objectives
- [ ] artifact tracking across sessions
- [ ] file-centric workflows beyond `SKILL.md`
- [ ] tighter personal knowledge ingestion

### Milestone 6: Multimodal and Retrieval Expansion

- [x] image attachment to user turns
- [ ] richer vision workflows
- [ ] retrieval adapters beyond the current metadata service
- [ ] local vector / index integrations
- [ ] source-aware answer synthesis

### Milestone 7: OSS Hardening

- [x] basic test coverage
- [ ] package metadata polish
- [ ] public docs expansion
- [ ] contributor workflow and issue templates
- [ ] release automation
- [ ] API stability and semantic versioning discipline

### Milestone 8: Platform Expansion

- [x] Apple Silicon baseline
- [ ] Linux support investigation
- [ ] Intel macOS feasibility review
- [ ] Windows support investigation
- [ ] platform-specific runtime and packaging strategy
- [ ] cross-platform test matrix

### Milestone 9: Agent Platform Direction

- [ ] pluggable tool registries
- [ ] multi-agent orchestration patterns
- [ ] remote execution adapters
- [ ] richer session analytics and evaluation
- [ ] portable agent workspace format

## What Is Actually Done Today

The project is still early-stage.

The completed work is mostly in:

- the terminal kernel loop
- the structured local tool-calling foundation
- `SKILL.md` ingestion
- session persistence
- database initialization and backend selection
- context compression
- basic tests around kernel input, metadata search, and DB behavior

That is a meaningful base, but it is still much closer to **a strong prototype and working personal tool** than a finished general-purpose agent platform.

## Near-Term Next Features

The most natural next steps from the current state are:

1. Package polish and public OSS hygiene.
2. Contribution, security, and release docs.
3. Expanded tool registry and safer tool execution controls.
4. More robust prompt presets and reusable workspace profiles.
5. Better retrieval and memory integrations.
6. Improved export, artifact, and task-management workflows.

## Philosophy

`nomadic` is opinionated in a few ways:

- local-first is a feature, not a fallback
- terminal workflows should still feel premium
- agent loops should be inspectable
- persistence matters
- prompting is work, and work deserves structure

## Appreciation

This project stands on the shoulders of excellent open-source work.

Special appreciation goes to the teams and contributors behind:

- **MLX**
- **MLX_VM**
- **MLX_VLM**

Their work has materially expanded what local-first AI workflows can look like on Apple Silicon. Projects like `nomadic` are only practical because that ecosystem keeps pushing inference, model serving, and developer ergonomics closer to everyday use. The open-source contribution here is not just technical plumbing, it is enabling a whole category of personal agent software to exist in a lightweight, local, and inspectable form.

## Project Status

Current status: **early public OSS / active prototype**.

Current support boundary: **Apple Silicon only**.

Expect:

- working flows
- uneven edges
- active iteration
- changing interfaces

## Documentation

For deeper command and usage details, see [GUIDE.md](GUIDE.md).

## License

This project should ship under Apache 2.0 for public OSS release, with the full license text in `LICENSE`.
