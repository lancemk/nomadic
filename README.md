# NOMADIC

**Agent-first REPL for local, Apple Silicon agentic work: tool use, multi-agent orchestration, human-in-the-loop safety review, and durable sessions.**

![alt text](banner.png)

`nomadic` is a terminal kernel where the agent is the primary interface. It talks to a local MLX model server (`mlx_vlm` / `mlx_lm`), gives the model structured tools, records everything in SQLite or Postgres, and lets you run, review, fork, and rewind agent work.

**Requires:** Python 3.12+ (when installing from source).

**Status:** early public OSS / active prototype. **Runtime support:** Apple Silicon (inference). The CLI itself ships for macOS arm64 and Linux x86_64/arm64 (e.g. to point at a remote OpenAI-compatible server); local MLX inference is macOS arm64 only.

## Install

One line, no Python needed (downloads a checksummed binary from the latest GitHub Release):

```bash
curl -fsSL https://raw.githubusercontent.com/lancemk/nomadic/main/install.sh | bash
```

On Apple Silicon, add `--with-mlx` to also create a managed MLX runtime so `nomadic serve` works:

```bash
curl -fsSL https://raw.githubusercontent.com/lancemk/nomadic/main/install.sh | bash -s -- --with-mlx
```

Options: `--version v0.2.0`, `--prefix DIR` (default `~/.local/bin`), `--uninstall`, `--dry-run`. Each release also attaches `SHA256SUMS` and the per-platform `nomadic-<version>-<platform>.tar.gz` bundles if you prefer to install by hand.

From source (development):

```bash
pip install -e ".[dev]"        # core + test/build tooling
pip install -e ".[mlx]"        # local inference runtime (Apple Silicon)
pip install -e ".[browser]"    # optional /browser tool (Playwright)
```

## Quick start

```bash
nomadic serve                                   # terminal 1: local model server
nomadic initialize --db-kind sqlite3 --db-path kernel.sqlite3   # once
nomadic kernel                                  # terminal 2: the agent REPL
nomadic kernel --session-id <uuid>              # resume a session
nomadic --version
```

`python -m nomadic …` and `uv run nomadic …` are equivalent when installed from source.

| Command | Purpose |
|---|---|
| `serve` | Run `mlx_vlm.server` (or `mlx_lm.server` for text-only models, chosen automatically) |
| `kernel` | Interactive agent REPL |
| `extract <SKILL.md>` | One-shot structured summary of a `SKILL.md` (optionally with `--image`) |
| `initialize` | Write `~/.config/nomadic/config.jsonc` and validate SQLite/Postgres |
| `list-skills` | Inventory `SKILL.md` files under a root |

## What you get

**Agent layer**
- `/goal`, `/go`, `/agents`: run parallel background agents against an objective, inspect and cancel them.
- `/review`, `/safety`: human-in-the-loop approval of risky tool calls via configurable safety profiles and per-tool policies.
- `/tasks`, `/task`: persistent, DB-backed task registry (status, assigned agent, artifacts).

**Sessions**
- SQLite or Postgres persistence, resume by id or name, `/save`, `/chat list`, `/fork`, `/rewind`, `/undo`, `/export`, `/usage`, `/context`.
- Automatic context compression with model-written memory summaries.

**Tools and workspace**
- Registry: `read_skill`, `exec_shell`, `metadata_search`, user tool packs in `~/.config/nomadic/tools`, and `browser_*` after `/browser on`.
- `/skill`, `/image`, `/exec`, `/add-dir`, `/diff`, `/artifacts`, `/btw`, `/preset`, `/profile`, `/template`, `/model` (picker), `/temp`, `/tokens`.

**Local runtime**
- Unified-memory audit and on-the-fly quantization (8/4/3/2-bit) so mid-sized models fit on constrained Macs.
- `key=value` inline CLI parameters, Hugging Face `hf://` / `models/` identifiers.

Full command details live in [GUIDE.md](GUIDE.md); history in [CHANGELOG.md](CHANGELOG.md).

## Why Nomadic

- **Local-first by construction.** The default path is an MLX model on your own Mac: no API key, no data leaving the machine.
- **Built-in oversight.** HITL review, safety profiles, and a task registry are part of the kernel, not add-ons. You see and gate what agents do.
- **Orchestration in the box.** Spawn, track, and cancel several background agents from one session.
- **Durable, queryable state.** Sessions, messages, events, and tasks live in a real database (SQLite or Postgres), so work is resumable, forkable, and inspectable with SQL.
- **Apple Silicon awareness.** Memory auditing and automatic quantization decide what can actually run on your hardware.
- **Monitoring-minded.** Usage, context budget, task and agent status are first-class commands.

## What Nomadic is not

- **Not a frontier-model client.** It targets local MLX servers; hosted providers (Anthropic, OpenAI, Google, …) are not a first-class path today.
- **Not cross-platform for inference.** Local serving is macOS arm64 only. Windows and Intel macOS are unsupported.
- **Not a minimal, hackable harness.** It is opinionated and monolithic (one large Python CLI). There is no extension API beyond tool packs and prompt presets.
- **Not battle-tested.** The orchestrator, HITL, and task layers are young; test coverage is thinner than for the core loop.
- **Not a plugin ecosystem.** No package registry, no MCP client, no themes or shareable extensions.
- **Not a model-quality fix.** Small local models are weaker at tool use than hosted ones; Nomadic structures the loop but cannot make a 4-bit local model reason like a frontier model.

## Why pi.dev is preferred (for most people)

[pi](https://pi.dev) is a minimal, extensible agent harness. Honest comparison, based on its public site:

| Dimension | Nomadic | pi.dev |
|---|---|---|
| Philosophy | Batteries-included kernel | Minimal harness; adapt it via primitives |
| Model access | Local MLX servers | 15+ providers (Anthropic, OpenAI, Google, Bedrock, Ollama, custom, …) |
| Platforms | macOS arm64 inference; CLI binaries for Linux | Cross-platform TypeScript tooling |
| Extensibility | Tool packs, presets, templates | TypeScript extensions, skills, prompt templates, themes, npm/git packages |
| Sessions | DB-backed (SQLite/Postgres), fork, rewind | Tree-structured history, `/tree` branching, HTML/gist export |
| Modes | Interactive REPL, one-shot extract | Interactive TUI, print/JSON, RPC, SDK embedding |
| Sub-agents / permissions | Built in (orchestrator, HITL review) | Deliberately omitted; build via extensions |
| Maturity and community | Early prototype, single maintainer | Established, broader ecosystem |

**Prefer pi.dev when** you want hosted or mixed models, cross-platform use, a large extension ecosystem, SDK/RPC embedding, or a mature, minimal core you can reshape.

**Choose Nomadic when** you want a fully local Apple Silicon workflow with orchestration, approval gates, and database-backed task and session state out of the box, and you accept a younger, more opinionated tool.

## CI/CD and releases

| Workflow | Trigger | Does |
|---|---|---|
| `CI` | PRs to `main` / `develop/**`, pushes | pytest on Ubuntu + macOS, package build, shellcheck, binary build + `install.sh` smoke test |
| `Post-Merge Version and Changelog` | PR merged to `main` / `develop/**` | Conventional-commit semver bump, changelog entry; on `main` also tags `vX.Y.Z` and dispatches the release |
| `Release` | `v*` tag or manual dispatch | Tests, builds PyInstaller bundles for `macos-arm64`, `linux-x86_64`, `linux-arm64`, publishes them with `SHA256SUMS` and `install.sh` to a GitHub Release |

Maintainer notes: the post-merge job pushes a version commit and tag to `main`, so branch protection must allow `github-actions[bot]` to push. Release bundles are named `nomadic-<version>-<platform>.tar.gz`. To build locally: `NOMADIC_REQUIRE_BINARIES=1 bash scripts/build_bundle.sh` (needs `pip install -e ".[dev]"`).

## Client-side file layout

A deployed `nomadic` (binary or pip install) writes to a fixed set of per-user locations, following the XDG convention. Run `nomadic paths` to see the resolved values on any machine.

| Kind | Default | Contents |
|---|---|---|
| **config** | `~/.config/nomadic/` | `config.jsonc` (DB target, profiles, presets, templates), `tools/` (user tool packs) |
| **data** | `~/.local/share/nomadic/` | `nomadic.sqlite3` (sessions, messages, events, tasks), `venv/` (MLX runtime from `install.sh --with-mlx`), `screenshots/` (browser tool) |
| **cache** | `~/.cache/nomadic/` | `quantized/<model>/<bits>bit/` runtime-quantized models (regenerable, can be tens of GB) |
| **state** | `~/.local/state/nomadic/` | `history.txt` (REPL input history, bounded by `NOMADIC_HISTORY_SIZE`, default 1000) |
| **binaries** | `~/.local/bin/` | `nomadic`, `nomadic-db` (override with `install.sh --prefix`) |

Not managed by nomadic: downloaded model weights live in the Hugging Face cache (`$HF_HOME`, default `~/.cache/huggingface`). Workspace outputs you request explicitly (`/save`, `/export`, `extract` writing `./evals/`, agent `write_file`) go where you point them or the current directory.

Relocate everything with one variable: `NOMADIC_HOME=/srv/nomadic` gives `config/ data/ cache/ state/` beneath it. Otherwise each kind honors its `XDG_*_HOME` variable. Other overrides: `NOMADIC_PYTHON` (interpreter hosting MLX), `NOMADIC_HISTORY_SIZE`, `--db-path` / `--postgres-dsn` / `--config`.

Upgrade notes: the default SQLite file moved from `./kernel.sqlite3` (working directory) to `~/.local/share/nomadic/nomadic.sqlite3`. A `kernel.sqlite3` already in the working directory, or a path stored in `config.jsonc`, keeps being used. Quantized models already under `~/.config/nomadic/quantized/` are reused, and history is copied from `~/.config/nomadic/history.txt` on first run.

Uninstall: `install.sh --uninstall` removes the binaries and the MLX venv, and leaves config, databases, and caches in place. Delete the directories above to remove everything.

## Session storage

SQLite for local-first use; Postgres for shared or durable setups (`docker compose up -d postgres`, then `nomadic initialize --db-kind postgres --postgres-dsn postgresql://nomadic:nomadic@127.0.0.1:5432/nomadic`). Persistence lives in `nomadic/storage/`. The schema ships inside the package (`nomadic/storage/sql/*.sql`) and in the binary, so `nomadic-db init` needs no external files. Any setting can be overridden per run with flags.

## Roadmap

Completed milestones (0-8 baselines and the agent layer) are tracked in [CHANGELOG.md](CHANGELOG.md). Open work:

- [ ] Tests for orchestrator, HITL, and task registry on par with the kernel loop
- [ ] Hosted/OpenAI-compatible remote provider as a first-class backend
- [ ] Pluggable tool registries and remote execution adapters
- [ ] Retrieval and memory integrations (vector/index, source-aware answers)
- [ ] File-centric workflows beyond `SKILL.md`
- [ ] Signed/notarized macOS binaries; Intel macOS and Windows feasibility
- [ ] Public docs, issue templates, API stability policy

## Philosophy

Local-first is a feature. Agent loops should be inspectable. Persistence matters. Prompting is work, and work deserves structure.

## Appreciation

Built on the shoulders of **MLX**, **mlx-lm**, and **mlx-vlm**, whose work makes local-first agent software practical on Apple Silicon. Thanks also to the pi project for demonstrating how far a minimal harness can go.

## License

Apache 2.0, see [LICENSE.md](LICENSE.md).
