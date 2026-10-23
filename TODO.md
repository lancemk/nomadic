# TODO

Status of the code against [specs-public/SRS-v0.3.md](specs-public/SRS-v0.3.md), plus open README roadmap items. The SRS is kept verbatim as the record; its header still says every FC is "planned", so this file is the live status.

Legend: **[x]** done and covered by tests (`tests/test_srs_v03.py`, `tests/test_srs_ext*.py`), **[~]** partly done, **[ ]** open. Items marked *unverified* were not exercised by tests or a run.

## Done in the reconcile pass

- [x] **Storage restructure:** `nomadic/db.py` and `nomadic/sql/` moved to `nomadic/storage/` (`storage/db.py`, `storage/sql/`). Entry point is now `nomadic.storage.db:main`; the root `db.py` stays as a thin wrapper. Stale root duplicates (`agent.py`, `sqlite_schema.sql`, `search/`) removed.
- [x] **§6 `agent_states` table** (SQLite + Postgres) with `upsert_agent_state` / `get_agent_state` / `list_agent_states`; `AgentInstance.to_state()/from_state()`, `persist()`, and `Orchestrator.restore_agents()` (re-creates pending HITL reviews after a restart).
- [x] **§6 Python version:** `requires-python` aligned to the spec's 3.12+ (was 3.14); CI now tests 3.12 and 3.14.
- [x] **FC-10:** FS tools moved to `nomadic/tools/fs.py`. Streaming line-range reads (large-file safe), unique-target `replace_content` (or `replace_all`), AST-aware `symbol` edits for `.py` files with a syntax check before writing, `list_dir` skips `.git`/`__pycache__`/`node_modules` and has a `recursive` flag.
- [x] **FC-12:** `nomadic/context.py` resolves `@path` / `@"quoted path"` relative to PWD, strips trailing punctuation, dedupes, skips binary files and directories, truncates per file (`max_skill_chars`) and caps the total at half of `max_context_tokens`. Wired into the REPL turn.
- [x] **FC-17:** `nomadic/status.py` `StatusBus` (thread-safe, bounded); `AgentRuntime.emit()`; each tool maps to its spec state (READING / WRITING / EDITING / EXECUTING); `REVIEWING` is now emitted when HITL blocks an agent; `THINKING` is emitted per step.
- [x] **FC-4:** file tools record real artifacts (`created` vs `modified`) on the task through the registry (persisted). The old heuristic that stored shell commands as artifacts is gone.
- [x] **FC-2 (partial, see below):** agent-to-agent hand-off (`Orchestrator.handoff` / `run_handoff`, `/agents handoff`), LLM calls and tool execution run on the orchestrator's thread pool, `run_blocking(..., use_process=True)` for process-pool jobs.
- [x] **FC-13/14/15/16:** pipeline extracted to `nomadic/quantize.py`. KV-cache term (`2·L·H_kv·D·ctx·2B`, GQA-aware) now feeds Q selection instead of a fixed 4 GB; HF token honoured for gated repos; converter verified before running; managed async subprocess streams log lines as `StatusEvent`s; failed runs delete the partial output.
- [x] **Bug fixes found on the way:** `/agents spawn` scheduled its task on a loop that never ran (background agents never started); `/settings`, `/switch`, `/undo` only printed a warning (aliases now resolve to `/config`, `/resume`, `/rewind`); orchestrator agents crashed on `read_file` because `AgentRuntime.state` was `None`; a paused-for-review agent had its task marked COMPLETED.

## Release management / startup performance (see [specs-public/PGO.md](specs-public/PGO.md))

Baseline completed: root cause of the slow post-install load was PyInstaller `--onefile` re-extracting ~114 MB on every launch, plus `--version` importing the whole CLI. Measured on Linux x86_64: `--version` 1.29 s → 0.10 s, `--help` 1.42 s → 0.25 s.

- [x] Frozen as onedir (`app/` + `bin/` symlinks); one app serves `nomadic` and `nomadic-db` (`nomadic/entry.py`); installer swaps `~/.local/share/nomadic/app` atomically.
- [x] `--version` fast path, lazy package exports, deferred `requests` / `prompt_toolkit` / `rich.markdown` / `rich.syntax` (`import nomadic.cli` 421 ms → ~150 ms); guarded by `tests/test_startup.py`.
- [x] Startup budget gate: `scripts/check_startup.py`, run in `build_bundle.sh` and CI `binary-smoke` on the installed copy.
- [x] Pre-release channels: tags with a pre-release segment publish as GitHub pre-releases; `install.sh --channel rc|staging`; PEP 440 tag/version check; installer reads the asset name from `SHA256SUMS`.
- [ ] Measure on `macos-14` (cold and warm) and record in PGO.md; confirm or refute per-launch scanning of onefile extractions.
- [ ] Developer ID signing + notarisation of `app/` on macOS.
- [ ] Relative regression gate (fail on >25 % slower than last release) and `$GITHUB_STEP_SUMMARY` publishing of startup numbers.
- [ ] Defer remaining eager imports for non-REPL commands (`rich.console`, `nomadic.agent` → `search.metadata` → `urllib.request`) to bring frozen `--help` under 0.2 s.
- [ ] Pilot Nuitka standalone (`--pgo-c`) against PyInstaller onedir; verify the CI interpreter was built with PGO/LTO.
- [ ] Sign checksums / attest provenance so `install.sh` verifies origin, not just integrity.

- [x] **MLX runtime discovery:** `nomadic/runtime.py` locates `mlx_lm` / `mlx_vlm` per module (env, pin file, managed venv, `uv tool` envs, PATH entry-point shebangs); `nomadic runtime [set|unset]`; actionable `uv tool install` errors. Model `config.json` is now read from the HF cache or the Hub over HTTP (`huggingface_hub` was never a dependency, so text models were mis-detected as VLMs in the binary).

## Open gaps

### Orchestration core
- [ ] **FC-1 / M1:** the normal REPL turn still goes through the synchronous `agent.run_agent_loop`; only `/goal` and `/agents` use `AgentInstance.step()`. Route REPL turns through `AgentInstance`.
- [ ] **FC-2:** the REPL is not an async kernel (spec §3 `AK`); background agents run on daemon threads with their own loops. Tool execution uses the thread pool only, not the process pool.
- [ ] **FC-2:** hand-off is orchestrator/command driven; the model cannot yet delegate by emitting a `handoff` tool call.

### Commands and TUI (FC-5 – FC-9)
- [ ] **FC-6:** `/agents` and `/tasks` are text panels, not live TUI dashboards.
- [ ] **FC-5:** `/goal` honours the safety profile through HITL; no "non-critical tool calls skip review" classification beyond the profile's policies (*unverified* against the spec's intent).
- [ ] **FC-8:** `/rewind` deletes messages/events but does not restore `agent_states` or undo tool side effects; `/fork` copies messages, workspace state is *unverified*.
- [ ] **FC-9:** `/usage` shows counts only (no quota/rate-limit tracking); `/context` checkpoint management *unverified*.

### Status feedback (FC-17)
- [ ] The REPL still renders via the `start_status` callback; nothing drains `StatusBus` yet and there is no `prompt_toolkit` toolbar renderer. The REPL turn path (`run_agent_loop`) does not publish to the bus.

### Quantization (FC-15)
- [ ] Cache layout is `quantized/<repo with / → -->/<Q>bit/` (documented in the README, legacy copies reused), not the spec's `<repo_id>-<Q>bit/`. Keep and amend the spec, or migrate.

### Non-functional (§6)
- [ ] `prompt_toolkit` async prompts / background status updates (tied to FC-2 async kernel and FC-17 renderer).

## Roadmap (README, not covered by the SRS)
- [ ] Tests for orchestrator, HITL, and task registry on par with the kernel loop (coverage grew in `tests/test_srs_v03.py`; kernel-loop parity not measured)
- [ ] Hosted/OpenAI-compatible remote provider as a first-class backend
- [ ] Pluggable tool registries and remote execution adapters
- [ ] Retrieval and memory integrations (vector/index, source-aware answers)
- [ ] File-centric workflows beyond `SKILL.md`
- [ ] Signed/notarized macOS binaries; Intel macOS and Windows feasibility
- [ ] Public docs, issue templates, API stability policy
- [ ] MCP client / extension API beyond tool packs and prompt presets
- [ ] Split `nomadic/cli.py` (~3.4k lines; quantization already extracted)
