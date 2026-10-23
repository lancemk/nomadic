# TODO

Gap analysis of the code against [specs-public/SRS-v0.3.md](specs-public/SRS-v0.3.md) (supersedes SRS-ext-v1.1 – v2.0 and the previous TODO.md), plus open README roadmap items.

The SRS header says every FC is "planned / not implemented", but a static read of the code shows most of it already exists. Statuses below come from reading and grepping the source, not from running it: **done** = found and matches the spec, **partial** = found with a specific shortfall, **gap** = not found. Anything marked "unverified" needs a run or a closer read.

## Spec gaps (SRS v0.3)

### Orchestration core (FC-1 – FC-4)
- [ ] **FC-1** (partial): `AgentInstance.step()` exists in `nomadic/orchestrator.py`; confirm the kernel's own turn loop (`nomadic/cli.py`) goes through it rather than the old synchronous path.
- [ ] **FC-2** (partial): `Orchestrator.run_parallel` and a `ThreadPoolExecutor` exist. Gaps: no agent-to-agent hand-off of sub-tasks/evidence (nothing found); no process pool for long local tools; the REPL itself is not an async kernel (spec §3 `AK`).
- [x] **FC-3**: `HITLManager`, `strictly_human` / `semi_autonomous` / `fully_autonomous` profiles, and approve/reject/modify/skip decisions are in `nomadic/hitl.py`.
- [ ] **FC-4** (partial): `tasks` table and `TaskRegistry` exist. Unverified: automatic tracking of files created/modified by a task into `artifacts_json`; Pending/Active/Completed/Failed state names.

### Autonomy, background, extended, state, DX (FC-5 – FC-9)
- [ ] **FC-5–FC-9** (present as commands): `/goal /go /grill-me /agents /tasks /browser /btw /fork /rewind /undo /rename /resume /switch /clear /diff /artifacts /add-dir /context /usage /export /config /settings /help /exit` are all registered in `nomadic/cli.py`. Needs a behaviour check against each bullet in §4.2–§4.6, in particular:
  - [ ] `/agents` and `/tasks` as TUI panels/dashboards (inspect, spawn, cancel), not just listings
  - [ ] `/rewind` restores agent state, not only messages; `/fork` copies workspace state
  - [ ] `/export` to Markdown/JSON and the Antigravity GUI hook; `/usage` quota/rate-limit tracking; `/context` checkpoint management
  - [ ] `/go` and `/goal` honour the FC-3 safety profile on non-critical tool calls

### FS tooling and context referencing (FC-10 – FC-12)
- [ ] **FC-10** (partial): `read_file` (with `start_line`/`end_line`), `write_file`, `replace_content`, `list_dir` exist but live in `nomadic/agent.py`, not `nomadic/tools/fs.py`. Gaps: `replace_content` is not AST-aware; large-file safety on `read_file` unverified.
- [ ] **FC-11** (partial): `KernelCompleter` in `nomadic/cli.py` combines `/` and `@` completion. Spec places it in `KernelInput._prompt_multiline`; check placement and that `@` is the only path trigger.
- [ ] **FC-12** (partial): `@path` is parsed and wrapped in `<file path="…">` blocks before `state.add_user()`. Gaps: truncation uses only `max_skill_chars`, `max_context_tokens` is not enforced; no PWD-relative resolution beyond Python's cwd and no hidden-system-message option; any `@token` is matched, including emails and mentions. Add tests.

### Runtime quantization (FC-13 – FC-16)
- [ ] **FC-13** (done): `sysctl -n hw.memsize` auditor in `nomadic/cli.py`; verify the `max(8, 0.25·M_sys)` reserve.
- [ ] **FC-14** (partial): `config.json` is fetched with `hf_hub_download` and parameters estimated; falls back to name parsing. Gap: no explicit auth-token handling; `M_kv_cache` term from the §4.8 model is not part of Q selection (unverified).
- [ ] **FC-15** (partial): `run_quantization` shells out to `mlx_lm`/`mlx_vlm` convert. Gaps: not a managed async subprocess; logs go to `ui.rule`, not `StatusEvent`s; verify cache path is `<cache>/quantized/<repo_id>-<Q>bit/` (README documents `<model>/<bits>bit/`).
- [ ] **FC-16** (partial): `--quantize`, `--q-bits`, `--cache-dir` exist; confirm both `cmd_serve` and `cmd_kernel` redirect `ServerConfig.model`.

### Status feedback (FC-17)
- [ ] **FC-17** (partial): `LifecycleState` (THINKING/READING/WRITING/EDITING/EXECUTING/REVIEWING) and `StatusEvent` exist in `nomadic/agent.py`; the CLI renders via `ui.start_status`. Gaps: `REVIEWING` is defined but never emitted (HITL pauses show no status); no event bus (`queue.Queue`/`asyncio.Queue`) — a bare callback is used; `StatusEvent` has `depth` but no tool context; spec's `AgentStatus` data class lives in `orchestrator.py` as a separate enum.

### Non-functional (§6)
- [ ] Python version: spec says 3.12+, `pyproject.toml` requires `>=3.14`. Decide which is right and align (3.14 shrinks the user base).
- [ ] Persistence: no `agent_states` table in `nomadic/sql/*.sql` (only `tasks`); add to SQLite and Postgres schemas.
- [ ] UI: `prompt_toolkit` background status updates and async prompts (ties to FC-2 async kernel).

## Spec housekeeping
- [ ] SRS v0.3 still says all FCs are "planned / not implemented" and leaves all milestones M1–M10 unchecked. Update it to match the code once the checks above are done.

## Roadmap (README, still open and not covered by the SRS)
- [ ] Tests for orchestrator, HITL, and task registry on par with the kernel loop (current: `tests/test_srs_ext.py`, `tests/test_srs_ext_v1_4.py`, `tests/test_roadmap_features.py`)
- [ ] Hosted/OpenAI-compatible remote provider as a first-class backend
- [ ] Pluggable tool registries and remote execution adapters
- [ ] Retrieval and memory integrations (vector/index, source-aware answers)
- [ ] File-centric workflows beyond `SKILL.md`
- [ ] Signed/notarized macOS binaries; Intel macOS and Windows feasibility
- [ ] Public docs, issue templates, API stability policy
- [ ] MCP client / extension API beyond tool packs and prompt presets
- [ ] Split `nomadic/cli.py` (~3.5k lines)
