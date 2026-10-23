# nomadic Kernel Guide

`nomadic.py` is a small Python CLI for three main flows:

1. Run an `mlx_vlm.server` instance locally.
2. Use `kernel` for an interactive multimodal chat and agent loop.
3. Use `extract` to turn a `SKILL.md` file into a structured context summary.

This guide walks through the fastest way to get started and explains the
context-window settings that now ship with the CLI.

```
/skill /Users/lance/_skills/Busines_Intel_Expert/SKILL.md
```

## Requirements

- macOS on Apple Silicon
- Python 3.12 or newer
- `mlx-vlm`
- `requests`
- `rich` is optional, but makes terminal output much easier to read

Install the Python packages:

```bash
pip install mlx-vlm requests rich
```

## Start The Server

The CLI wraps `python -m mlx_vlm.server` and starts the multimodal server that
the rest of the commands use.

```bash
python nomadic.py serve
```

You can override the model, host, and port either before or after the command:

```bash
python nomadic.py serve --model mlx-community/gemma-4-26b-a4b-it-4bit
python nomadic.py --port 8081 serve
```

## Interactive Kernel

Start the interactive structured agent loop after the server is running:

```bash
python nomadic.py kernel
```

Common shell-time examples:

```bash
python nomadic.py kernel --model mlx-community/Pixtral-12B-2409-4bit
python nomadic.py kernel --max-context-tokens 12000 --context-recompress-to 8000
```

Inside the kernel, useful commands include:

- `/skill <path>` load a `SKILL.md` file into the conversation
- `/image <path>` attach an image to the next user turn
- `/exec <cmd>` run a shell command and feed the result back into context
- `/history` inspect the current conversation
- `/config` print the current settings
- `/session` print the tracked chat session metadata
- `/tokens <n>` change the per-response output budget
- `/temp <value>` change the sampling temperature
- `/clear` reset the conversation
- `/save <file>` write the session history to JSON

The `you` input block supports multiline authoring when `prompt_toolkit` is
installed:

- `Enter` submits the current message
- `Shift+Enter` inserts a newline and keeps the editor growing vertically
- bullet lists, fenced code blocks, and indentation are preserved as typed
- slash commands are only treated as commands when the whole submission is a
  single line that starts with `/`

Interrupt behavior in `kernel` mode:

- first `Ctrl-C` during an active model/tool run interrupts that run and
  returns you to the kernel prompt
- first `Ctrl-C` while idle arms exit and shows a warning
- second `Ctrl-C` within 2 seconds exits the kernel process
- `Ctrl-D` still exits from an empty prompt

After each assistant reply, the kernel also shows a `User Action On Next Step`
menu to mimic the Codex flow:

- `1` is the recommended next action
- `2` and `3` are alternate likely next actions based on the current chat context
- `4` opens a free-text custom follow-up
- typing normal free text still works directly if you do not want to pick a menu option

If `prompt_toolkit` is not installed, the kernel falls back to the classic
single-line prompt so the CLI remains usable. Install it to enable the enhanced
multiline editor and themed border styling.

## Agent Loop

The `kernel` command now runs a four-stage tool-calling loop:

1. The CLI injects a small tool registry into the system prompt.
2. The model can emit one structured `tool_call` JSON object.
3. The CLI executes the requested local tool.
4. The tool result is fed back to the model until it returns a final answer.

The first registry is intentionally small:

- `read_skill` reads a local `SKILL.md`
- `exec_shell` runs one shell command and returns stdout and stderr
- `metadata_search` queries the configured metadata service

Tool calls use this envelope:

```json
{
  "tool_call": {
    "name": "read_skill",
    "arguments": {
      "path": "/absolute/or/relative/path/to/SKILL.md"
    }
  }
}
```

For metadata search, the model can emit:

```json
{
  "tool_call": {
    "name": "metadata_search",
    "arguments": {
      "query": "your search text"
    }
  }
}
```

The CLI handles this automatically. You do not need to type tool JSON yourself.

## Session Storage

Chat sessions are stored in either SQLite or Postgres and keyed by a UUIDv4.

Recommended first-time setup:

```bash
python nomadic.py initialize --db-kind sqlite3 --db-path kernel.sqlite3
```

This writes `~/.config/nomadic/config.jsonc`, remembers the selected backend,
and bootstraps the schema immediately.

You can also initialize the schema directly without writing config:

```bash
python db.py init --db-kind postgres --postgres-dsn postgresql://user:pass@host/db
```

Start a new session using the configured default database:

```bash
python nomadic.py kernel
```

Start a new session with a one-off SQLite override:

```bash
python nomadic.py kernel --db-path kernel.sqlite3
```

Start a new session with a one-off Postgres override:

```bash
python nomadic.py kernel --db-kind postgres --postgres-dsn postgresql://user:pass@host/db
```

Resume an existing session:

```bash
python nomadic.py kernel --session-id <uuidv4>
```

The session starts with a provisional name and auto-titles itself from the
first meaningful user turn.

## Extract A Skill

Use `extract` when you want a one-shot summary of a `SKILL.md` file:

```bash
python nomadic.py extract /mnt/skills/public/docx/SKILL.md
```

You can also attach an image:

```bash
python nomadic.py extract /mnt/skills/public/docx/SKILL.md \
  --image /path/to/diagram.png
```

The result is saved under `evals/` for downstream use.

## List Skills

Scan a tree and print every `SKILL.md` file:

```bash
python nomadic.py list-skills --root /mnt/skills
```

## CLI Flags

These are the flags you are most likely to tune:

- `--model` sets the `mlx-community/...` model to load
- `--host` and `--port` set the server bind
- `--temperature` controls randomness
- `--top-p` sets nucleus sampling
- `--max-tokens` limits the length of each model response
- `--max-skill-chars` limits how much `SKILL.md` content is injected at once
- `--max-context-tokens` sets the approximate kernel history budget
- `--context-recompress-to` sets the target budget after automatic compression
- `--db-kind` selects `sqlite3` or `postgres` for tracked chat sessions
- `--db-path` sets the SQLite database file for tracked chat sessions
- `--postgres-dsn` sets the Postgres DSN for tracked chat sessions
- `--config-path` points to an alternate `config.jsonc`
- `--session-id` resumes a specific UUIDv4 session
- `--metadata-base-url` sets the metadata service base URL

You can also set `METADATA_BASE_URL` in the environment as a default.

`--max-tokens` is for output length only. It is separate from the context budget.
It does not change the agent loop depth.

## Context Window Behavior

The kernel now manages history automatically.

- The system prompt is always preserved.
- The newest turns stay verbatim for as long as possible.
- Older turns are compressed into a compact memory note when the approximate
  non-system history budget is exceeded.
- The budget is approximate by design, so it does not require a tokenizer
  dependency.
- The system prompt is treated as fixed overhead and is not counted against
  the mutable history budget.

Recommended starting point:

```bash
python nomadic.py kernel \
  --max-context-tokens 12000 \
  --context-recompress-to 8000 \
  --max-skill-chars 12000
```

If compression happens too early:

- increase `--max-context-tokens`
- increase `--context-recompress-to`
- or shorten what you inject with `/skill`

If you want stricter memory use:

- lower `--max-context-tokens`
- lower `--context-recompress-to`

## Quick Workflow

1. Start the server with `serve`.
2. Open a second terminal and start `kernel`.
3. Load a `SKILL.md` with `/skill <path>`.
4. Attach an image with `/image <path>` if you need visual context.
5. Ask a question.
6. Let the kernel compress history automatically when the context budget is hit.

Example interaction:

```text
[1] you: Summarize /Users/me/skills/finance/SKILL.md and tell me what shell command lists the files here.
... model calls read_skill ...
... CLI injects the file contents ...
... model calls exec_shell ...
... CLI injects the command result ...
... model returns a normal final answer ...
```

## Notes

- The CLI accepts shared flags both before and after the subcommand.
- `serve`, `extract`, `kernel`, `initialize`, and `list-skills` all share the same model and
  server overrides.
- `extract` still uses `--max-skill-chars`, even though it does not use the
  interactive history budget.
- `nomadic/storage/` holds persistence: `db.py` is the backend-aware session connector and schema initializer (sessions, events, tasks, agent states); `sql/` has the SQLite/Postgres schemas. The root `db.py` is a thin wrapper for `nomadic-db`.
- `nomadic/tools/fs.py` provides the file tools (`read_file`, `write_file`, `replace_content`, `list_dir`).
- `nomadic/context.py` resolves `@path` references; `nomadic/quantize.py` is the runtime quantization pipeline; `nomadic/status.py` is the status event bus.
- Intermediate tool-call JSON is handled internally; the terminal prints the
  final assistant answer plus tool progress updates.
