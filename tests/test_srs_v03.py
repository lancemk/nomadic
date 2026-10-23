"""Tests for the SRS v0.3 gap work: storage, FS tools, context resolver, status bus,
orchestrator hand-off / persistence, and the quantization pipeline."""

from __future__ import annotations

import asyncio
import json

import pytest

from nomadic import agent as agent_mod
from nomadic.agent import AgentRuntime, LifecycleState, StatusEvent, ToolCall
from nomadic.context import extract_refs, resolve_file_references
from nomadic.hitl import HITLManager, SafetyProfile
from nomadic.orchestrator import AgentInstance, AgentStatus, Orchestrator
from nomadic.status import StatusBus
from nomadic.storage import db as chatdb
from nomadic.tasks import TaskRegistry
from nomadic.tools import fs


def _call(name, **arguments):
    return ToolCall(name=name, arguments=arguments, raw_json=json.dumps({"tool_call": {"name": name, "arguments": arguments}}))


def _scripted(*replies):
    queue = list(replies)

    def complete(history, stream=False):
        return queue.pop(0)

    return complete


def _tool(name, **arguments):
    return json.dumps({"tool_call": {"name": name, "arguments": arguments}})


@pytest.fixture
def conn(tmp_path):
    connection = chatdb.connect(chatdb.DatabaseSettings(kind="sqlite3", sqlite_path=str(tmp_path / "t.sqlite3")))
    chatdb.init_db(connection)
    yield connection
    connection.close()


# ── storage package ────────────────────────────────────────────────────────


def test_storage_package_exposes_db():
    import nomadic.storage as storage

    assert storage.db is chatdb
    assert chatdb.SQLITE_SCHEMA_RESOURCE.is_file()
    assert chatdb.TASKS_SCHEMA_RESOURCE.is_file()


def test_agent_state_round_trip(conn):
    state = {
        "agent_id": "agent-1", "status": "awaiting_review", "depth": 2, "max_depth": 5,
        "system_prompt": "sys", "history": [{"role": "user", "content": "hi"}],
        "current_tool_call": {"name": "exec_shell", "arguments": {"command": "ls"}, "raw_json": "{}"},
    }
    chatdb.upsert_agent_state(conn, state)
    loaded = chatdb.get_agent_state(conn, "agent-1")
    assert loaded["status"] == "awaiting_review"
    assert loaded["history"] == state["history"]
    assert loaded["current_tool_call"]["name"] == "exec_shell"

    chatdb.upsert_agent_state(conn, {**state, "status": "completed", "result": "done", "current_tool_call": None})
    assert chatdb.get_agent_state(conn, "agent-1")["result"] == "done"
    assert [s["agent_id"] for s in chatdb.list_agent_states(conn, status="completed")] == ["agent-1"]
    assert chatdb.list_agent_states(conn, status="idle") == []


# ── FC-10 FS tools ─────────────────────────────────────────────────────────


def test_read_file_streams_ranges_of_large_files(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("\n".join(f"line {i}" for i in range(1, 200_001)))
    assert big.stat().st_size > fs.MAX_READ_BYTES // 5
    runtime = AgentRuntime(state=None, ui=None)
    result = fs.handle_read_file({"path": str(big), "start_line": 100, "end_line": 102}, runtime)
    assert result.payload["content"] == "line 100\nline 101\nline 102"
    assert "lines 100-102" in result.content


def test_read_file_truncates_to_budget_and_validates(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x" * 50_000)
    runtime = AgentRuntime(state=None, ui=None)
    result = fs.handle_read_file({"path": str(f)}, runtime)
    assert result.payload["truncated"] is True
    assert len(result.payload["content"]) == runtime.max_skill_chars
    with pytest.raises(ValueError):
        fs.handle_read_file({"path": str(f), "start_line": 0}, runtime)
    with pytest.raises(ValueError):
        fs.handle_read_file({"path": str(f), "start_line": 5, "end_line": 2}, runtime)


def test_replace_content_requires_unique_target(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("foo foo")
    runtime = AgentRuntime(state=None, ui=None)
    with pytest.raises(ValueError, match="2 places"):
        fs.handle_replace_content({"path": str(f), "target": "foo", "replacement": "bar"}, runtime)
    fs.handle_replace_content({"path": str(f), "target": "foo", "replacement": "bar", "replace_all": True}, runtime)
    assert f.read_text() == "bar bar"


def test_replace_content_symbol_is_ast_aware(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def a():\n    return 1\n\nclass K:\n    def m(self):\n        return 2\n\ndef z():\n    return 3\n")
    runtime = AgentRuntime(state=None, ui=None)
    result = fs.handle_replace_content(
        {"path": str(f), "symbol": "K.m", "replacement": "def m(self):\n    return 20\n"}, runtime
    )
    assert result.payload["line"] == 5
    text = f.read_text()
    assert "        return 20" in text
    assert "def z():" in text and "return 2\n" not in text
    with pytest.raises(ValueError, match="Symbol not found"):
        fs.handle_replace_content({"path": str(f), "symbol": "nope", "replacement": "def nope(): pass"}, runtime)


def test_replace_content_rejects_syntax_errors(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n")
    with pytest.raises(ValueError, match="syntax error"):
        fs.handle_replace_content(
            {"path": str(f), "target": "x = 1", "replacement": "x = ("}, AgentRuntime(state=None, ui=None)
        )
    assert f.read_text() == "x = 1\n"


def test_list_dir_skips_noise_and_supports_flat(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    runtime = AgentRuntime(state=None, ui=None)
    deep = fs.handle_list_dir({"path": str(tmp_path)}, runtime).content
    assert "HEAD" not in deep and "sub/a.txt" in deep
    flat = fs.handle_list_dir({"path": str(tmp_path), "recursive": False}, runtime).content
    assert "sub/" in flat and "a.txt" not in flat


def test_fs_tools_live_in_default_registry():
    names = {t.name for t in agent_mod.build_default_registry().tools}
    assert {"read_file", "write_file", "replace_content", "list_dir"} <= names


# ── FC-17 status ───────────────────────────────────────────────────────────


def test_status_bus_is_fifo_bounded_and_notifies():
    bus = StatusBus(maxsize=2)
    seen = []
    bus.subscribe(seen.append)
    for n in range(3):
        bus.emit(LifecycleState.THINKING, str(n))
    assert [e.message for e in bus.drain()] == ["1", "2"]  # oldest dropped
    assert len(seen) == 3
    assert bus.poll() is None


def test_registry_emits_state_per_tool(tmp_path):
    events = []
    runtime = AgentRuntime(state=None, ui=None, status_callback=events.append)
    registry = agent_mod.build_default_registry()
    target = tmp_path / "f.py"
    registry.execute(_call("write_file", path=str(target), content="a = 1\n"), runtime)
    registry.execute(_call("replace_content", path=str(target), target="a = 1", replacement="a = 2"), runtime)
    registry.execute(_call("read_file", path=str(target)), runtime)
    assert [e.state for e in events] == [LifecycleState.WRITING, LifecycleState.EDITING, LifecycleState.READING]
    assert "line 1" in events[1].message


# ── FC-12 context resolver ─────────────────────────────────────────────────


def test_extract_refs_handles_quotes_punctuation_and_emails():
    assert extract_refs('see @a.py, @"my file.txt" and @a.py, mail me@x.com') == ["a.py,", "my file.txt"]


def test_resolver_relative_to_base_dir_and_punctuation(tmp_path):
    (tmp_path / "a.py").write_text("print(1)")
    (tmp_path / "my file.txt").write_text("hello")
    res = resolve_file_references('look at @a.py, and @"my file.txt".', base_dir=tmp_path)
    assert [f.ref for f in res.files] == ["a.py", "my file.txt"]
    assert res.files[0].render() == '<file path="a.py"><content>\nprint(1)\n</content></file>'
    assert "[Attached Context:]" in res.append_to("q")


def test_resolver_enforces_per_file_and_total_budget(tmp_path):
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("y" * 10_000)
    res = resolve_file_references("@a.txt @b.txt", base_dir=tmp_path, max_file_chars=6_000, max_context_tokens=2_000)
    # budget = 2000 tokens * 4 chars / 2 = 4000 chars total
    assert sum(len(f.content) - len("\n[... truncated ...]") for f in res.files) <= 4_000
    assert res.files[0].truncated
    assert any("truncated" in w or "budget" in w for w in res.warnings)


def test_resolver_skips_binary_dirs_missing_and_dupes(tmp_path):
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01")
    (tmp_path / "d").mkdir()
    (tmp_path / "t.txt").write_text("ok")
    res = resolve_file_references("@bin.dat @d @ghost @t.txt @./t.txt", base_dir=tmp_path)
    assert [f.ref for f in res.files] == ["t.txt"]
    assert any("binary" in w for w in res.warnings) and any("directory" in w for w in res.warnings)


# ── orchestrator: FC-2 / FC-3 / FC-4 / agent_states ────────────────────────


def _orch(**kw):
    return Orchestrator(default_system_prompt="sys", default_tool_registry=agent_mod.build_default_registry(), **kw)


def test_review_emits_reviewing_and_persists_state(conn):
    hitl = HITLManager(SafetyProfile.STRICTLY_HUMAN)
    orch = _orch(hitl_manager=hitl, db_conn=conn)
    inst = orch.spawn_agent(orch.task_registry.create_task("run it"))
    asyncio.run(orch.run_agent(inst.id, _scripted(_tool("exec_shell", command="ls")), None))

    assert inst.status == AgentStatus.AWAITING_REVIEW
    assert orch.task_registry.get_task(inst.task.id).status.value == "active"  # not marked completed
    states = [e.state for e in orch.status_bus.drain()]
    assert LifecycleState.THINKING in states and LifecycleState.REVIEWING in states
    assert chatdb.get_agent_state(conn, inst.id)["status"] == "awaiting_review"


def test_restore_agents_recreates_pending_review(conn):
    hitl = HITLManager(SafetyProfile.STRICTLY_HUMAN)
    orch = _orch(hitl_manager=hitl, db_conn=conn)
    inst = orch.spawn_agent(orch.task_registry.create_task("run it"))
    asyncio.run(orch.run_agent(inst.id, _scripted(_tool("exec_shell", command="ls")), None))

    fresh = _orch(hitl_manager=HITLManager(SafetyProfile.STRICTLY_HUMAN), db_conn=conn)
    restored = fresh.restore_agents()
    assert [a.id for a in restored] == [inst.id]
    assert restored[0].current_tool_call.name == "exec_shell"
    assert inst.id in fresh.hitl_manager.pending_reviews


def test_agent_state_round_trip_preserves_history():
    inst = AgentInstance(id="a", system_prompt="sys", tool_registry=agent_mod.build_default_registry())
    inst.add_user_message("hello")
    inst.depth = 3
    clone = AgentInstance.from_state(inst.to_state(), inst.tool_registry)
    assert clone.history == inst.history and clone.depth == 3 and clone.status == AgentStatus.IDLE


def test_file_tools_record_task_artifacts(tmp_path):
    orch = _orch(hitl_manager=HITLManager(SafetyProfile.FULLY_AUTONOMOUS))
    inst = orch.spawn_agent(orch.task_registry.create_task("write a file"))
    target = tmp_path / "out.txt"
    complete = _scripted(
        _tool("write_file", path=str(target), content="v1"),
        _tool("replace_content", path=str(target), target="v1", replacement="v2"),
        _tool("exec_shell", command="true"),
        "all done",
    )
    assert asyncio.run(orch.run_agent(inst.id, complete, None)) == "all done"
    assert [(a.path, a.action) for a in inst.task.artifacts] == [(str(target), "created"), (str(target), "modified")]
    assert target.read_text() == "v2"


def test_handoff_creates_child_task_and_returns_result():
    orch = _orch(hitl_manager=HITLManager(SafetyProfile.FULLY_AUTONOMOUS))
    parent = orch.spawn_agent(orch.task_registry.create_task("parent"))
    result = asyncio.run(
        orch.run_handoff(parent.id, "dig deeper", _scripted("child answer"), None, evidence="found X")
    )
    assert result == "child answer"
    child_task = next(t for t in orch.task_registry.tasks if t.objective == "dig deeper")
    assert child_task.parent_task_id == parent.task.id
    assert child_task.metadata["handoff_from"] == parent.id
    child = orch.get_agent(child_task.agent_id)
    assert any("Evidence from" in str(m["content"]) and "found X" in str(m["content"]) for m in child.history)
    assert parent.history[-1]["content"].endswith("child answer")
    with pytest.raises(ValueError):
        orch.handoff("missing", "x")


def test_run_blocking_offloads_work():
    orch = _orch()

    async def go():
        return await orch.run_blocking(sum, [1, 2, 3])

    assert asyncio.run(go()) == 6
    orch.shutdown()


# ── FC-13 – FC-16 quantization ─────────────────────────────────────────────


def test_kv_cache_estimate_is_gqa_aware():
    from nomadic import quantize

    cfg = {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8}
    gqa = quantize.estimate_kv_cache_gb(cfg, ctx_len=8192)
    mha = quantize.estimate_kv_cache_gb({**cfg, "num_key_value_heads": 32}, ctx_len=8192)
    assert gqa == pytest.approx(2 * 32 * 8 * 128 * 8192 * 2 / 1024**3)
    assert mha == pytest.approx(gqa * 4)
    assert quantize.estimate_kv_cache_gb(None) == quantize.FALLBACK_KV_CACHE_GB


def test_q_bits_choice_uses_kv_cache(monkeypatch):
    from nomadic import quantize

    cfg = {"num_hidden_layers": 64, "hidden_size": 8192, "num_attention_heads": 64, "num_key_value_heads": 64,
           "num_parameters": 20e9}
    monkeypatch.setattr(quantize, "load_model_config", lambda m: cfg)
    monkeypatch.setattr(quantize, "get_safe_memory_budget", lambda: 36.0)
    small_ctx = quantize.determine_optimal_q_bits("org/m", ctx_len=1024)
    huge_ctx = quantize.determine_optimal_q_bits("org/m", ctx_len=131072)
    assert small_ctx > huge_ctx


def test_resolve_q_bits():
    from nomadic import quantize

    warnings = []
    assert quantize.resolve_q_bits("3", "m") == 3
    assert quantize.resolve_q_bits("5", "m", warn=warnings.append) == 4 and warnings
    assert quantize.resolve_q_bits("garbage", "m") == 4


def test_quantization_streams_status_events_and_cleans_up(tmp_path, monkeypatch):
    from nomadic import quantize

    out = tmp_path / "out"
    script = tmp_path / "fake_convert.py"
    script.write_text("import sys\nprint('loading')\nprint('quantizing layer 1')\nsys.exit(int(sys.argv[-1] == 'fail'))\n")
    monkeypatch.setattr(quantize, "verify_converter", lambda m: None)
    monkeypatch.setattr(quantize, "is_text_only_model", lambda m: True)
    monkeypatch.setattr(quantize, "runtime_python", lambda: __import__("sys").executable)
    monkeypatch.setattr(
        quantize, "build_convert_command",
        lambda m, q, d: [__import__("sys").executable, str(script), "ok" if m == "good" else "fail"],
    )
    events = []
    logs = []
    assert quantize.run_quantization("good", 4, out, status=events.append, log=logs.append) == str(out)
    assert logs == ["loading", "quantizing layer 1"]
    assert all(e.state == LifecycleState.EXECUTING for e in events)
    assert any("quantizing layer 1" in e.message for e in events)

    out.mkdir()
    with pytest.raises(RuntimeError, match="exit code 1"):
        quantize.run_quantization("bad", 4, out, log=None)
    assert not out.exists()


# ── CLI wiring ─────────────────────────────────────────────────────────────


def test_command_aliases_resolve_to_single_handlers():
    from nomadic import cli

    assert cli.COMMAND_ALIASES == {"/settings": "/config", "/switch": "/resume", "/undo": "/rewind"}
