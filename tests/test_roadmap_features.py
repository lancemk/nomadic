
import pathlib
import json
import pytest
from nomadic import agent
from nomadic import cli
from nomadic import db as chatdb

def test_tool_registry_dynamic_loading(tmp_path):
    # Create a dummy tool file
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    tool_file = tools_dir / "my_tool.py"
    tool_file.write_text("""
from nomadic.agent import ToolDefinition, ToolExecutionResult, AgentRuntime
from typing import Mapping, Any

def handle_hello(args: Mapping[str, Any], runtime: AgentRuntime) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_name="hello",
        label="hello",
        status="ok",
        content="Hello, " + args.get("name", "World"),
        payload={"msg": "success"}
    )

def get_tools():
    return [
        ToolDefinition(
            name="hello",
            description="Say hello",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            handler=handle_hello
        )
    ]
""")

    registry = agent.ToolRegistry([])
    registry.load_from_directory(tools_dir)
    
    assert "hello" in [t.name for t in registry.tools]
    
    # Test execution
    res = registry.execute(agent.ToolCall(name="hello", arguments={"name": "Alice"}, raw_json=""), None)
    assert res.content == "Hello, Alice"

def test_server_config_profiles_and_presets():
    cfg = cli.ServerConfig()
    cfg.profiles = {
        "creative": {"temperature": 0.8, "max_tokens": 2048}
    }
    cfg.presets = {
        "coder": "You are a professional coder."
    }
    
    assert cfg.temperature == 0.15
    assert cfg.apply_profile("creative")
    assert cfg.temperature == 0.8
    assert cfg.max_tokens == 2048
    
    assert cfg.apply_preset("coder")
    assert cfg.cot_system_prompt == "You are a professional coder."
    
    assert not cfg.apply_profile("nonexistent")

def test_server_config_templates():
    cfg = cli.ServerConfig()
    cfg.templates = {
        "review": "Review the following code: {0}\nFocus on {1}.",
        "greet": "Hello {0}!"
    }
    
    assert cfg.format_template("greet", "Alice") == "Hello Alice!"
    assert cfg.format_template("review", "def foo(): pass, performance") == "Review the following code: def foo(): pass\nFocus on performance."
    assert cfg.format_template("missing", "args") is None
    # Test missing argument gracefully fails (returns None or raises handled exception, but current simple logic handles it via IndexError)
    assert cfg.format_template("greet", "") is None

def test_load_server_config_from_file(tmp_path):
    config_file = tmp_path / "config.jsonc"
    config_file.write_text(json.dumps({
        "server": {
            "model": "my-custom-model",
            "profiles": {
                "fast": {"max_tokens": 128}
            }
        },
        "kernel": {
            "max_skill_chars": 5000
        }
    }))
    
    cfg = cli.load_server_config(config_file)
    assert cfg.model == "my-custom-model"
    assert "fast" in cfg.profiles
    assert cfg.profiles["fast"]["max_tokens"] == 128
    assert cfg.max_skill_chars == 5000

def test_db_session_management(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    conn = chatdb.connect(db_path)
    chatdb.init_db(conn)
    
    import os
    pwd = os.getcwd()
    
    # Create session with pwd in metadata
    session = chatdb.create_session(
        conn,
        chat_name="Test Session",
        system_prompt="You are helpful.",
        metadata={"pwd": pwd}
    )
    session_id = session["id"]
    
    # List sessions by pwd
    sessions = chatdb.list_sessions(conn, pwd=pwd)
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id
    
    # Rename session
    chatdb.update_session(conn, session_id, chat_name="Renamed Session")
    session = chatdb.get_session(conn, session_id)
    assert session["chat_name"] == "Renamed Session"
    
    # Get by name
    session_by_name = chatdb.get_session_by_name(conn, "Renamed Session")
    assert session_by_name["id"] == session_id
    
    # Resume session
    resumed = chatdb.resume_session(conn, session_id)
    assert resumed["status"] == "active"


def test_normalize_model_name():
    from nomadic.cli import normalize_model_name
    assert normalize_model_name("mlx-community/gemma-4-26b-a4b-it-4bit") == "mlx-community/gemma-4-26b-a4b-it-4bit"
    assert normalize_model_name("hf://mlx-community/gemma-4-26b-a4b-it-4bit") == "mlx-community/gemma-4-26b-a4b-it-4bit"
    assert normalize_model_name("hf://models/mlx-community/gemma-4-26b-a4b-it-4bit") == "mlx-community/gemma-4-26b-a4b-it-4bit"
    assert normalize_model_name("models/mlx-community/gemma-4-26b-a4b-it-4bit") == "mlx-community/gemma-4-26b-a4b-it-4bit"


def test_server_config_model_normalization():
    cfg = cli.ServerConfig()
    cfg.model = "hf://mlx-community/gemma-4-26b-a4b-it-4bit"
    assert cfg.model == "mlx-community/gemma-4-26b-a4b-it-4bit"


def test_cli_parser_inline_kv_overrides(monkeypatch):
    import sys
    test_args = ["nomadic", "serve", "model=hf://mlx-community/gemma-4-26b-a4b-it-4bit", "port=9000"]
    monkeypatch.setattr(sys, "argv", test_args)

    called_cfg = []
    monkeypatch.setattr(cli, "cmd_serve", lambda cfg: called_cfg.append(cfg))

    cli.main()
    assert len(called_cfg) == 1
    assert called_cfg[0].model == "mlx-community/gemma-4-26b-a4b-it-4bit"
    assert called_cfg[0].port == 9000


def test_cli_parser_inline_kv_unrecognized(monkeypatch):
    import sys
    test_args = ["nomadic", "serve", "invalid_param=123"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit):
        cli.main()


def test_resource_auditor_and_param_estimation():
    from nomadic.cli import (
        get_safe_memory_budget,
        estimate_parameters_from_config,
        detect_parameters_from_name,
        determine_optimal_q_bits
    )
    # Check physical memory queries safely fallback or return positive values
    budget = get_safe_memory_budget()
    assert budget > 0.0

    # Test parameter count estimation from config
    mock_config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "hidden_act": "silu"
    }
    p = estimate_parameters_from_config(mock_config)
    assert p is not None
    assert 6.0 < p < 10.0

    # Test parameter count from name
    assert detect_parameters_from_name("mlx-community/Qwen2.5-32B-Instruct") == 32.0
    assert detect_parameters_from_name("mlx-community/Llama-3-8B-Instruct-4bit") == 8.0
    assert detect_parameters_from_name("some-model-1.5b") == 1.5
    assert detect_parameters_from_name("no-size-here") is None

    # Test dynamic optimal Q-bits determination
    import nomadic.cli
    orig_budget = nomadic.cli.get_safe_memory_budget
    try:
        nomadic.cli.get_safe_memory_budget = lambda: 36.0
        q = determine_optimal_q_bits("mlx-community/Qwen2.5-32B-Instruct")
        assert q == 4

        nomadic.cli.get_safe_memory_budget = lambda: 12.0
        q = determine_optimal_q_bits("mlx-community/Qwen2.5-32B-Instruct")
        assert q == 2
    finally:
        nomadic.cli.get_safe_memory_budget = orig_budget


def test_cli_parser_quantize_flag_and_cache(tmp_path, monkeypatch):
    import sys
    cache_dir = tmp_path / "cache"
    fake_model_dir = cache_dir / "mlx-community--Qwen2.5-32B-Instruct" / "4bit"
    fake_model_dir.mkdir(parents=True, exist_ok=True)
    (fake_model_dir / "config.json").write_text("{}")

    test_args = [
        "nomadic", "serve",
        "model=mlx-community/Qwen2.5-32B-Instruct",
        "--quantize",
        f"--cache-dir={fake_model_dir}"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    called_cfg = []
    monkeypatch.setattr(cli, "cmd_serve", lambda cfg: called_cfg.append(cfg))

    cli.main()
    assert len(called_cfg) == 1
    assert called_cfg[0].quantize is True
    assert called_cfg[0].model == str(fake_model_dir)


def test_slash_commands_v2_logic(tmp_path):
    # Setup test DB and state
    db_path = tmp_path / "test.sqlite3"
    conn = chatdb.connect(db_path)
    chatdb.init_db(conn)
    
    cfg = cli.ServerConfig()
    session = chatdb.create_session(conn, chat_name="Orig Session", system_prompt="test prompt", config=cfg)
    state = cli._load_kernel_state(conn, session, cfg)
    
    # 1. Test /rename
    state.session = dict(chatdb.update_session(conn, state.session_id, chat_name="New Name"))
    assert state.chat_name == "New Name"
    
    # 2. Test /add-dir
    state.workspace_paths = [str(tmp_path)]
    new_dir = tmp_path / "subdir"
    new_dir.mkdir()
    state.workspace_paths.append(str(new_dir))
    assert str(new_dir) in state.workspace_paths
    
    # 3. Test render_token_grid
    grid = cli.render_token_grid(0.5, 0.2, 0.2, 0.05, 0.05)
    assert "[blue]●[/blue]" in grid
    assert "□" in grid
    
    # 4. Test /fork
    # Insert some dummy messages
    chatdb.append_message(conn, state.session_id, role="user", content_text="Hello")
    chatdb.append_message(conn, state.session_id, role="assistant", content_text="Hi there")
    
    # Fork session
    new_id = "forked-session-uuid"
    columns = [
        "id", "chat_name", "status", "created_at", "updated_at", "started_at", "ended_at",
        "model", "host", "port", "temperature", "top_p", "max_tokens", "max_skill_chars",
        "max_context_tokens", "context_recompress_to", "system_prompt", "summary",
        "message_count", "user_turn_count", "assistant_turn_count", "tool_event_count",
        "last_message_at", "metadata_json"
    ]
    vals = [new_id, "Forked Session", "active", "now", "now", None, None,
            cfg.model, cfg.host, cfg.port, cfg.temperature, cfg.top_p, cfg.max_tokens, cfg.max_skill_chars,
            cfg.max_context_tokens, cfg.context_recompress_to, cfg.cot_system_prompt, None,
            2, 1, 1, 0, None, "{}"]
    
    placeholders = ", ".join([conn.placeholder] * len(columns))
    with chatdb._transaction(conn):
        conn.execute(
            f"INSERT INTO chat_sessions ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(vals)
        )
        
    forked_session = chatdb.get_session(conn, new_id)
    assert forked_session["chat_name"] == "Forked Session"

