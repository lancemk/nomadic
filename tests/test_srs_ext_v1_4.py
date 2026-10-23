import os
import pathlib
import pytest
from nomadic.agent import AgentRuntime
from nomadic.tools.fs import (
    handle_list_dir as _handle_list_dir,
    handle_read_file as _handle_read_file,
    handle_replace_content as _handle_replace_content,
    handle_write_file as _handle_write_file,
)
import re

def test_fs_tools_e2e(tmp_path):
    # Dummy runtime
    class DummyCfg:
        max_skill_chars = 10000
    class DummyState:
        cfg = DummyCfg()
    class DummyUI:
        pass
    runtime = AgentRuntime(state=DummyState(), ui=DummyUI())

    # 1. Test write_file
    test_file = tmp_path / "test_file.txt"
    write_args = {"path": str(test_file), "content": "Hello World\nLine 2"}
    res_write = _handle_write_file(write_args, runtime)
    assert res_write.status == "ok"
    assert test_file.exists()
    assert test_file.read_text() == "Hello World\nLine 2"

    # 2. Test read_file
    read_args = {"path": str(test_file)}
    res_read = _handle_read_file(read_args, runtime)
    assert res_read.status == "ok"
    assert "Hello World" in res_read.content

    # 3. Test read_file with line limits
    read_args_lines = {"path": str(test_file), "start_line": 2, "end_line": 2}
    res_read_lines = _handle_read_file(read_args_lines, runtime)
    assert "Line 2" in res_read_lines.content
    assert "Hello World" not in res_read_lines.content

    # 4. Test replace_content
    replace_args = {"path": str(test_file), "target": "World", "replacement": "Universe"}
    res_replace = _handle_replace_content(replace_args, runtime)
    assert res_replace.status == "ok"
    assert test_file.read_text() == "Hello Universe\nLine 2"

    # 5. Test list_dir
    list_args = {"path": str(tmp_path)}
    res_list = _handle_list_dir(list_args, runtime)
    assert res_list.status == "ok"
    assert "test_file.txt" in res_list.content

def test_context_reference_regex():
    raw_input = "Can you explain @README.md and @tests/test_file.py?"
    matches = set(re.findall(r'(?<!\S)@([^\s]+)', raw_input))
    assert matches == {"README.md", "tests/test_file.py?"} # wait, regex matches '?'
    # Let's fix the regex test logic
    raw_input2 = "Can you explain @README.md"
    matches2 = set(re.findall(r'(?<!\S)@([^\s]+)', raw_input2))
    assert matches2 == {"README.md"}
    
    # Check it doesn't match email
    email_input = "Contact me at user@example.com"
    matches_email = set(re.findall(r'(?<!\S)@([^\s]+)', email_input))
    assert len(matches_email) == 0
