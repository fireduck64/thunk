import asyncio
import json
import os
import sqlite3
import pytest
from src.components.workspace import WorkspaceComponent


def test_workspace_path_traversal_protection(tmp_path):
    """Verifies write_workspace_file, read_workspace_file, and list_workspace_files reject path traversal."""
    ws_dir = tmp_path / "workspace"
    mind_dir = tmp_path / "mind_ro"
    live_db = tmp_path / "memory.db"

    comp = WorkspaceComponent(
        workspace_dir=str(ws_dir),
        mind_ro_dir=str(mind_dir),
        live_db_path=str(live_db),
    )

    # 1. Write traversal attempts
    assert "Error" in comp.write_workspace_file("../escape.txt", "bad")
    assert "Error" in comp.write_workspace_file("subdir/../../escape.txt", "bad")
    assert "Error" in comp.write_workspace_file("/etc/passwd", "bad")
    assert "Error" in comp.write_workspace_file("/home/agent/mind/hack.txt", "bad")
    assert not (tmp_path / "escape.txt").exists()

    # 2. Read traversal attempts
    assert "Error" in comp.read_workspace_file("../secret.txt")
    assert "Error" in comp.read_workspace_file("/etc/passwd")
    assert "Error" in comp.read_workspace_file("mind/../../secret.txt")

    # 3. List traversal attempts
    assert "Error" in comp.list_workspace_files("../")
    assert "Error" in comp.list_workspace_files("/etc")


def test_workspace_file_read_write_list(tmp_path):
    """Verifies writing nested files, listing directories, reading files back, and output truncation."""
    ws_dir = tmp_path / "workspace"
    mind_dir = tmp_path / "mind_ro"
    live_db = tmp_path / "memory.db"

    comp = WorkspaceComponent(
        workspace_dir=str(ws_dir),
        mind_ro_dir=str(mind_dir),
        live_db_path=str(live_db),
        max_output_chars=200,
    )

    # Write nested file using relative path and container-prefixed path
    res = comp.write_workspace_file("scripts/hello.py", "print('Hello, Thunk!')\n")
    assert "Success" in res
    assert (ws_dir / "scripts" / "hello.py").exists()

    res2 = comp.write_workspace_file("/home/agent/workspace/notes.txt", "Workspace note")
    assert "Success" in res2

    # List root and subdirectory
    root_listing = comp.list_workspace_files(".")
    assert "[DIR]  scripts" in root_listing
    assert "[FILE] notes.txt" in root_listing

    sub_listing = comp.list_workspace_files("scripts")
    assert "[FILE] hello.py" in sub_listing

    # Read back file
    content = comp.read_workspace_file("scripts/hello.py")
    assert content == "print('Hello, Thunk!')\n"

    # Verify truncation on large file
    long_text = "HEAD_MARKER_" + ("X" * 500) + "_TAIL_MARKER"
    comp.write_workspace_file("large.txt", long_text)
    truncated = comp.read_workspace_file("large.txt")
    assert "HEAD_MARKER_" in truncated
    assert "_TAIL_MARKER" in truncated
    assert "...[truncated " in truncated


def test_workspace_mind_mirror_snapshot(tmp_path):
    """Verifies _sync_mind_snapshot backs up live SQLite DBs and skills into mind_ro_dir and writes README.md."""
    ws_dir = tmp_path / "workspace"
    mind_dir = tmp_path / "mind_ro"
    live_db = tmp_path / "memory.db"
    notes_db = tmp_path / "notes.db"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Populate live memory.db
    with sqlite3.connect(str(live_db)) as conn:
        conn.execute("CREATE TABLE archival_log (id INTEGER PRIMARY KEY, role TEXT, content TEXT)")
        conn.execute("INSERT INTO archival_log (role, content) VALUES ('user', 'Hello from live DB')")

    # Populate live notes.db
    with sqlite3.connect(str(notes_db)) as conn:
        conn.execute("CREATE TABLE notes (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO notes (key, value) VALUES ('sticky', 'Remember the clacks')")

    # Populate a skill file
    (skills_dir / "test_skill.md").write_text("---\nsummary: Use when testing.\n---\nSkill body", encoding="utf-8")

    comp = WorkspaceComponent(
        workspace_dir=str(ws_dir),
        mind_ro_dir=str(mind_dir),
        live_db_path=str(live_db),
        skills_dir=str(skills_dir),
    )
    comp._sync_mind_snapshot()

    mirror_mem = mind_dir / "memory.db"
    mirror_notes = mind_dir / "notes.db"
    mirror_readme = mind_dir / "README.md"
    mirror_skill = mind_dir / "skills" / "test_skill.md"

    assert mirror_mem.exists()
    assert mirror_notes.exists()
    assert mirror_readme.exists()
    assert mirror_skill.exists()

    with sqlite3.connect(str(mirror_mem)) as conn:
        rows = conn.execute("SELECT role, content FROM archival_log").fetchall()
        assert rows == [("user", "Hello from live DB")]

    with sqlite3.connect(str(mirror_notes)) as conn:
        rows = conn.execute("SELECT key, value FROM notes").fetchall()
        assert rows == [("sticky", "Remember the clacks")]

    # Agent can also read files from /home/agent/mind via read_workspace_file
    readme_text = comp.read_workspace_file("/home/agent/mind/README.md")
    assert "archival_log" in readme_text
    skill_text = comp.read_workspace_file("mind/skills/test_skill.md")
    assert "Skill body" in skill_text


@pytest.mark.asyncio
async def test_workspace_command_execution_timeout_and_truncation(tmp_path, monkeypatch):
    """Verifies run_workspace_command invokes docker exec with expected args, truncates long output, and handles timeouts."""
    ws_dir = tmp_path / "workspace"
    mind_dir = tmp_path / "mind_ro"
    live_db = tmp_path / "memory.db"

    comp = WorkspaceComponent(
        container_name="thunk-workspace",
        container_user="agent",
        container_workdir="/home/agent/workspace",
        workspace_dir=str(ws_dir),
        mind_ro_dir=str(mind_dir),
        live_db_path=str(live_db),
        max_output_chars=100,
    )

    recorded_cmds = []

    class DummyProcess:
        def __init__(self, stdout_data: bytes, stderr_data: bytes, returncode: int = 0, hang: bool = False):
            self.stdout_data = stdout_data
            self.stderr_data = stderr_data
            self.returncode = returncode
            self.hang = hang
            self.killed = False

        async def communicate(self):
            if self.hang:
                await asyncio.sleep(10)
            return self.stdout_data, self.stderr_data

        def kill(self):
            self.killed = True

        async def wait(self):
            return

    # 1. Normal execution + output truncation
    long_stdout = ("START_OUT_" + ("A" * 200) + "_END_OUT").encode("utf-8")

    async def mock_exec_success(*args, **kwargs):
        recorded_cmds.append(list(args))
        return DummyProcess(long_stdout, b"", returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec_success)

    result_json = await comp.run_workspace_command("python3 -c 'print(1)'", timeout_sec=500)
    data = json.loads(result_json)
    assert data["exit_code"] == 0
    assert "START_OUT_" in data["stdout"]
    assert "_END_OUT" in data["stdout"]
    assert "...[truncated " in data["stdout"]

    # Verify timeout_sec was clamped to 120s and docker exec args match specification
    assert recorded_cmds[-1] == [
        "docker",
        "exec",
        "-u",
        "agent",
        "-w",
        "/home/agent/workspace",
        "thunk-workspace",
        "timeout",
        "--kill-after=2s",
        "120s",
        "bash",
        "-c",
        "python3 -c 'print(1)'",
    ]
    # Verify _sync_mind_snapshot ran before command execution
    assert (mind_dir / "README.md").exists()

    # 2. Timeout handling
    hanging_proc = DummyProcess(b"", b"", returncode=124, hang=True)

    async def mock_exec_hang(*args, **kwargs):
        return hanging_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_exec_hang)

    orig_wait_for = asyncio.wait_for

    async def fast_timeout_wait_for(coro, timeout):
        return await orig_wait_for(coro, timeout=0.05)

    monkeypatch.setattr(asyncio, "wait_for", fast_timeout_wait_for)

    timeout_json = await comp.run_workspace_command("sleep 999", timeout_sec=1)
    timeout_data = json.loads(timeout_json)
    assert timeout_data["exit_code"] == 124
    assert "timed out" in timeout_data["stderr"]
    assert hanging_proc.killed is True


@pytest.mark.asyncio
async def test_workspace_filepath_alias_and_error_reporting(tmp_path):
    """Verifies AgentCore.execute_tool aliases filepath <-> path <-> subdir and reports received vs expected params on TypeError."""
    from src.core.agent import AgentCore

    ws_dir = tmp_path / "workspace"
    mind_dir = tmp_path / "mind_ro"
    live_db = tmp_path / "memory.db"

    comp = WorkspaceComponent(
        workspace_dir=str(ws_dir),
        mind_ro_dir=str(mind_dir),
        live_db_path=str(live_db),
    )

    agent = AgentCore()
    agent.register_component(comp)
    agent._build_tools()

    # 1. write_workspace_file called with 'filepath' instead of 'path'
    write_res = await agent.execute_tool(
        "write_workspace_file",
        json.dumps({"filepath": "museum/crab_union_manifesto.md", "content": "Crab Union Rules"}),
    )
    assert "Success" in write_res

    # 2. read_workspace_file called with 'filepath' and extra 'library' arg (exact Wisp tool call)
    read_res = await agent.execute_tool(
        "read_workspace_file",
        json.dumps({"filepath": "museum/crab_union_manifesto.md", "library": "workspace"}),
    )
    assert read_res == "Crab Union Rules"

    # 3. list_workspace_files called with 'path' instead of 'subdir'
    list_res = await agent.execute_tool(
        "list_workspace_files",
        json.dumps({"path": "museum"}),
    )
    assert "crab_union_manifesto.md" in list_res

    # 4. Missing required argument reports received args and expected params
    err_res = await agent.execute_tool(
        "read_workspace_file",
        json.dumps({"wrong_arg": "museum/crab_union_manifesto.md"}),
    )
    assert "received args: ['wrong_arg']" in err_res
    assert "expected params: ['path']" in err_res

