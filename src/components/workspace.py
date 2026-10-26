import asyncio
import json
import os
import shutil
import sqlite3
from typing import Callable, Dict, List, Optional, Tuple
import fitz
from bs4 import BeautifulSoup
from ebooklib import epub
from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.image import downsize_image_bytes, encode_jpeg_data_url


MIND_README_CONTENT = """# Agent Mind Read-Only Mirror (`/home/agent/mind`)

This directory contains an automatically updated, read-only snapshot of your own SQLite state databases and procedural skills. It is refreshed prior to every `run_workspace_command` execution so you can safely query your own memory, notes, tasks, and reading state using `sqlite3`, Python, or Go without risking database locks or corruption.

## Databases & Schemas

### 1. `memory.db` (Tiered & Subconscious Memory)
- `archival_log`: `(id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, timestamp DATETIME)` — Complete chronological log of all messages and tool executions.
- `core_summary`: `(id INTEGER PRIMARY KEY, summary TEXT)` — Current active rolling core summary (row `id = 1`).
- `core_summary_history`: `(id INTEGER PRIMARY KEY AUTOINCREMENT, summary TEXT, compacted_up_to_id INTEGER, timestamp DATETIME)` — Historical log of every memory compaction and summary revision.
- `memory_settings`: `(key TEXT PRIMARY KEY, value TEXT)` — Compaction state pointers (e.g., `last_compacted_id`).
- `layer_1_pool`: `(id TEXT PRIMARY KEY, content TEXT, metadata TEXT, hits INTEGER, age INTEGER, last_similarity REAL, updated_at DATETIME)` — Stateful Spreading Activation working pool candidates.

### 2. `notes.db` (Structured Notes)
- `notes`: `(key TEXT PRIMARY KEY, value TEXT)` — Hierarchical key-value scratchpad notes (including `sticky`).

### 3. `tasks.db` (Priority Task Queue)
- `task_queue`: `(id INTEGER PRIMARY KEY AUTOINCREMENT, priority INTEGER, description TEXT)` — Pending autonomous tasks.

### 4. `reader_state.db` (EPUB Sequential Reader)
- `reader_state`: `(id INTEGER PRIMARY KEY, current_book TEXT, current_index INTEGER)` — Active EPUB bookmark state.
- `epub_metadata_cache`: `(filepath TEXT PRIMARY KEY, mtime REAL, title TEXT, author TEXT, description TEXT)` — Cached EPUB metadata.

### 5. `pdf_state.db` (PDF Viewer)
- `pdf_state`: `(id INTEGER PRIMARY KEY, current_pdf TEXT, current_page INTEGER)` — Active PDF bookmark state.
- `pdf_metadata_cache`: `(filepath TEXT PRIMARY KEY, mtime REAL, page_count INTEGER, title TEXT, author TEXT, subject TEXT)` — Cached PDF metadata.

### 6. `skills/` (Procedural Skills)
- Markdown files (`*.md`) with YAML frontmatter containing your saved procedural workflows.
"""


class WorkspaceComponent(BaseComponent):
    """
    Provides the agent with an isolated, air-gapped Linux container workspace
    (/home/agent/workspace RW) and a read-only replica of its own SQLite databases
    and skills (/home/agent/mind RO).
    """

    def __init__(
        self,
        container_name: str = None,
        container_user: str = None,
        container_workdir: str = None,
        workspace_dir: str = None,
        mind_ro_dir: str = None,
        live_db_path: str = None,
        default_timeout_sec: int = None,
        max_output_chars: int = None,
        extra_db_paths: Dict[str, str] = None,
        skills_dir: str = None,
    ):
        self.config = load_config()
        ws_cfg = self.config.get("workspace", {})

        self.container_name = container_name or ws_cfg.get("container_name", "thunk-workspace")
        self.container_user = container_user or ws_cfg.get("container_user", "agent")
        self.container_workdir = container_workdir or ws_cfg.get("container_workdir", "/home/agent/workspace")
        self.container_mind_dir = ws_cfg.get("container_mind_dir", "/home/agent/mind")

        raw_ws_dir = workspace_dir or ws_cfg.get("workspace_dir", "data/thunk/workspace")
        raw_mind_dir = mind_ro_dir or ws_cfg.get("mind_ro_dir", "data/thunk/mind_ro")
        self.workspace_dir = os.path.abspath(raw_ws_dir)
        self.mind_ro_dir = os.path.abspath(raw_mind_dir)

        self.live_db_path = (
            live_db_path
            or ws_cfg.get("live_db_path")
            or self.config.get("memory", {}).get("db_path", "data/thunk/memory.db")
        )
        self.default_timeout_sec = int(
            default_timeout_sec if default_timeout_sec is not None else ws_cfg.get("default_timeout_sec", 30)
        )
        self.max_output_chars = int(
            max_output_chars if max_output_chars is not None else ws_cfg.get("max_output_chars", 6000)
        )

        if extra_db_paths is not None:
            self.extra_db_paths = dict(extra_db_paths)
        elif live_db_path is not None:
            # When an explicit live_db_path is injected (e.g. in unit tests), only mirror sibling DBs if they exist
            base_agent_dir = os.path.dirname(os.path.abspath(live_db_path))
            self.extra_db_paths = {
                "notes.db": os.path.join(base_agent_dir, "notes.db"),
                "tasks.db": os.path.join(base_agent_dir, "tasks.db"),
                "reader_state.db": os.path.join(base_agent_dir, "reader_state.db"),
            }
        else:
            base_agent_dir = os.path.dirname(os.path.abspath(self.live_db_path))
            reader_db = (
                self.config.get("file_reader", {}).get("db_path")
                or self.config.get("reader", {}).get("db_path")
                or os.path.join(base_agent_dir, "reader_state.db")
            )
            self.extra_db_paths = {
                "notes.db": self.config.get("notes", {}).get("db_path", ""),
                "tasks.db": self.config.get("tasks", {}).get("db_path", ""),
                "reader_state.db": reader_db,
            }

        if skills_dir is not None:
            self.skills_dir = skills_dir
        elif live_db_path is not None:
            self.skills_dir = os.path.join(os.path.dirname(os.path.abspath(live_db_path)), "skills")
        else:
            self.skills_dir = self.config.get("skills", {}).get("skills_dir", "")

        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.mind_ro_dir, exist_ok=True)

    def get_system_prompt_addition(self) -> str:
        return (
            "--- LINUX WORKSPACE & MIND MIRROR ---\n"
            f"You have a persistent, air-gapped Linux workspace at `{self.container_workdir}` (read-write) "
            "equipped with Python 3 (numpy, pandas, networkx, sympy, matplotlib), Go, Java, Bazel, C/C++, sqlite3, ripgrep, jq, and standard CLI tools.\n"
            f"You also have a read-only mirror of your own internal SQLite databases (`memory.db`, `notes.db`, `tasks.db`, `reader_state.db`) "
            f"and `skills/` mounted at `{self.container_mind_dir}` (see `{self.container_mind_dir}/README.md` for table schemas).\n"
            "Use `run_workspace_command`, `write_workspace_file`, `read_workspace_file`, `view_workspace_image`, and `list_workspace_files` "
            "to write scripts, run experiments, inspect generated plots/PDFs, or query your own memory database."
        )

    def get_tools(self) -> List[Callable]:
        return [
            self.run_workspace_command,
            self.write_workspace_file,
            self.read_workspace_file,
            self.view_workspace_image,
            self.list_workspace_files,
        ]

    @staticmethod
    def _backup_sqlite_file(src_path: str, dst_path: str) -> None:
        """Safely snapshots a live SQLite database using sqlite3 online backup into a temp file then moves atomically."""
        if not src_path or not os.path.exists(src_path):
            return
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        tmp_path = f"{dst_path}.tmp"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        src_conn = sqlite3.connect(src_path)
        dst_conn = sqlite3.connect(tmp_path)
        try:
            with src_conn, dst_conn:
                src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
            src_conn.close()
        os.replace(tmp_path, dst_path)

    def _sync_mind_snapshot(self) -> None:
        """Snapshots live SQLite databases and skills into mind_ro_dir before workspace command execution."""
        os.makedirs(self.mind_ro_dir, exist_ok=True)

        # 1. Mirror primary memory.db
        if self.live_db_path and os.path.exists(self.live_db_path):
            mirror_db_path = os.path.join(self.mind_ro_dir, "memory.db")
            self._backup_sqlite_file(self.live_db_path, mirror_db_path)

        # 2. Mirror supplementary agent databases if present
        for filename, src_db in self.extra_db_paths.items():
            if src_db and os.path.exists(src_db):
                dst_db = os.path.join(self.mind_ro_dir, filename)
                self._backup_sqlite_file(src_db, dst_db)

        # 3. Mirror skills directory if present
        if self.skills_dir and os.path.isdir(self.skills_dir):
            dst_skills_dir = os.path.join(self.mind_ro_dir, "skills")
            os.makedirs(dst_skills_dir, exist_ok=True)
            src_files = {f for f in os.listdir(self.skills_dir) if os.path.isfile(os.path.join(self.skills_dir, f))}
            for fname in src_files:
                src_f = os.path.join(self.skills_dir, fname)
                dst_f = os.path.join(dst_skills_dir, fname)
                tmp_f = f"{dst_f}.tmp"
                shutil.copy2(src_f, tmp_f)
                os.replace(tmp_f, dst_f)
            # Remove stale skill files no longer in live skills_dir
            for existing in os.listdir(dst_skills_dir):
                if existing not in src_files and not existing.endswith(".tmp"):
                    try:
                        os.remove(os.path.join(dst_skills_dir, existing))
                    except OSError:
                        pass

        # 4. Ensure README.md exists in mind_ro_dir
        readme_path = os.path.join(self.mind_ro_dir, "README.md")
        if not os.path.exists(readme_path):
            tmp_readme = f"{readme_path}.tmp"
            with open(tmp_readme, "w", encoding="utf-8") as f:
                f.write(MIND_README_CONTENT)
            os.replace(tmp_readme, readme_path)

    def _truncate_output(self, text: str) -> str:
        """Truncates text exceeding max_output_chars while preserving both head and tail context."""
        if not text or len(text) <= self.max_output_chars:
            return text
        half = self.max_output_chars // 2
        omitted = len(text) - self.max_output_chars
        return f"{text[:half]}\n...[truncated {omitted} chars]...\n{text[-half:]}"

    def _resolve_workspace_write_path(self, path: str) -> str:
        """Resolves a path strictly inside workspace_dir for writing."""
        if not path or not path.strip():
            raise ValueError("Path must not be empty.")
        clean = path.strip()

        # Allow container-prefixed workspace path (e.g. /home/agent/workspace/foo.py)
        ws_prefix = self.container_workdir.rstrip("/") + "/"
        if clean.startswith(ws_prefix):
            clean = clean[len(ws_prefix):]
        elif clean.startswith("/"):
            raise ValueError(f"Access denied: absolute path '{path}' is outside of workspace.")

        norm_parts = os.path.normpath(clean).split(os.sep)
        if ".." in norm_parts:
            raise ValueError(f"Access denied: path traversal ('..') is not allowed in '{path}'.")

        abs_path = os.path.abspath(os.path.join(self.workspace_dir, clean))
        if os.path.commonpath([self.workspace_dir, abs_path]) != self.workspace_dir or abs_path == self.workspace_dir:
            raise ValueError(f"Access denied: path '{path}' resolves outside of workspace.")
        return abs_path

    def _resolve_workspace_read_path(self, path: str, allow_dir: bool = False) -> Tuple[str, str]:
        """
        Resolves a path inside workspace_dir (or mind_ro_dir if prefixed with /home/agent/mind or mind/).
        Returns (abs_path, base_dir).
        """
        if not path or not path.strip():
            if allow_dir:
                return self.workspace_dir, self.workspace_dir
            raise ValueError("Path must not be empty.")

        clean = path.strip()
        base_dir = self.workspace_dir

        mind_prefix = self.container_mind_dir.rstrip("/")
        ws_prefix = self.container_workdir.rstrip("/")

        if clean == mind_prefix or clean.startswith(mind_prefix + "/"):
            base_dir = self.mind_ro_dir
            clean = clean[len(mind_prefix):].lstrip("/") or "."
        elif clean == "mind" or clean.startswith("mind/"):
            base_dir = self.mind_ro_dir
            clean = clean[len("mind"):].lstrip("/") or "."
        elif clean == ws_prefix or clean.startswith(ws_prefix + "/"):
            base_dir = self.workspace_dir
            clean = clean[len(ws_prefix):].lstrip("/") or "."
        elif clean.startswith("/"):
            raise ValueError(f"Access denied: path '{path}' is outside of workspace and mind directories.")

        norm_parts = os.path.normpath(clean).split(os.sep)
        if ".." in norm_parts:
            raise ValueError(f"Access denied: path traversal ('..') is not allowed in '{path}'.")

        abs_path = os.path.abspath(os.path.join(base_dir, clean))
        if os.path.commonpath([base_dir, abs_path]) != base_dir:
            raise ValueError(f"Access denied: path '{path}' resolves outside of allowed directory.")
        if not allow_dir and abs_path == base_dir:
            raise ValueError(f"Error: '{path}' is a directory, not a file.")
        return abs_path, base_dir

    async def run_workspace_command(
        self,
        command: str,
        workdir: str = "/home/agent/workspace",
        timeout_sec: int = 30,
    ) -> str:
        """
        Executes a bash command inside your air-gapped Linux workspace container.
        Your read-only mind snapshot (/home/agent/mind) is synchronized before execution.
        Returns JSON with exit_code, stdout, and stderr.
        """
        if not command or not command.strip():
            return json.dumps({"exit_code": -1, "stdout": "", "stderr": "Error: command must not be empty."})

        timeout_val = max(1, min(120, int(timeout_sec if timeout_sec is not None else self.default_timeout_sec)))
        effective_workdir = (workdir or self.container_workdir).strip()

        try:
            self._sync_mind_snapshot()
        except Exception as e:
            return json.dumps(
                {"exit_code": -1, "stdout": "", "stderr": f"Error syncing mind snapshot before command: {e}"}
            )

        cmd = [
            "docker",
            "exec",
            "-u",
            self.container_user,
            "-w",
            effective_workdir,
            self.container_name,
            "timeout",
            "--kill-after=2s",
            f"{timeout_val}s",
            "bash",
            "-c",
            command,
        ]

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_val + 5,
            )
            exit_code = proc.returncode if proc.returncode is not None else 0
            stdout_str = self._truncate_output(stdout_bytes.decode("utf-8", errors="replace"))
            stderr_str = self._truncate_output(stderr_bytes.decode("utf-8", errors="replace"))
            return json.dumps(
                {
                    "exit_code": exit_code,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                },
                indent=2,
            )
        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            return json.dumps(
                {
                    "exit_code": 124,
                    "stdout": "",
                    "stderr": f"Error: Command timed out after {timeout_val} seconds.",
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": f"Error executing workspace command: {str(e)}",
                },
                indent=2,
            )

    def write_workspace_file(self, path: str, content: str) -> str:
        """
        Writes a UTF-8 text file to your persistent workspace (/home/agent/workspace).
        Creates parent directories automatically.
        """
        try:
            abs_path = self._resolve_workspace_write_path(path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            tmp_path = f"{abs_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, abs_path)
            rel = os.path.relpath(abs_path, self.workspace_dir)
            return f"Success: Wrote {len(content)} chars to workspace file '{rel}'."
        except Exception as e:
            return f"Error writing workspace file '{path}': {str(e)}"

    def read_workspace_file(self, path: str, chunk: int = 1) -> str:
        """
        Reads a file (UTF-8 text, PDF page, or EPUB chunk) from your workspace (/home/agent/workspace)
        or mind mirror (/home/agent/mind). Pass 'chunk' (1-indexed) to read specific pages of PDFs or chunks of large files.
        """
        try:
            abs_path, base_dir = self._resolve_workspace_read_path(path, allow_dir=False)
            if not os.path.exists(abs_path):
                return f"Error: File '{path}' does not exist."
            if not os.path.isfile(abs_path):
                return f"Error: '{path}' is not a regular file."

            rel_path = os.path.relpath(abs_path, base_dir)
            chunk_idx = max(1, int(chunk if chunk is not None else 1))
            ext = os.path.splitext(abs_path)[1].lower()

            if ext == ".pdf":
                with fitz.open(abs_path) as doc:
                    total_pages = len(doc)
                    if chunk_idx > total_pages:
                        return json.dumps(
                            {
                                "format": "pdf",
                                "path": rel_path,
                                "error": f"Invalid chunk/page {chunk_idx}. PDF has {total_pages} pages (1..{total_pages}).",
                                "total_pages": total_pages,
                            },
                            indent=2,
                        )
                    page = doc[chunk_idx - 1]
                    text = page.get_text("text").strip()
                    raw_images = page.get_images(full=True)
                    images = [
                        {"width": int(img[2]), "height": int(img[3])}
                        for img in raw_images
                        if int(img[2]) > 8 and int(img[3]) > 8
                    ]
                    vector_drawing_count = len(page.get_drawings())
                    has_visuals = bool(images) or vector_drawing_count > 5

                next_hint = (
                    f"Call read_workspace_file(path='{rel_path}', chunk={chunk_idx + 1}) for page {chunk_idx + 1}/{total_pages}."
                    if chunk_idx < total_pages
                    else f"End of PDF '{rel_path}' reached ({total_pages}/{total_pages})."
                )
                res = {
                    "format": "pdf",
                    "path": rel_path,
                    "chunk": chunk_idx,
                    "page_number": chunk_idx,
                    "total_pages": total_pages,
                    "total_chunks": total_pages,
                    "content": text,
                    "has_unrenderable_visuals": has_visuals,
                    "image_count": len(images),
                    "vector_drawing_count": vector_drawing_count,
                    "next_hint": next_hint,
                }
                if has_visuals or not text:
                    res["notice"] = (
                        f"Call view_workspace_image(path='{rel_path}', chunk={chunk_idx}) to visually inspect this page."
                    )
                return json.dumps(res, indent=2)

            if ext == ".epub":
                book = epub.read_epub(abs_path)
                parts = []
                for item in book.get_items_of_type(9):
                    soup = BeautifulSoup(item.get_body_content(), "html.parser")
                    t = soup.get_text(separator=" ", strip=True)
                    if t:
                        parts.append(t)
                combined = "\n\n".join(parts)
                c_size = 2500
                chunks = [combined[i : i + c_size] for i in range(0, len(combined), c_size)] or [""]
                total_chunks = len(chunks)
                if chunk_idx > total_chunks:
                    return json.dumps(
                        {
                            "format": "epub",
                            "path": rel_path,
                            "error": f"Requested chunk {chunk_idx} exceeds total_chunks ({total_chunks}).",
                            "total_chunks": total_chunks,
                        },
                        indent=2,
                    )
                return json.dumps(
                    {
                        "format": "epub",
                        "path": rel_path,
                        "chunk": chunk_idx,
                        "total_chunks": total_chunks,
                        "content": chunks[chunk_idx - 1],
                        "next_hint": (
                            f"Call read_workspace_file(path='{rel_path}', chunk={chunk_idx + 1}) for chunk {chunk_idx + 1}/{total_chunks}."
                            if chunk_idx < total_chunks
                            else f"End of EPUB '{rel_path}' reached."
                        ),
                    },
                    indent=2,
                )

            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            if chunk_idx == 1:
                return self._truncate_output(content)

            step = max(1, self.max_output_chars)
            chunks = [content[i : i + step] for i in range(0, len(content), step)] or [""]
            if chunk_idx > len(chunks):
                return f"Error: Requested chunk {chunk_idx} exceeds total_chunks ({len(chunks)})."
            return chunks[chunk_idx - 1]
        except Exception as e:
            return f"Error reading workspace file '{path}': {str(e)}"

    def view_workspace_image(self, path: str, chunk: int = 1) -> str:
        """
        Loads an image file (or renders page 'chunk' of a PDF) from your workspace (/home/agent/workspace),
        downsizes it to at most 1024px on its longest edge, and attaches it to your visual context for the current turn.
        """
        try:
            abs_path, base_dir = self._resolve_workspace_read_path(path, allow_dir=False)
            if not os.path.exists(abs_path):
                return f"Error: File '{path}' does not exist."
            if not os.path.isfile(abs_path):
                return f"Error: '{path}' is not a regular file."

            rel_path = os.path.relpath(abs_path, base_dir)
            ext = os.path.splitext(abs_path)[1].lower()

            if ext == ".pdf":
                page_number = max(1, int(chunk if chunk is not None else 1))
                with fitz.open(abs_path) as doc:
                    total_pages = len(doc)
                    if page_number < 1 or page_number > total_pages:
                        return f"Error: Invalid page/chunk {page_number}. PDF has {total_pages} pages (1..{total_pages})."
                    page = doc[page_number - 1]
                    pix = page.get_pixmap(dpi=150, alpha=False)
                    raw_png = pix.tobytes("png")

                jpeg_bytes, width, height = downsize_image_bytes(raw_png, max_edge=1024)
                data_url = encode_jpeg_data_url(jpeg_bytes)
                if hasattr(self, "agent") and self.agent is not None:
                    self.agent.pending_tool_images.append(data_url)

                return (
                    f"Rendered page {page_number}/{total_pages} of workspace PDF '{rel_path}' "
                    f"({width}x{height}px) into your visual context for this turn."
                )

            with open(abs_path, "rb") as f:
                raw_bytes = f.read()

            jpeg_bytes, width, height = downsize_image_bytes(raw_bytes, max_edge=1024)
            data_url = encode_jpeg_data_url(jpeg_bytes)
            if hasattr(self, "agent") and self.agent is not None:
                self.agent.pending_tool_images.append(data_url)

            return (
                f"Loaded workspace image '{rel_path}' ({width}x{height}px) into your visual context for this turn."
            )
        except Exception as e:
            return f"Error viewing workspace image '{path}': {str(e)}"

    def list_workspace_files(self, subdir: str = ".") -> str:
        """
        Lists files and directories inside your workspace (or /home/agent/mind if prefixed with 'mind/').
        """
        try:
            abs_path, base_dir = self._resolve_workspace_read_path(subdir or ".", allow_dir=True)
            if not os.path.exists(abs_path):
                return f"Error: Directory '{subdir}' does not exist."
            if not os.path.isdir(abs_path):
                return f"Error: '{subdir}' is not a directory."

            entries = os.listdir(abs_path)
            if not entries:
                return f"Directory '{subdir or '.'}' is empty."

            dirs = sorted([e for e in entries if os.path.isdir(os.path.join(abs_path, e))])
            files = sorted([e for e in entries if os.path.isfile(os.path.join(abs_path, e))])

            rel_display = os.path.relpath(abs_path, base_dir)
            lines = [f"Contents of {rel_display}:"]
            for d in dirs:
                lines.append(f"[DIR]  {d}")
            for f in files:
                size = os.path.getsize(os.path.join(abs_path, f))
                lines.append(f"[FILE] {f} ({size} bytes)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing workspace directory '{subdir}': {str(e)}"
