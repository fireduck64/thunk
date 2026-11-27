import glob
import io
import json
import os
import re
import shutil
import sqlite3
import uuid
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
from ebooklib import epub
import fitz
from PIL import Image

from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.image import downsize_image_bytes, encode_jpeg_data_url

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

LIBRARY_ALIASES = {
    "ebooks": "ebook",
    "ebook": "ebooks",
    "pdfs": "pdf",
    "pdf": "pdfs",
    "discord_images": "discord_file",
    "discord_files": "discord_file",
    "discord_file": "discord_images",
    "gutenburg": "gutenberg",
    "project_gutenberg": "gutenberg",
}

GUTINDEX_ENTRY_RE = re.compile(r"^(\S.+?)\s{2,}(\d+)C?\s*$")
MAX_WALK_FILES_PER_LIBRARY = 10000
MAX_DIR_LIST_ENTRIES = 100


class FileReaderComponent(BaseComponent):
    """
    Unified file, document (EPUB/PDF/Gutenberg text), and image reader across all configured libraries.
    Supports cross-library search (`library='all'`) and zero-walk Project Gutenberg indexing via GUTINDEX.ALL.
    Maintains persistent reading bookmark state in SQLite for sequential reading via `read_next_chunk()`.
    """

    def __init__(
        self,
        libraries: Dict[str, str] = None,
        db_path: str = None,
        chunk_size: int = None,
        lines_per_chunk: int = 200,
        shared_workspace_dir: str = None,
        thread_id: str = "main",
    ):
        self.config = load_config()
        self.thread_id = (thread_id or "main").strip() or "main"
        file_cfg = self.config.get("file_reader", {})
        explicit_libs_passed = libraries is not None

        if libraries is None:
            raw_libs = dict(file_cfg.get("libraries") or {})
            legacy_ebook = self.config.get("reader", {}).get("library_dir")
            if legacy_ebook and "ebook" not in raw_libs and "ebooks" not in raw_libs:
                raw_libs["ebook"] = legacy_ebook
            legacy_pdf = self.config.get("pdf_viewer", {}).get("library_dir")
            if legacy_pdf and "pdf" not in raw_libs and "pdfs" not in raw_libs:
                raw_libs["pdf"] = legacy_pdf
            ws_dir_cfg = self.config.get("workspace", {}).get("workspace_dir")
            if ws_dir_cfg and "workspace" not in raw_libs:
                raw_libs["workspace"] = ws_dir_cfg
            if not raw_libs:
                raise ValueError(
                    "FileReaderComponent requires 'libraries' to be explicitly defined in config under [file_reader.libraries]"
                )
            libraries = raw_libs

        self.libraries: Dict[str, str] = {}
        for name, path in libraries.items():
            if path:
                self.libraries[name] = os.path.abspath(path)

        ws_cfg = self.config.get("workspace", {})
        self.container_workdir = (ws_cfg.get("container_workdir") or "/home/agent/workspace").rstrip("/")
        if shared_workspace_dir is not None:
            self.shared_workspace_dir: Optional[str] = os.path.abspath(shared_workspace_dir)
        elif "workspace" in self.libraries:
            ws_abs = self.libraries["workspace"]
            if explicit_libs_passed or os.path.basename(os.path.dirname(ws_abs)) == "test":
                self.shared_workspace_dir = os.path.abspath(
                    os.path.join(os.path.dirname(ws_abs), "shared_workspace")
                )
            elif ws_cfg.get("shared_workspace_dir"):
                self.shared_workspace_dir = os.path.abspath(ws_cfg["shared_workspace_dir"])
            else:
                self.shared_workspace_dir = os.path.abspath("data/shared/workspace")
        else:
            self.shared_workspace_dir = None
        if self.shared_workspace_dir:
            os.makedirs(self.shared_workspace_dir, exist_ok=True)

        if db_path is not None:
            self.db_path = db_path
        elif file_cfg.get("db_path"):
            self.db_path = file_cfg["db_path"]
        elif self.config.get("reader", {}).get("db_path"):
            self.db_path = self.config["reader"]["db_path"]
        else:
            mem_db = self.config.get("memory", {}).get("db_path", "data/thunk/memory.db")
            self.db_path = os.path.join(os.path.dirname(os.path.abspath(mem_db)), "reader_state.db")

        self.chunk_size = int(
            chunk_size if chunk_size is not None else file_cfg.get("chunk_size", 9000)
        )
        self.lines_per_chunk = int(
            lines_per_chunk if lines_per_chunk is not None else file_cfg.get("lines_per_chunk", 200)
        )

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        for lib_key in ("ebook", "ebooks", "pdf", "pdfs", "discord_file", "discord_images"):
            if lib_key in self.libraries:
                os.makedirs(self.libraries[lib_key], exist_ok=True)

        self.current_book: Optional[str] = None
        self.current_index: int = 0
        self.current_library: Optional[str] = None
        self.current_rel_path: Optional[str] = None
        self.current_book_label: Optional[str] = None
        self.current_chunks: List[str] = []
        self.current_chunk_titles: List[str] = []

        self._init_db()
        self._load_state()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reader_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        current_book TEXT,
                        current_index INTEGER,
                        current_library TEXT,
                        current_rel_path TEXT
                    )
                    """
                )
                cols = [r[1] for r in conn.execute("PRAGMA table_info(reader_state)").fetchall()]
                if "current_library" not in cols:
                    conn.execute("ALTER TABLE reader_state ADD COLUMN current_library TEXT")
                if "current_rel_path" not in cols:
                    conn.execute("ALTER TABLE reader_state ADD COLUMN current_rel_path TEXT")

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS book_metadata (
                        rel_path TEXT PRIMARY KEY,
                        mtime REAL,
                        title TEXT,
                        author TEXT,
                        label TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pdf_metadata (
                        rel_path TEXT PRIMARY KEY,
                        mtime REAL,
                        title TEXT,
                        author TEXT,
                        total_pages INTEGER,
                        label TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gutenberg_meta (
                        library TEXT PRIMARY KEY,
                        gutindex_mtime REAL,
                        entry_count INTEGER
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gutenberg_catalog (
                        library TEXT NOT NULL,
                        ebook_id INTEGER NOT NULL,
                        description TEXT NOT NULL,
                        PRIMARY KEY (library, ebook_id)
                    )
                    """
                )
                conn.execute(
                    "INSERT OR IGNORE INTO reader_state (id, current_book, current_index, current_library, current_rel_path) "
                    "VALUES (1, NULL, 0, NULL, NULL)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reader_thread_state (
                        thread_id TEXT PRIMARY KEY,
                        current_book TEXT,
                        current_index INTEGER,
                        current_library TEXT,
                        current_rel_path TEXT
                    )
                    """
                )
                if self.thread_id in ("main", "mind"):
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO reader_thread_state (
                            thread_id, current_book, current_index, current_library, current_rel_path
                        )
                        SELECT ?, current_book, current_index, current_library, current_rel_path
                        FROM reader_state
                        WHERE id = 1
                        """,
                        (self.thread_id,),
                    )
                else:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO reader_thread_state (
                            thread_id, current_book, current_index, current_library, current_rel_path
                        )
                        VALUES (?, NULL, 0, NULL, NULL)
                        """,
                        (self.thread_id,),
                    )
        finally:
            conn.close()

    def _save_state(self) -> None:
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO reader_thread_state (
                        thread_id, current_book, current_index, current_library, current_rel_path
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        current_book = excluded.current_book,
                        current_index = excluded.current_index,
                        current_library = excluded.current_library,
                        current_rel_path = excluded.current_rel_path
                    """,
                    (
                        self.thread_id,
                        self.current_book,
                        self.current_index,
                        self.current_library,
                        self.current_rel_path,
                    ),
                )
                if self.thread_id in ("main", "mind"):
                    conn.execute(
                        """
                        UPDATE reader_state
                        SET current_book = ?, current_index = ?, current_library = ?, current_rel_path = ?
                        WHERE id = 1
                        """,
                        (self.current_book, self.current_index, self.current_library, self.current_rel_path),
                    )
        finally:
            conn.close()

    def _infer_library_and_rel_path(self, abs_file_path: str) -> Tuple[Optional[str], Optional[str]]:
        abs_target = os.path.abspath(abs_file_path)
        if self.shared_workspace_dir and "workspace" in self.libraries:
            try:
                if os.path.commonpath([self.shared_workspace_dir, abs_target]) == self.shared_workspace_dir:
                    rel_s = os.path.relpath(abs_target, self.shared_workspace_dir)
                    return "workspace", ("shared" if rel_s == "." else f"shared/{rel_s}")
            except ValueError:
                pass
        for lib_name, base_dir in self.libraries.items():
            try:
                if os.path.commonpath([base_dir, abs_target]) == base_dir:
                    return lib_name, os.path.relpath(abs_target, base_dir)
            except ValueError:
                continue
        return None, None

    def _load_state(self) -> None:
        conn = self._connect_db()
        try:
            with conn:
                row = conn.execute(
                    "SELECT current_book, current_index, current_library, current_rel_path "
                    "FROM reader_thread_state WHERE thread_id = ?",
                    (self.thread_id,),
                ).fetchone()
                if row is None and self.thread_id == "main":
                    row = conn.execute(
                        "SELECT current_book, current_index, current_library, current_rel_path FROM reader_state WHERE id = 1"
                    ).fetchone()
        finally:
            conn.close()

        if not row:
            return

        self.current_book = row[0]
        self.current_index = int(row[1] or 0)
        self.current_library = row[2]
        self.current_rel_path = row[3]
        self.current_book_label = None
        self.current_chunks = []
        self.current_chunk_titles = []

        if self.current_book and os.path.exists(self.current_book):
            if not self.current_library or not self.current_rel_path:
                lib_name, rel_p = self._infer_library_and_rel_path(self.current_book)
                self.current_library = lib_name
                self.current_rel_path = rel_p

            ext = os.path.splitext(self.current_book)[1].lower()
            if ext == ".epub":
                self.current_book_label = self._format_epub_label(
                    self.current_book, self.current_rel_path or os.path.basename(self.current_book)
                )
                try:
                    self._parse_epub_chunks(self.current_book)
                except Exception as e:
                    print(f"[FileReader Warning] Failed to restore EPUB '{self.current_book}': {e}")
            elif ext in (".htm", ".html", ".xhtml") and self._is_book_library(self.current_library):
                self.current_book_label = self.current_rel_path or os.path.basename(self.current_book)
                try:
                    self._parse_html_book_chunks(self.current_book)
                except Exception as e:
                    print(f"[FileReader Warning] Failed to restore HTML book '{self.current_book}': {e}")
            elif ext == ".pdf":
                self.current_book_label = self._format_pdf_label(
                    self.current_book, self.current_rel_path or os.path.basename(self.current_book)
                )
            else:
                self.current_book_label = self.current_rel_path or os.path.basename(self.current_book)

    def _resolve_library_name(self, library: str) -> str:
        clean = (library or "").strip()
        if clean in self.libraries:
            return clean
        lower = clean.lower()
        for k in self.libraries:
            if k.lower() == lower:
                return k
        alias = LIBRARY_ALIASES.get(lower)
        if alias and alias in self.libraries:
            return alias
        raise ValueError(f"Library '{library}' is not configured. Available libraries: {list(self.libraries.keys())}")

    @staticmethod
    def _clean_filepath_arg(filepath: str) -> str:
        clean = (filepath or "").strip()
        if "[path: " in clean and clean.endswith("]"):
            clean = clean.split("[path: ", 1)[1][:-1].strip()
        return clean

    # --- Project Gutenberg Catalog & Path Helpers ---

    @staticmethod
    def _gutenberg_id_to_dir(ebook_id: int) -> str:
        eid = int(ebook_id)
        if eid < 10:
            return f"0/{eid}"
        s = str(eid)
        return "/".join(list(s[:-1]) + [s])

    def _resolve_gutenberg_book_rel_path(self, base_dir: str, ebook_id: int) -> str:
        book_dir = self._gutenberg_id_to_dir(ebook_id)
        abs_book_dir = os.path.join(base_dir, book_dir)
        if not os.path.isdir(abs_book_dir):
            return f"{book_dir}/{ebook_id}-0.txt"

        try:
            entries = sorted(os.listdir(abs_book_dir))
        except OSError:
            return f"{book_dir}/{ebook_id}-0.txt"

        # Prefer UTF-8 text, then standard text, then EPUB, then HTML subdir
        preferred_names = [f"{ebook_id}-0.txt", f"{ebook_id}-8.txt", f"{ebook_id}.txt"]
        for cand in preferred_names:
            if cand in entries and os.path.isfile(os.path.join(abs_book_dir, cand)):
                return f"{book_dir}/{cand}"

        for entry in entries:
            lower = entry.lower()
            if (lower.endswith(".txt") or lower.endswith(".epub")) and lower != "license.txt":
                if os.path.isfile(os.path.join(abs_book_dir, entry)):
                    return f"{book_dir}/{entry}"

        html_dir = os.path.join(abs_book_dir, f"{ebook_id}-h")
        if os.path.isdir(html_dir):
            for h_cand in (f"{ebook_id}-h.htm", f"{ebook_id}-h.html"):
                if os.path.isfile(os.path.join(html_dir, h_cand)):
                    return f"{book_dir}/{ebook_id}-h/{h_cand}"

        return book_dir

    @staticmethod
    def _parse_gutindex_file(gutindex_path: str) -> List[Tuple[int, str]]:
        entries_by_id: Dict[int, str] = {}
        current_id: Optional[int] = None
        current_lines: List[str] = []

        with open(gutindex_path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if "<==End of GUTINDEX.ALL==>" in line:
                    break
                m = GUTINDEX_ENTRY_RE.match(line)
                if m:
                    if current_id is not None and current_id not in entries_by_id:
                        entries_by_id[current_id] = " ".join(current_lines)
                    title_part, ebook_id = m.group(1).strip(), int(m.group(2))
                    current_id = ebook_id
                    current_lines = [title_part]
                elif current_id is not None:
                    if line.startswith(" ") and line.strip():
                        current_lines.append(line.strip())
                    else:
                        if current_id not in entries_by_id:
                            entries_by_id[current_id] = " ".join(current_lines)
                        current_id = None
                        current_lines = []
            if current_id is not None and current_id not in entries_by_id:
                entries_by_id[current_id] = " ".join(current_lines)

        return list(entries_by_id.items())

    def _ensure_gutenberg_catalog(self, library: str, base_dir: str) -> bool:
        gutindex_path = os.path.join(base_dir, "GUTINDEX.ALL")
        if not os.path.isfile(gutindex_path):
            return False

        try:
            mtime = os.path.getmtime(gutindex_path)
        except OSError:
            return False

        conn = self._connect_db()
        try:
            with conn:
                row = conn.execute(
                    "SELECT gutindex_mtime, entry_count FROM gutenberg_meta WHERE library = ?",
                    (library,),
                ).fetchone()
                if row and row[0] == mtime and (row[1] or 0) > 0:
                    return True

            print(f"[FileReader] Indexing Project Gutenberg catalog from '{gutindex_path}'...")
            parsed = self._parse_gutindex_file(gutindex_path)
            with conn:
                conn.execute("DELETE FROM gutenberg_catalog WHERE library = ?", (library,))
                conn.executemany(
                    "INSERT OR REPLACE INTO gutenberg_catalog (library, ebook_id, description) VALUES (?, ?, ?)",
                    [(library, eid, desc) for eid, desc in parsed],
                )
                conn.execute(
                    "INSERT OR REPLACE INTO gutenberg_meta (library, gutindex_mtime, entry_count) VALUES (?, ?, ?)",
                    (library, mtime, len(parsed)),
                )
            print(f"[FileReader] Indexed {len(parsed)} Project Gutenberg entries for library '{library}'.")
            return True
        finally:
            conn.close()

    def _search_gutenberg_catalog(self, library: str, base_dir: str, terms: List[str], limit: int = 50) -> List[str]:
        if not self._ensure_gutenberg_catalog(library, base_dir):
            return []

        where_clauses = []
        params: List[Any] = [library]
        for term in terms:
            if term.lstrip("#").isdigit():
                where_clauses.append("(CAST(ebook_id AS TEXT) = ? OR lower(description) LIKE ?)")
                params.extend([term.lstrip("#"), f"%{term}%"])
            else:
                where_clauses.append("lower(description) LIKE ?")
                params.append(f"%{term}%")

        sql = (
            "SELECT ebook_id, description FROM gutenberg_catalog "
            f"WHERE library = ? AND {' AND '.join(where_clauses)} "
            "ORDER BY ebook_id ASC LIMIT ?"
        )
        params.append(limit + 1)

        conn = self._connect_db()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        results = []
        for eid, desc in rows[:limit]:
            rel_path = self._resolve_gutenberg_book_rel_path(base_dir, int(eid))
            results.append(f"{desc} (#{eid}) [path: {rel_path}]")
        if len(rows) > limit:
            results.append(f"... and more matches in '{library}'. Please refine your query.")
        return results

    def _resolve_and_check_path(self, library: str, rel_path: str) -> Tuple[str, str, str]:
        raw_lib_lower = (library or "").strip().lower()
        cleaned_rel = self._clean_filepath_arg(rel_path)

        if raw_lib_lower in ("shared", "shared_workspace") and "workspace" in self.libraries:
            resolved_lib = "workspace"
            if cleaned_rel.startswith(self.container_workdir + "/"):
                cleaned_rel = cleaned_rel[len(self.container_workdir) + 1 :]
            if cleaned_rel == "shared" or cleaned_rel.startswith("shared/"):
                pass
            else:
                cleaned_rel = f"shared/{cleaned_rel.lstrip('/')}" if cleaned_rel.lstrip("/") else "shared"
        else:
            resolved_lib = self._resolve_library_name(library)

        base_dir = self.libraries[resolved_lib]

        if resolved_lib == "workspace":
            if cleaned_rel == self.container_workdir:
                cleaned_rel = ""
            elif cleaned_rel.startswith(self.container_workdir + "/"):
                cleaned_rel = cleaned_rel[len(self.container_workdir) + 1 :]

        # If this is a Gutenberg library and the caller passed an eBook ID (e.g. "1342" or "#1342"), resolve it
        if os.path.isfile(os.path.join(base_dir, "GUTINDEX.ALL")):
            id_candidate = cleaned_rel.lstrip("#").strip()
            if id_candidate.isdigit() and int(id_candidate) >= 1:
                # Only treat single-digit 0..9 as a directory when listing, or book ID when >= 10 / prefixed with #
                if cleaned_rel.startswith("#") or int(id_candidate) >= 10:
                    cleaned_rel = self._resolve_gutenberg_book_rel_path(base_dir, int(id_candidate))

        safe_rel_path = cleaned_rel.lstrip(os.sep) if cleaned_rel else ""
        if resolved_lib == "workspace" and self.shared_workspace_dir:
            norm_parts = os.path.normpath(safe_rel_path or ".").split(os.sep)
            if ".." not in norm_parts and (safe_rel_path == "shared" or safe_rel_path.startswith("shared/")):
                sub_rel = safe_rel_path[len("shared") :].lstrip(os.sep)
                abs_path = os.path.abspath(os.path.join(self.shared_workspace_dir, sub_rel))
                if os.path.commonpath([self.shared_workspace_dir, abs_path]) != self.shared_workspace_dir:
                    raise ValueError(f"Access denied. Path '{rel_path}' is outside of library '{resolved_lib}'.")
                sub_norm = os.path.relpath(abs_path, self.shared_workspace_dir)
                display_rel = "shared" if sub_norm == "." else f"shared/{sub_norm}"
                return resolved_lib, abs_path, display_rel

        abs_path = os.path.abspath(os.path.join(base_dir, safe_rel_path))

        if os.path.commonpath([base_dir, abs_path]) != base_dir:
            raise ValueError(f"Access denied. Path '{rel_path}' is outside of library '{resolved_lib}'.")

        norm_rel = os.path.relpath(abs_path, base_dir)
        return resolved_lib, abs_path, ("" if norm_rel == "." else norm_rel)

    # --- Metadata & Parsing Helpers ---

    @staticmethod
    def _extract_epub_metadata(abs_file_path: str) -> Tuple[str, str]:
        title, author = "", ""
        try:
            with zipfile.ZipFile(abs_file_path, "r") as z:
                container = ET.fromstring(z.read("META-INF/container.xml"))
                rf = container.find(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile")
                if rf is not None and "full-path" in rf.attrib:
                    opf = ET.fromstring(z.read(rf.attrib["full-path"]))
                    t_el = opf.find(".//{http://purl.org/dc/elements/1.1/}title")
                    c_el = opf.find(".//{http://purl.org/dc/elements/1.1/}creator")
                    if t_el is not None and t_el.text:
                        title = t_el.text.strip()
                    if c_el is not None and c_el.text:
                        author = c_el.text.strip()
        except Exception:
            pass
        return title, author

    def _format_epub_label(self, abs_file_path: str, rel_path: str) -> str:
        title, author = self._extract_epub_metadata(abs_file_path)
        if title and author:
            return f"{title} by {author} [path: {rel_path}]"
        elif title:
            return f"{title} [path: {rel_path}]"
        return rel_path

    @staticmethod
    def _extract_pdf_metadata(abs_file_path: str) -> Tuple[str, str, int]:
        title, author, total_pages = "", "", 0
        try:
            with fitz.open(abs_file_path) as doc:
                total_pages = len(doc)
                meta = doc.metadata or {}
                title = (meta.get("title") or "").strip()
                author = (meta.get("author") or "").strip()
        except Exception:
            pass
        return title, author, total_pages

    @staticmethod
    def _build_pdf_label(rel_path: str, title: str, author: str, total_pages: int) -> str:
        if title and author:
            return f"{title} by {author} ({total_pages} pages) [path: {rel_path}]"
        elif title:
            return f"{title} ({total_pages} pages) [path: {rel_path}]"
        return f"{os.path.basename(rel_path)} ({total_pages} pages) [path: {rel_path}]"

    def _format_pdf_label(self, abs_file_path: str, rel_path: str) -> str:
        title, author, total_pages = self._extract_pdf_metadata(abs_file_path)
        return self._build_pdf_label(rel_path, title, author, total_pages)

    def _get_cached_file_label(self, abs_file_path: str, rel_path: str) -> str:
        ext = os.path.splitext(abs_file_path)[1].lower()
        if ext not in (".epub", ".pdf"):
            return rel_path

        try:
            mtime = os.path.getmtime(abs_file_path)
        except OSError:
            return rel_path

        conn = self._connect_db()
        try:
            with conn:
                if ext == ".epub":
                    row = conn.execute(
                        "SELECT mtime, label FROM book_metadata WHERE rel_path = ?",
                        (rel_path,),
                    ).fetchone()
                    if row and row[0] == mtime:
                        return row[1]
                    title, author = self._extract_epub_metadata(abs_file_path)
                    label = self._format_epub_label(abs_file_path, rel_path)
                    conn.execute(
                        "INSERT OR REPLACE INTO book_metadata (rel_path, mtime, title, author, label) VALUES (?, ?, ?, ?, ?)",
                        (rel_path, mtime, title, author, label),
                    )
                    return label
                else:
                    row = conn.execute(
                        "SELECT mtime, label FROM pdf_metadata WHERE rel_path = ?",
                        (rel_path,),
                    ).fetchone()
                    if row and row[0] == mtime:
                        return row[1]
                    title, author, total_pages = self._extract_pdf_metadata(abs_file_path)
                    label = self._build_pdf_label(rel_path, title, author, total_pages)
                    conn.execute(
                        "INSERT OR REPLACE INTO pdf_metadata (rel_path, mtime, title, author, total_pages, label) VALUES (?, ?, ?, ?, ?, ?)",
                        (rel_path, mtime, title, author, total_pages, label),
                    )
                    return label
        finally:
            conn.close()

    @staticmethod
    def _is_book_library(lib_name: Optional[str]) -> bool:
        low = (lib_name or "").strip().lower()
        return low in ("ebook", "ebooks", "gutenberg", "gutenburg", "project_gutenberg")

    @staticmethod
    def _iter_epub_spine_items(book: Any) -> List[Any]:
        spine_items: List[Any] = []
        seen_ids = set()
        for entry in getattr(book, "spine", None) or []:
            item_id = entry[0] if isinstance(entry, (tuple, list)) and entry else entry
            if not item_id or item_id in seen_ids:
                continue
            item = book.get_item_with_id(item_id)
            if item is not None and item.get_type() == 9:  # ITEM_DOCUMENT
                seen_ids.add(item_id)
                spine_items.append(item)
        if spine_items:
            return spine_items
        return list(book.get_items_of_type(9))

    @classmethod
    def _extract_html_sections(
        cls,
        raw_html: Any,
        fallback_title: str = "",
    ) -> List[Tuple[str, List[str]]]:
        soup = BeautifulSoup(raw_html, "html.parser")
        doc_title = ""
        if soup.title and soup.title.string:
            t_cand = " ".join(soup.title.string.split()).strip()
            if t_cand and t_cand.lower() not in ("unknown", "untitled", "index", "document"):
                doc_title = t_cand

        for noisy in soup(["script", "style", "head", "svg", "nav"]):
            noisy.decompose()

        root = soup.body if soup.body is not None else soup

        for br in root.find_all("br"):
            br.replace_with("\n")
        for hr in root.find_all("hr"):
            hr.replace_with("\n\n* * *\n\n")

        for htag in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            htext = " ".join(htag.get_text(" ", strip=True).split()).strip()
            if not htext:
                htag.decompose()
                continue
            if htag.name in ("h1", "h2"):
                htag.replace_with(f"\n\n__SECTION_H12__ {htext}\n\n")
            else:
                htag.replace_with(f"\n\n### {htext}\n\n")

        for btag in root.find_all(
            ["p", "div", "section", "article", "blockquote", "li", "tr", "pre", "table"]
        ):
            btag.insert_before("\n\n")
            btag.insert_after("\n\n")

        raw_text = root.get_text()
        norm_lines = [re.sub(r"[ \t\f\v]+", " ", ln).strip() for ln in raw_text.splitlines()]
        cleaned_blob = "\n".join(norm_lines)
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned_blob) if p.strip()]
        if not raw_paras:
            return []

        default_title = doc_title or fallback_title or ""
        sections: List[Tuple[str, List[str]]] = []
        cur_title = default_title
        cur_paras: List[str] = []

        for para in raw_paras:
            if para.startswith("__SECTION_H12__ "):
                h_title = para[len("__SECTION_H12__ ") :].strip()
                heading_line = f"# {h_title}"
                # If previous paragraphs only contain headings (e.g. "Part I" before "Chapter 1"), merge them
                has_body = any(not p.startswith("# ") for p in cur_paras)
                if cur_paras and has_body:
                    sections.append((cur_title, cur_paras))
                    cur_title = h_title
                    cur_paras = [heading_line]
                else:
                    cur_title = f"{cur_title} — {h_title}" if (cur_title and cur_paras) else h_title
                    cur_paras.append(heading_line)
            else:
                cur_paras.append(para)

        if cur_paras:
            sections.append((cur_title, cur_paras))
        return sections

    @staticmethod
    def _split_oversized_paragraph(para: str, max_chars: int) -> List[str]:
        if len(para) <= max_chars:
            return [para]

        # Split along sentence boundaries while keeping closing punctuation attached
        parts = re.split(r'([.!?]["\'”’)\]]*)\s+', para)
        sentences: List[str] = []
        idx = 0
        while idx < len(parts):
            seg = parts[idx]
            punct = parts[idx + 1] if (idx + 1) < len(parts) else ""
            combined = (seg + punct).strip()
            if combined:
                sentences.append(combined)
            idx += 2

        out: List[str] = []
        cur = ""
        for sent in sentences:
            if len(sent) > max_chars:
                if cur:
                    out.append(cur)
                    cur = ""
                rem = sent
                while len(rem) > max_chars:
                    cut = rem.rfind(" ", 0, max_chars)
                    if cut <= 0:
                        cut = max_chars
                    out.append(rem[:cut].strip())
                    rem = rem[cut:].strip()
                if rem:
                    cur = rem
            elif not cur:
                cur = sent
            elif len(cur) + 1 + len(sent) <= max_chars:
                cur = f"{cur} {sent}"
            else:
                out.append(cur)
                cur = sent
        if cur:
            out.append(cur)
        return out or [para[:max_chars]]

    @classmethod
    def _pack_sections_into_chunks(
        cls,
        sections: List[Tuple[str, List[str]]],
        max_chars: int,
    ) -> Tuple[List[str], List[str]]:
        budget = max(100, int(max_chars))
        min_standalone = min(600, max(80, budget // 8))

        # Coalesce tiny frontmatter/divider sections with the following section when combined <= budget
        merged_sections: List[Tuple[str, List[str]]] = []
        for sec_title, paras in sections:
            if not paras:
                continue
            if merged_sections:
                prev_title, prev_paras = merged_sections[-1]
                prev_text = "\n\n".join(prev_paras)
                cur_text = "\n\n".join(paras)
                if len(prev_text) < min_standalone and (len(prev_text) + 2 + len(cur_text)) <= budget:
                    combined_title = sec_title or prev_title
                    if prev_title and sec_title and prev_title != sec_title:
                        combined_title = f"{prev_title} / {sec_title}"
                    merged_sections[-1] = (combined_title, prev_paras + paras)
                    continue
            merged_sections.append((sec_title, list(paras)))

        chunks: List[str] = []
        titles: List[str] = []

        for sec_title, paras in merged_sections:
            sec_chunks: List[str] = []
            cur_paras: List[str] = []
            cur_len = 0

            for p in paras:
                sep = 2 if cur_paras else 0
                # Never emit a chunk that consists solely of headings without body prose
                only_headings_so_far = bool(cur_paras) and all(cp.startswith("#") for cp in cur_paras)
                if only_headings_so_far and (cur_len + sep + len(p)) > budget:
                    avail = max(40, budget - cur_len - sep)
                    pieces = cls._split_oversized_paragraph(p, avail)
                    cur_paras.append(pieces[0])
                    sec_chunks.append("\n\n".join(cur_paras))
                    cur_paras = []
                    cur_len = 0
                    rest_pieces: List[str] = []
                    if len(pieces) > 1:
                        rest_pieces.extend(cls._split_oversized_paragraph(" ".join(pieces[1:]), budget))
                    for rp in rest_pieces:
                        r_sep = 2 if cur_paras else 0
                        if cur_paras and (cur_len + r_sep + len(rp)) > budget:
                            sec_chunks.append("\n\n".join(cur_paras))
                            cur_paras = [rp]
                            cur_len = len(rp)
                        else:
                            cur_paras.append(rp)
                            cur_len += r_sep + len(rp)
                    continue

                sub_paras = cls._split_oversized_paragraph(p, budget) if len(p) > budget else [p]
                for sp in sub_paras:
                    s_sep = 2 if cur_paras else 0
                    if cur_paras and (cur_len + s_sep + len(sp)) > budget:
                        sec_chunks.append("\n\n".join(cur_paras))
                        cur_paras = [sp]
                        cur_len = len(sp)
                    else:
                        cur_paras.append(sp)
                        cur_len += s_sep + len(sp)

            if cur_paras:
                sec_chunks.append("\n\n".join(cur_paras))

            total_parts = len(sec_chunks)
            for idx, ch_text in enumerate(sec_chunks, start=1):
                chunks.append(ch_text)
                if sec_title:
                    if total_parts > 1:
                        titles.append(f"{sec_title} (Part {idx}/{total_parts})")
                    else:
                        titles.append(sec_title)
                else:
                    titles.append(f"Part {idx}/{total_parts}" if total_parts > 1 else "")

        return chunks, titles

    def _parse_epub_chunks(self, abs_file_path: str) -> List[str]:
        book = epub.read_epub(abs_file_path)
        all_sections: List[Tuple[str, List[str]]] = []
        for item in self._iter_epub_spine_items(book):
            raw_body = item.get_content() if hasattr(item, "get_content") else item.get_body_content()
            item_sections = self._extract_html_sections(raw_body)
            all_sections.extend(item_sections)

        chunks, titles = self._pack_sections_into_chunks(all_sections, self.chunk_size)
        self.current_chunks = chunks
        self.current_chunk_titles = titles
        return self.current_chunks

    def _parse_html_book_chunks(self, abs_file_path: str) -> List[str]:
        with open(abs_file_path, "rb") as f:
            raw_bytes = f.read()
        sections = self._extract_html_sections(raw_bytes)
        chunks, titles = self._pack_sections_into_chunks(sections, self.chunk_size)
        self.current_chunks = chunks
        self.current_chunk_titles = titles
        return self.current_chunks

    def get_system_prompt_addition(self) -> str:
        libs_str = ", ".join(f"`{k}`" for k in self.libraries.keys())
        prompt = (
            "--- UNIFIED FILE, BOOK, PDF & IMAGE READER ---\n"
            f"You have access to a unified file and media reader across libraries: {libs_str}.\n"
            "- Use `list_libraries()` to see all mounted library directories.\n"
            "- Use `list_files(library='all', path='', query='')` to list files/directories in a specific library, "
            "or pass `query` (with `library='all'` or a specific library name like `'gutenberg'` or `'ebook'`) to search by title, author, or filename across libraries.\n"
            "- Use `read_file(library, filepath, chunk=1)` to read any file (EPUB books, PDF documents, Project Gutenberg `.txt` books, or code files). "
            "It returns the requested chunk/page along with `total_chunks` (or `total_pages`) and a `next_hint`.\n"
            "- Use `read_next_chunk()` to advance and read the next chunk or page of the most recently read book or document.\n"
            "- Use `view_image(library, filepath, chunk=1)` to visually inspect an image file (e.g., in `discord_file` or `workspace`) "
            "or render page `chunk` of a PDF into your visual context."
        )
        if "workspace" in self.libraries:
            prompt += (
                "\n- Use `copy_to_workspace(library, filepath, dest_path='')` to atomically copy any file or image from a library "
                "(e.g., `'discord_file'`, `'ebook'`, `'pdf'`, or `'workspace'`) into your workspace or `'shared/...'` directory."
            )
        if self.current_book and os.path.exists(self.current_book):
            label = self.current_book_label or os.path.basename(self.current_book)
            ext = os.path.splitext(self.current_book)[1].lower()
            if ext == ".epub":
                total = max(1, len(self.current_chunks))
                prompt += f"\n[Reader Status: Active EPUB is '{label}' (library '{self.current_library}') at chunk {self.current_index}/{total}.]"
            elif ext in (".htm", ".html", ".xhtml") and self._is_book_library(self.current_library):
                total = max(1, len(self.current_chunks))
                prompt += f"\n[Reader Status: Active HTML book is '{label}' (library '{self.current_library}') at chunk {self.current_index}/{total}.]"
            elif ext == ".pdf":
                _, _, total_pages = self._extract_pdf_metadata(self.current_book)
                prompt += f"\n[Reader Status: Active PDF is '{label}' (library '{self.current_library}') at page {self.current_index}/{max(1, total_pages)}.]"
            else:
                prompt += f"\n[Reader Status: Active document is '{label}' (library '{self.current_library}') at chunk {self.current_index}.]"
        else:
            prompt += "\n[Reader Status: No active book or document bookmark.]"
        return prompt

    def get_tools(self) -> List[Callable]:
        tools: List[Callable] = [
            self.list_libraries,
            self.list_files,
            self.read_file,
            self.read_next_chunk,
            self.view_image,
        ]
        if "workspace" in self.libraries:
            tools.append(self.copy_to_workspace)
        return tools

    # --- Tool Implementations ---

    def list_libraries(self) -> str:
        """Lists all configured library names available for reading."""
        if not self.libraries:
            return "No libraries are currently configured."

        result = "Available libraries:\n"
        for name, path in self.libraries.items():
            result += f"- {name}: mounted at {path}\n"
        return result

    def _rel_path_for_library_file(self, lib_name: str, abs_f: str) -> str:
        base_dir = self.libraries[lib_name]
        if lib_name == "workspace" and self.shared_workspace_dir:
            try:
                if os.path.commonpath([self.shared_workspace_dir, abs_f]) == self.shared_workspace_dir:
                    sub = os.path.relpath(abs_f, self.shared_workspace_dir)
                    return "shared" if sub == "." else f"shared/{sub}"
            except ValueError:
                pass
        return os.path.relpath(abs_f, base_dir)

    def _search_single_library(self, lib_name: str, target_path: str, terms: List[str], limit: int = 50) -> List[str]:
        base_dir = self.libraries[lib_name]
        if target_path == base_dir and os.path.isfile(os.path.join(base_dir, "GUTINDEX.ALL")):
            return self._search_gutenberg_catalog(lib_name, base_dir, terms, limit=limit)

        walk_roots = [target_path]
        if (
            lib_name == "workspace"
            and target_path == base_dir
            and self.shared_workspace_dir
            and os.path.isdir(self.shared_workspace_dir)
        ):
            walk_roots.append(self.shared_workspace_dir)

        matches: List[str] = []
        walked_files = 0
        for walk_root in walk_roots:
            for root, dirs, files in os.walk(walk_root):
                if walk_root == base_dir and lib_name == "workspace" and self.shared_workspace_dir and root == base_dir:
                    if "shared" in dirs:
                        dirs.remove("shared")
                for fname in sorted(files):
                    walked_files += 1
                    if walked_files > MAX_WALK_FILES_PER_LIBRARY:
                        matches.append(
                            f"... search stopped after scanning {MAX_WALK_FILES_PER_LIBRARY} files in '{lib_name}'."
                        )
                        return matches
                    abs_f = os.path.join(root, fname)
                    rel_f = self._rel_path_for_library_file(lib_name, abs_f)
                    label = self._get_cached_file_label(abs_f, rel_f)
                    if all(term in label.lower() or term in rel_f.lower() for term in terms):
                        matches.append(label)
                        if len(matches) >= limit:
                            matches.append(f"... and more matches in '{lib_name}'. Please refine your query.")
                            return matches
        return matches

    def list_files(self, library: str = "all", path: str = "", query: str = "") -> str:
        """
        Lists files and directories inside a specified library, or searches by filename,
        book title, or author when 'query' is provided. Pass library='all' with a 'query'
        to search across all configured libraries simultaneously.
        """
        try:
            clean_lib = (library or "all").strip()
            clean_query = (query or "").strip()

            # Cross-library search when library is 'all' / '*' / empty
            if clean_lib.lower() in ("all", "*", ""):
                if not clean_query:
                    return self.list_libraries()
                terms = [t for t in re.split(r"[\s,]+", clean_query.lower()) if t]
                if not terms:
                    return self.list_libraries()

                combined_matches = []
                for lib_name, base_dir in self.libraries.items():
                    if not os.path.exists(base_dir):
                        continue
                    sub_matches = self._search_single_library(lib_name, base_dir, terms, limit=25)
                    for m in sub_matches:
                        combined_matches.append(f"- [{lib_name}] {m}")
                if not combined_matches:
                    return f"No files matching '{clean_query}' found across any library."
                return f"Cross-library search results for '{clean_query}':\n" + "\n".join(combined_matches[:75])

            resolved_lib, target_path, rel_base = self._resolve_and_check_path(clean_lib, path)
            base_dir = self.libraries[resolved_lib]

            if not os.path.exists(target_path):
                return f"Error: Path '{path}' does not exist in library '{resolved_lib}'."

            if clean_query:
                terms = [t for t in re.split(r"[\s,]+", clean_query.lower()) if t]
                matches = self._search_single_library(resolved_lib, target_path, terms, limit=50)
                if not matches:
                    return f"No files matching '{clean_query}' found in library '{resolved_lib}'."
                return f"Search results in '{resolved_lib}' for '{clean_query}':\n" + "\n".join(
                    f"- {m}" for m in matches
                )

            if not os.path.isdir(target_path):
                return f"Error: '{path}' is not a directory."

            entries = os.listdir(target_path)
            dirs_set = {e for e in entries if os.path.isdir(os.path.join(target_path, e))}
            if (
                resolved_lib == "workspace"
                and target_path == base_dir
                and self.shared_workspace_dir
                and os.path.isdir(self.shared_workspace_dir)
            ):
                dirs_set.add("shared")
            dirs = sorted(dirs_set)
            files = sorted([e for e in entries if os.path.isfile(os.path.join(target_path, e))])

            if not dirs and not files:
                return f"Directory '{rel_base or '/'}' in library '{resolved_lib}' is empty."

            result = [f"Contents of {resolved_lib} : {rel_base or '/'}\n"]
            shown = 0
            total_entries = len(dirs) + len(files)
            for d in dirs:
                if shown >= MAX_DIR_LIST_ENTRIES:
                    break
                result.append(f"[DIR]  {d}")
                shown += 1
            for f in files:
                if shown >= MAX_DIR_LIST_ENTRIES:
                    break
                abs_f = os.path.join(target_path, f)
                rel_f = self._rel_path_for_library_file(resolved_lib, abs_f)
                ext = os.path.splitext(f)[1].lower()
                if ext in (".epub", ".pdf") and len(files) <= 200:
                    label = self._get_cached_file_label(abs_f, rel_f)
                    result.append(f"[FILE] {label}")
                else:
                    result.append(f"[FILE] {f}")
                shown += 1

            if total_entries > shown:
                result.append(f"... and {total_entries - shown} more entries (use 'query' to filter).")

            return "\n".join(result)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def read_file(self, library: str, filepath: str, chunk: int = 1) -> str:
        """
        Reads a chunk or page (1-indexed) of any file (EPUB, PDF, or text/code) from a library.
        Returns JSON containing the content, chunk/page index, total_chunks/total_pages, and a next_hint.
        Updates your active bookmark so you can call read_next_chunk() to continue reading sequentially.
        """
        try:
            resolved_lib, abs_path, rel_path = self._resolve_and_check_path(library, filepath)
            base_dir = self.libraries[resolved_lib]

            # If a Gutenberg book directory was passed (e.g. '1/3/4/1342'), auto-resolve the book text file inside it
            if os.path.isdir(abs_path):
                tail = os.path.basename(abs_path)
                if tail.isdigit():
                    resolved_rel = self._resolve_gutenberg_book_rel_path(base_dir, int(tail))
                    abs_path = os.path.abspath(os.path.join(base_dir, resolved_rel))
                    rel_path = resolved_rel

            if not os.path.exists(abs_path):
                return json.dumps({"error": f"File '{filepath}' does not exist in library '{resolved_lib}'."})
            if not os.path.isfile(abs_path):
                return json.dumps({"error": f"'{filepath}' is not a file."})

            chunk_idx = max(1, int(chunk if chunk is not None else 1))
            ext = os.path.splitext(abs_path)[1].lower()

            if ext == ".epub":
                return self._read_epub_chunk(resolved_lib, abs_path, rel_path, chunk_idx)
            elif ext in (".htm", ".html", ".xhtml") and self._is_book_library(resolved_lib):
                return self._read_html_book_chunk(resolved_lib, abs_path, rel_path, chunk_idx)
            elif ext == ".pdf":
                return self._read_pdf_chunk(resolved_lib, abs_path, rel_path, chunk_idx)
            elif ext in IMAGE_EXTENSIONS:
                return self._read_image_metadata(resolved_lib, abs_path, rel_path)
            else:
                return self._read_text_chunk(resolved_lib, abs_path, rel_path, chunk_idx)
        except Exception as e:
            return json.dumps({"error": f"Error reading file '{filepath}': {str(e)}"})

    def _read_epub_chunk(self, library: str, abs_path: str, rel_path: str, chunk: int) -> str:
        if self.current_book != abs_path or not self.current_chunks:
            self._parse_epub_chunks(abs_path)
        label = self._get_cached_file_label(abs_path, rel_path)
        return self._format_book_chunk_response(
            fmt="epub",
            library=library,
            abs_path=abs_path,
            rel_path=rel_path,
            label=label,
            chunk=chunk,
        )

    def _read_html_book_chunk(self, library: str, abs_path: str, rel_path: str, chunk: int) -> str:
        if self.current_book != abs_path or not self.current_chunks:
            self._parse_html_book_chunks(abs_path)
        return self._format_book_chunk_response(
            fmt="html_book",
            library=library,
            abs_path=abs_path,
            rel_path=rel_path,
            label=rel_path,
            chunk=chunk,
        )

    def _format_book_chunk_response(
        self,
        fmt: str,
        library: str,
        abs_path: str,
        rel_path: str,
        label: str,
        chunk: int,
    ) -> str:
        total_chunks = max(1, len(self.current_chunks))

        if chunk > total_chunks:
            return json.dumps(
                {
                    "format": fmt,
                    "library": library,
                    "filepath": rel_path,
                    "document": label,
                    "chunk": chunk,
                    "total_chunks": total_chunks,
                    "error": f"You have reached the end of '{label}' (requested chunk {chunk}, total_chunks={total_chunks}).",
                },
                indent=2,
            )

        text = self.current_chunks[chunk - 1] if self.current_chunks else ""
        sec_title = (
            self.current_chunk_titles[chunk - 1]
            if (chunk - 1) < len(self.current_chunk_titles)
            else ""
        )
        self.current_book = abs_path
        self.current_library = library
        self.current_rel_path = rel_path
        self.current_book_label = label
        self.current_index = chunk
        self._save_state()

        if chunk < total_chunks:
            next_hint = (
                f"Call read_next_chunk() to read chunk {chunk + 1}/{total_chunks}, "
                f"or call read_file(library='{library}', filepath='{rel_path}', chunk={chunk + 1})."
            )
        else:
            next_hint = f"End of '{label}' reached ({total_chunks}/{total_chunks} chunks)."

        payload: Dict[str, Any] = {
            "format": fmt,
            "library": library,
            "filepath": rel_path,
            "document": label,
            "chunk": chunk,
            "total_chunks": total_chunks,
        }
        if sec_title:
            payload["section_title"] = sec_title
        payload["content"] = text
        payload["next_hint"] = next_hint
        return json.dumps(payload, indent=2)

    def _read_pdf_chunk(self, library: str, abs_path: str, rel_path: str, page_number: int) -> str:
        label = self._get_cached_file_label(abs_path, rel_path)
        with fitz.open(abs_path) as doc:
            total_pages = len(doc)
            if page_number < 1 or page_number > total_pages:
                return json.dumps(
                    {
                        "format": "pdf",
                        "library": library,
                        "filepath": rel_path,
                        "document": label,
                        "error": f"Invalid chunk/page {page_number}. Document has {total_pages} pages (1..{total_pages}).",
                        "total_pages": total_pages,
                        "total_chunks": total_pages,
                    },
                    indent=2,
                )

            page = doc[page_number - 1]
            text = page.get_text("text").strip()

            raw_images = page.get_images(full=True)
            images = [
                {"width": int(img[2]), "height": int(img[3])}
                for img in raw_images
                if int(img[2]) > 8 and int(img[3]) > 8
            ]
            drawings = page.get_drawings()
            vector_drawing_count = len(drawings)
            has_unrenderable_visuals = bool(images) or vector_drawing_count > 5

        self.current_book = abs_path
        self.current_library = library
        self.current_rel_path = rel_path
        self.current_book_label = label
        self.current_index = page_number
        self._save_state()

        if page_number < total_pages:
            next_hint = (
                f"Call read_next_chunk() to read page {page_number + 1}/{total_pages}, "
                f"or call read_file(library='{library}', filepath='{rel_path}', chunk={page_number + 1})."
            )
        else:
            next_hint = f"End of PDF '{label}' reached (page {total_pages}/{total_pages})."

        result: Dict[str, Any] = {
            "format": "pdf",
            "library": library,
            "filepath": rel_path,
            "document": label,
            "chunk": page_number,
            "page_number": page_number,
            "total_chunks": total_pages,
            "total_pages": total_pages,
            "content": text,
            "text": text,
            "has_unrenderable_visuals": has_unrenderable_visuals,
            "image_count": len(images),
            "images": images,
            "vector_drawing_count": vector_drawing_count,
            "next_hint": next_hint,
        }
        if has_unrenderable_visuals:
            result["notice"] = (
                f"This page contains {len(images)} embedded image(s) and {vector_drawing_count} vector drawing elements "
                f"that cannot be represented as text. Call view_image(library='{library}', filepath='{rel_path}', chunk={page_number}) "
                "to visually inspect the page."
            )
        elif not text:
            result["notice"] = (
                f"No extractable text found on page {page_number}. "
                f"Call view_image(library='{library}', filepath='{rel_path}', chunk={page_number}) to visually inspect the page."
            )

        return json.dumps(result, indent=2)

    def _read_text_chunk(self, library: str, abs_path: str, rel_path: str, chunk: int) -> str:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = [line.rstrip("\r\n") for line in f]

        total_lines = len(all_lines)
        total_chunks = max(1, (total_lines + self.lines_per_chunk - 1) // self.lines_per_chunk)
        if chunk > total_chunks:
            return json.dumps(
                {
                    "format": "text",
                    "library": library,
                    "filepath": rel_path,
                    "chunk": chunk,
                    "total_chunks": total_chunks,
                    "total_lines": total_lines,
                    "error": f"Requested chunk {chunk} exceeds total_chunks ({total_chunks}).",
                },
                indent=2,
            )

        start_idx = (chunk - 1) * self.lines_per_chunk
        end_idx = min(total_lines, start_idx + self.lines_per_chunk)
        numbered_lines = [f"{i + 1}: {all_lines[i]}" for i in range(start_idx, end_idx)]
        content = "\n".join(numbered_lines)

        self.current_book = abs_path
        self.current_library = library
        self.current_rel_path = rel_path
        self.current_book_label = rel_path
        self.current_index = chunk
        self._save_state()

        if chunk < total_chunks:
            next_hint = (
                f"Call read_next_chunk() to read chunk {chunk + 1}/{total_chunks}, "
                f"or call read_file(library='{library}', filepath='{rel_path}', chunk={chunk + 1})."
            )
        else:
            next_hint = f"End of file '{rel_path}' reached ({total_chunks}/{total_chunks} chunks)."

        return json.dumps(
            {
                "format": "text",
                "library": library,
                "filepath": rel_path,
                "chunk": chunk,
                "total_chunks": total_chunks,
                "start_line": start_idx + 1 if total_lines else 0,
                "end_line": end_idx,
                "total_lines": total_lines,
                "content": content,
                "next_hint": next_hint,
            },
            indent=2,
        )

    @staticmethod
    def _read_image_metadata(library: str, abs_path: str, rel_path: str) -> str:
        with open(abs_path, "rb") as f:
            raw_bytes = f.read()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            width, height = img.size
            fmt = img.format or "IMAGE"
        return json.dumps(
            {
                "format": "image",
                "library": library,
                "filepath": rel_path,
                "image_format": fmt,
                "width": width,
                "height": height,
                "chunk": 1,
                "total_chunks": 1,
                "next_hint": f"Call view_image(library='{library}', filepath='{rel_path}') to attach this image to your visual context.",
            },
            indent=2,
        )

    def read_next_chunk(self) -> str:
        """
        Advances and returns the next chunk or page of the most recently read book or document (EPUB, PDF, or text).
        """
        if not self.current_book or not os.path.exists(self.current_book):
            return json.dumps(
                {
                    "error": "No active book or document is currently open. Call read_file(library, filepath, chunk=1) first."
                }
            )

        if not self.current_library or not self.current_rel_path:
            lib_name, rel_p = self._infer_library_and_rel_path(self.current_book)
            self.current_library = lib_name
            self.current_rel_path = rel_p

        if not self.current_library or not self.current_rel_path:
            return json.dumps(
                {
                    "error": f"Active document '{self.current_book}' does not belong to any configured library."
                }
            )

        next_chunk = self.current_index + 1
        return self.read_file(self.current_library, self.current_rel_path, chunk=next_chunk)

    def view_image(self, library: str, filepath: str, chunk: int = 1) -> str:
        """
        Loads an image file (or renders page 'chunk' of a PDF) from a configured library,
        downsizes it to at most 1024px on its longest edge, and attaches it to your
        visual context for the current turn.
        """
        try:
            resolved_lib, abs_path, rel_path = self._resolve_and_check_path(library, filepath)

            if not os.path.exists(abs_path):
                return f"Error: File '{filepath}' does not exist in library '{resolved_lib}'."

            if not os.path.isfile(abs_path):
                return f"Error: '{filepath}' is not a file."

            ext = os.path.splitext(abs_path)[1].lower()
            if ext == ".pdf":
                page_number = max(1, int(chunk if chunk is not None else 1))
                label = self._get_cached_file_label(abs_path, rel_path)
                with fitz.open(abs_path) as doc:
                    total_pages = len(doc)
                    if page_number < 1 or page_number > total_pages:
                        return f"Error: Invalid page/chunk {page_number}. PDF has {total_pages} pages (1..{total_pages})."
                    page = doc[page_number - 1]
                    pix = page.get_pixmap(dpi=150, alpha=False)
                    raw_png = pix.tobytes("png")

                jpeg_bytes, width, height = downsize_image_bytes(raw_png, max_edge=1024)
                data_url = encode_jpeg_data_url(jpeg_bytes)

                self.current_book = abs_path
                self.current_library = resolved_lib
                self.current_rel_path = rel_path
                self.current_book_label = label
                self.current_index = page_number
                self._save_state()

                if hasattr(self, "agent") and self.agent is not None:
                    self.agent.pending_tool_images.append(data_url)

                return (
                    f"Rendered page {page_number}/{total_pages} of '{label}' from library '{resolved_lib}' "
                    f"({width}x{height}px) into your visual context for this turn."
                )

            with open(abs_path, "rb") as f:
                raw_bytes = f.read()

            jpeg_bytes, width, height = downsize_image_bytes(raw_bytes, max_edge=1024)
            data_url = encode_jpeg_data_url(jpeg_bytes)

            if hasattr(self, "agent") and self.agent is not None:
                self.agent.pending_tool_images.append(data_url)

            return (
                f"Loaded image '{rel_path}' from library '{resolved_lib}' "
                f"({width}x{height}px) into your visual context for this turn."
            )
        except Exception as e:
            return f"Error viewing image '{filepath}': {str(e)}"

    def _resolve_workspace_copy_dest(self, dest_path: str, src_basename: str) -> Tuple[str, str]:
        if "workspace" not in self.libraries:
            raise ValueError("Library 'workspace' is not configured.")

        ws_base = self.libraries["workspace"]
        raw_dest = (dest_path or "").strip()

        if raw_dest == self.container_workdir:
            raw_dest = ""
        elif raw_dest.startswith(self.container_workdir + "/"):
            raw_dest = raw_dest[len(self.container_workdir) + 1 :]
        elif raw_dest.startswith("/"):
            raise ValueError(
                f"Access denied. Destination '{dest_path}' must be relative to workspace ('{self.container_workdir}')."
            )

        treat_as_dir = raw_dest.endswith("/")
        clean_dest = raw_dest.strip("/")

        if clean_dest:
            norm_parts = os.path.normpath(clean_dest).split(os.sep)
            if ".." in norm_parts:
                raise ValueError(
                    f"Access denied. Path traversal ('..') is not allowed in destination '{dest_path}'."
                )

        if not clean_dest or clean_dest == ".":
            effective_rel = src_basename
        elif clean_dest == "shared":
            effective_rel = f"shared/{src_basename}"
        elif treat_as_dir:
            effective_rel = f"{clean_dest}/{src_basename}"
        else:
            effective_rel = clean_dest

        if self.shared_workspace_dir and (
            effective_rel == "shared" or effective_rel.startswith("shared/")
        ):
            sub_rel = effective_rel[len("shared") :].lstrip(os.sep)
            if not sub_rel:
                sub_rel = src_basename
            dst_abs = os.path.abspath(os.path.join(self.shared_workspace_dir, sub_rel))
            if os.path.commonpath([self.shared_workspace_dir, dst_abs]) != self.shared_workspace_dir:
                raise ValueError(f"Access denied. Destination '{dest_path}' is outside shared workspace.")
            if os.path.isdir(dst_abs):
                dst_abs = os.path.abspath(os.path.join(dst_abs, src_basename))
                if os.path.commonpath([self.shared_workspace_dir, dst_abs]) != self.shared_workspace_dir:
                    raise ValueError(f"Access denied. Destination '{dest_path}' is outside shared workspace.")
            dst_rel = f"shared/{os.path.relpath(dst_abs, self.shared_workspace_dir)}"
            return dst_abs, dst_rel

        dst_abs = os.path.abspath(os.path.join(ws_base, effective_rel))
        if os.path.commonpath([ws_base, dst_abs]) != ws_base:
            raise ValueError(f"Access denied. Destination '{dest_path}' is outside workspace.")
        if os.path.isdir(dst_abs):
            dst_abs = os.path.abspath(os.path.join(dst_abs, src_basename))
            if os.path.commonpath([ws_base, dst_abs]) != ws_base:
                raise ValueError(f"Access denied. Destination '{dest_path}' is outside workspace.")
        dst_rel = os.path.relpath(dst_abs, ws_base)
        return dst_abs, dst_rel

    def copy_to_workspace(
        self,
        library: str,
        filepath: str,
        dest_path: str = "",
        overwrite: bool = True,
    ) -> str:
        """
        Atomically copies a file or image from any configured library (e.g. 'discord_file',
        'ebook', 'pdf', 'gutenberg', or 'workspace') into your workspace ('/home/agent/workspace')
        or shared workspace ('shared/...').

        Args:
            library: Source library name (e.g. 'discord_file', 'workspace', 'pdf', 'ebook').
            filepath: Relative path of the source file inside the library.
            dest_path: Optional destination file or directory path inside workspace (e.g. 'book/page_01.jpg' or 'shared/incoming/'). Defaults to the source filename in workspace root.
            overwrite: Whether to overwrite an existing destination file (default True).
        """
        try:
            src_lib, src_abs, src_rel = self._resolve_and_check_path(library, filepath)
            if not os.path.exists(src_abs):
                return f"Error: File '{filepath}' does not exist in library '{src_lib}'."
            if not os.path.isfile(src_abs):
                return f"Error: '{filepath}' in library '{src_lib}' is not a regular file."

            src_basename = os.path.basename(src_abs)
            dst_abs, dst_rel = self._resolve_workspace_copy_dest(dest_path, src_basename)

            if os.path.abspath(src_abs) == os.path.abspath(dst_abs):
                size_bytes = os.path.getsize(src_abs)
                return (
                    f"File '{dst_rel}' is already at '{self.container_workdir}/{dst_rel}' "
                    f"({size_bytes} bytes; no copy needed)."
                )

            if not overwrite and os.path.exists(dst_abs):
                return (
                    f"Error: Destination '{dst_rel}' already exists in workspace "
                    "(pass overwrite=True to replace)."
                )

            os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
            tmp_path = f"{dst_abs}.{uuid.uuid4().hex[:8]}.tmp"
            try:
                shutil.copyfile(src_abs, tmp_path)
                os.replace(tmp_path, dst_abs)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

            size_bytes = os.path.getsize(dst_abs)
            return (
                f"Copied '{src_rel}' from library '{src_lib}' to workspace '{dst_rel}' "
                f"({self.container_workdir}/{dst_rel}, {size_bytes} bytes)."
            )
        except Exception as e:
            return f"Error copying '{filepath}' to workspace: {str(e)}"
