import glob
import json
import os
import sqlite3
from typing import Any, Callable, Dict, List, Tuple
import fitz
from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.image import downsize_image_bytes, encode_jpeg_data_url


class PdfViewerComponent(BaseComponent):
    """
    Page-oriented PDF reader and visual page inspector.
    Allows the agent to list/search PDFs, extract page text as structured JSON
    (flagging any embedded images or vector drawings), and render pages as
    1024px images into the LLM's visual context.
    """

    def __init__(self, library_dir: str = None, db_path: str = None):
        self.config = load_config()
        pdf_cfg = self.config.get("pdf_viewer", {})

        if library_dir is None:
            if "library_dir" not in pdf_cfg or not pdf_cfg["library_dir"]:
                raise ValueError(
                    "PdfViewerComponent requires 'library_dir' to be explicitly defined in config under [pdf_viewer].library_dir"
                )
            self.library_dir = pdf_cfg["library_dir"]
        else:
            self.library_dir = library_dir

        if db_path is None:
            if "db_path" not in pdf_cfg or not pdf_cfg["db_path"]:
                raise ValueError(
                    "PdfViewerComponent requires 'db_path' to be explicitly defined in config under [pdf_viewer].db_path"
                )
            self.db_path = pdf_cfg["db_path"]
        else:
            self.db_path = db_path

        os.makedirs(self.library_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        self.current_pdf = None
        self.current_page = 1
        self.current_pdf_label = None

        self._init_db()
        self._load_state()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pdf_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        current_pdf TEXT,
                        current_page INTEGER
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pdf_metadata (
                        rel_path TEXT PRIMARY KEY,
                        mtime REAL,
                        title TEXT,
                        author TEXT,
                        total_pages INTEGER,
                        label TEXT
                    )
                """)
                conn.execute(
                    "INSERT OR IGNORE INTO pdf_state (id, current_pdf, current_page) VALUES (1, NULL, 1)"
                )
        finally:
            conn.close()

    def _save_state(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE pdf_state SET current_pdf = ?, current_page = ? WHERE id = 1",
                    (self.current_pdf, self.current_page),
                )
        finally:
            conn.close()

    def _load_state(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                row = conn.execute(
                    "SELECT current_pdf, current_page FROM pdf_state WHERE id = 1"
                ).fetchone()
        finally:
            conn.close()

        if row:
            self.current_pdf = row[0]
            self.current_page = row[1] or 1
            if self.current_pdf and os.path.exists(self.current_pdf):
                self.current_pdf_label = self._format_pdf_label(self.current_pdf)

    def _extract_pdf_metadata(self, abs_file_path: str) -> Tuple[str, str, int]:
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

    def _build_label(self, rel_path: str, title: str, author: str, total_pages: int) -> str:
        if title and author:
            return f"{title} by {author} ({total_pages} pages) [path: {rel_path}]"
        elif title:
            return f"{title} ({total_pages} pages) [path: {rel_path}]"
        return f"{os.path.basename(rel_path)} ({total_pages} pages) [path: {rel_path}]"

    def _format_pdf_label(self, abs_file_path: str) -> str:
        rel_path = os.path.relpath(abs_file_path, self.library_dir)
        title, author, total_pages = self._extract_pdf_metadata(abs_file_path)
        return self._build_label(rel_path, title, author, total_pages)

    def _get_pdf_catalog(self) -> List[str]:
        search_pattern = os.path.join(self.library_dir, "**", "*.pdf")
        files = glob.glob(search_pattern, recursive=True)

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                rows = conn.execute("SELECT rel_path, mtime, label FROM pdf_metadata").fetchall()
                cache = {r[0]: (r[1], r[2]) for r in rows}

                updates = []
                catalog = []
                for f in files:
                    rel_path = os.path.relpath(f, self.library_dir)
                    try:
                        mtime = os.path.getmtime(f)
                    except OSError:
                        continue

                    cached = cache.get(rel_path)
                    if cached and cached[0] == mtime:
                        catalog.append(cached[1])
                    else:
                        title, author, total_pages = self._extract_pdf_metadata(f)
                        label = self._build_label(rel_path, title, author, total_pages)
                        updates.append((rel_path, mtime, title, author, total_pages, label))
                        catalog.append(label)

                if updates:
                    conn.executemany(
                        "INSERT OR REPLACE INTO pdf_metadata (rel_path, mtime, title, author, total_pages, label) VALUES (?, ?, ?, ?, ?, ?)",
                        updates,
                    )
        finally:
            conn.close()

        return sorted(catalog)

    def _resolve_pdf_path(self, filename: str = "") -> str:
        clean_name = (filename or "").strip()
        if clean_name:
            if "[path: " in clean_name and clean_name.endswith("]"):
                clean_name = clean_name.split("[path: ", 1)[1][:-1].strip()

            abs_lib = os.path.abspath(self.library_dir)
            abs_file = os.path.abspath(os.path.join(self.library_dir, clean_name))
            if os.path.commonpath([abs_lib, abs_file]) != abs_lib:
                raise ValueError(f"Invalid PDF filename '{filename}'. Path traversal is not allowed.")
            if not os.path.exists(abs_file):
                raise FileNotFoundError(f"PDF '{clean_name}' not found in {abs_lib}.")
            return abs_file

        if self.current_pdf and os.path.exists(self.current_pdf):
            return self.current_pdf

        raise ValueError(
            "No PDF is currently active. Please provide the 'filename' argument (from list_pdfs or search_pdfs)."
        )

    def get_system_prompt_addition(self) -> str:
        prompt = (
            "You have access to a page-oriented PDF Viewer. "
            "Use `list_pdfs` or `search_pdfs` to discover PDF documents. "
            "Use `read_pdf_page(page_number, filename)` to extract text and check for images/diagrams on a specific page (1-indexed). "
            "Once a PDF has been accessed, `filename` is optional on subsequent calls to `read_pdf_page` and `view_pdf_page`. "
            "If `read_pdf_page` reports `has_unrenderable_visuals: true` (or if a page is a scan, diagram, map, or table), "
            "use `view_pdf_page(page_number, filename)` to render the page as an image into your visual context."
        )
        if self.current_pdf and os.path.exists(self.current_pdf):
            label = self.current_pdf_label or os.path.basename(self.current_pdf)
            prompt += f"\n[PDF Status: Active PDF is '{label}', last accessed at page {self.current_page}]."
        else:
            prompt += "\n[PDF Status: No PDF currently active.]"
        return prompt

    def get_tools(self) -> List[Callable]:
        return [
            self.list_pdfs,
            self.search_pdfs,
            self.read_pdf_page,
            self.view_pdf_page,
        ]

    # --- Tool Implementations ---

    def list_pdfs(self) -> List[str]:
        """Returns a list of all PDF documents available in the PDF library (with title, author, page count, and relative path)."""
        return self._get_pdf_catalog()

    def search_pdfs(self, query: str) -> List[str]:
        """
        Searches for PDF documents whose title, author, or filename match the query.
        Returns matching entries showing 'Title by Author (N pages) [path: relative/path.pdf]'.
        """
        catalog = self._get_pdf_catalog()
        query_lower = query.lower()
        matches = [entry for entry in catalog if query_lower in entry.lower()]
        if len(matches) > 50:
            return matches[:50] + [f"... and {len(matches) - 50} more. Please refine your search."]
        return matches

    def read_pdf_page(self, page_number: int, filename: str = "") -> str:
        """
        Extracts text from a 1-indexed PDF page and returns a JSON document detailing
        the text, total pages, and whether the page contains un-text-renderable visuals
        (embedded images or vector drawings). 'filename' is optional if a PDF is already active.
        """
        try:
            abs_file_path = self._resolve_pdf_path(filename)
            with fitz.open(abs_file_path) as doc:
                total_pages = len(doc)
                if page_number < 1 or page_number > total_pages:
                    return json.dumps(
                        {
                            "error": f"Invalid page_number {page_number}. Document has {total_pages} pages (1..{total_pages})."
                        }
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

            self.current_pdf = abs_file_path
            self.current_page = page_number
            self.current_pdf_label = self._format_pdf_label(abs_file_path)
            self._save_state()

            if hasattr(self, "agent") and self.agent:
                self.agent.rebuild_system_prompt()

            result: Dict[str, Any] = {
                "document": self.current_pdf_label,
                "page_number": page_number,
                "total_pages": total_pages,
                "text": text,
                "has_unrenderable_visuals": has_unrenderable_visuals,
                "image_count": len(images),
                "images": images,
                "vector_drawing_count": vector_drawing_count,
            }
            if has_unrenderable_visuals:
                result["notice"] = (
                    f"This page contains {len(images)} embedded image(s) and {vector_drawing_count} vector drawing elements "
                    f"that cannot be represented as text. Call view_pdf_page(page_number={page_number}) to visually inspect the page."
                )
            elif not text:
                result["notice"] = (
                    f"No extractable text found on page {page_number}. Call view_pdf_page(page_number={page_number}) to visually inspect the page."
                )

            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def view_pdf_page(self, page_number: int, filename: str = "") -> str:
        """
        Renders a 1-indexed PDF page into an image (downsized to at most 1024px on its longest edge)
        and attaches it to your visual context for the current turn so you can see diagrams,
        illustrations, layouts, or scanned text. 'filename' is optional if a PDF is already active.
        """
        try:
            abs_file_path = self._resolve_pdf_path(filename)
            with fitz.open(abs_file_path) as doc:
                total_pages = len(doc)
                if page_number < 1 or page_number > total_pages:
                    return f"Error: Invalid page_number {page_number}. Document has {total_pages} pages (1..{total_pages})."

                page = doc[page_number - 1]
                pix = page.get_pixmap(dpi=150, alpha=False)
                raw_png = pix.tobytes("png")

            jpeg_bytes, width, height = downsize_image_bytes(raw_png, max_edge=1024)
            data_url = encode_jpeg_data_url(jpeg_bytes)

            self.current_pdf = abs_file_path
            self.current_page = page_number
            self.current_pdf_label = self._format_pdf_label(abs_file_path)
            self._save_state()

            if hasattr(self, "agent") and self.agent is not None:
                self.agent.pending_tool_images.append(data_url)
                self.agent.rebuild_system_prompt()

            return (
                f"Rendered page {page_number}/{total_pages} of '{self.current_pdf_label}' "
                f"({width}x{height}px) into your visual context for this turn."
            )
        except Exception as e:
            return f"Error viewing PDF page: {str(e)}"
