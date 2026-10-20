import os
import re
import glob
import json
import sqlite3
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Callable, Dict, Any, Tuple
from ebooklib import epub
from bs4 import BeautifulSoup
from src.components.base import BaseComponent
from src.utils.config import load_config

class SequentialReaderComponent(BaseComponent):
    """
    Allows the agent to open EPUB books and read them sequentially, chunk by chunk.
    This is useful for deep comprehension tasks where semantic search is insufficient.
    Maintains bookmark state across restarts.
    """
    def __init__(self, library_dir: str = None, db_path: str = None, chunk_size: int = 2500):
        self.config = load_config()
        
        if library_dir is None:
            reader_cfg = self.config.get("reader", {})
            if "library_dir" not in reader_cfg or not reader_cfg["library_dir"]:
                raise ValueError("SequentialReaderComponent requires 'library_dir' to be explicitly defined in config under [reader].library_dir")
            self.library_dir = reader_cfg["library_dir"]
        else:
            self.library_dir = library_dir

        if db_path is None:
            reader_cfg = self.config.get("reader", {})
            if "db_path" not in reader_cfg or not reader_cfg["db_path"]:
                raise ValueError("SequentialReaderComponent requires 'db_path' to be explicitly defined in config under [reader].db_path")
            self.db_path = reader_cfg["db_path"]
        else:
            self.db_path = db_path
            
        self.chunk_size = chunk_size
        
        # Ensure library and database parent directories exist
        os.makedirs(self.library_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        self._init_db()
        self._load_state()

    def _init_db(self):
        """Initializes the SQLite DB used to store the reader's state."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reader_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        current_book TEXT,
                        current_index INTEGER
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS book_metadata (
                        rel_path TEXT PRIMARY KEY,
                        mtime REAL,
                        title TEXT,
                        author TEXT,
                        label TEXT
                    )
                """)
                # Ensure there is always a row to update
                conn.execute("INSERT OR IGNORE INTO reader_state (id, current_book, current_index) VALUES (1, NULL, 0)")

        finally:
            conn.close()

    def _extract_epub_metadata(self, abs_file_path: str) -> Tuple[str, str]:
        """Fast OPF XML extraction of (title, author) from an EPUB without decompressing chapters."""
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

    def _format_book_label(self, abs_file_path: str) -> str:
        """Returns a descriptive label combining EPUB title, author, and library-relative path."""
        rel_path = os.path.relpath(abs_file_path, self.library_dir)
        title, author = self._extract_epub_metadata(abs_file_path)
        if title and author:
            return f"{title} by {author} [path: {rel_path}]"
        elif title:
            return f"{title} [path: {rel_path}]"
        return rel_path

    def _get_library_catalog(self) -> List[str]:
        """Returns cached descriptive labels for all EPUBs in library_dir, updating any new/modified files."""
        search_pattern = os.path.join(self.library_dir, "**", "*.epub")
        files = glob.glob(search_pattern, recursive=True)

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                rows = conn.execute("SELECT rel_path, mtime, label FROM book_metadata").fetchall()
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
                        title, author = self._extract_epub_metadata(f)
                        if title and author:
                            label = f"{title} by {author} [path: {rel_path}]"
                        elif title:
                            label = f"{title} [path: {rel_path}]"
                        else:
                            label = rel_path
                        updates.append((rel_path, mtime, title, author, label))
                        catalog.append(label)

                if updates:
                    conn.executemany(
                        "INSERT OR REPLACE INTO book_metadata (rel_path, mtime, title, author, label) VALUES (?, ?, ?, ?, ?)",
                        updates
                    )
        finally:
            conn.close()

        return sorted(catalog)
    def _save_state(self):
        """Saves the current bookmark to SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE reader_state SET current_book = ?, current_index = ? WHERE id = 1",
                    (self.current_book, self.current_index)
                )

        finally:
            conn.close()
    def _load_state(self):
        """Loads the bookmark from SQLite and rebuilds the text chunks if a book was open."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute("SELECT current_book, current_index FROM reader_state WHERE id = 1")
                row = cursor.fetchone()

        finally:
            conn.close()
        self.current_book = row[0]
        self.current_index = row[1]
        self.current_book_label = None
        self.current_chunks = []
        
        if self.current_book and os.path.exists(self.current_book):
            self.current_book_label = self._format_book_label(self.current_book)
            print(f"[Reader] Restoring state: Re-parsing '{self.current_book_label}'...")
            self._parse_book(self.current_book)

    def _parse_book(self, abs_file_path: str):
        """Internal helper to extract text and chunk it without resetting the index."""
        self.current_book_label = self._format_book_label(abs_file_path)
        book = epub.read_epub(abs_file_path)
        full_text = []
        
        for item in book.get_items_of_type(9): # ITEM_DOCUMENT
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            if text:
                full_text.append(text)
                
        combined_text = "\n\n".join(full_text)
        
        self.current_chunks = [
            combined_text[i:i+self.chunk_size] 
            for i in range(0, len(combined_text), self.chunk_size)
        ]

    def get_system_prompt_addition(self) -> str:
        prompt = (
            "You have access to a sequential document reader. While your knowledge base search "
            "is good for finding facts, the sequential reader allows you to sit down and read "
            "an entire book from start to finish.\n"
            "Use `search_library` to find books by title or author. Use `open_book` (passing the relative path) "
            "to load a book into your active reading state. Use `read_next_chunk` and `read_previous_chunk` "
            "to navigate through the open book."
        )
        if self.current_book:
            label = self.current_book_label or os.path.basename(self.current_book)
            prompt += f"\n[Status: You currently have '{label}' open at chunk {self.current_index}/{max(1, len(self.current_chunks))}]."
        else:
            prompt += "\n[Status: You do not have a book open.]"
            
        return prompt

    def get_tools(self) -> List[Callable]:
        return [
            self.list_library,
            self.search_library,
            self.open_book,
            self.read_next_chunk,
            self.read_previous_chunk
        ]

    # --- Tool Implementations ---

    def list_library(self) -> List[str]:
        """Returns a list of all EPUB books available to read (with title, author, and relative path)."""
        return self._get_library_catalog()

    def search_library(self, query: str) -> List[str]:
        """
        Searches for books in the library whose title, author, or filename match all words in the query.
        Returns a list of matching entries showing 'Title by Author [path: relative/path.epub]'.
        Pass the relative path inside [path: ...] to open_book.
        """
        catalog = self._get_library_catalog()
        terms = [t for t in re.split(r"[\s,]+", query.lower()) if t]
        if not terms:
            return []

        matches = []
        for entry in catalog:
            entry_lower = entry.lower()
            if all(term in entry_lower for term in terms):
                matches.append(entry)
                
        if len(matches) > 50:
            return matches[:50] + [f"... and {len(matches) - 50} more. Please refine your search."]
        return matches

    def open_book(self, filename: str) -> str:
        """
        Opens an EPUB file from the library, extracts its text, and prepares it for reading.
        Resets your reading position to the beginning.
        """
        # If the agent passed the full formatted catalog entry "... [path: foo/bar.epub]", extract the path
        if "[path: " in filename and filename.endswith("]"):
            filename = filename.split("[path: ", 1)[1][:-1].strip()

        file_path = os.path.join(self.library_dir, filename)
        
        abs_library_dir = os.path.abspath(self.library_dir)
        abs_file_path = os.path.abspath(file_path)
        
        if not abs_file_path.startswith(abs_library_dir):
            return f"Error: Invalid filename '{filename}'. Path traversal is not allowed."

        if not os.path.exists(abs_file_path):
            return f"Error: Book '{filename}' not found at {abs_file_path}."

        try:
            print(f"[Reader] Extracting text from {abs_file_path}...")
            self._parse_book(abs_file_path)
            self.current_book = abs_file_path
            self.current_index = 0
            self._save_state()
            
            # Request the core to rebuild the prompt so the status updates immediately
            if hasattr(self, 'agent') and self.agent:
                self.agent.rebuild_system_prompt()
            
            return f"Successfully opened '{self.current_book_label}'. The book has been split into {len(self.current_chunks)} reading chunks. Use read_next_chunk to begin."
            
        except Exception as e:
            return f"Error opening book: {str(e)}"

    def read_next_chunk(self) -> str:
        """Returns the next chunk of text from the currently open book and advances your bookmark."""
        if not self.current_book:
            return "Error: No book is currently open. Use open_book first."
            
        if self.current_index >= len(self.current_chunks):
            return f"You have reached the end of '{self.current_book_label}'."
            
        chunk = self.current_chunks[self.current_index]
        self.current_index += 1
        self._save_state()
        
        if hasattr(self, 'agent') and self.agent:
            self.agent.rebuild_system_prompt()
            
        return f"--- [Book: {self.current_book_label}] Chunk {self.current_index}/{len(self.current_chunks)} ---\n{chunk}"

    def read_previous_chunk(self) -> str:
        """Moves your bookmark back one chunk and returns that text."""
        if not self.current_book:
            return "Error: No book is currently open."
            
        if self.current_index <= 1:
            self.current_index = 0
            self._save_state()
            if hasattr(self, 'agent') and self.agent:
                self.agent.rebuild_system_prompt()
            return f"You are already at the beginning of '{self.current_book_label}'."
            
        self.current_index -= 2 
        if self.current_index < 0:
            self.current_index = 0
            
        chunk = self.current_chunks[self.current_index]
        self.current_index += 1 
        self._save_state()
        
        if hasattr(self, 'agent') and self.agent:
            self.agent.rebuild_system_prompt()
            
        return f"--- [Book: {self.current_book_label}] Chunk {self.current_index}/{len(self.current_chunks)} ---\n{chunk}"
