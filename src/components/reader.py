import os
import glob
import json
import sqlite3
from typing import List, Callable, Dict, Any
from ebooklib import epub
from bs4 import BeautifulSoup
from src.components.base import BaseComponent

class SequentialReaderComponent(BaseComponent):
    """
    Allows the agent to open EPUB books and read them sequentially, chunk by chunk.
    This is useful for deep comprehension tasks where semantic search is insufficient.
    Maintains bookmark state across restarts.
    """
    def __init__(self, library_dir: str = "books", chunk_size: int = 2500, db_path: str = "reader_state.db"):
        self.library_dir = library_dir
        self.chunk_size = chunk_size
        self.db_path = db_path
        
        # Ensure library directory exists
        os.makedirs(self.library_dir, exist_ok=True)
        
        self._init_db()
        self._load_state()

    def _init_db(self):
        """Initializes the SQLite DB used to store the reader's state."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reader_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    current_book TEXT,
                    current_index INTEGER
                )
            """)
            # Ensure there is always a row to update
            conn.execute("INSERT OR IGNORE INTO reader_state (id, current_book, current_index) VALUES (1, NULL, 0)")

    def _save_state(self):
        """Saves the current bookmark to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE reader_state SET current_book = ?, current_index = ? WHERE id = 1",
                (self.current_book, self.current_index)
            )

    def _load_state(self):
        """Loads the bookmark from SQLite and rebuilds the text chunks if a book was open."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT current_book, current_index FROM reader_state WHERE id = 1")
            row = cursor.fetchone()
            
        self.current_book = row[0]
        self.current_index = row[1]
        self.current_chunks = []
        
        if self.current_book and os.path.exists(self.current_book):
            print(f"[Reader] Restoring state: Re-parsing '{os.path.basename(self.current_book)}'...")
            self._parse_book(self.current_book)

    def _parse_book(self, abs_file_path: str):
        """Internal helper to extract text and chunk it without resetting the index."""
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
            "Use `search_library` to find books by title. Use `open_book` to load a book into "
            "your active reading state. Use `read_next_chunk` and `read_previous_chunk` to navigate "
            "through the open book."
        )
        if self.current_book:
            prompt += f"\n[Status: You currently have '{os.path.basename(self.current_book)}' open at chunk {self.current_index}/{max(1, len(self.current_chunks))}]."
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
        """Returns a list of all EPUB filenames available to read."""
        search_pattern = os.path.join(self.library_dir, "**", "*.epub")
        files = glob.glob(search_pattern, recursive=True)
        return [os.path.relpath(f, self.library_dir) for f in files]

    def search_library(self, query: str) -> List[str]:
        """
        Searches for books in the library whose filenames match the query.
        Returns a list of matching relative file paths that can be passed to open_book.
        """
        search_pattern = os.path.join(self.library_dir, "**", "*.epub")
        files = glob.glob(search_pattern, recursive=True)
        
        matches = []
        query_lower = query.lower()
        for f in files:
            rel_path = os.path.relpath(f, self.library_dir)
            if query_lower in rel_path.lower():
                matches.append(rel_path)
                
        if len(matches) > 50:
            return matches[:50] + [f"... and {len(matches) - 50} more. Please refine your search."]
        return matches

    def open_book(self, filename: str) -> str:
        """
        Opens an EPUB file from the library, extracts its text, and prepares it for reading.
        Resets your reading position to the beginning.
        """
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
            
            return f"Successfully opened '{filename}'. The book has been split into {len(self.current_chunks)} reading chunks. Use read_next_chunk to begin."
            
        except Exception as e:
            return f"Error opening book: {str(e)}"

    def read_next_chunk(self) -> str:
        """Returns the next chunk of text from the currently open book and advances your bookmark."""
        if not self.current_book:
            return "Error: No book is currently open. Use open_book first."
            
        if self.current_index >= len(self.current_chunks):
            return "You have reached the end of the book."
            
        chunk = self.current_chunks[self.current_index]
        self.current_index += 1
        self._save_state()
        
        if hasattr(self, 'agent') and self.agent:
            self.agent.rebuild_system_prompt()
            
        return f"--- Chunk {self.current_index}/{len(self.current_chunks)} ---\n{chunk}"

    def read_previous_chunk(self) -> str:
        """Moves your bookmark back one chunk and returns that text."""
        if not self.current_book:
            return "Error: No book is currently open."
            
        if self.current_index <= 1:
            self.current_index = 0
            self._save_state()
            if hasattr(self, 'agent') and self.agent:
                self.agent.rebuild_system_prompt()
            return "You are already at the beginning of the book."
            
        self.current_index -= 2 
        if self.current_index < 0:
            self.current_index = 0
            
        chunk = self.current_chunks[self.current_index]
        self.current_index += 1 
        self._save_state()
        
        if hasattr(self, 'agent') and self.agent:
            self.agent.rebuild_system_prompt()
            
        return f"--- Chunk {self.current_index}/{len(self.current_chunks)} ---\n{chunk}"
