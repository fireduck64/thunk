from typing import List, Callable
import sqlite3
from src.components.base import BaseComponent

class StructuredNotesComponent(BaseComponent):
    """
    Provides a key-value scratchpad for the agent to deliberately store information.
    """
    def __init__(self, db_path: str = "notes.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS notes (key TEXT PRIMARY KEY, value TEXT)"
            )

    def get_system_prompt_addition(self) -> str:
        return (
            "You have access to a Structured Notes system. Treat this like a desk where "
            "you can store specific information blocks for later retrieval. Use `put_note` "
            "to save things you don't want to lose from your working memory, and `list_note_keys` "
            "to see what you have saved."
        )

    def get_tools(self) -> List[Callable]:
        return [
            self.list_note_keys,
            self.get_note,
            self.put_note,
            self.delete_note
        ]

    # --- Tool Implementations ---

    def list_note_keys(self) -> List[str]:
        """Returns a list of all currently saved note keys."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT key FROM notes")
            return [row[0] for row in cursor.fetchall()]

    def get_note(self, key: str) -> str:
        """Retrieves the contents of a specific note."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM notes WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return row[0]
            return f"Error: Note with key '{key}' not found."

    def put_note(self, key: str, value: str) -> str:
        """Creates or overwrites a note with the given key and value."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO notes (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", 
                (key, value)
            )
        return f"Success: Note '{key}' saved."

    def delete_note(self, key: str) -> str:
        """Deletes a note."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM notes WHERE key = ?", (key,))
        return f"Success: Note '{key}' deleted if it existed."
