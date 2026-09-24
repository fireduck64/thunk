from typing import List, Callable, Dict, Any
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
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS notes (key TEXT PRIMARY KEY, value TEXT)"
                )

        finally:
            conn.close()
    def get_system_prompt_addition(self) -> str:
        return (
            "You have access to a Structured Notes system. Treat this like a desk where "
            "you can store specific information blocks for later retrieval. Use `put_note` "
            "to save things you don't want to lose.\n"
            "You can organize notes hierarchically using '/' in your keys (e.g., 'book/character/name'). "
            "Use `list_note_keys` with an optional 'path' argument to navigate folders. "
            "Use `rename_note` to move or rename notes. "
            "(You can also retrieve notes via the `global_search` tool)."
        )

    def get_tools(self) -> List[Callable]:
        # We still expose get_note for explicit precise lookups, 
        # but the agent can also rely on global_search.
        return [
            self.list_note_keys,
            self.get_note,
            self.put_note,
            self.delete_note,
            self.rename_note
        ]

    def execute_search(self, query: str) -> str:
        """Standardized interface for GlobalSearchComponent."""
        # Simple string matching across all note keys and values
        query_lower = query.lower()
        results = []
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute("SELECT key, value FROM notes")
                for key, value in cursor.fetchall():
                    if query_lower in key.lower() or query_lower in value.lower():
                        results.append(f"Note '{key}':\n{value}")

        finally:
            conn.close()
        if not results:
            return "" # Return empty string so GlobalSearch skips this component
        return "\n\n".join(results)

    # --- Tool Implementations ---

    def list_note_keys(self, path: str = "") -> List[Dict[str, Any]]:
        """
        Returns a structured list of note keys and pseudo-directories under the specified path.
        Leave path empty to list the root level.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute("SELECT key, LENGTH(value) FROM notes")
                rows = cursor.fetchall()
        finally:
            conn.close()

        prefix = path if not path or path.endswith('/') else path + '/'
        
        dirs = {}
        files = []
        
        for key, length in rows:
            if key.startswith(prefix):
                remainder = key[len(prefix):]
                if '/' in remainder:
                    # It's a pseudo-directory
                    dir_name = remainder.split('/')[0]
                    dirs[dir_name] = dirs.get(dir_name, 0) + 1
                else:
                    # It's a direct note at this level
                    files.append({"type": "note", "name": remainder, "size_chars": length or 0})
                    
        results = []
        # Sort directories and append
        for d in sorted(dirs.keys()):
            results.append({"type": "dir", "name": d, "count": dirs[d]})
            
        # Sort files by name and append
        results.extend(sorted(files, key=lambda x: x["name"]))
        
        return results

    def get_note(self, key: str) -> str:
        """Retrieves the contents of a specific note."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute("SELECT value FROM notes WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    return row[0]
                return f"Error: Note with key '{key}' not found."

        finally:
            conn.close()
            
    def put_note(self, key: str, value: str) -> str:
        """Creates or overwrites a note with the given key and value."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "INSERT INTO notes (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", 
                    (key, value)
                )
        finally:
            conn.close()
        return f"Success: Note '{key}' saved."

    def delete_note(self, key: str) -> str:
        """Deletes a note."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("DELETE FROM notes WHERE key = ?", (key,))
        finally:
            conn.close()
        return f"Success: Note '{key}' deleted if it existed."
        
    def rename_note(self, old_key: str, new_key: str) -> str:
        """Renames or moves a note from old_key to new_key."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                # Check if old_key exists
                cursor = conn.execute("SELECT 1 FROM notes WHERE key = ?", (old_key,))
                if not cursor.fetchone():
                    return f"Error: Source note '{old_key}' does not exist."
                    
                # Check if new_key already exists to prevent accidental overwrite
                cursor = conn.execute("SELECT 1 FROM notes WHERE key = ?", (new_key,))
                if cursor.fetchone():
                    return f"Error: Destination note '{new_key}' already exists. Delete it first if you want to overwrite."
                    
                # Perform the rename
                conn.execute("UPDATE notes SET key = ? WHERE key = ?", (new_key, old_key))
        finally:
            conn.close()
            
        return f"Success: Note renamed from '{old_key}' to '{new_key}'."
