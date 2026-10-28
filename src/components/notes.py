from typing import List, Callable, Dict, Any
import os
import re
import sqlite3
from src.components.base import BaseComponent
from src.utils.config import load_config

class StructuredNotesComponent(BaseComponent):
    """
    Provides a key-value scratchpad for the agent to deliberately store information.
    """
    def __init__(self, db_path: str = None):
        if db_path is None:
            config = load_config()
            notes_cfg = config.get("notes", {})
            if "db_path" not in notes_cfg or not notes_cfg["db_path"]:
                raise ValueError("StructuredNotesComponent requires 'db_path' to be explicitly defined in config under [notes].db_path")
            self.db_path = notes_cfg["db_path"]
        else:
            self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self):
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS notes (key TEXT PRIMARY KEY, value TEXT)"
                )
                try:
                    conn.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(key, value)"
                    )
                    n_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
                    fts_count = conn.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0]
                    if n_count != fts_count:
                        conn.execute("DELETE FROM notes_fts")
                        conn.execute("INSERT INTO notes_fts (key, value) SELECT key, value FROM notes")
                except Exception:
                    pass
        finally:
            conn.close()
    def get_system_prompt_addition(self) -> str:
        prompt = (
            "You have access to a Structured Notes system. Treat this like a desk where "
            "you can store specific information blocks for later retrieval. Use `put_note` "
            "to save things you don't want to lose.\n"
            "You can organize notes hierarchically using '/' in your keys (e.g., 'book/character/name'). "
            "Use `list_note_keys` with an optional 'path' argument to navigate folders. "
            "Use `rename_note` to move or rename notes. "
            "(You can also retrieve notes via the `global_search` tool).\n"
            "A note named 'sticky' is always included in your system prompt if it exists."
        )

        conn = self._connect_db()
        try:
            with conn:
                cursor = conn.execute("SELECT value FROM notes WHERE key = ?", ("sticky",))
                row = cursor.fetchone()
                if row:
                    prompt += f"\n\n--- STICKY NOTE ---\n{row[0]}"
        finally:
            conn.close()

        return prompt

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

    @staticmethod
    def _extract_snippet(value: str, tokens: List[str], max_chars: int = 600) -> str:
        text = value or ""
        if len(text) <= max_chars:
            return text
        lower = text.lower()
        match_pos = -1
        for tok in tokens:
            idx = lower.find(tok.lower())
            if idx != -1:
                match_pos = idx
                break
        if match_pos == -1:
            return text[:max_chars] + "\n...[truncated]..."
        half = max_chars // 2
        start = max(0, match_pos - half)
        end = min(len(text), start + max_chars)
        start = max(0, end - max_chars)
        prefix = "...[truncated]...\n" if start > 0 else ""
        suffix = "\n...[truncated]..." if end < len(text) else ""
        return f"{prefix}{text[start:end]}{suffix}"

    def execute_search(self, query: str) -> str:
        """Standardized interface for GlobalSearchComponent using FTS5 BM25 + tokenized keyword ranking."""
        clean_query = (query or "").strip()
        if not clean_query:
            return ""

        tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", clean_query) if len(t) >= 2]
        if not tokens:
            tokens = [clean_query.lower()]

        scored_notes: Dict[str, tuple[float, str]] = {}
        conn = self._connect_db()
        try:
            with conn:
                # 1. FTS5 BM25 match (with prefix matching per token)
                try:
                    fts_expr = " OR ".join(f'"{t}"*' for t in tokens)
                    fts_rows = conn.execute(
                        "SELECT key, value, bm25(notes_fts) AS rank FROM notes_fts "
                        "WHERE notes_fts MATCH ? ORDER BY rank ASC LIMIT 10",
                        (fts_expr,),
                    ).fetchall()
                    for idx, (k, v, _) in enumerate(fts_rows):
                        # Higher score for better FTS rank
                        scored_notes[str(k)] = (10.0 - idx * 0.5, str(v or ""))
                except Exception:
                    pass

                # 2. Tokenized multi-keyword scoring across all notes (catches path slashes, substrings, etc.)
                query_lower = clean_query.lower()
                rows = conn.execute("SELECT key, value FROM notes").fetchall()
                for k, v in rows:
                    key_str = str(k or "")
                    val_str = str(v or "")
                    k_low = key_str.lower()
                    v_low = val_str.lower()

                    score = 0.0
                    if query_lower in k_low:
                        score += 6.0
                    if query_lower in v_low:
                        score += 4.0
                    for tok in tokens:
                        if tok in k_low:
                            score += 2.0
                        if tok in v_low:
                            score += 1.0

                    if score > 0:
                        prev_score = scored_notes.get(key_str, (0.0, val_str))[0]
                        scored_notes[key_str] = (prev_score + score, val_str)
        finally:
            conn.close()

        if not scored_notes:
            return ""

        ranked = sorted(
            scored_notes.items(),
            key=lambda kv: (kv[1][0], kv[0]),
            reverse=True,
        )[:5]

        results = []
        for key, (_, value) in ranked:
            snippet = self._extract_snippet(value, tokens, max_chars=600)
            results.append(f"Note '{key}':\n{snippet}")
        return "\n\n".join(results)

    # --- Tool Implementations ---

    def list_note_keys(self, path: str = "") -> List[Dict[str, Any]]:
        """
        Returns a structured list of note keys and pseudo-directories under the specified path.
        Leave path empty to list the root level.
        """
        conn = self._connect_db()
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
        conn = self._connect_db()
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
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO notes (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value", 
                    (key, value)
                )
                try:
                    conn.execute("DELETE FROM notes_fts WHERE key = ?", (key,))
                    conn.execute("INSERT INTO notes_fts (key, value) VALUES (?, ?)", (key, value))
                except Exception:
                    pass
        finally:
            conn.close()
        return f"Success: Note '{key}' saved."

    def delete_note(self, key: str) -> str:
        """Deletes a note."""
        conn = self._connect_db()
        try:
            with conn:
                conn.execute("DELETE FROM notes WHERE key = ?", (key,))
                try:
                    conn.execute("DELETE FROM notes_fts WHERE key = ?", (key,))
                except Exception:
                    pass
        finally:
            conn.close()
        return f"Success: Note '{key}' deleted if it existed."
        
    def rename_note(self, old_key: str, new_key: str) -> str:
        """Renames or moves a note from old_key to new_key."""
        conn = self._connect_db()
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
                try:
                    conn.execute("UPDATE notes_fts SET key = ? WHERE key = ?", (new_key, old_key))
                except Exception:
                    pass
        finally:
            conn.close()
            
        return f"Success: Note renamed from '{old_key}' to '{new_key}'."
