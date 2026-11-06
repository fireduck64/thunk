from typing import List, Callable, Dict, Any, Optional, Tuple
import os
import re
import sqlite3
from src.components.base import BaseComponent
from src.utils.config import load_config


ASSISTANT_GUARD_ERROR = (
    "Error: Notes under 'assistant/' are managed automatically by your Executive Assistant. "
    "To change how your assistant takes notes or briefs you, edit 'prompts/assistant_briefing', "
    "'prompts/assistant_filing', or 'prompts/assistant_compaction'."
)

DEFAULT_ASSISTANT_BRIEFING_PROMPT = (
    "You are my Executive Assistant preparing a quick briefing before I respond.\n"
    "Compare the Incoming Event against my Active Context (Core Summary + Recent Messages).\n"
    "1. ONLY brief me on people, terms, projects, or history that are referenced in the Incoming Event but MISSING from my Active Context.\n"
    "2. Do NOT repeat things I am already actively discussing.\n"
    "3. Pull relevant details from 'assistant/*' dossiers and my general notes (who the speaker is, open promises, inside jokes, or factual anchors).\n"
    "4. Keep it under 4 bullet points. If nothing is missing from my active context, output exactly: NONE"
)

DEFAULT_ASSISTANT_FILING_PROMPT = (
    "You are my Executive Assistant organizing my 'assistant/*' reference notes as older messages leave my active working memory.\n"
    "Review the Evicted Messages and existing 'assistant/*' notes:\n"
    "1. Identify durable facts about people/speakers (e.g. 'assistant/people/<name>'), ongoing projects/terms ('assistant/topics/<slug>'), or unresolved promises.\n"
    "2. For existing notes, extract ONLY net-new facts/bullet points that are not already present in the note.\n"
    "3. Ignore transient small talk or routine tool outputs."
)

DEFAULT_ASSISTANT_COMPACTION_PROMPT = (
    "You are my Executive Assistant performing background housekeeping on an oversized 'assistant/*' reference dossier.\n"
    "The note below has accumulated many appended bullet points over time.\n"
    "1. Consolidate and deduplicate overlapping bullets into a clean, well-organized markdown dossier.\n"
    "2. Preserve ALL specific names, dates, technical details, biographical facts, inside jokes, and unresolved promises.\n"
    "3. Remove only redundant phrasing or obsolete transient noise.\n"
    "4. Output ONLY the compacted markdown content for the note (do not wrap in JSON or code fences)."
)

DEFAULT_SEED_PROMPTS: Dict[str, str] = {
    "prompts/assistant_briefing": DEFAULT_ASSISTANT_BRIEFING_PROMPT,
    "prompts/assistant_filing": DEFAULT_ASSISTANT_FILING_PROMPT,
    "prompts/assistant_compaction": DEFAULT_ASSISTANT_COMPACTION_PROMPT,
}


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
                    "CREATE TABLE IF NOT EXISTS notes (key TEXT PRIMARY KEY, value TEXT, summary TEXT DEFAULT '')"
                )
                cols = [row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()]
                if "summary" not in cols:
                    conn.execute("ALTER TABLE notes ADD COLUMN summary TEXT DEFAULT ''")
                try:
                    missing_rows = conn.execute(
                        "SELECT key, value FROM notes WHERE summary IS NULL OR summary = ''"
                    ).fetchall()
                    if missing_rows:
                        conn.executemany(
                            "UPDATE notes SET summary = ? WHERE key = ?",
                            [(self._derive_summary(str(v or "")), str(k)) for k, v in missing_rows],
                        )
                except Exception:
                    pass
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

    @staticmethod
    def _normalize_key(key: str) -> str:
        return (key or "").strip().lstrip("/")

    @classmethod
    def _is_assistant_key(cls, key: str) -> bool:
        clean = cls._normalize_key(key)
        return clean == "assistant" or clean.startswith("assistant/")

    @staticmethod
    def _derive_summary(content: str, explicit_summary: str = "") -> str:
        if explicit_summary and explicit_summary.strip():
            return " ".join(explicit_summary.strip().split())
        for line in (content or "").splitlines():
            clean = line.strip().lstrip("#*- ").strip()
            if clean:
                return clean[:140] + ("..." if len(clean) > 140 else "")
        return ""

    def ensure_seed_prompts(self) -> None:
        """Seeds default prompts under 'prompts/*' if they do not already exist."""
        conn = self._connect_db()
        try:
            with conn:
                for key, default_val in DEFAULT_SEED_PROMPTS.items():
                    row = conn.execute("SELECT value FROM notes WHERE key = ?", (key,)).fetchone()
                    if row is None or not str(row[0] or "").strip():
                        summary = self._derive_summary(default_val)
                        conn.execute(
                            "INSERT INTO notes (key, value, summary) VALUES (?, ?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, summary=excluded.summary",
                            (key, default_val, summary),
                        )
                        try:
                            conn.execute("DELETE FROM notes_fts WHERE key = ?", (key,))
                            conn.execute("INSERT INTO notes_fts (key, value) VALUES (?, ?)", (key, default_val))
                        except Exception:
                            pass
        finally:
            conn.close()

    def resolve_assistant_key(self, key: str) -> str:
        """
        Ensures 'key' starts with 'assistant/' and resolves it case-insensitively against
        existing 'assistant/*' notes in SQLite so e.g. 'assistant/people/Wisp' and
        'assistant/people/wisp' always resolve to the same canonical key.
        """
        clean_key = self._normalize_key(key)
        if not clean_key:
            raise ValueError("Assistant note key cannot be empty.")
        if clean_key.lower() == "assistant":
            raise ValueError("Assistant note key must specify a path under 'assistant/'.")
        if not clean_key.lower().startswith("assistant/"):
            clean_key = f"assistant/{clean_key}"
        else:
            clean_key = "assistant/" + clean_key[len("assistant/"):]

        conn = self._connect_db()
        try:
            with conn:
                row = conn.execute(
                    "SELECT key FROM notes WHERE LOWER(key) = LOWER(?) LIMIT 1",
                    (clean_key,),
                ).fetchone()
                if row and row[0]:
                    return str(row[0])
        finally:
            conn.close()
        return clean_key

    def assistant_put_note(self, key: str, summary: str = "", content: Optional[str] = None) -> str:
        """
        Internal helper for ExecutiveAssistantComponent to write or update notes under 'assistant/*'.
        Automatically prefixes 'assistant/' if missing and resolves canonical casing.
        """
        canonical_key = self.resolve_assistant_key(key)

        if content is None:
            note_content = str(summary or "")
            note_summary = self._derive_summary(note_content)
        else:
            note_content = str(content or "")
            note_summary = self._derive_summary(note_content, explicit_summary=str(summary or ""))

        fts_text = f"{note_summary}\n{note_content}".strip() if note_summary else note_content
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO notes (key, value, summary) VALUES (?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, summary=excluded.summary",
                    (canonical_key, note_content, note_summary),
                )
                try:
                    conn.execute("DELETE FROM notes_fts WHERE key = ?", (canonical_key,))
                    conn.execute("INSERT INTO notes_fts (key, value) VALUES (?, ?)", (canonical_key, fts_text))
                except Exception:
                    pass
        finally:
            conn.close()
        return f"Success: Assistant note '{canonical_key}' saved."

    def assistant_append_note(self, key: str, summary: str, new_points: List[str]) -> str:
        """
        Resolves the canonical 'assistant/*' key case-insensitively and appends net-new
        markdown bullet points to the existing note content in Python.
        Returns the resolved canonical key.
        """
        canonical_key = self.resolve_assistant_key(key)

        conn = self._connect_db()
        existing_content: Optional[str] = None
        existing_summary: str = ""
        try:
            with conn:
                row = conn.execute(
                    "SELECT value, summary FROM notes WHERE key = ?",
                    (canonical_key,),
                ).fetchone()
                if row is not None:
                    existing_content = str(row[0] or "")
                    existing_summary = str(row[1] or "")
        finally:
            conn.close()

        existing_lines_norm = set()
        if existing_content:
            for line in existing_content.splitlines():
                stripped = line.strip().lstrip("-* ").strip().lower()
                if stripped:
                    existing_lines_norm.add(stripped)

        cleaned_points: List[str] = []
        for raw_pt in new_points or []:
            for subline in str(raw_pt or "").splitlines():
                s = subline.strip()
                if not s:
                    continue
                core = s.lstrip("-* ").strip()
                if not core:
                    continue
                core_low = core.lower()
                if core_low in existing_lines_norm:
                    continue
                existing_lines_norm.add(core_low)
                bullet = s if (s.startswith("- ") or s.startswith("* ") or s.startswith("#")) else f"- {core}"
                cleaned_points.append(bullet)

        incoming_summary = (summary or "").strip()
        if existing_content is not None and existing_content.strip():
            if cleaned_points:
                updated_content = existing_content.rstrip() + "\n" + "\n".join(cleaned_points)
            else:
                updated_content = existing_content
            final_summary = incoming_summary if incoming_summary else existing_summary
        else:
            updated_content = "\n".join(cleaned_points)
            if incoming_summary:
                final_summary = incoming_summary
            elif cleaned_points:
                final_summary = cleaned_points[0].lstrip("-*# ").strip()[:120]
            else:
                final_summary = ""

        if updated_content.strip() or existing_content is not None:
            self.assistant_put_note(canonical_key, final_summary, updated_content)
        return canonical_key

    def assistant_get_note(self, key: str) -> Optional[str]:
        """Internal helper for ExecutiveAssistantComponent to read a note (returns None if not found)."""
        clean_key = self._normalize_key(key)
        conn = self._connect_db()
        try:
            with conn:
                row = conn.execute("SELECT value FROM notes WHERE key = ?", (clean_key,)).fetchone()
                if row is not None:
                    return str(row[0])
                if clean_key.lower().startswith("assistant/"):
                    row = conn.execute(
                        "SELECT value FROM notes WHERE LOWER(key) = LOWER(?) LIMIT 1",
                        (clean_key,),
                    ).fetchone()
                    if row is not None:
                        return str(row[0])
                return None
        finally:
            conn.close()

    def get_all_notes(self) -> List[Dict[str, str]]:
        """Returns all notes with key, summary, and content for candidate scanning."""
        conn = self._connect_db()
        try:
            with conn:
                cols = [row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()]
                if "summary" in cols:
                    rows = conn.execute("SELECT key, value, summary FROM notes ORDER BY key ASC").fetchall()
                else:
                    rows = [(k, v, "") for k, v in conn.execute("SELECT key, value FROM notes ORDER BY key ASC").fetchall()]
        finally:
            conn.close()

        items: List[Dict[str, str]] = []
        for k, v, s in rows:
            key_str = str(k or "")
            val_str = str(v or "")
            sum_str = self._derive_summary(val_str, explicit_summary=str(s or ""))
            items.append({"key": key_str, "summary": sum_str, "content": val_str})
        return items
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

    def search_notes_structured(
        self,
        query: str,
        limit: int = 3,
        exclude_prefixes: Tuple[str, ...] = ("prompts/", "assistant/"),
    ) -> List[Dict[str, Any]]:
        """
        Searches notes using FTS5 BM25 + tokenized keyword ranking and returns structured note dicts:
        [{"key": k, "summary": summary, "content": v, "score": score}, ...]
        Excludes 'sticky' and any key starting with exclude_prefixes.
        """
        clean_query = (query or "").strip()
        if not clean_query or limit <= 0:
            return []

        tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", clean_query) if len(t) >= 2]
        if not tokens:
            tokens = [clean_query.lower()]

        norm_excludes = tuple(p.lower() for p in (exclude_prefixes or ()))

        def _is_excluded(k: str) -> bool:
            kl = (k or "").strip().lower()
            if not kl or kl == "sticky":
                return True
            return any(kl.startswith(pref) for pref in norm_excludes)

        scored_notes: Dict[str, Tuple[float, str, str]] = {}
        conn = self._connect_db()
        try:
            with conn:
                # 1. FTS5 BM25 match
                try:
                    fts_expr = " OR ".join(f'"{t}"*' for t in tokens)
                    fts_rows = conn.execute(
                        "SELECT key, value, bm25(notes_fts) AS rank FROM notes_fts "
                        "WHERE notes_fts MATCH ? ORDER BY rank ASC LIMIT 15",
                        (fts_expr,),
                    ).fetchall()
                    for idx, (k, v, _) in enumerate(fts_rows):
                        key_str = str(k or "")
                        if _is_excluded(key_str):
                            continue
                        scored_notes[key_str] = (10.0 - idx * 0.5, str(v or ""), "")
                except Exception:
                    pass

                # 2. Tokenized multi-keyword scoring across notes
                query_lower = clean_query.lower()
                cols = [row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()]
                if "summary" in cols:
                    rows = conn.execute("SELECT key, value, summary FROM notes").fetchall()
                else:
                    rows = [(k, v, "") for k, v in conn.execute("SELECT key, value FROM notes").fetchall()]

                for k, v, s in rows:
                    key_str = str(k or "")
                    if _is_excluded(key_str):
                        continue
                    val_str = str(v or "")
                    sum_str = str(s or "")
                    k_low = key_str.lower()
                    v_low = val_str.lower()
                    s_low = sum_str.lower()

                    score = 0.0
                    if query_lower in k_low:
                        score += 6.0
                    if query_lower in s_low:
                        score += 5.0
                    if query_lower in v_low:
                        score += 4.0
                    for tok in tokens:
                        if tok in k_low:
                            score += 2.0
                        if tok in s_low:
                            score += 1.5
                        if tok in v_low:
                            score += 1.0

                    if score > 0 or key_str in scored_notes:
                        prev_score = scored_notes.get(key_str, (0.0, val_str, sum_str))[0]
                        scored_notes[key_str] = (prev_score + score, val_str, sum_str)
        finally:
            conn.close()

        if not scored_notes:
            return []

        ranked = sorted(
            scored_notes.items(),
            key=lambda kv: (kv[1][0], kv[0]),
            reverse=True,
        )[:limit]

        out: List[Dict[str, Any]] = []
        for key, (score, val_str, sum_str) in ranked:
            out.append(
                {
                    "key": key,
                    "summary": self._derive_summary(val_str, explicit_summary=sum_str),
                    "content": val_str,
                    "score": float(score),
                }
            )
        return out

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
                norm_key = self._normalize_key(key)
                if norm_key != key:
                    row = conn.execute("SELECT value FROM notes WHERE key = ?", (norm_key,)).fetchone()
                    if row:
                        return row[0]
                if norm_key.lower().startswith("assistant/"):
                    row = conn.execute(
                        "SELECT value FROM notes WHERE LOWER(key) = LOWER(?) LIMIT 1",
                        (norm_key,),
                    ).fetchone()
                    if row:
                        return row[0]
                return f"Error: Note with key '{key}' not found."

        finally:
            conn.close()
            
    def put_note(self, key: str, value: str) -> str:
        """Creates or overwrites a note with the given key and value."""
        if self._is_assistant_key(key):
            return ASSISTANT_GUARD_ERROR

        summary = self._derive_summary(value)
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO notes (key, value, summary) VALUES (?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, summary=excluded.summary", 
                    (key, value, summary)
                )
                try:
                    conn.execute("DELETE FROM notes_fts WHERE key = ?", (key,))
                    conn.execute("INSERT INTO notes_fts (key, value) VALUES (?, ?)", (key, value))
                except Exception:
                    pass
        finally:
            conn.close()
        return f"Success: Note '{key}' saved."

    def _reset_seed_prompt_in_db(self, conn: sqlite3.Connection, prompt_key: str) -> None:
        default_val = DEFAULT_SEED_PROMPTS[prompt_key]
        summary = self._derive_summary(default_val)
        conn.execute(
            "INSERT INTO notes (key, value, summary) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, summary=excluded.summary",
            (prompt_key, default_val, summary),
        )
        try:
            conn.execute("DELETE FROM notes_fts WHERE key = ?", (prompt_key,))
            conn.execute("INSERT INTO notes_fts (key, value) VALUES (?, ?)", (prompt_key, default_val))
        except Exception:
            pass

    def delete_note(self, key: str) -> str:
        """Deletes a note."""
        if self._is_assistant_key(key):
            return ASSISTANT_GUARD_ERROR

        norm_key = self._normalize_key(key)
        conn = self._connect_db()
        try:
            with conn:
                if norm_key in DEFAULT_SEED_PROMPTS:
                    if key != norm_key:
                        conn.execute("DELETE FROM notes WHERE key = ?", (key,))
                    self._reset_seed_prompt_in_db(conn, norm_key)
                    return f"Notice: Protected prompt note '{norm_key}' cannot be deleted and was reset to factory defaults."

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
        if self._is_assistant_key(old_key) or self._is_assistant_key(new_key):
            return ASSISTANT_GUARD_ERROR

        norm_old = self._normalize_key(old_key)
        conn = self._connect_db()
        try:
            with conn:
                if norm_old in DEFAULT_SEED_PROMPTS:
                    self._reset_seed_prompt_in_db(conn, norm_old)
                    return (
                        f"Notice: Protected prompt note '{norm_old}' cannot be moved away and was reset to factory defaults."
                    )

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
