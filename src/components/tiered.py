import asyncio
import fcntl
import inspect
import os
import sqlite3
import sys
import time
import json
from typing import List, Dict, Callable, Any
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from transformers import AutoTokenizer
from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.metrics import MetricSpan, count_tokens, infer_exec_location, set_shared_tokenizer

class TieredMemoryComponent(BaseComponent):
    """
    Manages the agent's memory tiers:
    1. Archival Memory (Logs all messages to SQLite, scoped by thread_id)
    2. Core Summary (Compresses old messages into a sliding summary based on token count, scoped by thread_id)
    """
    def __init__(self, db_path: str = None, thread_id: str = "main", tokenizer: Any = None):
        self.config = load_config()
        self.thread_id = (thread_id or "main").strip() or "main"
        if db_path is None:
            memory_cfg = self.config.get("memory", {})
            if "db_path" not in memory_cfg or not memory_cfg["db_path"]:
                raise ValueError("TieredMemoryComponent requires 'db_path' to be explicitly defined in config under [memory].db_path")
            self.db_path = memory_cfg["db_path"]
        else:
            self.db_path = db_path

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._acquire_lock()
        
        tokenizer_model = self.config.get("tokenizer", {}).get("model_name", "unsloth/gemma-7b")
        self.high_watermark = self.config.get("memory", {}).get("high_watermark", 16000)
        self.low_watermark = self.config.get("memory", {}).get("low_watermark", 8000)
        self.max_summarize_tokens = self.config.get("memory", {}).get("max_summarize_tokens", 6000)
        self.max_summary_tokens = self.config.get("memory", {}).get("max_summary_tokens", 8192)
        self.max_retries = 5
        self.retry_base_delay = 2.0
        
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                # We set a large model_max_length to prevent warnings when counting tokens 
                # for sequences larger than the tokenizer's default limit (e.g. 8192 for Gemma 7B)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_model, 
                    model_max_length=100000
                )
            except Exception as e:
                print(f"[Memory Warning] Could not load tokenizer '{tokenizer_model}': {e}")
                print("[Memory Warning] Falling back to 'gpt2' tokenizer for approximate token counting.")
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=100000)

        set_shared_tokenizer(self.tokenizer)
        self._init_db()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _settings_key(self, base_key: str) -> str:
        return base_key if self.thread_id == "main" else f"{self.thread_id}:{base_key}"

    def _acquire_lock(self):
        """Acquires an exclusive non-blocking OS file lock per (db_path, thread_id)."""
        lock_path = (
            f"{self.db_path}.lock"
            if self.thread_id == "main"
            else f"{self.db_path}.{self.thread_id}.lock"
        )
        self._lock_path = lock_path
        self._lock_file = open(lock_path, "w")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise RuntimeError(
                f"Database '{self.db_path}' (thread '{self.thread_id}') is already locked by another running process (lockfile: '{lock_path}')."
            )

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _init_db(self):
        conn = self._connect_db()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS archival_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        role TEXT, 
                        content TEXT, 
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        thread_id TEXT NOT NULL DEFAULT 'main',
                        tool_calls TEXT,
                        tool_call_id TEXT,
                        name TEXT
                    )
                """)
                self._ensure_column(conn, "archival_log", "thread_id", "TEXT NOT NULL DEFAULT 'main'")
                self._ensure_column(conn, "archival_log", "tool_calls", "TEXT")
                self._ensure_column(conn, "archival_log", "tool_call_id", "TEXT")
                self._ensure_column(conn, "archival_log", "name", "TEXT")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_archival_log_thread_id ON archival_log(thread_id, id)")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS core_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        summary TEXT,
                        thread_id TEXT NOT NULL DEFAULT 'main'
                    )
                """)
                self._ensure_column(conn, "core_summary", "thread_id", "TEXT NOT NULL DEFAULT 'main'")
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_core_summary_thread_id ON core_summary(thread_id)")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS core_summary_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        summary TEXT,
                        compacted_up_to_id INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        thread_id TEXT NOT NULL DEFAULT 'main'
                    )
                """)
                self._ensure_column(conn, "core_summary_history", "thread_id", "TEXT NOT NULL DEFAULT 'main'")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_settings (
                        key TEXT PRIMARY KEY, 
                        value TEXT
                    )
                """)
                if self.thread_id == "main":
                    conn.execute(
                        "INSERT OR IGNORE INTO core_summary (id, summary, thread_id) VALUES (1, 'No summary yet.', 'main')"
                    )
                else:
                    conn.execute(
                        "INSERT OR IGNORE INTO core_summary (summary, thread_id) VALUES ('No summary yet.', ?)",
                        (self.thread_id,),
                    )
                conn.execute(
                    "INSERT OR IGNORE INTO memory_settings (key, value) VALUES (?, '0')",
                    (self._settings_key("last_compacted_id"),),
                )

        finally:
            conn.close()

    def _get_core_summary(self) -> str:
        conn = self._connect_db()
        try:
            with conn:
                cursor = conn.execute(
                    "SELECT summary FROM core_summary WHERE thread_id = ?",
                    (self.thread_id,),
                )
                row = cursor.fetchone()
                return row[0] if row else "No summary yet."

        finally:
            conn.close()

    def _get_last_compacted_id(self) -> int:
        conn = self._connect_db()
        try:
            with conn:
                cursor = conn.execute(
                    "SELECT value FROM memory_settings WHERE key = ?",
                    (self._settings_key("last_compacted_id"),),
                )
                row = cursor.fetchone()
                return int(row[0]) if row else 0
        finally:
            conn.close()

    def _set_last_compacted_id(self, last_id: int):
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_settings (key, value) VALUES (?, ?)",
                    (self._settings_key("last_compacted_id"), str(last_id))
                )
        finally:
            conn.close()

    def _update_core_summary(self, new_summary: str, max_evicted_id: int = 0):
        if not new_summary or not new_summary.strip():
            raise ValueError("Refusing to overwrite core_summary with an empty summary.")

        # Hard limit to prevent LLM hallucinations from exploding the system prompt
        tokens = self.tokenizer.encode(new_summary)
        if len(tokens) > self.max_summary_tokens:
            print(f"[Memory Warning] Summary exceeded {self.max_summary_tokens} tokens. Truncating.")
            # Decode only the tokens up to the limit
            truncated_tokens = tokens[:self.max_summary_tokens]
            new_summary = self.tokenizer.decode(truncated_tokens) + "... [TRUNCATED]"
            
        for attempt in range(self.max_retries):
            conn = None
            try:
                conn = self._connect_db()
                with conn:
                    conn.execute(
                        "UPDATE core_summary SET summary = ? WHERE thread_id = ?",
                        (new_summary, self.thread_id),
                    )
                    conn.execute(
                        "INSERT INTO core_summary_history (summary, compacted_up_to_id, thread_id) VALUES (?, ?, ?)",
                        (new_summary, max_evicted_id, self.thread_id)
                    )
                    if max_evicted_id > 0:
                        conn.execute(
                            "INSERT OR REPLACE INTO memory_settings (key, value) VALUES (?, ?)",
                            (self._settings_key("last_compacted_id"), str(max_evicted_id))
                        )
                return
            except Exception as e:
                print(
                    f"[Memory ERROR] Core summary SQLite write attempt {attempt + 1}/{self.max_retries} failed: {e}",
                    flush=True,
                )
                if attempt + 1 < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    if delay > 0:
                        time.sleep(delay)
            finally:
                if conn is not None:
                    conn.close()

        print(
            f"[Memory FATAL] Failed to update core_summary in SQLite after {self.max_retries} attempts. Exiting process.",
            flush=True,
        )
        sys.exit(1)

    def get_system_prompt_addition(self) -> str:
        summary = self._get_core_summary()
        return (
            "--- CURRENT CORE SUMMARY ---\n"
            f"{summary}\n"
            "----------------------------\n"
            "Use this summary to remember past interactions and your current overarching goals."
        )

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Helper to roughly approximate the token count of a list of messages."""
        content = ""
        for msg in messages:
            content += f"Role: {msg.get('role', 'unknown')}\n"
            text = msg.get("content") or ""
            content += f"{text}\n"
            if "tool_calls" in msg and msg["tool_calls"]:
                # Approximation of tool calls JSON string
                content += str(msg["tool_calls"]) + "\n"
                
        return len(self.tokenizer.encode(content))

    async def on_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == "agent_started":
            agent = payload["agent"]
            self._restore_working_memory(agent)
            
        elif event_name == "message_added":
            msg = payload["message"]
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""

            tool_calls_json = None
            raw_tcs = msg.get("tool_calls")
            if raw_tcs:
                serialized_tcs = []
                for tc in raw_tcs:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        tc_entry = {
                            "id": str(tc.get("id") or ""),
                            "name": str(fn.get("name") or tc.get("name") or ""),
                            "arguments": str(fn.get("arguments") if "function" in tc else tc.get("arguments") or "{}"),
                        }
                        sig = tc.get("thought_signature") or tc.get("_thought_signature")
                        if sig:
                            tc_entry["thought_signature"] = str(sig)
                        serialized_tcs.append(tc_entry)
                    else:
                        fn = getattr(tc, "function", None)
                        tc_entry = {
                            "id": str(getattr(tc, "id", "") or ""),
                            "name": str(getattr(fn, "name", "") if fn else ""),
                            "arguments": str(getattr(fn, "arguments", "{}") if fn else "{}"),
                        }
                        sig = getattr(tc, "_thought_signature", None) or getattr(tc, "thought_signature", None)
                        if sig:
                            tc_entry["thought_signature"] = str(sig)
                        serialized_tcs.append(tc_entry)
                tool_calls_json = json.dumps(serialized_tcs)
                if not content:
                    content = "[Tool Calls Requested]"

            tool_call_id = str(msg.get("tool_call_id")) if msg.get("tool_call_id") else None
            tool_name = str(msg.get("name")) if msg.get("name") else None
                
            saved = False
            for attempt in range(self.max_retries):
                conn = None
                try:
                    conn = self._connect_db()
                    with conn:
                        cursor = conn.execute(
                            "INSERT INTO archival_log (role, content, thread_id, tool_calls, tool_call_id, name) VALUES (?, ?, ?, ?, ?, ?)", 
                            (role, str(content), self.thread_id, tool_calls_json, tool_call_id, tool_name)
                        )
                        msg["_db_id"] = cursor.lastrowid
                    saved = True
                    break
                except Exception as e:
                    print(
                        f"[Memory ERROR] Archival log write attempt {attempt + 1}/{self.max_retries} failed: {e}",
                        flush=True,
                    )
                    if attempt + 1 < self.max_retries:
                        delay = self.retry_base_delay * (2 ** attempt)
                        if delay > 0:
                            await asyncio.sleep(delay)
                finally:
                    if conn is not None:
                        conn.close()

            if not saved:
                print(
                    f"[Memory FATAL] Failed to write message to archival_log after {self.max_retries} attempts. Exiting process.",
                    flush=True,
                )
                sys.exit(1)
        elif event_name == "context_window_check":
            agent = payload["agent"]
            
            total_tokens = self._count_tokens(agent.messages)
            last_tokens = total_tokens
            
            while total_tokens > self.high_watermark and len(agent.messages) > 1:
                print(f"[Memory] Context window exceeded ({total_tokens} tokens > {self.high_watermark}). Compressing...")
                await self._compress_memory(agent)
                total_tokens = self._count_tokens(agent.messages)
                
                if total_tokens == last_tokens:
                    print("[Memory CRITICAL] Compression did not reduce token count. Breaking to prevent infinite loop.")
                    break
                last_tokens = total_tokens
                
            if total_tokens > self.high_watermark:
                print(f"[Memory WARNING] Context window is {total_tokens} tokens, exceeding {self.high_watermark}, but no further messages can be compressed.")

    @staticmethod
    def _row_to_restored_message(
        row_id: int,
        role: str,
        content: str,
        tool_calls_json: Any = None,
        tool_call_id: Any = None,
        tool_name: Any = None,
    ) -> Dict[str, Any]:
        if content == "[Tool Calls Requested]" and not tool_calls_json:
            return {}
        if not content and not tool_calls_json:
            return {}

        msg: Dict[str, Any] = {"role": role, "_db_id": row_id}
        if content and content != "[Tool Calls Requested]":
            msg["content"] = content
        elif role != "assistant":
            msg["content"] = content or ""

        if tool_calls_json and role == "assistant":
            try:
                parsed_tcs = json.loads(tool_calls_json)
                reconstructed = []
                for idx, tc in enumerate(parsed_tcs):
                    tc_obj = ChatCompletionMessageToolCall(
                        id=str(tc.get("id") or f"restored_tc_{row_id}_{idx}"),
                        type="function",
                        function=Function(
                            name=str(tc.get("name") or "tool"),
                            arguments=str(tc.get("arguments") or "{}"),
                        ),
                    )
                    if tc.get("thought_signature"):
                        tc_obj._thought_signature = str(tc["thought_signature"])
                    reconstructed.append(tc_obj)
                if reconstructed:
                    msg["tool_calls"] = reconstructed
            except Exception:
                if "content" not in msg:
                    return {}

        if role == "tool":
            if tool_call_id:
                msg["tool_call_id"] = str(tool_call_id)
            if tool_name:
                msg["name"] = str(tool_name)

        if "content" not in msg and "tool_calls" not in msg:
            return {}
        return msg

    def _restore_working_memory(self, agent: Any):
        """
        On startup, re-populates the agent's short-term context window (Tier 1 memory)
        from the archival log so it can seamlessly resume its previous thought process.
        """
        print("[Memory] Restoring agent's working memory from previous session...")
        
        last_compacted_id = self._get_last_compacted_id()
        
        conn = self._connect_db()
        try:
            with conn:
                if last_compacted_id == 0:
                    # Legacy fallback: fetch a chunk of recent messages for this thread
                    cursor = conn.execute(
                        "SELECT id, role, content, tool_calls, tool_call_id, name "
                        "FROM archival_log WHERE thread_id = ? ORDER BY id DESC LIMIT 500",
                        (self.thread_id,),
                    )
                    rows = cursor.fetchall()
                else:
                    # Modern restore: load everything for this thread since the last successful compaction
                    cursor = conn.execute(
                        "SELECT id, role, content, tool_calls, tool_call_id, name "
                        "FROM archival_log WHERE thread_id = ? AND id > ? ORDER BY id ASC",
                        (self.thread_id, last_compacted_id)
                    )
                    rows = cursor.fetchall()
        finally:
            conn.close()
            
        if not rows:
            return
            
        messages_to_restore = []
        # Current tokens is just the system prompt at this point
        current_tokens = self._count_tokens(agent.messages)
        
        if last_compacted_id == 0:
            # Legacy fallback: restore backwards until low_watermark
            for row_id, role, content, tc_json, tc_id, t_name in rows:
                msg = self._row_to_restored_message(row_id, role, content, tc_json, tc_id, t_name)
                if not msg:
                    continue
                msg_tokens = self._count_tokens([msg])
                
                if current_tokens + msg_tokens > self.low_watermark:
                    break
                    
                messages_to_restore.append(msg)
                current_tokens += msg_tokens
                
            messages_to_restore.reverse()
            
            # Now set the last_compacted_id so next time it works perfectly
            if messages_to_restore:
                first_restored_id = messages_to_restore[0]["_db_id"]
                self._set_last_compacted_id(first_restored_id - 1)
        else:
            for row_id, role, content, tc_json, tc_id, t_name in rows:
                msg = self._row_to_restored_message(row_id, role, content, tc_json, tc_id, t_name)
                if not msg:
                    continue
                messages_to_restore.append(msg)
                current_tokens += self._count_tokens([msg])
                
        agent.messages.extend(messages_to_restore)
        print(f"[Memory] Restored {len(messages_to_restore)} messages ({current_tokens} tokens).")
            
    async def _compress_memory(self, agent: Any):
        """Extracts older messages, asks the LLM to summarize them, and evicts them."""
        # Index 0 is System Prompt. 
        # We want to evict messages starting from index 1 until we drop below low_watermark.
        current_tokens = self._count_tokens(agent.messages)
        target_tokens_to_evict = current_tokens - self.low_watermark
        
        if target_tokens_to_evict <= 0:
            return
            
        messages_to_compress = []
        accumulated_tokens = 0
        num_to_evict = 0
        max_evicted_id = 0
        
        for i, msg in enumerate(agent.messages[1:]):
            msg_tokens = self._count_tokens([msg])
            messages_to_compress.append(msg)
            accumulated_tokens += msg_tokens
            num_to_evict += 1
            
            if "_db_id" in msg:
                max_evicted_id = max(max_evicted_id, msg["_db_id"])
            
            # Stop if we have gathered enough tokens to hit our low watermark target
            if accumulated_tokens >= target_tokens_to_evict:
                break
                
            # Stop if we hit the safe chunk limit for the summarizer
            if accumulated_tokens >= self.max_summarize_tokens:
                break

        # Align eviction boundary so we don't slice mid-tool-turn or leave an assistant tool_call
        # at the very top of working memory when a subsequent user turn exists.
        remaining_slice = agent.messages[1 + num_to_evict :]
        if any(m.get("role") == "user" for m in remaining_slice):
            while (1 + num_to_evict) < len(agent.messages):
                next_msg = agent.messages[1 + num_to_evict]
                if next_msg.get("role") == "user":
                    break
                msg_tokens = self._count_tokens([next_msg])
                messages_to_compress.append(next_msg)
                accumulated_tokens += msg_tokens
                num_to_evict += 1
                if "_db_id" in next_msg:
                    max_evicted_id = max(max_evicted_id, next_msg["_db_id"])
                
        if num_to_evict == 0:
            return
            
        # Build a transcript
        transcript = ""
        for msg in messages_to_compress:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not content and msg.get("tool_calls"):
                content = "Called tools."
                
            # Strip transient subconscious recall blocks from transcript
            if role == "user" and content.startswith("[Subconscious Recall Triggered"):
                parts = content.split("--- End Recall ---\n\n", 1)
                if len(parts) == 2:
                    content = parts[1]
                    
            transcript += f"[{role.upper()}]: {content}\n"

        current_summary = self._get_core_summary()
        
        prompt = (
            "You are a background memory compression process for an AI agent.\n"
            "Here is the agent's current Core Summary:\n"
            f"<current_summary>\n{current_summary}\n</current_summary>\n\n"
            "Here is a transcript of the oldest recent interactions that are being evicted from the active context window:\n"
            f"<transcript>\n{transcript}\n</transcript>\n\n"
            "Please rewrite and update the Core Summary to incorporate the new facts, lore, or goal updates from the transcript.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Aggressively condense or completely discard older, stale information (e.g., books you have already finished, resolved quests, or old context) to make room for current priorities.\n"
            "- Prioritize recent events, current state, and active goals over past history.\n"
            "- Remember that evicted facts are safely archived in Vector Memory, so it is safe to drop them from this active summary.\n"
            "- Keep the summary highly condensed and written from the perspective of the agent (e.g., 'I am currently reading...').\n"
            "Return ONLY the raw text of the new summary. Do not include introductory text."
        )

        new_summary = None
        agent_name = getattr(agent, "name", None) or self.config.get("agent", {}).get("name", "Thunk")
        thread_id = getattr(agent, "thread_id", None) or self.thread_id or "main"
        exec_loc = getattr(agent, "exec_location", None) or (
            "cloud"
            if self.config.get("llm", {}).get("provider", "").strip().lower() == "gemini"
            else infer_exec_location(self.config.get("llm", {}).get("api_url"))
        )

        with MetricSpan(
            operation="llm_call",
            agent=agent_name,
            thread=thread_id,
            component="tiered_memory",
            type="memory_compaction",
            model=getattr(agent, "model", None),
            exec_location=exec_loc,
        ) as span:
            span.inc("evicted_messages", num_to_evict)
            span.inc("evicted_tokens", accumulated_tokens)
            span.inc("request_tokens", count_tokens(prompt, tokenizer=self.tokenizer))
            span.inc("request_bytes", len(prompt.encode("utf-8")))

            for attempt in range(self.max_retries):
                try:
                    with span.subtimer("llm_call_ms"):
                        response_stream = await agent.client.chat.completions.create(
                            model=agent.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=self.max_summary_tokens,
                            extra_headers={"X-Agent-ID": f"{agent_name}-{thread_id}-memory"},
                            stream=True,
                        )
                        message = await agent._accumulate_stream(response_stream)
                    candidate_summary = (message.content or "").strip()
                    if not candidate_summary:
                        raise ValueError("Refusing to overwrite core_summary with an empty summary.")
                    new_summary = candidate_summary
                    break
                except Exception as e:
                    print(
                        f"[Memory ERROR] Compression attempt {attempt + 1}/{self.max_retries} failed: {e}",
                        flush=True,
                    )
                    if attempt + 1 < self.max_retries:
                        span.inc("retries", 1)
                        delay = self.retry_base_delay * (2 ** attempt)
                        if delay > 0:
                            await asyncio.sleep(delay)

            if not new_summary:
                span.fail()
                span.finish()
                print(
                    f"[Memory FATAL] All {self.max_retries} compression attempts failed. Refusing to evict memory without summary. Exiting process.",
                    flush=True,
                )
                sys.exit(1)

            span.inc("response_tokens", count_tokens(new_summary, tokenizer=self.tokenizer))
            span.inc("response_bytes", len(new_summary.encode("utf-8")))

            # --- MEMORY CONSOLIDATION (Subconscious Integration) ---
            for comp in agent.components:
                if comp.__class__.__name__ == "VectorMemoryComponent":
                    try:
                        consolidation = f"Consolidated Memory Block:\nSummary: {new_summary}\nRaw Events:\n{transcript}"
                        sig_params = inspect.signature(comp.save_semantic_memory).parameters
                        supports_kwargs = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD or p.name == "metadata"
                            for p in sig_params.values()
                        )
                        with span.subtimer("embedding_ms"):
                            if supports_kwargs:
                                result = comp.save_semantic_memory(
                                    consolidation,
                                    metadata={"thread_id": self.thread_id},
                                )
                            else:
                                result = comp.save_semantic_memory(consolidation)
                        if isinstance(result, str) and result.startswith("Error"):
                            raise RuntimeError(result)
                        print("[Memory] Sent consolidated block to long-term Vector Memory.")
                    except BaseException as e:
                        span.fail()
                        span.finish()
                        print(
                            f"[Memory FATAL] Failed to consolidate to Vector Memory: {e}. Exiting process.",
                            flush=True,
                        )
                        sys.exit(1)

            # Atomically save the new summary, append to history, and advance last_compacted_id
            with span.subtimer("db_save_ms"):
                self._update_core_summary(new_summary, max_evicted_id=max_evicted_id)
            span.inc("db_save", 1)
            span.inc("db_save_records", 1)
            print(f"[Memory] New Core Summary Generated: {new_summary[:50]}...")

            # Evict compacted messages only after Vector Memory and SQLite commits have succeeded
            del agent.messages[1 : 1 + num_to_evict]
            agent.rebuild_system_prompt()
