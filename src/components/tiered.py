import sqlite3
import json
from typing import List, Dict, Callable, Any
from src.components.base import BaseComponent

class TieredMemoryComponent(BaseComponent):
    """
    Manages the agent's memory tiers:
    1. Archival Memory (Logs all messages to SQLite)
    2. Core Summary (Compresses old messages into a sliding summary)
    """
    def __init__(self, db_path: str = "memory.db", max_messages: int = 10, summarize_chunk: int = 4):
        self.db_path = db_path
        self.max_messages = max_messages
        self.summarize_chunk = summarize_chunk
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS archival_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    role TEXT, 
                    content TEXT, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS core_summary (
                    id INTEGER PRIMARY KEY, 
                    summary TEXT
                )
            """)
            # Ensure there is always a row 1 for the summary
            conn.execute("INSERT OR IGNORE INTO core_summary (id, summary) VALUES (1, 'No summary yet.')")

    def _get_core_summary(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT summary FROM core_summary WHERE id = 1")
            return cursor.fetchone()[0]

    def _update_core_summary(self, new_summary: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE core_summary SET summary = ? WHERE id = 1", (new_summary,))

    def get_system_prompt_addition(self) -> str:
        summary = self._get_core_summary()
        return (
            "--- CURRENT CORE SUMMARY ---\n"
            f"{summary}\n"
            "----------------------------\n"
            "Use this summary to remember past interactions and your current overarching goals."
        )

    async def on_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == "message_added":
            msg = payload["message"]
            role = msg.get("role", "unknown")
            
            # Tools sometimes return complex objects or we want to save the raw JSON
            content = msg.get("content", "")
            if not content and "tool_calls" in msg:
                # Need to handle pydantic/OpenAI objects safely for logging
                content = f"[Tool Calls Requested]"
                
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO archival_log (role, content) VALUES (?, ?)", 
                    (role, str(content))
                )

        elif event_name == "context_window_check":
            agent = payload["agent"]
            
            # Count how many messages we have (excluding the system prompt at index 0)
            if len(agent.messages) > self.max_messages + 1:
                print(f"[Memory] Context window exceeded ({len(agent.messages)} msgs). Compressing...")
                await self._compress_memory(agent)

    async def _compress_memory(self, agent: Any):
        """Extracts the oldest M messages, asks the LLM to summarize them, and evicts them."""
        # Index 0 is System Prompt. We want to pop index 1 through summarize_chunk
        messages_to_compress = agent.messages[1:self.summarize_chunk + 1]
        
        # Build a transcript
        transcript = ""
        for msg in messages_to_compress:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not content and msg.get("tool_calls"):
                content = "Called tools."
            transcript += f"[{role.upper()}]: {content}\n"

        current_summary = self._get_core_summary()
        
        prompt = (
            "You are a background memory compression process for an AI agent.\n"
            "Here is the agent's current Core Summary:\n"
            f"<current_summary>\n{current_summary}\n</current_summary>\n\n"
            "Here is a transcript of the oldest recent interactions that are being evicted from the active context window:\n"
            f"<transcript>\n{transcript}\n</transcript>\n\n"
            "Please update the Core Summary to incorporate any important new facts, user preferences, or goal updates from the transcript. "
            "Keep the summary concise and written from the perspective of the agent (e.g., 'The user asked me to...'). "
            "Return ONLY the raw text of the new summary. Do not include introductory text."
        )

        try:
            # We make a direct LLM call using the agent's client
            response = await agent.client.chat.completions.create(
                model=agent.model,
                messages=[{"role": "user", "content": prompt}]
            )
            new_summary = response.choices[0].message.content.strip()
            
            # Save the new summary
            self._update_core_summary(new_summary)
            print(f"[Memory] New Core Summary Generated: {new_summary[:50]}...")
            
            # Safely remove the compressed messages from the agent's working memory
            del agent.messages[1:self.summarize_chunk + 1]
            
            # Ask the agent to rebuild its system prompt so the new summary is injected immediately
            agent.rebuild_system_prompt()
            
        except Exception as e:
            print(f"[Memory Error] Compression failed: {e}")
