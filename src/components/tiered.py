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
        if event_name == "agent_started":
            agent = payload["agent"]
            self._restore_working_memory(agent)
            
        elif event_name == "message_added":
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

    def _restore_working_memory(self, agent: Any):
        """
        On startup, re-populates the agent's short-term context window (Tier 1 memory)
        from the archival log so it can seamlessly resume its previous thought process.
        """
        print("[Memory] Restoring agent's working memory from previous session...")
        with sqlite3.connect(self.db_path) as conn:
            # We fetch the last N messages, where N is max_messages
            # We sort descending to get the newest, then reverse them to chronological order
            cursor = conn.execute(
                "SELECT role, content FROM archival_log ORDER BY id DESC LIMIT ?",
                (self.max_messages,)
            )
            rows = cursor.fetchall()
            
        if not rows:
            return
            
        rows.reverse() # Put them back in chronological order
        
        for role, content in rows:
            # The archival log saves tool calls as a string "[Tool Calls Requested]" for humans,
            # which breaks OpenAI's strict tool_call format if we just shove it back into context.
            # For now, if it was a tool call that we didn't serialize perfectly, we skip it or 
            # insert it as a generic assistant message to provide context without breaking the API.
            if content == "[Tool Calls Requested]":
                # We skip injecting broken tool calls into the strict OpenAI context window on restore
                continue
                
            # If the database stored a raw empty string, and it's not a tool call (because we skipped those),
            # OpenAI will crash if we try to send {"role": "assistant", "content": ""}. 
            # We must normalize empty content to a string with at least one space or skip it.
            if not content:
                continue
                
            agent.messages.append({
                "role": role,
                "content": content
            })
            
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
                
            # If this message was a user message that had a [Subconscious Recall] block prepended to it,
            # we want to strip that block out of the transcript before we summarize it!
            # The subconscious recall is just transient context, not a permanent event that happened.
            if role == "user" and content.startswith("[Subconscious Recall Triggered"):
                parts = content.split("--- End Recall ---\n\n", 1)
                if len(parts) == 2:
                    content = parts[1] # Keep only what the Operator actually said
                    
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

        max_retries = 3
        for attempt in range(max_retries):
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
                
                # --- MEMORY CONSOLIDATION (Subconscious Integration) ---
                # As discussed in MEMORY_REMODEL, we want synthesized summaries to automatically
                # flow into the long-term semantic vector database, much like human sleep consolidation.
                for comp in agent.components:
                    if comp.__class__.__name__ == "VectorMemoryComponent":
                        try:
                            # Save the transcript + summary as a consolidated block
                            consolidation = f"Consolidated Memory Block:\nSummary: {new_summary}\nRaw Events:\n{transcript}"
                            comp.save_semantic_memory(consolidation)
                            print("[Memory] Sent consolidated block to long-term Vector Memory.")
                        except Exception as e:
                            print(f"[Memory] Failed to consolidate to Vector Memory: {e}")
                
                # Safely remove the compressed messages from the agent's working memory
                del agent.messages[1:self.summarize_chunk + 1]
                
                # Ask the agent to rebuild its system prompt so the new summary is injected immediately
                agent.rebuild_system_prompt()
                return # Success, exit the compression function
                
            except Exception as e:
                print(f"[Memory Error] Compression attempt {attempt + 1}/{max_retries} failed: {e}")
                import asyncio
                await asyncio.sleep(5) # Wait before retrying
                
        # --- Emergency Eviction (Circuit Breaker) ---
        # If we failed all 3 times (e.g. Nginx is completely dead, or context is so large 
        # that even the compression prompt exceeds OpenAI limits), we must forcefully 
        # evict the oldest messages without summarizing them. Otherwise, the agent's 
        # context window will remain over the threshold, and it will be deadlocked forever.
        print("[Memory CRITICAL] All compression retries failed. Forcefully evicting oldest messages to prevent deadlock.")
        del agent.messages[1:self.summarize_chunk + 1]
        agent.rebuild_system_prompt()
