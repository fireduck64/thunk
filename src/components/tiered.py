import sqlite3
import json
from typing import List, Dict, Callable, Any
from transformers import AutoTokenizer
from src.components.base import BaseComponent
from src.utils.config import load_config

class TieredMemoryComponent(BaseComponent):
    """
    Manages the agent's memory tiers:
    1. Archival Memory (Logs all messages to SQLite)
    2. Core Summary (Compresses old messages into a sliding summary based on token count)
    """
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.config = load_config()
        
        tokenizer_model = self.config.get("tokenizer", {}).get("model_name", "unsloth/gemma-7b")
        self.high_watermark = self.config.get("memory", {}).get("high_watermark", 16000)
        self.low_watermark = self.config.get("memory", {}).get("low_watermark", 8000)
        self.max_summarize_tokens = self.config.get("memory", {}).get("max_summarize_tokens", 6000)
        self.max_summary_tokens = self.config.get("memory", {}).get("max_summary_tokens", 4000)
        
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
            
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
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

        finally:
            conn.close()

    def _get_core_summary(self) -> str:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute("SELECT summary FROM core_summary WHERE id = 1")
                return cursor.fetchone()[0]

        finally:
            conn.close()

    def _update_core_summary(self, new_summary: str):
        # Hard limit to prevent LLM hallucinations from exploding the system prompt
        tokens = self.tokenizer.encode(new_summary)
        if len(tokens) > self.max_summary_tokens:
            print(f"[Memory Warning] Summary exceeded {self.max_summary_tokens} tokens. Truncating.")
            # Decode only the tokens up to the limit
            truncated_tokens = tokens[:self.max_summary_tokens]
            new_summary = self.tokenizer.decode(truncated_tokens) + "... [TRUNCATED]"
            
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("UPDATE core_summary SET summary = ? WHERE id = 1", (new_summary,))

        finally:
            conn.close()

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
            
            # Tools sometimes return complex objects or we want to save the raw JSON
            content = msg.get("content", "")
            if not content and "tool_calls" in msg:
                # Need to handle pydantic/OpenAI objects safely for logging
                content = f"[Tool Calls Requested]"
                
            conn = sqlite3.connect(self.db_path)
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO archival_log (role, content) VALUES (?, ?)", 
                        (role, str(content))
                    )

            finally:
                conn.close()
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

    def _restore_working_memory(self, agent: Any):
        """
        On startup, re-populates the agent's short-term context window (Tier 1 memory)
        from the archival log so it can seamlessly resume its previous thought process.
        """
        print("[Memory] Restoring agent's working memory from previous session...")
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                # We fetch a large chunk of recent messages
                cursor = conn.execute(
                    "SELECT role, content FROM archival_log ORDER BY id DESC LIMIT 500"
                )
                rows = cursor.fetchall()

        finally:
            conn.close()
            
        if not rows:
            return
            
        messages_to_restore = []
        # Current tokens is just the system prompt at this point
        current_tokens = self._count_tokens(agent.messages)
        
        for role, content in rows:
            # Skip invalid/unserializable legacy tool calls
            if content == "[Tool Calls Requested]" or not content:
                continue
                
            msg = {"role": role, "content": content}
            msg_tokens = self._count_tokens([msg])
            
            # Stop restoring if it would exceed our low watermark
            if current_tokens + msg_tokens > self.low_watermark:
                break
                
            messages_to_restore.append(msg)
            current_tokens += msg_tokens
            
        # Put them back in chronological order
        messages_to_restore.reverse() 
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
        
        for i, msg in enumerate(agent.messages[1:]):
            msg_tokens = self._count_tokens([msg])
            messages_to_compress.append(msg)
            accumulated_tokens += msg_tokens
            num_to_evict += 1
            
            # Stop if we have gathered enough tokens to hit our low watermark target
            if accumulated_tokens >= target_tokens_to_evict:
                break
                
            # Stop if we hit the safe chunk limit for the summarizer
            if accumulated_tokens >= self.max_summarize_tokens:
                break
                
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

        # IMPORTANT FIX: We aggressively evict the messages from the agent's context 
        # BEFORE making the LLM call. This guarantees the context window shrinks 
        # even if the summarization LLM call times out or crashes.
        del agent.messages[1 : 1 + num_to_evict]

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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # We make a direct LLM call using the agent's client with a stream
                response_stream = await agent.client.chat.completions.create(
                    model=agent.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )
                message = await agent._accumulate_stream(response_stream)
                new_summary = message.content.strip()
                
                # Save the new summary
                self._update_core_summary(new_summary)
                print(f"[Memory] New Core Summary Generated: {new_summary[:50]}...")
                
                # --- MEMORY CONSOLIDATION (Subconscious Integration) ---
                for comp in agent.components:
                    if comp.__class__.__name__ == "VectorMemoryComponent":
                        try:
                            # Save the transcript + summary as a consolidated block
                            consolidation = f"Consolidated Memory Block:\nSummary: {new_summary}\nRaw Events:\n{transcript}"
                            comp.save_semantic_memory(consolidation)
                            print("[Memory] Sent consolidated block to long-term Vector Memory.")
                        except Exception as e:
                            print(f"[Memory] Failed to consolidate to Vector Memory: {e}")
                
                # Ask the agent to rebuild its system prompt so the new summary is injected immediately
                agent.rebuild_system_prompt()
                return # Success, exit the compression function
                
            except Exception as e:
                print(f"[Memory Error] Compression attempt {attempt + 1}/{max_retries} failed: {e}")
                import asyncio
                await asyncio.sleep(5) # Wait before retrying
                
        # --- Emergency Eviction (Circuit Breaker) ---
        print("[Memory CRITICAL] All compression retries failed. The messages were evicted, but the summary was lost.")
        agent.rebuild_system_prompt()
