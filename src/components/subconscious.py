import asyncio
from typing import Dict, Any
import httpx
from openai import AsyncOpenAI
from src.components.base import BaseComponent
from src.utils.config import load_config

class SubconsciousRecallComponent(BaseComponent):
    """
    Implements associative memory.
    Listens to incoming user messages or new content.
    Queries INTERNAL memory (top-N from VectorMemory & Notes, excluding external KnowledgeBases),
    synthesizes and filters the candidates using a lightweight LLM (e.g. gemma4:e2b),
    and quietly prepends relevant associative recall to the LLM's context window.
    """
    def __init__(self):
        self.config = load_config()
        sub_cfg = self.config.get("subconscious", {})
        llm_cfg = self.config.get("llm", {})

        self.api_url = sub_cfg.get("api_url") or llm_cfg.get("api_url")
        self.api_key = sub_cfg.get("api_key") or llm_cfg.get("api_key")
        self.model_name = sub_cfg.get("model_name", "gemma4:e2b")
        self.top_k = int(sub_cfg.get("top_k", 8))

        self.client = AsyncOpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(None)
        )
        self.internal_memory_components = []

    async def _synthesize_memories(self, incoming_message: str, raw_memories: str) -> str:
        """Uses the lightweight subconscious model to filter and synthesize top-N candidate memories."""
        prompt = (
            "You are the associative subconscious memory filter for an AI agent.\n"
            "Given the incoming message and a set of candidate memories retrieved from the vector database, "
            "distill and synthesize ONLY the facts, themes, and specific details from the candidate memories "
            "that are relevant to the incoming message.\n"
            "If NONE of the candidate memories are genuinely relevant to the incoming message, reply with the single word: NONE\n"
            "Otherwise, write a compact, high-density associative recall summary (3-6 bullet points or a tight paragraph) "
            "connecting the most relevant details across the candidate memories. Do not invent facts outside the memories.\n\n"
            f"<incoming_message>\n{incoming_message}\n</incoming_message>\n\n"
            f"<candidate_memories>\n{raw_memories}\n</candidate_memories>"
        )

        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            stream=True
        )

        content = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content

        synthesized = content.strip()
        if synthesized.upper() == "NONE" or synthesized.upper().startswith("NONE\n"):
            return ""
        return synthesized

    async def on_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == "agent_started":
            # Find the VectorMemory and StructuredNotes components
            for comp in self.agent.components:
                # We specifically exclude KnowledgeBaseComponent to prevent automatic library lookups
                if comp.__class__.__name__ in ["VectorMemoryComponent", "StructuredNotesComponent"]:
                    if hasattr(comp, "execute_search"):
                        self.internal_memory_components.append(comp)
            print(f"[Subconscious] Wired up to {len(self.internal_memory_components)} internal memory stores (model: {self.model_name}, top_k: {self.top_k}).")
            
        elif event_name == "before_user_message_added":
            # A user message just arrived! Let's do an associative search.
            original_text = payload.get("content", "")
            if not original_text or not self.internal_memory_components:
                return

            # Skip routine clock ticks so they don't trigger unnecessary vector + LLM synthesis calls
            if original_text.startswith("[System Clock]"):
                return

            print(f"[Subconscious] Associating top-{self.top_k} concepts for incoming message...")
            
            tasks = []
            loop = asyncio.get_event_loop()
            for comp in self.internal_memory_components:
                if hasattr(comp, "search_semantic_memory"):
                    tasks.append(loop.run_in_executor(None, comp.search_semantic_memory, original_text, self.top_k))
                elif asyncio.iscoroutinefunction(comp.execute_search):
                    tasks.append(comp.execute_search(original_text))
                else:
                    tasks.append(loop.run_in_executor(None, comp.execute_search, original_text))
                    
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            raw_candidates = []
            for result in results:
                if isinstance(result, Exception) or not result:
                    continue
                text = result.strip()
                if text and not text.startswith("No relevant memories found") and not text.startswith("Error "):
                    raw_candidates.append(text)

            if not raw_candidates:
                return

            combined_raw = "\n\n".join(raw_candidates)
            try:
                print(f"[Subconscious] Synthesizing candidate memories with {self.model_name}...")
                subconscious_thoughts = await self._synthesize_memories(original_text, combined_raw)
            except Exception as e:
                print(f"[Subconscious Warning] Synthesis failed ({e}); falling back to raw recall.")
                subconscious_thoughts = combined_raw

            # If we found relevant internal memories, we prepend them to the message.
            # This injected text goes into Working Memory (LLM context window)
            # without getting permanently written to the Archival Log.
            if subconscious_thoughts.strip():
                print("[Subconscious] Found relevant associative memories. Injecting.")
                new_content = (
                    f"[Subconscious Recall Triggered by below message:]\n"
                    f"{subconscious_thoughts.strip()}\n"
                    f"--- End Recall ---\n\n"
                    f"{original_text}"
                )
                payload["content"] = new_content
