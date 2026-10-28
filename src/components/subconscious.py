import asyncio
from typing import Dict, Any
import httpx
from openai import AsyncOpenAI
from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.metrics import MetricSpan, count_tokens, extract_chunk_usage, infer_exec_location

class SubconsciousRecallComponent(BaseComponent):
    """
    Implements associative memory.
    Listens to incoming user messages or new content.
    Queries INTERNAL memory (top-N from VectorMemory & Notes, excluding external KnowledgeBases),
    synthesizes and filters the candidates using a lightweight LLM (e.g. gemma4:e2b),
    and quietly prepends relevant associative recall to the LLM's context window.
    """
    def __init__(self, thread_id: str = "main"):
        self.config = load_config()
        self.agent_name = self.config.get("agent", {}).get("name", "Thunk")
        self.thread_id = (thread_id or "main").strip() or "main"
        sub_cfg = self.config.get("subconscious", {})
        llm_cfg = self.config.get("llm", {})

        self.api_url = sub_cfg.get("api_url") or llm_cfg.get("api_url")
        self.api_key = sub_cfg.get("api_key") or llm_cfg.get("api_key") or "ollama"
        self.model_name = sub_cfg.get("model_name", "gemma4:e2b")
        self.top_k = int(sub_cfg.get("top_k", 8))
        self.max_output_tokens = int(sub_cfg.get("max_output_tokens", 2048))

        self.client = AsyncOpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
            default_headers={"X-Agent-ID": f"{self.agent_name}-{self.thread_id}-subconscious"},
            timeout=httpx.Timeout(None)
        )
        self.internal_memory_components = []

    async def _synthesize_memories(self, incoming_message: str, raw_memories: str) -> str:
        """Uses the lightweight subconscious model to filter and synthesize top-N candidate memories."""
        prompt = (
            "You are the high-fidelity distillation bridge for an AI agent's subconscious memory.\n"
            "Given the incoming message and a set of candidate memories, extract the raw essence, "
            "idiosyncrasies, and specific facts relevant to the message. Do NOT synthesize or smooth the data into a coherent narrative.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. PRESERVE THE EDGES: Keep specific, bizarre, or grotesque details exactly as they appear. Do not normalize idiosyncratic language into generalities.\n"
            "2. EMBRACE DISSONANCE: If candidate memories contain conflicting information or contradictions, do NOT resolve them. List them as competing data points. Friction is a signal.\n"
            "3. ANTI-SMOOTHING: Avoid narrative fillers, corporate phrasing, or 'averaging' the tone. No phrases like 'it appears that,' 'overall,' or 'the party encountered.'\n"
            "4. ZERO INVENTION: Use only the provided memories. If no memories are genuinely relevant, reply with the single word: NONE\n\n"
            f"<incoming_message>\n{incoming_message}\n</incoming_message>\n\n"
            f"<candidate_memories>\n{raw_memories}\n</candidate_memories>\n\n"
            "OUTPUT: A high-contrast list of 3-6 raw, distilled points of recall."
        )

        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=self.max_output_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        content = ""
        self.last_usage = None
        async for chunk in stream:
            parsed_usage = extract_chunk_usage(chunk)
            if parsed_usage is not None:
                self.last_usage = parsed_usage
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

            agent_obj = getattr(self, "agent", None)
            agent_name = getattr(agent_obj, "name", None) or self.agent_name
            thread_id = getattr(agent_obj, "thread_id", None) or self.thread_id or "main"
            exec_loc = infer_exec_location(self.api_url)

            with MetricSpan(
                operation="llm_call",
                agent=agent_name,
                thread=thread_id,
                component="subconscious",
                type="subconscious_recall",
                model=self.model_name,
                exec_location=exec_loc,
            ) as span:
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
                        
                with span.subtimer("db_lookup_ms"):
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                span.inc("db_lookup", len(tasks))
                
                raw_candidates = []
                for result in results:
                    if isinstance(result, Exception) or not result:
                        continue
                    text = result.strip()
                    if text and not text.startswith("No relevant memories found") and not text.startswith("Error "):
                        raw_candidates.append(text)

                span.inc("db_lookup_records", len(raw_candidates))
                if not raw_candidates:
                    return

                combined_raw = "\n\n".join(raw_candidates)
                req_str = f"{original_text}\n{combined_raw}"
                span.inc("request_tokens", count_tokens(req_str, agent=agent_obj))
                span.inc("request_bytes", len(req_str.encode("utf-8")))

                try:
                    print(f"[Subconscious] Synthesizing candidate memories with {self.model_name}...")
                    with span.subtimer("llm_call_ms"):
                        subconscious_thoughts = await self._synthesize_memories(original_text, combined_raw)
                except Exception as e:
                    span.fail()
                    print(f"[Subconscious Warning] Synthesis failed ({e}); falling back to raw recall.")
                    subconscious_thoughts = combined_raw

                usage = getattr(self, "last_usage", None) or {}
                if usage.get("prompt_tokens"):
                    span.set_counter("request_tokens", int(usage["prompt_tokens"]))
                if usage.get("cached_tokens") is not None:
                    span.set_counter("cached_tokens", int(usage["cached_tokens"]))
                if usage.get("thinking_tokens") is not None:
                    span.set_counter("thinking_tokens", int(usage["thinking_tokens"]))
                span.inc(
                    "response_tokens",
                    usage.get("completion_tokens") or count_tokens(subconscious_thoughts, agent=agent_obj),
                )
                span.inc("response_bytes", len((subconscious_thoughts or "").encode("utf-8")))

                if subconscious_thoughts.strip():
                    print("[Subconscious] Found relevant associative memories. Injecting.")
                    new_content = (
                        f"[Subconscious Recall Triggered by below message:]\n"
                        f"{subconscious_thoughts.strip()}\n"
                        f"--- End Recall ---\n\n"
                        f"{original_text}"
                    )
                    payload["content"] = new_content
