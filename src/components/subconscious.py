import asyncio
from typing import Dict, Any
from src.components.base import BaseComponent

class SubconsciousRecallComponent(BaseComponent):
    """
    Implements associative memory.
    Listens to incoming user messages or new content.
    Automatically queries INTERNAL memory (VectorMemory & Notes, but NOT external KnowledgeBases)
    and quietly prepends relevant thoughts to the LLM's context window.
    """
    def __init__(self):
        self.internal_memory_components = []

    async def on_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == "agent_started":
            # Find the VectorMemory and StructuredNotes components
            for comp in self.agent.components:
                # We specifically exclude KnowledgeBaseComponent to prevent automatic library lookups
                if comp.__class__.__name__ in ["VectorMemoryComponent", "StructuredNotesComponent"]:
                    if hasattr(comp, "execute_search"):
                        self.internal_memory_components.append(comp)
            print(f"[Subconscious] Wired up to {len(self.internal_memory_components)} internal memory stores.")
            
        elif event_name == "before_user_message_added":
            # A user message just arrived! Let's do an associative search.
            original_text = payload.get("content", "")
            if not original_text or not self.internal_memory_components:
                return

            print(f"[Subconscious] Associating concepts for incoming message...")
            
            tasks = []
            for comp in self.internal_memory_components:
                if asyncio.iscoroutinefunction(comp.execute_search):
                    tasks.append(comp.execute_search(original_text))
                else:
                    loop = asyncio.get_event_loop()
                    tasks.append(loop.run_in_executor(None, comp.execute_search, original_text))
                    
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            subconscious_thoughts = ""
            for result in results:
                if isinstance(result, Exception):
                    continue
                if result:
                    subconscious_thoughts += f"{result}\n"

            # If we found relevant internal memories, we prepend them to the message.
            # Because of how we updated AgentCore, this injected text goes into the Working Memory (LLM context window)
            # but does NOT get permanently written to the Archival Log!
            if subconscious_thoughts.strip():
                print("[Subconscious] Found relevant associative memories. Injecting.")
                new_content = (
                    f"[Subconscious Recall Triggered by below message:]\n"
                    f"{subconscious_thoughts.strip()}\n"
                    f"--- End Recall ---\n\n"
                    f"Operator says: {original_text}"
                )
                payload["content"] = new_content
