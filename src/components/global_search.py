import asyncio
from typing import List, Callable, Dict
from src.components.base import BaseComponent

class GlobalSearchComponent(BaseComponent):
    """
    A 'Meta-Component' that exposes a single `global_search` tool to the LLM.
    When executed, it fans out the search query to all attached components 
    that implement the `execute_search(query)` method.
    """
    def __init__(self):
        # We will populate this when the agent starts
        self.searchable_components = []

    async def on_event(self, event_name: str, payload: dict) -> None:
        if event_name == "agent_started":
            # Discover which components support searching
            self.searchable_components = [
                comp for comp in self.agent.components 
                if hasattr(comp, "execute_search") and callable(getattr(comp, "execute_search"))
            ]
            print(f"[GlobalSearch] Registered {len(self.searchable_components)} searchable components.")

    def get_system_prompt_addition(self) -> str:
        return (
            "You have access to a `global_search` tool. This will simultaneously search your "
            "internal semantic memory, your structured notes, and your massive external knowledge base. "
            "Always use this when you need to recall facts, lore, or past thoughts."
        )

    def get_tools(self) -> List[Callable]:
        return [self.global_search]

    async def global_search(self, query: str) -> str:
        """
        Simultaneously searches all available databases (Knowledge Base, Semantic Memory, Structured Notes).
        Returns an aggregated list of results.
        """
        print(f"[GlobalSearch] Fanning out query: '{query}'")
        
        if not self.searchable_components:
            return "Error: No searchable databases are currently attached to the system."

        # Execute all searches concurrently
        tasks = []
        for comp in self.searchable_components:
            # Check if the component's execute_search is async
            if asyncio.iscoroutinefunction(comp.execute_search):
                tasks.append(comp.execute_search(query))
            else:
                # If it's sync, wrap it in a thread so it doesn't block
                loop = asyncio.get_event_loop()
                tasks.append(loop.run_in_executor(None, comp.execute_search, query))
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate the results
        aggregated_output = ""
        for comp, result in zip(self.searchable_components, results):
            comp_name = comp.__class__.__name__
            if isinstance(result, Exception):
                aggregated_output += f"\n=== Results from {comp_name} ===\nError: {str(result)}\n"
            elif not result:
                # If a component explicitly returns empty/None, skip it to save tokens
                continue
            else:
                aggregated_output += f"\n=== Results from {comp_name} ===\n{result}\n"
                
        if not aggregated_output.strip():
            return f"No results found in any database for query: '{query}'"
            
        return aggregated_output.strip()
