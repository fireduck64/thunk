from typing import Any, Callable, Dict, List

class BaseComponent:
    """
    The foundational class for all modules attached to the Agent Core.
    """
    agent: Any = None # Will be populated by AgentCore during registration
    
    def get_system_prompt_addition(self) -> str:
        """Return text to be appended to the agent's main system prompt."""
        return ""

    def get_tools(self) -> List[Callable]:
        """Return a list of Python functions to be exposed as tools to the LLM."""
        return []

    async def on_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """
        Hook into the event bus. 
        Examples: 'agent_start', 'before_llm_call', 'tool_executed'
        """
        pass
