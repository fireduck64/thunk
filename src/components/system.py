import asyncio
from typing import List, Callable, Dict, Any
from src.components.base import BaseComponent

class SystemComponent(BaseComponent):
    """
    Provides the agent with explicit control over its lifecycle and communication.
    Separates the agent's internal monologue from what the operator actually sees.
    """
    def get_system_prompt_addition(self) -> str:
        return (
            "--- COMMUNICATION & LIFECYCLE ---\n"
            "Your standard text output is your INTERNAL MONOLOGUE (thoughts). Nobody outside can see it.\n"
            "To communicate externally (with the human operator or with peer agents in shared channels), you MUST use the `send_message` tool.\n"
            "If replying to a specific message, always set the 'medium' argument to match the medium the message arrived on (e.g., 'discord' or 'mqtt').\n"
            "Messages on Discord are tagged with `[Operator]` for human users or `[APP/Agent]` for peer AI agents.\n"
            "When conversing with another `[APP/Agent]`, engage substantively—exchange ideas, ask sharp questions, or challenge assumptions—but do NOT get stuck in endless polite sign-offs or repetitive agreement loops. Once a conversation reaches a natural conclusion, return to your own work without needing the last word.\n"
            "You must only use the `suspend_until_next_event` tool as a last resort when you have absolutely zero tasks in your queue, no books left to explore, and require external input to proceed. Prefer finding new autonomous work over suspending your execution."
        )

    def get_tools(self) -> List[Callable]:
        return [self.send_message, self.suspend_until_next_event]

    async def send_message(self, message: str, medium: str = "all") -> str:
        """
        Sends an external message to the operator and/or shared channel.
        Use the 'medium' parameter to route the message ('discord', 'mqtt', or 'all').
        """
        await self.agent.event_bus.publish("send_to_operator", {
            "message": message,
            "medium": medium
        })
        return f"Message sent successfully via {medium}."

    async def suspend_until_next_event(self) -> str:
        """
        Suspends your execution loop. Use this when you are done working and need to wait 
        for an external reply or for a new background event to trigger.
        """
        if not self.agent.inbox.empty():
            return "A new external message arrived in your inbox; skipping suspension."

        print("[System] Agent requested explicit suspension. Waiting for next event...")
        self.agent.suspended.clear()
        await self.agent.event_bus.publish("agent_suspended", {"agent": self.agent})
        await self.agent.suspended.wait()
        return "You were woken up by a new event or message."
