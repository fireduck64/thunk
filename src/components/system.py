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
            "Your standard text output is your INTERNAL MONOLOGUE (thoughts). The operator CANNOT see it.\n"
            "To communicate with the operator, you MUST use the `send_message_to_operator` tool.\n"
            "When you have finished a task and are waiting for a reply, use the `wait_for_next_event` tool to suspend your execution."
        )

    def get_tools(self) -> List[Callable]:
        return [self.send_message_to_operator, self.wait_for_next_event]

    async def send_message_to_operator(self, message: str) -> str:
        """Sends a message to the human operator."""
        await self.agent.event_bus.publish("send_to_operator", {"message": message})
        return "Message sent successfully."

    async def wait_for_next_event(self) -> str:
        """
        Suspends your execution loop. Use this when you are done working and need to wait 
        for the operator to reply or for a new background event to trigger.
        """
        print("[System] Agent requested explicit suspension. Waiting for next event...")
        self.agent.suspended.clear()
        await self.agent.event_bus.publish("agent_suspended", {"agent": self.agent})
        await self.agent.suspended.wait()
        return "You were woken up by a new event or message."
