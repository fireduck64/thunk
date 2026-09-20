import asyncio
from typing import List, Callable
from src.core.agent import AgentCore
from src.components.critic import CriticComponent
from src.components.base import BaseComponent

class StubbornDoorComponent(BaseComponent):
    """A dummy component representing an impossible task to trigger a loop."""
    def get_system_prompt_addition(self) -> str:
        return (
            "Your overarching goal is to open the stubborn door. "
            "You have an `open_door` tool and an `inspect_door` tool. "
            "If it fails, keep trying."
        )
        
    def get_tools(self) -> List[Callable]:
        return [self.open_door, self.inspect_door]
        
    def open_door(self) -> str:
        return "Error: The door is stuck. It will not open."
        
    def inspect_door(self) -> str:
        return "The door is locked with a rusty padlock. Brute force will not work."

async def main():
    agent = AgentCore()
    
    agent.register_component(StubbornDoorComponent())
    agent.register_component(CriticComponent(frequency=3))

    agent_task = asyncio.create_task(agent.start())
    await asyncio.sleep(1)

    print("\n--- Sending User Message ---")
    await agent.resume("Please open the door. Try multiple times if necessary.")

    await asyncio.sleep(60)
    agent_task.cancel()

asyncio.run(main())
