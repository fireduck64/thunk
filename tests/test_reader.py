import asyncio
import pytest
from src.core.agent import AgentCore
from src.components.reader import SequentialReaderComponent

@pytest.mark.asyncio
async def test_sequential_reader():
    agent = AgentCore()
    
    reader_component = SequentialReaderComponent()
    agent.register_component(reader_component)
    
    agent_finished_event = asyncio.Event()
    async def on_agent_suspended(*args, **kwargs):
        agent_finished_event.set()
    agent.event_bus.subscribe("agent_suspended", on_agent_suspended)
    
    agent_task = asyncio.create_task(agent.start())
    await asyncio.sleep(0.5)
    
    print("\n--- Sending User Message 1 ---")
    await agent.resume("What books are available in your library?")
    await agent_finished_event.wait()
    agent_finished_event.clear()
    
    print("\n--- Sending User Message 2 ---")
    await agent.resume("Please open MrDarcy.epub and read the first two chunks. Summarize the beginning for me.")
    await agent_finished_event.wait()
    agent_finished_event.clear()
    
    agent_task.cancel()
    
    # Verify the component state
    assert reader_component.current_book is not None
    assert "MrDarcy.epub" in reader_component.current_book
    assert reader_component.current_index == 2

