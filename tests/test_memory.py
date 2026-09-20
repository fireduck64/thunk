import asyncio
import pytest
import os
from src.core.agent import AgentCore
from src.components.tiered import TieredMemoryComponent

@pytest.mark.asyncio
async def test_tiered_memory():
    # Cleanup old test db if it exists
    if os.path.exists("test_memory.db"):
        os.remove("test_memory.db")
        
    agent = AgentCore()
    
    # We set max_messages to 4 so it compresses very quickly for the test
    mem_component = TieredMemoryComponent(db_path="test_memory.db", max_messages=4, summarize_chunk=2)
    agent.register_component(mem_component)
    
    # We need a way to know when the agent finishes processing a message
    agent_finished_event = asyncio.Event()
    
    async def on_agent_suspended(*args, **kwargs):
        agent_finished_event.set()
        
    agent.event_bus.subscribe("agent_suspended", on_agent_suspended)
    
    agent_task = asyncio.create_task(agent.start())
    await asyncio.sleep(0.5)
    
    print("\n--- Message 1 ---")
    await agent.resume("My name is John. I like apples.")
    await agent_finished_event.wait()
    agent_finished_event.clear()
    
    print("\n--- Message 2 ---")
    await agent.resume("Can you remember my name?")
    await agent_finished_event.wait()
    agent_finished_event.clear()
    
    print("\n--- Message 3 (Should trigger compression) ---")
    await agent.resume("My favorite color is blue.")
    await agent_finished_event.wait()
    agent_finished_event.clear()
    
    # Let the background compression task finish
    # Since compression involves an LLM call, it might take 10-20 seconds.
    # In a proper test suite we might mock the LLM call, but for this integration test
    # we just wait a bit longer or monitor the summary
    for _ in range(30):
        if "John" in mem_component._get_core_summary():
            break
        await asyncio.sleep(1)
    
    print("\n--- Final Context Size ---")
    print(f"Messages in working memory: {len(agent.messages)}")
    print(f"Current Core Summary: {mem_component._get_core_summary()}")
    
    agent_task.cancel()
    
    # Basic assertions to ensure the memory component actually compressed
    assert len(agent.messages) <= 6 # It shouldn't grow boundlessly
    assert "John" in mem_component._get_core_summary()

