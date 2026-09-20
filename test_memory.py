import asyncio
from src.core.agent import AgentCore
from src.memory.tiered import TieredMemoryComponent

async def main():
    agent = AgentCore()
    
    # We set max_messages to 4 so it compresses very quickly for the test
    mem_component = TieredMemoryComponent(db_path="test_memory.db", max_messages=4, summarize_chunk=2)
    agent.register_component(mem_component)
    
    agent_task = asyncio.create_task(agent.start())
    await asyncio.sleep(1)
    
    print("\n--- Message 1 ---")
    await agent.resume("My name is John. I like apples.")
    await asyncio.sleep(10)
    
    print("\n--- Message 2 ---")
    await agent.resume("Can you remember my name?")
    await asyncio.sleep(10)
    
    print("\n--- Message 3 (Should trigger compression) ---")
    await agent.resume("My favorite color is blue.")
    await asyncio.sleep(15)
    
    print("\n--- Final Context Size ---")
    print(f"Messages in working memory: {len(agent.messages)}")
    print(f"Current Core Summary: {mem_component._get_core_summary()}")
    
    agent_task.cancel()

asyncio.run(main())
