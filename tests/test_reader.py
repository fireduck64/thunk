import asyncio
from src.core.agent import AgentCore
from src.components.reader import SequentialReaderComponent

async def main():
    agent = AgentCore()
    
    reader_component = SequentialReaderComponent()
    agent.register_component(reader_component)
    
    agent_task = asyncio.create_task(agent.start())
    await asyncio.sleep(1)
    
    print("\n--- Sending User Message 1 ---")
    await agent.resume("What books are available in your library?")
    await asyncio.sleep(10)
    
    print("\n--- Sending User Message 2 ---")
    await agent.resume("Please open MrDarcy.epub and read the first two chunks. Summarize the beginning for me.")
    await asyncio.sleep(30)
    
    agent_task.cancel()

asyncio.run(main())
