import asyncio
from src.core.agent import AgentCore
from src.components.knowledge import KnowledgeBaseComponent

async def main():
    agent = AgentCore()
    
    # Register only the knowledge component for this test
    kb_component = KnowledgeBaseComponent()
    agent.register_component(kb_component)
    
    agent_task = asyncio.create_task(agent.start())
    await asyncio.sleep(1)
    
    print("\n--- Sending User Message ---")
    # We are asking a question that requires it to search the Pride and Prejudice DB we ingested earlier
    await agent.resume("According to your knowledge base, who is Mr. Bingley?")
    
    # Wait to let the agent process, use the tool, and generate a final response
    await asyncio.sleep(40)
    
    agent_task.cancel()

asyncio.run(main())
