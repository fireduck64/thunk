import asyncio
from src.core.agent import AgentCore
from src.components.notes import StructuredNotesComponent

async def main():
    # Initialize the core
    agent = AgentCore()
    
    # Register components
    notes_component = StructuredNotesComponent(db_path="test_notes.db")
    agent.register_component(notes_component)
    
    # Run the agent in the background
    # We use asyncio.create_task so we can send it messages while it runs
    agent_task = asyncio.create_task(agent.start())
    
    # Give the agent a moment to start
    await asyncio.sleep(1)
    
    # 1. Ask the agent to save a note using the tool
    print("\n--- Sending User Message 1 ---")
    agent.resume("Please save a note with the key 'project_ideas' and the value '1. Learn async. 2. Build robots.'")
    
    # Wait to let the agent process and use the tool
    await asyncio.sleep(10)
    
    # 2. Ask the agent to recall the note
    print("\n--- Sending User Message 2 ---")
    agent.resume("What were the project ideas I just asked you to save? Use your tools to check.")
    
    await asyncio.sleep(10)
    
    print("\n--- Checking the SQLite DB directly to verify ---")
    print("Database contents:", notes_component.get_note("project_ideas"))
    
    # Cancel the infinite loop
    agent_task.cancel()

if __name__ == "__main__":
    # Windows/Linux specific asyncio boilerplate for graceful shutdown
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
