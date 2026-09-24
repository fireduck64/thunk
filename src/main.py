import asyncio
import sys
from src.core.agent import AgentCore
from src.components.notes import StructuredNotesComponent
from src.components.knowledge import KnowledgeBaseComponent
from src.components.logger import AuditLogComponent
from src.components.tiered import TieredMemoryComponent
from src.components.critic import CriticComponent
from src.components.reader import SequentialReaderComponent
from src.components.system import SystemComponent
from src.components.tasks import TaskQueueComponent
from src.components.clock import ClockComponent
from src.components.directives import DirectivesComponent
from src.components.vector_memory import VectorMemoryComponent
from src.components.global_search import GlobalSearchComponent
from src.components.subconscious import SubconsciousRecallComponent
from src.components.file_reader import FileReaderComponent
from src.adapters.mqtt import MQTTAdapter
from src.adapters.discord_adapter import DiscordAdapter

async def main():
    print("Initializing Thunk Agent...")
    agent = AgentCore()
    
    # Register all components to create a fully capable agent
    agent.register_component(SubconsciousRecallComponent())
    agent.register_component(ClockComponent())
    agent.register_component(DirectivesComponent())
    agent.register_component(SystemComponent())
    agent.register_component(AuditLogComponent(log_dir="logs", verbose=True))
    
    # We aggressively tune the memory component because large context windows
    # on local models can cause timeouts. 
    # Max messages: 12. Chunk to summarize: 6.
    agent.register_component(TieredMemoryComponent(db_path="memory.db", max_messages=12, summarize_chunk=6))
    
    agent.register_component(CriticComponent(frequency=8))
    agent.register_component(TaskQueueComponent(db_path="tasks.db"))
    agent.register_component(StructuredNotesComponent(db_path="notes.db"))
    agent.register_component(KnowledgeBaseComponent())
    agent.register_component(VectorMemoryComponent(collection_name="agent_memory"))
    agent.register_component(SequentialReaderComponent(library_dir="/vault/ebook"))
    agent.register_component(GlobalSearchComponent())
    agent.register_component(FileReaderComponent(libraries={"thunk_source": "/home/fireduck/projects/thunk/src"}))
    
    # Create adapters
    mqtt_adapter = MQTTAdapter(agent)
    discord_adapter = DiscordAdapter(agent)
    
    # Run the agent loop and all listeners concurrently
    try:
        await asyncio.gather(
            agent.start(),
            mqtt_adapter.start(),
            discord_adapter.start()
        )
    except asyncio.CancelledError:
        print("\nShutting down gracefully...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
