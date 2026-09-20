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
from src.adapters.mqtt import MQTTAdapter

async def main():
    print("Initializing Thunk Agent...")
    agent = AgentCore()
    
    # Register all components to create a fully capable agent
    agent.register_component(ClockComponent())
    agent.register_component(DirectivesComponent())
    agent.register_component(SystemComponent())
    agent.register_component(AuditLogComponent(log_dir="logs"))
    agent.register_component(TieredMemoryComponent(db_path="memory.db", max_messages=20, summarize_chunk=10))
    agent.register_component(CriticComponent(frequency=5))
    agent.register_component(TaskQueueComponent(db_path="tasks.db"))
    agent.register_component(StructuredNotesComponent(db_path="notes.db"))
    agent.register_component(KnowledgeBaseComponent())
    agent.register_component(VectorMemoryComponent(collection_name="agent_memory"))
    agent.register_component(SequentialReaderComponent(library_dir="/vault/ebook"))
    
    # Create the MQTT adapter
    mqtt_adapter = MQTTAdapter(agent)
    
    # Run both the agent loop and the MQTT listener concurrently
    try:
        await asyncio.gather(
            agent.start(),
            mqtt_adapter.start()
        )
    except asyncio.CancelledError:
        print("\nShutting down gracefully...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
