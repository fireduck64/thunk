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
from src.components.oracle import OracleComponent
from src.components.skills import SkillComponent
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
    agent.register_component(AuditLogComponent())
    agent.register_component(TieredMemoryComponent())
    agent.register_component(CriticComponent(frequency=8))
    agent.register_component(TaskQueueComponent())
    agent.register_component(StructuredNotesComponent())
    agent.register_component(KnowledgeBaseComponent())
    agent.register_component(VectorMemoryComponent())
    agent.register_component(SequentialReaderComponent())
    agent.register_component(GlobalSearchComponent())
    agent.register_component(FileReaderComponent())
    agent.register_component(OracleComponent())
    agent.register_component(SkillComponent())
    
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
