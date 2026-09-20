import asyncio
import sys
from src.core.agent import AgentCore
from src.components.notes import StructuredNotesComponent
from src.components.knowledge import KnowledgeBaseComponent
from src.components.logger import AuditLogComponent
from src.adapters.mqtt import MQTTAdapter

async def main():
    print("Initializing Thunk Agent...")
    agent = AgentCore()
    
    # Register components
    agent.register_component(AuditLogComponent())
    agent.register_component(StructuredNotesComponent(db_path="notes.db"))
    agent.register_component(KnowledgeBaseComponent())
    
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
