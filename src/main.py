#!./venv/bin/python3
import argparse
import asyncio
import os
import sys

# Ensure the project root is on sys.path when invoked directly as ./src/main.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import set_config_path, load_config
from src.core.agent import AgentCore
from src.components.notes import StructuredNotesComponent
from src.components.knowledge import KnowledgeBaseComponent
from src.components.logger import AuditLogComponent
from src.components.tiered import TieredMemoryComponent
from src.components.critic import CriticComponent
from src.components.reader import SequentialReaderComponent
from src.components.pdf_viewer import PdfViewerComponent
from src.components.system import SystemComponent
from src.components.tasks import TaskQueueComponent
from src.components.clock import ClockComponent
from src.components.directives import DirectivesComponent
from src.components.vector_memory import VectorMemoryComponent
from src.components.global_search import GlobalSearchComponent
from src.components.subconscious import SubconsciousRecallComponent
from src.components.spreading_activation import SpreadingActivationComponent
from src.components.file_reader import FileReaderComponent
from src.components.oracle import OracleComponent
from src.components.skills import SkillComponent
from src.components.workspace import WorkspaceComponent
from src.adapters.mqtt import MQTTAdapter
from src.adapters.discord_adapter import DiscordAdapter

async def main():
    print("Initializing Thunk Agent...")
    config = load_config()
    agent = AgentCore()

    # Select mutually exclusive subconscious recall strategy
    sub_mode = config.get("subconscious", {}).get("mode", "standard")
    sa_enabled = config.get("spreading_activation", {}).get("enabled", False)
    if sub_mode == "spreading_activation" or sa_enabled:
        agent.register_component(SpreadingActivationComponent())
    else:
        agent.register_component(SubconsciousRecallComponent())

    # Register remaining components to create a fully capable agent
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
    agent.register_component(PdfViewerComponent())
    agent.register_component(GlobalSearchComponent())
    agent.register_component(FileReaderComponent())
    agent.register_component(OracleComponent())
    agent.register_component(SkillComponent())
    if config.get("workspace", {}).get("enabled", False):
        agent.register_component(WorkspaceComponent())
    
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
    parser = argparse.ArgumentParser(description="Thunk Autonomous Agent")
    parser.add_argument("-c", "--config", type=str, help="Path to the TOML configuration file")
    args = parser.parse_args()
    if args.config:
        set_config_path(args.config)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
