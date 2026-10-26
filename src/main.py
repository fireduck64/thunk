#!./venv/bin/python3
import argparse
import asyncio
from datetime import datetime
import os
import sys
from transformers import AutoTokenizer

# Ensure the project root is on sys.path when invoked directly as ./src/main.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import set_config_path, load_config
from src.utils.metrics import set_shared_tokenizer
from src.core.agent import AgentCore
from src.components.notes import StructuredNotesComponent
from src.components.knowledge import KnowledgeBaseComponent
from src.components.logger import AuditLogComponent
from src.components.tiered import TieredMemoryComponent
from src.components.critic import CriticComponent
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
from src.components.thread_presence import ThreadRegistry, ThreadPresenceComponent
from src.adapters.mqtt import MQTTAdapter
from src.adapters.discord_adapter import DiscordAdapter


def _load_shared_tokenizer(config: dict):
    tokenizer_model = config.get("tokenizer", {}).get("model_name", "unsloth/gemma-7b")
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_model, model_max_length=100000)
    except Exception as e:
        print(f"[Memory Warning] Could not load tokenizer '{tokenizer_model}': {e}")
        print("[Memory Warning] Falling back to 'gpt2' tokenizer for approximate token counting.")
        tok = AutoTokenizer.from_pretrained("gpt2", model_max_length=100000)
    set_shared_tokenizer(tok)
    return tok


def _create_shared_log_file(config: dict) -> str:
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"session_{session_id}.jsonl")


def _register_thread_components(
    agent: AgentCore,
    config: dict,
    thread_id: str = "main",
    shared_tokenizer=None,
    shared_log_file: str = None,
    registry: ThreadRegistry = None,
    role: str = "General consciousness thread",
    enabled_components: list = None,
):
    sub_mode = config.get("subconscious", {}).get("mode", "standard")
    sa_enabled = config.get("spreading_activation", {}).get("enabled", False)
    if sub_mode == "spreading_activation" or sa_enabled:
        agent.register_component(SpreadingActivationComponent(thread_id=thread_id))
    else:
        agent.register_component(SubconsciousRecallComponent(thread_id=thread_id))

    agent.register_component(ClockComponent())
    agent.register_component(DirectivesComponent())
    if registry is not None:
        agent.register_component(
            ThreadPresenceComponent(registry=registry, thread_id=thread_id, role=role)
        )
    agent.register_component(SystemComponent())
    agent.register_component(AuditLogComponent(thread_id=thread_id, log_file=shared_log_file))
    agent.register_component(
        TieredMemoryComponent(thread_id=thread_id, tokenizer=shared_tokenizer)
    )
    agent.register_component(CriticComponent(frequency=8))

    if enabled_components is not None:
        requested = {str(c).strip().lower() for c in enabled_components}
    else:
        requested = None

    def _wants(*aliases: str) -> bool:
        if requested is None:
            return True
        return any(a.lower() in requested for a in aliases)

    if _wants("tasks", "task_queue", "taskqueuecomponent"):
        agent.register_component(TaskQueueComponent())
    if _wants("notes", "structured_notes", "structurednotescomponent"):
        agent.register_component(StructuredNotesComponent())
    if _wants("knowledge", "knowledge_base", "knowledgebasecomponent"):
        agent.register_component(KnowledgeBaseComponent(thread_id=thread_id))
    if _wants("vector_memory", "vectormemorycomponent"):
        agent.register_component(VectorMemoryComponent(thread_id=thread_id))
    if _wants("global_search", "globalsearchcomponent"):
        agent.register_component(GlobalSearchComponent())
    if _wants(
        "file_reader",
        "filereadercomponent",
        "reader",
        "sequential_reader",
        "sequentialreadercomponent",
        "pdf_viewer",
        "pdf",
        "pdfviewercomponent",
    ):
        agent.register_component(FileReaderComponent())
    if _wants("oracle", "oraclecomponent"):
        agent.register_component(OracleComponent())
    if _wants("skills", "skillcomponent"):
        agent.register_component(SkillComponent())

    ws_globally_enabled = config.get("workspace", {}).get("enabled", False)
    if requested is None:
        if ws_globally_enabled:
            agent.register_component(WorkspaceComponent())
    elif _wants("workspace", "workspacecomponent") and ws_globally_enabled:
        agent.register_component(WorkspaceComponent())


async def main():
    print("Initializing Thunk Agent...")
    config = load_config()
    shared_tokenizer = _load_shared_tokenizer(config)
    shared_log_file = _create_shared_log_file(config)
    threads_cfg = config.get("threads")

    if not isinstance(threads_cfg, dict) or not threads_cfg:
        agent = AgentCore(thread_id="main")
        _register_thread_components(
            agent,
            config,
            thread_id="main",
            shared_tokenizer=shared_tokenizer,
            shared_log_file=shared_log_file,
        )
        mqtt_adapter = MQTTAdapter(agent)
        discord_adapter = DiscordAdapter(agent)

        try:
            await asyncio.gather(
                agent.start(),
                mqtt_adapter.start(),
                discord_adapter.start(),
            )
        except asyncio.CancelledError:
            print("\nShutting down gracefully...")
        return

    # Multi-thread initialization
    registry = ThreadRegistry()
    for thread_id, t_cfg in threads_cfg.items():
        if not isinstance(t_cfg, dict):
            continue
        role = t_cfg.get("role", "General consciousness thread")
        registry.register_thread(thread_id, role)

    coroutines = []
    for thread_id, t_cfg in threads_cfg.items():
        if not isinstance(t_cfg, dict):
            continue
        role = t_cfg.get("role", "General consciousness thread")
        enabled_components = t_cfg.get("components", None)
        adapters = [str(a).strip().lower() for a in t_cfg.get("adapters", [])]

        agent = AgentCore(thread_id=thread_id)
        registry.register_thread(thread_id, role, core=agent)
        _register_thread_components(
            agent,
            config,
            thread_id=thread_id,
            shared_tokenizer=shared_tokenizer,
            shared_log_file=shared_log_file,
            registry=registry,
            role=role,
            enabled_components=enabled_components,
        )
        coroutines.append(agent.start())

        if "mqtt" in adapters:
            mqtt_adapter = MQTTAdapter(agent)
            coroutines.append(mqtt_adapter.start())
        if "discord" in adapters:
            discord_adapter = DiscordAdapter(agent)
            coroutines.append(discord_adapter.start())

    try:
        await asyncio.gather(*coroutines)
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
