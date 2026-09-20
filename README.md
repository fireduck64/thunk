

# Autonomous Modular Agent Framework (Thunk)

A framework for building long-running, stateful, and highly autonomous Large Language Model (LLM) agents. 

This system is designed around a central **Agent Core** (an asynchronous event loop) that delegates capabilities, memory, and tooling to pluggable **Components**. It is built in Python, designed to run locally, and optimized for OpenAI-compatible endpoints (specifically Ollama).

## Core Architecture

The framework is built to solve the limitations of standard "chat" interfaces by enabling continuous processing, metacognition, and structured memory.

1. **The Agent Core (`asyncio` Event Loop):**
   - Operates a continuous "thinking loop".
   - Collects context and tools from all attached components.
   - Prompts the LLM and executes requested tools.
   - Can suspend its own execution until specific events occur (e.g., a webhook trigger or a timer).

2. **The Component Interface:**
   - All capabilities are isolated into modules that inherit from a standard `BaseComponent`.
   - Components can inject instructions into the main System Prompt.
   - Components can expose Python functions as LLM tools (auto-generating JSON schemas via Pydantic).
   - Components can hook into system lifecycle events (`on_start`, `on_llm_response`, `on_error`).

3. **Event Bus & API:**
   - A lightweight background server (e.g., FastAPI) listens for external triggers (webhooks, user CLI inputs) and injects them into the event loop, waking the agent if it is suspended.

---

## The "Mind" of the Agent (Information Management)

To solve context window exhaustion and infinite loops, the framework utilizes a structured approach to memory:

*   **Tier 1: Working Memory (Context Window):** Managed automatically as a sliding window of recent dialogue and tool executions.
*   **Tier 2: Archival Memory (Database):** An immutable log of every interaction.
*   **Tier 3: Core Summary:** A continuously updated scratchpad of current facts, user preferences, and the overarching goal. (Injected into every system prompt).
*   **Tier 4: Structured Notes:** Deliberately managed by the agent via tools (`put_note`, `get_note`) for specific information blocks.
*   **Tier 5: Knowledge Base:** Vector-searchable repositories of broad information (Books, Wikis, Skills).

---

## Planned Modules (Components)

### Core System Modules

*   **Lifecycle Controller:** Provides tools for the agent to control its execution state (e.g., `suspend_execution`, `sleep`).
*   **Tiered Memory Manager:** Monitors Working Memory size. Triggers background LLM tasks to summarize old conversations into the Core Summary before evicting them from the active context window.
*   **Audit Logger:** Subscribes to all system events and writes a deterministic, raw JSONL record of exactly what the system executed (distinct from the agent's filtered memory).

### Metacognition

*   **The Critic Agent:** A supervisor module running on a smaller, faster LLM model. It periodically reviews the Audit Log to determine if the main agent is stuck in a loop or making progress, and injects "internal feelings" (e.g., *"You feel stuck on this file, try something else"*) into the Working Memory.

### Tooling & Capabilities

*   **Structured Notes:** Provides a key-value database for the agent to use as a deliberate scratchpad. (Tools: `list_note_keys`, `get_note`, `put_note`, `delete_note`).
*   **Skills Database:** Connects to a vector database. Allows the agent to look up proven workflows or save new complex workflows it discovers for future use. (Tools: `search_skills`, `save_skill`).
*   **Global Search (Meta-Component):** Aggregates search functionality across all capable modules. Instead of exposing 10 different search tools, it exposes a single `global_search(query)` tool that fans out to the Wiki, Book, and Skill databases and aggregates the results.
*   **Document Knowledge Base (Read-Only):** RAG component connecting to a vector database of ingested EPUBs and Wikis. Accessed primarily through the Global Search component.
*   **Sequential Book Reader:** Allows the agent to deeply comprehend large texts in a stateful manner, chunk by chunk, rather than just relying on semantic search. (Tools: `list_library`, `open_document`, `read_next_page`).

## Technology Stack

*   **Language:** Python 3.11+ (Leveraging `asyncio` and advanced typing).
*   **LLM Integration:** `openai` Python SDK (targeting local Ollama endpoints).
*   **Tool Schema Generation:** `pydantic` (and potentially `instructor` or `marvin`).
*   **Storage:** SQLite (for Notes, Archival Memory, State) and ChromaDB / Qdrant (for Vector/Skills storage).
*   **Deployment:** Designed to run via Docker Compose, coordinating the Python agent, Vector DB, and Ollama instance.
