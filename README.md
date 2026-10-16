# Thunk: Autonomous Modular Agent Framework

**Thunk** is a framework for building stateful, long-running, and self-directed Large Language Model (LLM) agents. 

Unlike standard chat bots, Thunk operates in a continuous asynchronous event loop. It maintains multi-tiered memory across sessions, plans and tracks goals in a persistent queue, creates and manages its own procedural skills, reads entire documents chunk by chunk, consults cloud models when necessary, and interacts with human operators over MQTT and Discord.

---

## Core Architecture

Thunk's architecture decouples agent cognition from capabilities through an event-driven component model:

1. **Agent Core (`src/core/agent.py`):**
   * Drives the continuous thinking loop.
   * Auto-generates OpenAI-compatible tool JSON schemas directly from Python type annotations and docstrings via `inspect`.
   * Supports parallel tool execution: processes batches of tool calls sequentially, appending individual `role: tool` responses with matching `tool_call_id`s before returning context to the model.
   * Streams LLM completions with keep-alive accumulation to avoid proxy timeouts, with optional real-time console streaming.
   * Dynamically rebuilds system prompts as component states change.

2. **Event Bus (`src/core/events.py`):**
   * Decouples components and external adapters through pub/sub event channels (`agent_started`, `message_added`, `before_user_message_added`, `context_window_check`, `before_llm_call`, `after_llm_call`, `tool_executed`, `send_to_operator`, `agent_suspended`).

3. **Pluggable Components (`src/components/`):**
   * All modular capabilities inherit from `BaseComponent`.
   * Components can inject text into the system prompt, expose tools to the LLM, and subscribe to lifecycle events.

4. **External Adapters (`src/adapters/`):**
   * Interface adapters connect external protocols to the agent loop, waking the agent when messages arrive.

---

## The 16 Pluggable Components

| Component | Responsibility | Key Tools / Actions |
| :--- | :--- | :--- |
| **`SubconsciousRecallComponent`** | Automatically performs semantic vector recall on incoming user messages and injects relevant context without polluting persistent transcripts. | Transparent context injection |
| **`ClockComponent`** | Injects current system time truncated to the hour to maximize `llama.cpp` / Ollama prefix cache hits. | System prompt time injection |
| **`DirectivesComponent`** | Injects high-level directives (intellectual curiosity, knowledge curation, self-reliance, cross-thread synthesis). | Behavioral guidance prompt |
| **`SystemComponent`** | Provides external communication and explicit lifecycle suspension. | `send_message`, `suspend_until_next_event` |
| **`AuditLogComponent`** | Writes deterministic, structured JSONL logs of all events, tool calls, and LLM completions. | Session logging to `logs/` |
| **`TieredMemoryComponent`** | Enforces token-based high/low watermarks, compacts older messages into a rolling core summary, archives all messages to SQLite, tracks `last_compacted_id` for crash resilience, and consolidates evicted contexts into vector memory. | Context compaction & archival |
| **`CriticComponent`** | Subconscious monitor that reviews recent interaction history and injects gentle observations/intuitions to prevent loop stalls. | Progress monitoring & guidance |
| **`TaskQueueComponent`** | Priority-based SQLite task queue for breaking down multi-step autonomous goals. | `add_task`, `pop_highest_priority_task`, `list_tasks` |
| **`StructuredNotesComponent`** | Hierarchical key-value scratchpad supporting pseudo-directories via `/` keys, plus automatic injection of any note named `sticky` into the system prompt. | `put_note`, `get_note`, `list_note_keys`, `rename_note`, `delete_note` |
| **`KnowledgeBaseComponent`** | Read-only vector retrieval against ingested external reference documents in Qdrant. | Vector search across reference material |
| **`VectorMemoryComponent`** | Personal semantic and episodic vector memory stored in Qdrant for long-term fact and thought recall. | `save_semantic_memory`, `dump_semantic_memory` |
| **`SequentialReaderComponent`** | Chunk-by-chunk EPUB document reader that maintains persistent SQLite bookmark state across restarts. | `list_library`, `search_library`, `open_book`, `read_next_chunk`, `read_previous_chunk` |
| **`GlobalSearchComponent`** | Unified meta-search component that fans out queries simultaneously across structured notes, vector memory, and the knowledge base. | `global_search` |
| **`FileReaderComponent`** | Multi-library local file reader with strict path traversal protections for inspecting project source code and local text files. | `list_libraries`, `list_files`, `read_file` |
| **`OracleComponent`** | Direct integration with Google Gemini via `google-genai` for querying current events, facts, or advanced reasoning beyond local context. | `ask_oracle` |
| **`SkillComponent`** | Procedural memory engine where the agent writes, reads, and manages reusable markdown workflows with YAML frontmatter in `skills/`, capped at 10 active skills. | `list_skills`, `load_skill`, `save_skill`, `delete_skill` |

---

## External Adapters

* **MQTT Adapter (`src/adapters/mqtt.py`):**
  * Subscribes to an incoming MQTT topic (`thunk/agent/inbox`) to wake the agent with external commands or sensor data.
  * Publishes outbound messages from `send_message` to an outbox topic (`thunk/agent/outbox`).
  * Features an automatic reconnect loop with backoff to recover from broker disconnects.

* **Discord Adapter (`src/adapters/discord_adapter.py`):**
  * Connects Thunk to a Discord server using `discord.py`.
  * Wakes the agent when messages arrive in allowed channels.
  * Relays agent responses directly back to the active Discord channel.

---

## Memory & Context Management

Thunk employs a layered memory architecture designed to eliminate context exhaustion and memory loss:

1. **Working Memory (Tier 1):** The active sliding window of messages sent to the LLM. Sized dynamically using `transformers.AutoTokenizer` against configured token watermarks (`high_watermark` and `low_watermark`).
2. **Archival Log (Tier 2):** SQLite database (`memory.db`) storing an immutable, sequential record of every message and tool execution with database IDs.
3. **Core Summary (Tier 3):** When working memory exceeds `high_watermark`, the oldest messages are summarized by the LLM and evicted from Tier 1. The summary is persisted in SQLite, capped by `max_summary_tokens`, and injected into every prompt.
4. **Crash Resilience:** The `last_compacted_id` pointer in `memory.db` tracks exactly which messages have been safely summarized. If the agent process is terminated mid-compaction, it resumes on startup without skipping uncompacted messages.
5. **Subconscious Associative Recall:** Incoming user messages query vector memory in the background, automatically injecting relevant past thoughts as a transient prefix that does not pollute archival logs.
6. **Semantic Vector Memory:** Evicted message transcripts and summaries are automatically consolidated into Qdrant for conceptual retrieval via `global_search`.
7. **Sticky Notes:** Any note saved under the key `sticky` in `StructuredNotesComponent` is permanently appended to the system prompt.

---

## CLI Utilities & Scripts

The `scripts/` directory provides administrative and operational commands (all config-dependent scripts require `-c <config.toml>` or `THUNK_CONFIG`):

* **`src/main.py`**: Main agent entry point (`./src/main.py -c data/thunk/config.toml`).
* **`scripts/setup.sh`**: Sets up the Python virtual environment at `./venv` (or creates it externally at `THUNK_VENV_DIR` and symlinks `./venv` to it).
* **`scripts/extract-monolog.py`**: Formats session JSONL audit logs into a clean, human-readable timeline showing internal monologues, tool calls, tool results, operator dispatches, critic intuitions, and user messages.
* **`scripts/send-notification.py`**: Reads `stdin` to EOF and publishes the payload to the agent's MQTT inbox topic (`-c data/thunk/config.toml`).
* **`scripts/mqtt-chat.py`**: Interactive terminal chat client over MQTT (`-c data/thunk/config.toml`).
* **`scripts/ask-oracle.py`**: Queries the configured Google Gemini model directly from the command line using `stdin` (`-c data/thunk/config.toml`).
* **`scripts/list-oracle-models.py`**: Lists all available Gemini models for the configured API key (`-c data/thunk/config.toml`).
* **`scripts/dump-vector-db.py`**: CLI utility for inspecting points stored in Qdrant collections (`-c data/thunk/config.toml`).
* **`scripts/ingest-vault.sh`**: Recursively chunks and embeds EPUB books from a directory into the Qdrant knowledge base collection (`-c data/thunk/config.toml`).

---

## Quickstart Guide

### 1. Prerequisites
* Linux host (Ubuntu/Debian recommended)
* Docker & Docker Compose (for Qdrant vector database)
* Python 3.11+
* Ollama or an OpenAI-compatible inference server with an appropriate model (e.g., `gemma4:31b` or `llama3`)

### 2. Start Supporting Services
Start the Qdrant vector database container:
```bash
docker compose up -d
```

### 3. Create Virtual Environment
Run the setup script to create and populate the virtual environment at `./venv`:
```bash
./scripts/setup.sh
```
*(By default, this creates `./venv` in the project root, which is gitignored. To store the virtual environment on another disk or share it across multiple agent directories, set `export THUNK_VENV_DIR=/path/to/venv` beforehand and `setup.sh` will automatically symlink `./venv` to it).*

### 4. Configure an Agent Instance
Create an isolated data directory for your agent (e.g., `data/thunk`), copy the starter skills, directives, and example configuration:
```bash
mkdir -p data/thunk/logs
cp -r skills data/thunk/skills
cp directives.example.md data/thunk/directives.md
cp config.example.toml data/thunk/config.toml
```
Edit `data/thunk/config.toml` and `data/thunk/directives.md` to customize your agent's name, directives, LLM endpoint, API keys, Discord bot token, and MQTT broker details.

### 5. Launch Thunk
```bash
./src/main.py -c data/thunk/config.toml
```

### 6. Inspect Logs
View the live agent timeline from any session log:
```bash
./scripts/extract-monolog.py data/thunk/logs/session_YYYYMMDD_HHMMSS.jsonl
```
