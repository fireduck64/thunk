# Thunk: Future Development Concepts & Backlog

This document tracks upcoming concepts, architectural plans, and completed milestones for the Thunk agent framework.

---

## Active Backlog & Future Concepts

### 1. ZIM File Support (Offline Wikipedia)
* **Concept:** Provide the agent with access to massive offline encyclopedias (like Wikipedia or WikiVoyage) stored in compressed `.zim` format.
* **Architectural Options to Consider:**
  * **Option A: Full Vector DB Ingestion (Semantic Search)**
    * *Approach:* Use `libzim` to extract HTML articles, strip boilerplate, chunk the text, compute embeddings via Ollama, and batch-upload to Qdrant.
    * *Pros:* Allows the agent to query broad knowledge via `global_search` using abstract concepts rather than exact page titles.
    * *Cons:* Ingestion is computationally heavy and time-consuming for full Wikipedia (6.5M+ articles).
  * **Option B: Lazy-Load `ZimReaderComponent` (Real-Time Lookup)**
    * *Approach:* Create a native `ZimReaderComponent` with tools like `search_wiki_titles(keyword)` and `get_wiki_article(title)`. The agent queries the local ZIM file index on demand.
    * *Pros:* Instant setup. Zero embedding computation time or vector storage bloat.
    * *Cons:* Searches must be title/keyword based. Relies on the agent to summarize and persist important findings to `VectorMemoryComponent` for long-term conceptual recall.

### 2. Productionization & Portability
* **Concept:** Standardize deployment for automated host provisioning and self-contained environments.
* **Implementation Goals:**
  * **Configurable Environment Management:** Standardize virtual environment paths across heterogeneous hosts while keeping the git workspace clean.
  * **Database Strategy:** Evaluate consolidating multi-file SQLite databases (`memory.db`, `notes.db`, `tasks.db`, `reader_state.db`) into a unified PostgreSQL or MySQL service container for multi-agent support.
  * **Docker Storage Standard:** Ensure all persistent state (Qdrant, databases, logs, reader cache) is backed by named Docker volumes with standardized prefixes.
  * **Vector Database Separation:** Split the Qdrant infrastructure into two distinct collections/instances:
    1. *Knowledge Store:* High storage footprint, static (re-ingestible from books/wikis).
    2. *Memory Vector Store:* Low storage footprint, high intrinsic value (contains agent reflections, episodic memories, and learned skills). Backing up the agent's "brain" remains lightweight and independent of static libraries.

### 3. Proactive Skill Execution
* **Concept:** Expand `SkillComponent` so the agent can autonomously trigger specialized skills based on pattern matches in its task queue or incoming user intents, rather than requiring explicit manual tool lookups.

---

## Completed Architecture Milestones

* [x] **Discord Integration:** Full bidirectional `DiscordAdapter` using `discord.py` supporting channel whitelisting and direct message relay.
* [x] **Cloud AI Oracle:** Integrated `OracleComponent` connecting to Google Gemini (`google-genai` / `gemini-3.8-flash`) for live web knowledge, fact verification, and current events.
* [x] **Procedural Skills Engine:** Built `SkillComponent` to store reusable multi-step markdown procedures in `skills/` with YAML frontmatter, 10-skill cap, and dynamic prompt injection.
* [x] **Multi-Library Local File Reader:** Implemented `FileReaderComponent` with path traversal protections and named library mappings for examining source code and local files.
* [x] **Real-Time Streaming Output & Log Formatting:** Added `stream_output` debug configuration for live token streaming in the console and built `scripts/extract-monolog.sh` to parse JSONL session logs into clean, human-readable timelines.
* [x] **MQTT Auto-Reconnect:** Implemented resilient reconnection loops with backoff in `MQTTAdapter` and built `scripts/send-notification.py` to pipe stdin directly to the agent.
* [x] **Crash-Resilient Token Watermark Memory:** Replaced message-count limits with `transformers.AutoTokenizer` token counting (`high_watermark` / `low_watermark`), token-capped core summaries, and SQLite `last_compacted_id` tracking to prevent memory holes on process termination.
* [x] **Subconscious Associative Recall:** Added `SubconsciousRecallComponent` to query semantic vector memory on incoming user messages, injecting memories into the prompt without polluting persistent logs.
* [x] **Hierarchical Notes & Sticky Injection:** Upgraded `StructuredNotesComponent` to support pseudo-directory navigation (`/`) and automatic persistent injection of notes named `sticky`.
