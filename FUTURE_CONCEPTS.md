# Thunk: Future Development Concepts & Backlog

This document tracks concepts, features, and integrations planned for future development.

## 1. Discord Integration
*   **Concept:** Build a `DiscordAdapter` (similar to the current `MQTTAdapter`).
*   **Goal:** Allow the agent to reside in a Discord server, read channel history, and interact with users natively.
*   **Implementation Notes:** Would likely require the `discord.py` library and an event hook to wake the agent when `@mentioned` or when specific channels receive activity.

## 2. ZIM File Imports (Offline Wikipedia)
*   **Concept:** Expand the Knowledge Base capabilities to ingest `.zim` files.
*   **Goal:** Allow the agent to RAG against massive offline repositories like the entirety of Wikipedia, StackExchange, or medical encyclopedias without relying on web search.
*   **Implementation Notes:** Need a tool/script similar to `ingest_dir.py` but utilizing `libzim` to parse and chunk the compressed archives before embedding them into Qdrant.

## 3. Comprehensive Audit Logging (Debugging Mode)
*   **Concept:** Enhance the `AuditLogComponent`.
*   **Goal:** Add a configuration flag (e.g., `debug_mode = true`) that forces the logger to serialize and dump *everything*, including the full `AgentCore` state, memory tiers, and raw API responses, rather than stripping it down for clean JSONL.
*   **Implementation Notes:** Helpful for tracing exact token usage, context window limits, and diagnosing complex LLM hallucination chains.

## 4. Local File Reader Module
*   **Concept:** A new `FileReaderComponent`.
*   **Goal:** Give the agent a tool to explicitly read arbitrary local text/code files (e.g., `read_file(path="/var/log/syslog")`). 
*   **Implementation Notes:** 
    *   Needs strict path traversal protections (similar to the `SequentialReaderComponent` fix).
    *   Should probably support line-offset reading (e.g., `read_file(path, start_line=100, end_line=150)`) to prevent standard files from exploding the context window.
