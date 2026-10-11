# Memory Remodeling: Automatic vs. Manual Cognitive Architecture

This document outlines the conceptual shift in Thunk's memory design from manual tool queries toward automatic associative recall, alongside the current production architecture implementing it.

---

## The Original Concept

I think we have been thinking about memory wrong.

I don't want to over-anthropomorphize, but lacking a better example we can use how humans seem to work.

When a human experiences an event, they don't think "let me see if this triggers a memory" which in our system would be doing a search. That is automatic for humans.

* **Event:** I see a butterfly near a castle.
* **Triggered memory:** I recall something about the butterfly city in the *Shadows of the Apt* series.

### Automatic Retrieval
For whatever our events are, like a chat message or a section of a book just read, we can automatically do a search and include those along with the data sent to the agent.

### Automatic vs. Manual
* **Automatic:** Should cover things the agent has experienced: its memory log, notes it has made, and facts it has explicitly saved. 
* **Manual:** It should *not* automatically search the external knowledge database. That information has not been internalized; it hasn't been processed by the agent. It is outside, and it should be an active decision for the agent to check that or not.

### Automatic Recording
Log entries should automatically feed a vector memory store. Humans don't need to deliberately think about recording an experience—it happens organically. However, there must be boundaries: the persistent memory log should record the event ("I see a butterfly near a castle"), but should *not* pollute the log with the transient memory that was automatically recalled.

---

## Production Implementation & Architecture

All core principles of this document are implemented across Thunk's component suite:

### 1. The Subconscious Associative Recall Engine (`SubconsciousRecallComponent`)
* Subscribes to the `before_user_message_added` event fired by `AgentCore.resume()`.
* When an operator sends a message (via Discord, MQTT, or CLI), the component queries `VectorMemoryComponent` in the background for semantically relevant memories.
* If relevant memories are found (similarity above threshold), it prepends a `[Subconscious Recall Triggered]` block to the message in working memory before the LLM processes it.

### 2. Strict Separation: Internal Mind vs. External Knowledge
* **Internal (Automatic):** `SubconsciousRecallComponent` only searches `VectorMemoryComponent` (the agent's lived experiences, reflections, and past consolidated transcripts).
* **External (Manual):** `KnowledgeBaseComponent` (containing ingested reference books and documents) is strictly excluded from subconscious recall. The agent must consciously choose to search external knowledge using `global_search`.

### 3. Log Purity & Transcript Stripping
* In `AgentCore`, clean user messages are committed to SQLite `archival_log`, while the subconscious memory prefix is only attached in-memory for the immediate LLM inference step.
* During memory compaction in `TieredMemoryComponent`, any lingering subconscious blocks are stripped from the transcript:
  ```python
  if role == "user" and content.startswith("[Subconscious Recall Triggered"):
      parts = content.split("--- End Recall ---\n\n", 1)
      if len(parts) == 2:
          content = parts[1]
  ```
  This prevents the agent's recalled memories from recursively amplifying and bloating future summaries.

### 4. Automatic Consolidation into Vector Memory
* When `TieredMemoryComponent` evicts messages from Tier 1 working memory, it automatically pairs the evicted raw transcript with the updated core summary and saves it as a consolidated block into `VectorMemoryComponent`.
* As a result, the agent continuously and automatically builds an episodic long-term memory of its past operations without needing to remember to save every individual action.
