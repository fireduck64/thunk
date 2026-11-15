# Memory System Improvements Backlog

Notes from the Nov 14, 2026 log & metrics audit across `data/{thunk,wisp}/{notes.db,memory.db,metrics}`.

## Completed / In Progress
- **Fix `ExecutiveAssistantComponent` Note Compaction & Briefing Deduplication:**
  - Explicit target character budget (`<= 75%` of `max_note_chars`), `finish_reason == "length"` / mid-line truncation guard, shrink-check guard, and per-note compaction cooldown/hash tracking so a single oversized dossier cannot loop forever or starve other oversized notes.
  - Sliding-window recent-briefing memory (`recent_briefing_window` turns) + bullet-level overlap deduplication + removal of redundant full dossier index from Pass 2 so `[Assistant Briefing]` stops repeating the same dossiers on ~99% of turns.

---

## Remaining Ideas

### 1. Stop "Self-Poisoning Memory" (Operator Ground Truth vs. Agent Hypotheses)
- **Problem:** Neither `TieredMemoryComponent` (`core_summary`) nor `ExecutiveAssistantComponent` (`assistant_filing`) distinguishes between **Operator Directives / Immutable Environment Rules** and **Agent Hypotheses / Transient Tool Failures**.
  - `TieredMemoryComponent` instructs the summarizer to *"Aggressively condense or completely discard older, stale information... Prioritize recent events"*, causing older instructions from `fireduck` (e.g., *"use `broken.git` as your shared writable repo"*) to age out while recent failed agent guesses (e.g., trying to `git push` to reference clone `everybody-codes-data.git` on GitHub or SCP to `hoffman2.idre.ucla.edu`) get written into `core_summary` as active blockers.
  - `ExecutiveAssistantComponent` (`assistant_filing`) appends agent error speculation into `assistant/system/infrastructure` (e.g., *"GitHub is blocked by missing PAT / DNS failure"*, *"Vector DB (Qdrant) is DOWN with Errno 111"* after a sandboxed container script tried connecting to `127.0.0.1:6333`), and `assistant_briefing` then repeats those false conclusions back to the agent.
- **Proposed Fix:**
  1. **Seed/Pin `assistant/system/environment` (or Environment Ground Truth):** A read-only or factory-seeded reference note stating the actual runtime rules:
     - `broken.git` (`ssh://git@spark-git/home/git/broken.git`) is the **only** writable shared git repository; other repos/clones in the workspace (`everybody-codes-data.git`, `TDCOSMO`, etc.) are read-only reference material.
     - The workspace container is intentionally firewalled behind `workspace-proxy` (outbound internet/SSH to GitHub, UCLA, etc. is blocked by policy, not "down").
     - Host services like Qdrant (`:6333`) and Ollama are accessed via native agent tools (`save_semantic_memory`, `global_search`), not via `localhost` inside `run_workspace_command`.
  2. **Provenance-Aware Filing (`prompts/assistant_filing`):**
     - Distinguish **Operator (`fireduck`) facts/directives** from **unverified agent hypotheses / transient tool errors**.
     - Never file transient shell/network errors from workspace experiments as permanent system infrastructure facts.
     - Tag operator-stated standing rules with `[Operator Directive]` so compaction preserves them.
  3. **Core Summary Preservation (`src/components/tiered.py`):**
     - Update `TieredMemoryComponent._compress_memory()` so standing operator instructions, workflow rules, and environment constraints are preserved across compactions instead of discarded for recency.
  4. **Data Scrub:**
     - Clean poisoned `"GitHub DNS / PAT / Qdrant is down / broken.git is a LURE"` bullets out of `data/{thunk,wisp}/notes.db` and `core_summary`.

---

### 2. Decouple `SpreadingActivationComponent` (Qdrant) from `ExecutiveAssistantComponent` (`notes.db`)
- **Problem:**
  - `SpreadingActivationComponent` currently searches both `VectorMemoryComponent` (Qdrant) and `StructuredNotesComponent.execute_search()`.
  - `StructuredNotesComponent.execute_search()` returns formatted note blobs (`"Note 'assistant/...': ..."`) which `SpreadingActivationComponent` hashes into 16-char SHA256 hex IDs in `layer_1_pool`.
  - `MemoryGardenerComponent` (`clutter_merger`) pulls IDs from `layer_1_pool` and passes them to `QdrantClient.retrieve(ids=[point_id])`, triggering `400 Bad Request` when given a 16-char note hash instead of a Qdrant UUID.
  - Seconds later on the same turn, `ExecutiveAssistantComponent` searches `notes.db` a second time on `before_llm_call`.
  - Additionally, `StructuredNotesComponent.execute_search()` falls back to a full-table token count scan (`+1.0` per query word found in `value`), biasing results toward the longest 5,000-char `assistant/*` notes.
- **Proposed Fix:**
  - Let `SpreadingActivationComponent` own Layer-2 Qdrant semantic/episodic recall (`VectorMemoryComponent`), and let `ExecutiveAssistantComponent` own `StructuredNotesComponent` (`notes.db`).
  - Ignore/purge non-UUID 16-char hashes in `layer_1_pool` / `MemoryGardenerComponent`.
  - Tighten `StructuredNotesComponent.execute_search()` so long notes do not win purely by word-count surface area.

---

### 3. Workspace Path Normalization & Reference Repo Hygiene
- **Problem:**
  - ~25% of `workspace` tool calls return non-zero exit codes. Agents frequently call `write_workspace_file("workspace/foo.py")`, creating `/home/agent/workspace/workspace/foo.py`, and then fail when running `python3 foo.py` in `/home/agent/workspace`.
  - Read-only reference clones like `everybody-codes-data.git` contain `.git` directories pointing to GitHub (`wimglenn/everybody-codes-data.git`), luring agents into committing/pushing there when searching `find . -name ".git"`.
- **Proposed Fix:**
  - Normalize/strip redundant leading `workspace/` or `/home/agent/workspace/` prefixes in `WorkspaceComponent` file path resolution.
  - Strip or rename `.git` inside read-only reference clones so `broken.git` is unambiguously the only Git repo in `/home/agent/workspace`.
