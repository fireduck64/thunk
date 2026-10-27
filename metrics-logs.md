# Metrics Logging Specification (`metrics-YYYY-MM-DD.jsonl`)

## 1. File Storage & Daily Rotation

- **Append-Only:** Metric log files are strictly append-only JSON Lines (`.jsonl`). Each completed operation writes a single JSON object followed by a newline (`\n`). Existing lines are never modified or overwritten.
- **Directory:** Configured via `[logging].metrics_dir` in `config.toml` (e.g., `data/thunk/metrics/`).
- **Daily UTC Rotation:** Each agent writes to one file per UTC calendar day named `metrics-YYYY-MM-DD.jsonl`.
  - When the UTC date rolls over at `00:00:00Z`, new entries are automatically written to the new day's file.
  - On process restart, if the current UTC day's file already exists, new entries are appended to it.

---

## 2. Required Fields

Every JSONL entry **must** include the following top-level fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | `string` | When the operation completed, formatted as RFC 3339 in UTC with millisecond precision (e.g., `"2026-10-26T20:24:15.123Z"`). |
| `operation` | `string` | Low-cardinality name of the operation performed (e.g., `"llm_call"`, `"tool_call"`, `"vector_search"`, `"vector_save"`, `"workspace_exec"`, `"speaker_call"`, `"db_query"`). Avoid encoding dynamic parameters in `operation` (use `"db_query"`, not `"db_query_for_books"`). |
| `duration_ms` | `number` | Total wall-clock duration of the operation in milliseconds. |
| `error` | `integer` | `0` if the operation succeeded (or recovered via retries); `1` if the operation failed with an unrecovered error. |

---

## 3. Standard Optional Fields

Depending on the operation, entries may include the following optional top-level fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `agent` | `string` | Name of the agent (e.g., `"Thunk"`, `"Wisp"`). |
| `thread` | `string` | Consciousness thread identifier (e.g., `"main"`, `"social"`, `"mind"`). |
| `component` | `string` | Software component performing the operation (`"core"`, `"tiered_memory"`, `"spreading_activation"`, `"subconscious"`, `"critic"`, `"oracle"`, `"vector_memory"`, `"knowledge"`, `"workspace"`). |
| `type` | `string` | Sub-classification of the operation:<br>• For `operation == "llm_call"`: `"agent_turn"`, `"memory_compaction"`, `"subconscious_recall"`, `"critic"`, or `"oracle"`.<br>• For `operation == "tool_call"`: the tool function name (e.g., `"read_file"`, `"send_message"`, `"run_workspace_command"`). |
| `model` | `string` | Model identifier used for LLM or embedding calls (e.g., `"gemma4:31b"`, `"gemma4:e2b"`, `"gemini-3.8-flash"`, `"nomic-embed-text"`). |
| `exec_location` | `string` | For `operation == "llm_call"`, where inference executed: `"local"` (local Ollama / vLLM / llama.cpp) or `"cloud"` (e.g., `"oracle"`). |
| `finish_reason` | `string` | For `operation == "llm_call"`, the completion finish reason (e.g., `"stop"`, `"tool_calls"`, `"length"`). |
| `timers` | `object` | Map of sub-operation timer names (`*_ms`) to accumulated milliseconds (float). |
| `counters` | `object` | Map of counter names to accumulated integer counts. |

---

## 4. `timers` and `counters` Accumulators

### `timers` (Sub-Operation Durations in Milliseconds)
`timers` tracks the time spent inside specific sub-steps of an operation. Values act as **accumulators**: if an operation performs multiple `db_save` calls (or retries an LLM/embedding call), `db_save_ms` is the sum of all of those calls.

```json
"timers": {
  "db_lookup_ms": 9.5,
  "llm_call_ms": 790.2,
  "db_save_ms": 3.0
}
```

**Standard `timers` keys:**
- `ttft_ms` — Time to first streamed token on an LLM call.
- `llm_call_ms` — Time spent waiting on LLM completion generation.
- `embedding_ms` — Time spent generating vector embeddings (or consolidating to vector memory).
- `db_lookup_ms` — Time spent querying SQLite or Qdrant.
- `db_save_ms` — Time spent writing/committing to SQLite or Qdrant.
- `exec_ms` — Time spent executing a subprocess or container command.

### `counters` (Integer Counts & Volumes)
`counters` tracks integer quantities processed or produced during the operation (such as database operations, record counts, token counts, and payload sizes in bytes).

```json
"counters": {
  "db_lookup": 1,
  "db_lookup_records": 8,
  "db_save": 1,
  "db_save_records": 1,
  "request_tokens": 412,
  "response_tokens": 96
}
```

**Standard `counters` keys:**
- `request_tokens`, `response_tokens` — Input and output token counts (from endpoint usage metadata if provided, or computed via the shared `AutoTokenizer`).
- `request_bytes`, `response_bytes` — UTF-8 byte lengths of the request and response payloads.
- `request_messages` — Number of messages in the LLM context window for `agent_turn`.
- `tool_calls` — Number of tool calls emitted by the assistant in `agent_turn`.
- `evicted_messages`, `evicted_tokens` — Messages and tokens compacted during `memory_compaction`.
- `promoted_records`, `recurring_records` — Candidate memories promoted during `subconscious_recall`.
- `db_lookup`, `db_lookup_records` — Number of database read queries and total records retrieved.
- `db_save`, `db_save_records` — Number of database write operations and total records written.
- `retries` — Number of retry attempts performed before succeeding or failing.
- `exit_code` — Process exit code for `workspace_exec`.

---

## 5. Example Log Entries

### Agent Turn (`operation: "llm_call"`, `type: "agent_turn"`)
```json
{
  "timestamp": "2026-10-26T20:24:15.123Z",
  "operation": "llm_call",
  "duration_ms": 4812.65,
  "error": 0,
  "agent": "Thunk",
  "thread": "mind",
  "component": "core",
  "type": "agent_turn",
  "model": "gemma4:31b",
  "exec_location": "local",
  "finish_reason": "tool_calls",
  "timers": {
    "llm_call_ms": 4809.1,
    "ttft_ms": 612.4
  },
  "counters": {
    "request_messages": 18,
    "request_bytes": 34910,
    "request_tokens": 8740,
    "response_tokens": 142,
    "response_bytes": 588,
    "tool_calls": 1
  }
}
```

### Subconscious Recall (`operation: "llm_call"`, `type: "subconscious_recall"`)
```json
{
  "timestamp": "2026-10-26T20:24:10.305Z",
  "operation": "llm_call",
  "duration_ms": 842.4,
  "error": 0,
  "agent": "Thunk",
  "thread": "social",
  "component": "spreading_activation",
  "type": "subconscious_recall",
  "model": "gemma4:e2b",
  "exec_location": "local",
  "timers": {
    "db_lookup_ms": 48.2,
    "db_save_ms": 3.1,
    "llm_call_ms": 790.5
  },
  "counters": {
    "db_lookup": 2,
    "db_lookup_records": 14,
    "promoted_records": 8,
    "recurring_records": 3,
    "db_save": 1,
    "db_save_records": 32,
    "request_tokens": 1120,
    "request_bytes": 4510,
    "response_tokens": 115,
    "response_bytes": 462
  }
}
```

---

## 6. Extending the Schema

Additional fields may be added to the top-level JSON object or inside `timers` / `counters` as new components or operations are introduced. When multiple components emit similar measurements, reuse existing field names from this specification and document any new shared fields here.
