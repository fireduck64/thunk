Metrics Logs

Each metric log file should be append only.  Only write new lines to the end of a file, no other modifications.

Each agent should have a new metric log each day.  As the data starts (in UTC) open a new file with the day's date and append the logs there.
On a restart, if such a file already exists, append to it for the day.

Each log entry must have the following fields:

 timestamp - When the operation ended in RFC 3339 in UTC with millisecond precision (2026-10-26T20:24:15.123Z)
 operation - The name of the operation done (low cardinality).  Examples: "db_query" not "db_query_for_books"
 duration_ms - How long the operation took in miliseconds - total clock time
 error - 0 or 1.  0 means the operation had no non-recoverable error.  1 means the operation failed.

Depending on the situation the following optional fields may apply:
 agent - name of the agent (e.g. "Thunk", "Wisp")
 thread - consciousness thread identifier (e.g. "main", "social", "mind")
 component - software component doing the operation (e.g. "core", "tiered_memory", "spreading_activation", "subconscious", "critic", "oracle", "vector_memory", "knowledge", "workspace")
 type - sub-classification of the operation:
   - For operation == "llm_call": "agent_turn", "memory_compaction", "subconscious_recall", "critic", or "oracle"
   - For operation == "tool_call": the tool function name (e.g. "read_file", "send_message", "run_workspace_command")
 model - model identifier used for LLM or embedding operations (e.g. "gemma4:31b", "gemma4:e2b", "nomic-embed-text")
 exec_location - for operation == "llm_call", indicates where inference ran: "local" or "cloud" (e.g. "cloud" for "oracle")
 finish_reason - for operation == "llm_call", the completion finish reason (e.g. "stop", "tool_calls", "length", "error")
 timers - times of sub-components of this operation.  Example:
   "timers": {
    "db_lookup_ms": 9.5,
    "simulation_ms": 2.0,
    "db_save_ms": 3.0
  }
  These should be accumulators.  For example, if we do multiple db_save sub operations as part of this operation, db_save_ms should be the sum of the times.
  Common timer keys: "ttft_ms", "llm_call_ms", "embedding_ms", "db_lookup_ms", "db_save_ms", "exec_ms".

 counters - counts of things done or handled by the operation. Example:
  "counters": {
    "db_lookup": 1,
    "db_lookup_records": 8,
    "db_save": 1,
    "db_save_records": 1
   }
   // In this example, it tells the story that we did one db lookup, retreiving 8 records and did one db save, saving one record.
   // counters may also be used for any integer things, like request bytes, result bytes, request tokens, result tokens.
   Common counter keys: "request_tokens", "response_tokens", "request_bytes", "response_bytes", "request_messages", "tool_calls", "evicted_messages", "evicted_tokens", "promoted_records", "recurring_records", "db_lookup", "db_lookup_records", "db_save", "db_save_records", "retries".


Additional fields may be added onto the base json structure or nested as makes sense.  If multiple components are outputting the same sort of data, try to keep the naming consistent.  In that case, add them as optional fields to this file to help keep track.



