import json
import os
from datetime import datetime
from typing import Dict, Any
from src.components.base import BaseComponent

class AuditLogComponent(BaseComponent):
    """
    Subscribes to all system events and writes a deterministic, 
    raw JSONL record of exactly what the system executed.
    """
    def __init__(self, log_dir: str = "logs", verbose: bool = False):
        self.log_dir = log_dir
        self.verbose = verbose
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a new log file for each session
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"session_{session_id}.jsonl")
        print(f"[AuditLogger] Writing session log to {self.log_file}")

    def _write_log(self, event_name: str, payload: Dict[str, Any]):
        """Helper to serialize and write an event to the JSONL file."""
        try:
            # We need to make sure the payload is JSON serializable
            # Pydantic objects or complex classes might choke json.dumps
            serializable_payload = {}
            for k, v in payload.items():
                if k == "agent":
                    if self.verbose:
                        # In verbose mode, we dump the entire working memory of the agent
                        # to see exactly what context it is operating on
                        try:
                            # Try to cleanly extract the messages without breaking on pydantic objects
                            messages = []
                            for msg in v.messages:
                                safe_msg = {"role": msg.get("role"), "content": msg.get("content")}
                                if "tool_calls" in msg and msg["tool_calls"]:
                                    safe_msg["tool_calls"] = [
                                        {"name": tc.function.name, "arguments": tc.function.arguments}
                                        for tc in msg["tool_calls"]
                                    ]
                                messages.append(safe_msg)
                            serializable_payload["agent_state"] = {"messages": messages}
                        except Exception:
                            serializable_payload["agent_state"] = "Error serializing agent state"
                    continue # Always skip serializing the raw python class object itself
                
                # Handle OpenAI message structures safely
                if k == "message" and isinstance(v, dict):
                    safe_msg = {
                        "role": v.get("role"),
                        "content": v.get("content")
                    }
                    if "tool_calls" in v and v["tool_calls"]:
                        safe_msg["tool_calls"] = [
                            {"name": tc.function.name, "arguments": tc.function.arguments}
                            for tc in v["tool_calls"]
                        ]
                    serializable_payload[k] = safe_msg
                elif k == "response":
                    # Serialize the raw OpenAI ChatCompletion response object
                    try:
                        serializable_payload[k] = v.model_dump()
                    except Exception:
                        serializable_payload[k] = "Error serializing raw response object"
                else:
                    serializable_payload[k] = v

            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "event": event_name,
                "payload": serializable_payload
            }
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            print(f"[AuditLogger Error] Failed to serialize event '{event_name}': {e}")

    async def on_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        # We don't want to log the context_window_check on every loop, it's too noisy
        if event_name in ["context_window_check", "agent_suspended"]:
            return
            
        # We only log before/after llm_call if verbose is True, because they are massive
        if event_name in ["before_llm_call", "after_llm_call"] and not self.verbose:
            return
            
        self._write_log(event_name, payload)
