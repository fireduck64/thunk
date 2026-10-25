import json
import os
from datetime import datetime
from typing import Dict, Any
from src.components.base import BaseComponent
from src.utils.config import load_config

class AuditLogComponent(BaseComponent):
    """
    Subscribes to all system events and writes a deterministic, 
    raw JSONL record of exactly what the system executed.
    """
    def __init__(
        self,
        log_dir: str = None,
        verbose: bool = None,
        thread_id: str = "main",
        log_file: str = None,
    ):
        self.thread_id = (thread_id or "main").strip() or "main"
        if log_dir is None:
            config = load_config()
            logging_cfg = config.get("logging", {})
            if not log_file and ("log_dir" not in logging_cfg or not logging_cfg["log_dir"]):
                raise ValueError("AuditLogComponent requires 'log_dir' to be explicitly defined in config under [logging].log_dir")
            self.log_dir = logging_cfg.get("log_dir") or (os.path.dirname(os.path.abspath(log_file)) if log_file else ".")
            if verbose is None:
                self.verbose = logging_cfg.get("verbose", False)
            else:
                self.verbose = verbose
        else:
            self.log_dir = log_dir
            self.verbose = verbose if verbose is not None else False

        if log_file:
            self.log_file = log_file
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
        else:
            os.makedirs(self.log_dir, exist_ok=True)
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(self.log_dir, f"session_{session_id}.jsonl")
        print(f"[AuditLogger:{self.thread_id}] Writing session log to {self.log_file}")

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
                elif k == "messages" and isinstance(v, list):
                    safe_messages = []
                    for msg in v:
                        safe_msg = {"role": msg.get("role"), "content": msg.get("content")}
                        if "tool_calls" in msg and msg["tool_calls"]:
                            safe_msg["tool_calls"] = [
                                {"name": tc.function.name, "arguments": tc.function.arguments}
                                for tc in msg["tool_calls"]
                            ]
                        safe_messages.append(safe_msg)
                    serializable_payload[k] = safe_messages
                elif k == "response":
                    # Serialize the reconstructed ChatCompletionMessage object
                    try:
                        serializable_payload[k] = v.model_dump()
                    except Exception:
                        serializable_payload[k] = "Error serializing raw response object"
                else:
                    serializable_payload[k] = v

            agent_obj = payload.get("agent") or getattr(self, "agent", None)
            resolved_thread_id = (
                payload.get("thread_id")
                or getattr(agent_obj, "thread_id", None)
                or self.thread_id
                or "main"
            )
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "thread_id": resolved_thread_id,
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
