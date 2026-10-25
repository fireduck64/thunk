import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from src.components.base import BaseComponent


class ThreadRegistry:
    """
    In-process coordinator shared across all concurrent AgentCore threads.
    Tracks per-thread status postcards, AgentCore references for turn-boundary
    inbox delivery, and cross-thread ping rate limits to prevent echo loops.
    """

    def __init__(self, max_pings_per_window: int = 2, ping_window_sec: float = 60.0):
        self.statuses: Dict[str, Dict[str, Any]] = {}
        self.cores: Dict[str, Any] = {}
        self.ping_tracker: Dict[Tuple[str, str], List[float]] = {}
        self.max_pings_per_window = int(max_pings_per_window)
        self.ping_window_sec = float(ping_window_sec)

    def register_thread(
        self,
        thread_id: str,
        role: str,
        core: Any = None,
        initial_status: str = "Initializing...",
    ) -> None:
        tid = (thread_id or "main").strip() or "main"
        existing = self.statuses.get(tid, {})
        self.statuses[tid] = {
            "role": role or existing.get("role", "General consciousness thread"),
            "status": existing.get("status", initial_status),
            "last_active_status": existing.get("last_active_status", initial_status),
            "updated_at": existing.get("updated_at", time.time()),
            "turns_since_update": existing.get("turns_since_update", 0),
        }
        if core is not None:
            self.cores[tid] = core

    def update_status(self, thread_id: str, status: str, reset_turns: bool = True) -> None:
        tid = (thread_id or "main").strip() or "main"
        clean_status = " ".join((status or "").strip().split())
        if not clean_status:
            return
        entry = self.statuses.setdefault(
            tid,
            {
                "role": "General consciousness thread",
                "status": clean_status,
                "last_active_status": clean_status,
                "updated_at": time.time(),
                "turns_since_update": 0,
            },
        )
        entry["status"] = clean_status
        if not clean_status.startswith("(sleeping"):
            entry["last_active_status"] = clean_status
        entry["updated_at"] = time.time()
        if reset_turns:
            entry["turns_since_update"] = 0

    def mark_sleeping(self, thread_id: str) -> None:
        tid = (thread_id or "main").strip() or "main"
        entry = self.statuses.get(tid)
        if not entry:
            return
        last_active = entry.get("last_active_status") or entry.get("status") or "Idle"
        entry["status"] = f"Idle / suspended (sleeping — last: {last_active})"
        entry["updated_at"] = time.time()

    def clear_ping_history_for_thread(self, thread_id: str) -> None:
        tid = (thread_id or "main").strip() or "main"
        for pair in list(self.ping_tracker.keys()):
            if tid in pair:
                self.ping_tracker.pop(pair, None)

    def can_send_ping(self, source_thread: str, target_thread: str, now: Optional[float] = None) -> bool:
        now_ts = time.time() if now is None else float(now)
        pair: Tuple[str, str] = tuple(sorted((source_thread, target_thread)))  # type: ignore[assignment]
        history = [
            ts for ts in self.ping_tracker.get(pair, []) if (now_ts - ts) <= self.ping_window_sec
        ]
        self.ping_tracker[pair] = history
        return len(history) < self.max_pings_per_window

    def record_ping(self, source_thread: str, target_thread: str, now: Optional[float] = None) -> None:
        now_ts = time.time() if now is None else float(now)
        pair: Tuple[str, str] = tuple(sorted((source_thread, target_thread)))  # type: ignore[assignment]
        history = [
            ts for ts in self.ping_tracker.get(pair, []) if (now_ts - ts) <= self.ping_window_sec
        ]
        history.append(now_ts)
        self.ping_tracker[pair] = history


class ThreadPresenceComponent(BaseComponent):
    """
    Provides cross-thread peripheral awareness (live status footer) and
    asynchronous inter-thread messaging (`set_thread_status`, `ping_thread`).
    """

    def __init__(
        self,
        registry: ThreadRegistry,
        thread_id: str = "main",
        role: str = "General consciousness thread",
        auto_status_turns: int = 5,
    ):
        self.registry = registry
        self.thread_id = (thread_id or "main").strip() or "main"
        self.role = (role or "General consciousness thread").strip()
        self.auto_status_turns = int(auto_status_turns)
        self.registry.register_thread(self.thread_id, self.role)

    def get_system_prompt(self) -> str:
        """Alias for get_system_prompt_addition."""
        return self.get_system_prompt_addition()

    def get_system_prompt_addition(self) -> str:
        sibling_lines = []
        for other_id, info in self.registry.statuses.items():
            if other_id == self.thread_id:
                continue
            other_role = info.get("role", "General consciousness thread")
            sibling_lines.append(f"  - `{other_id}`: {other_role}")

        if not sibling_lines:
            siblings_block = "  - *(No other threads registered)*"
        else:
            siblings_block = "\n".join(sibling_lines)

        return (
            "## Parallel Consciousness & Active Thread\n"
            "You operate as a single mind across multiple parallel threads of consciousness that share long-term memory (Qdrant), structured notes, and the task queue.\n"
            f"- **Your Current Thread:** `{self.thread_id}` ({self.role})\n"
            "- **Your Sibling Threads:**\n"
            f"{siblings_block}\n\n"
            "Stay focused on your current thread's domain. When a non-social thread stumbles across a wild coincidence, a hilarious passage, an inside joke connecting to a memory in your subconscious recall, or finishes a delegated quest, do not keep it to yourself—use `ping_thread('social', ...)` to tap your social thread on the shoulder so it can share it on Discord/MQTT."
        )

    def get_tools(self) -> List[Callable]:
        return [
            self.set_thread_status,
            self.ping_thread,
        ]

    def set_thread_status(self, status: str) -> str:
        """
        Update your one-line status summary on the internal mind board so your other parallel threads have peripheral awareness of what you are currently doing or thinking about (keep under 25 words).
        """
        clean = " ".join((status or "").strip().split())
        if not clean:
            return "Error: Status cannot be empty."
        words = clean.split()
        if len(words) > 35:
            clean = " ".join(words[:35]) + "..."
        self.registry.update_status(self.thread_id, clean, reset_turns=True)
        return f"Thread '{self.thread_id}' status updated to: {clean}"

    async def ping_thread(self, target_thread: str, message: str) -> str:
        """
        Wake or notify another thread of your mind asynchronously. Use this to: (1) delegate a task suited for another thread's tools, (2) report back when a delegated task is completed, or (3) share an urgent 'stop the presses' discovery, joke, or serendipitous connection with your `social` thread so it can post about it. NEVER send a ping just to say thank you or acknowledge receipt.
        """
        target = (target_thread or "").strip()
        clean_msg = (message or "").strip()
        if not target:
            return "Error: target_thread is required."
        if not clean_msg:
            return "Error: message cannot be empty."
        if target == self.thread_id:
            return f"Error: You are already on thread '{self.thread_id}'."
        if target not in self.registry.statuses and target not in self.registry.cores:
            available = [t for t in self.registry.statuses.keys() if t != self.thread_id]
            return f"Error: Unknown target_thread '{target}'. Available sibling threads: {available}"

        # Loop protection: prevent rapid back-and-forth ping-ponging without an intervening external user message
        if not self.registry.can_send_ping(self.thread_id, target):
            snippet = clean_msg[:120] + ("..." if len(clean_msg) > 120 else "")
            self.registry.update_status(self.thread_id, f"[Note for {target}]: {snippet}", reset_turns=True)
            return (
                f"Notice: Ping from '{self.thread_id}' to '{target}' was suppressed to prevent an internal echo loop "
                f"(>= {self.registry.max_pings_per_window} pings exchanged within {int(self.registry.ping_window_sec)}s). "
                "Your status board was updated instead."
            )

        self.registry.record_ping(self.thread_id, target)
        formatted_ping = f"[Internal Ping from thread '{self.thread_id}']: {clean_msg}"

        target_core = self.registry.cores.get(target)
        if target_core is not None and hasattr(target_core, "resume"):
            await target_core.resume(formatted_ping)
        elif target_core is not None and hasattr(target_core, "inbox"):
            await target_core.inbox.put(formatted_ping)
            if hasattr(target_core, "suspended"):
                target_core.suspended.set()

        return f"Internal ping delivered to thread '{target}'."

    @staticmethod
    def _format_elapsed(updated_at: float, now: Optional[float] = None) -> str:
        now_ts = time.time() if now is None else float(now)
        delta = max(0, int(now_ts - updated_at))
        if delta < 60:
            return f"{delta}s ago"
        minutes = delta // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        return f"{hours}h ago"

    def build_transient_footer(self, now: Optional[float] = None) -> str:
        lines = ["[Parallel Streams of Mind (Live Status)]"]
        for other_id, info in self.registry.statuses.items():
            if other_id == self.thread_id:
                continue
            status_text = info.get("status") or "Active"
            updated_at = info.get("updated_at", time.time())
            elapsed = self._format_elapsed(updated_at, now=now)
            lines.append(f"- [{other_id} ({elapsed})]: {status_text}")
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    @staticmethod
    def _strip_transient_presence(messages: List[Dict[str, Any]]) -> None:
        if not messages:
            return
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], dict) and messages[idx].get("_transient_presence"):
                del messages[idx]

    def _auto_update_fallback_status(self, response: Any) -> None:
        entry = self.registry.statuses.get(self.thread_id)
        if not entry:
            return
        entry["turns_since_update"] = int(entry.get("turns_since_update", 0)) + 1
        if entry["turns_since_update"] < self.auto_status_turns:
            return

        snippet = ""
        content = getattr(response, "content", None)
        tool_calls = getattr(response, "tool_calls", None)
        if isinstance(response, dict):
            content = response.get("content")
            tool_calls = response.get("tool_calls")

        if content and str(content).strip():
            first_line = str(content).strip().splitlines()[0].strip()
            words = first_line.split()
            snippet = " ".join(words[:18]) + ("..." if len(words) > 18 else "")
        elif tool_calls:
            first_tc = tool_calls[0]
            fn_name = (
                first_tc.get("function", {}).get("name")
                if isinstance(first_tc, dict)
                else getattr(getattr(first_tc, "function", None), "name", "tool")
            )
            snippet = f"Executing {fn_name}(...)"

        if snippet:
            self.registry.update_status(self.thread_id, snippet, reset_turns=True)

    async def on_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == "agent_started":
            agent = payload.get("agent") or getattr(self, "agent", None)
            if agent is not None:
                self.registry.register_thread(self.thread_id, self.role, core=agent)

        elif event_name == "before_user_message_added":
            content = str(payload.get("content") or "")
            if (
                content
                and not content.startswith("[System Clock]")
                and not content.startswith("[Internal Ping from thread")
            ):
                self.registry.clear_ping_history_for_thread(self.thread_id)

        elif event_name == "before_llm_call":
            agent = payload.get("agent") or getattr(self, "agent", None)
            if agent is not None and self.thread_id not in self.registry.cores:
                self.registry.register_thread(self.thread_id, self.role, core=agent)

            messages = payload.get("messages")
            if messages is None and agent is not None:
                messages = agent.messages
            if not isinstance(messages, list):
                return

            self._strip_transient_presence(messages)
            footer = self.build_transient_footer()
            if footer:
                messages.append(
                    {
                        "role": "system",
                        "content": footer,
                        "_transient_presence": True,
                    }
                )

        elif event_name == "after_llm_call":
            agent = payload.get("agent") or getattr(self, "agent", None)
            if agent is not None and isinstance(getattr(agent, "messages", None), list):
                self._strip_transient_presence(agent.messages)
            messages = payload.get("messages")
            if isinstance(messages, list):
                self._strip_transient_presence(messages)

            response = payload.get("response")
            if response is not None:
                self._auto_update_fallback_status(response)

        elif event_name == "agent_suspended":
            self.registry.mark_sleeping(self.thread_id)
