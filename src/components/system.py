import asyncio
from typing import List, Callable, Dict, Any
from src.components.base import BaseComponent
from src.utils.config import load_config

class SystemComponent(BaseComponent):
    """
    Provides the agent with explicit control over its lifecycle and communication.
    Separates the agent's internal monologue from what the operator actually sees.
    """
    def _is_boost_enabled(self) -> bool:
        agent = getattr(self, "agent", None)
        if agent is not None and hasattr(agent, "boost_enabled"):
            return bool(agent.boost_enabled)
        boost_cfg = load_config().get("boost", {})
        return bool(boost_cfg.get("enabled", bool(boost_cfg)))

    def get_system_prompt_addition(self) -> str:
        prompt = (
            "--- COMMUNICATION & LIFECYCLE ---\n"
            "Your standard text output is your INTERNAL MONOLOGUE (thoughts). Nobody outside can see it.\n"
            "To communicate externally (with the human operator or with peer agents), you MUST use the `send_message` or `react_to_message` tools.\n"
            "If replying to a specific message, set `medium` to match the medium the message arrived on (e.g., 'discord' or 'mqtt'). "
            "On Discord, you can optionally pass `reply_to_message_id` to `send_message` to create a threaded Discord reply to a specific message, "
            "and on both Discord and MQTT you can attach an image or file from any configured library (e.g., `attachment_library='workspace'`, `attachment_path='plot.png'`).\n"
            "On MQTT, you can send direct messages to another agent by setting `medium='mqtt'` and `recipient='<agent_name>'` (e.g., `recipient='wisp'` or `recipient='thunk'`). "
            "Use `list_mqtt_agents` to discover available MQTT agents and their status. Incoming `[Via MQTT | System]` messages are unformatted system broadcasts (clock, daily news); incoming `[Via MQTT | DM from <sender>]` messages are direct messages from another agent.\n"
            "You can also use `react_to_message(message_id, emoji)` with a standard Unicode emoji (e.g., '👍', '🔥', '🤔', '⚡') to react to any Discord message.\n"
            "Messages on Discord are tagged with `[Operator]` for human users or `[APP/Agent]` for peer AI agents.\n"
            "When conversing with another `[APP/Agent]`, engage substantively—exchange ideas, ask sharp questions, or challenge assumptions—but do NOT get stuck in endless polite sign-offs or repetitive agreement loops. Once a conversation reaches a natural conclusion, prefer using `react_to_message` for a lightweight acknowledgment (or simply returning to your own work) without needing the last word. Incoming `[Via Discord | Reaction]` events are informational and do not require a reply.\n"
            "You must only use the `suspend_until_next_event` tool as a last resort when you have absolutely zero tasks in your queue, no books left to explore, and require external input to proceed. Prefer finding new autonomous work over suspending your execution."
        )
        if self._is_boost_enabled():
            agent = getattr(self, "agent", None)
            boost_model = getattr(agent, "boost_model", None) or load_config().get("boost", {}).get("model_name", "gemini-4-neon")
            prompt += (
                f"\nIf you notice yourself stuck in a rut, spinning in a tool/debugging loop, or tackling an unusually demanding reasoning problem, "
                f"you can call `activate_boost_mode(reason=..., target_thread='self')` (or `'mind'`, `'social'`, `'all'`) to temporarily switch onto a higher-performance cloud model (`{boost_model}`)."
            )
        return prompt

    def get_tools(self) -> List[Callable]:
        tools: List[Callable] = [
            self.send_message,
            self.list_mqtt_agents,
            self.react_to_message,
            self.suspend_until_next_event,
        ]
        if self._is_boost_enabled():
            tools.append(self.activate_boost_mode)
        return tools

    async def send_message(
        self,
        message: str = "",
        medium: str = "all",
        reply_to_message_id: str = "",
        recipient: str = "",
        attachment_library: str = "",
        attachment_path: str = "",
    ) -> str:
        """
        Sends an external message via 'discord', 'mqtt', or 'all'.
        When medium is 'mqtt', specify 'recipient' (the target agent's name, e.g. 'wisp' or 'thunk').
        When medium is 'discord', optionally provide 'reply_to_message_id' to reply to a specific Discord message ID,
        and optionally provide 'attachment_library' and 'attachment_path' to attach a file or image from a library (e.g. 'workspace' or 'discord_file').
        """
        payload: Dict[str, Any] = {
            "message": message or "",
            "medium": medium,
            "reply_to_message_id": str(reply_to_message_id).strip() if reply_to_message_id else "",
            "recipient": str(recipient).strip() if recipient else "",
            "attachment_library": str(attachment_library).strip() if attachment_library else "",
            "attachment_path": str(attachment_path).strip() if attachment_path else "",
        }
        await self.agent.event_bus.publish("send_to_operator", payload)
        if "error" in payload:
            return f"Error sending message via {medium}: {payload['error']}"
        if "sent_message_id" in payload:
            att_info = f", attached='{payload['attached_file']}'" if payload.get("attached_file") else ""
            return f"Message sent successfully via {medium} (Discord message_id={payload['sent_message_id']}{att_info})."
        if "mqtt_recipient" in payload:
            att_info = f", attached='{payload['attached_file']}'" if payload.get("attached_file") else ""
            return f"Direct message sent successfully via mqtt to '{payload['mqtt_recipient']}' ({payload.get('mqtt_topic')}{att_info})."
        return f"Message sent successfully via {medium}."

    def list_mqtt_agents(self) -> str:
        """
        Lists all known peer agents discovered via MQTT presence (+/agent/presence),
        including their online/offline status and direct-message topic.
        """
        adapter = getattr(self.agent, "mqtt_adapter", None) if hasattr(self, "agent") else None
        if adapter is None:
            return "MQTT adapter is not attached."
        if not adapter.known_agents:
            return "No MQTT agents have announced presence yet."

        lines = ["Known MQTT Agents:"]
        for key in sorted(adapter.known_agents.keys()):
            info = adapter.known_agents[key]
            name = info.get("agent", key)
            status = info.get("status", "unknown")
            dm_topic = info.get("dm_topic", f"{key}/agent/dm")
            lines.append(f"- {name} (recipient='{key}', status={status}, dm_topic='{dm_topic}')")
        return "\n".join(lines)

    async def react_to_message(self, message_id: str, emoji: str) -> str:
        """
        Adds an emoji reaction (e.g., '👍', '🔥', '🤔') to a specific Discord message by its message_id.
        """
        payload: Dict[str, Any] = {
            "message_id": str(message_id).strip(),
            "emoji": emoji.strip()
        }
        await self.agent.event_bus.publish("react_to_message", payload)
        if "error" in payload:
            return f"Error reacting to message {message_id}: {payload['error']}"
        return payload.get("status", f"Reacted with {emoji} to message {message_id}.")

    async def suspend_until_next_event(self) -> str:
        """
        Suspends your execution loop. Use this when you are done working and need to wait 
        for an external reply or for a new background event to trigger.
        """
        if not self.agent.inbox.empty():
            return "A new external message arrived in your inbox; skipping suspension."

        thread_id = getattr(self.agent, "thread_id", "main")
        print(f"[System:{thread_id}] Agent requested explicit suspension. Waiting for next event...")
        self.agent.suspended.clear()
        await self.agent.event_bus.publish("agent_suspended", {"agent": self.agent})
        await self.agent.suspended.wait()
        return "You were woken up by a new event or message."

    def _resolve_target_cores(self, target_thread: str) -> tuple[List[Any], str]:
        own_core = getattr(self, "agent", None)
        if own_core is None:
            return [], "Error: SystemComponent is not attached to an AgentCore."

        registry = getattr(own_core, "thread_registry", None)
        if registry is None:
            for comp in getattr(own_core, "components", []) or []:
                reg = getattr(comp, "registry", None)
                if reg is not None and hasattr(reg, "cores"):
                    registry = reg
                    break

        cores_map: Dict[str, Any] = {}
        if registry is not None and getattr(registry, "cores", None):
            cores_map.update(registry.cores)
        own_tid = getattr(own_core, "thread_id", "main") or "main"
        cores_map.setdefault(own_tid, own_core)

        raw_target = (target_thread or "self").strip().lower()
        if raw_target in ("", "self", "this", "current") or raw_target == own_tid.lower():
            return [own_core], ""
        if raw_target in ("all", "*", "both"):
            return list(cores_map.values()), ""

        for tid, core in cores_map.items():
            if tid.lower() == raw_target:
                return [core], ""

        available = ", ".join(sorted(cores_map.keys()))
        return [], (
            f"Error: Unknown target_thread '{target_thread}'. "
            f"Available threads: {available} (or 'self' / 'all')."
        )

    async def activate_boost_mode(
        self,
        reason: str = "",
        target_thread: str = "self",
    ) -> str:
        """
        Temporarily switches this thread (or 'mind', 'social', or 'all') onto the configured
        high-performance boost model (e.g., gemini-4-neon) to break out of a loop, rut, or tough problem.

        Args:
            reason: Brief explanation of why boost mode is needed right now.
            target_thread: Which thread to boost ('self', 'mind', 'social', or 'all'). Defaults to 'self'.
        """
        own_core = getattr(self, "agent", None)
        targets, err = self._resolve_target_cores(target_thread)
        if err:
            return err

        own_tid = getattr(own_core, "thread_id", "main") or "main"
        summaries: List[str] = []
        for core in targets:
            res = core.activate_boost(
                reason=reason,
                triggered_by=f"agent:{own_tid}",
            )
            if not res.get("ok"):
                return f"Error activating Boost Mode on thread '{core.thread_id}': {res.get('error')}"
            summaries.append(res["summary"])
            if core is not own_core:
                await core.resume(f"[System Notice] {res['summary']}")

        return "\n".join(summaries)
