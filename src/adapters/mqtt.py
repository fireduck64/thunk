import asyncio
import json
import aiomqtt
from src.core.agent import AgentCore
from src.utils.config import load_config

class MQTTAdapter:
    """
    Connects the AgentCore to an MQTT broker.
    - Listens on a system inbox topic (`<agent>/agent/inbox`) for unformatted system events (clock, news).
    - Listens on a direct message topic (`<agent>/agent/dm`) for JSON agent-to-agent messages (`{"sender", "text"}`).
    - Publishes and subscribes to `+/agent/presence` with retained LWT so agents can discover peers.
    """
    def __init__(self, agent: AgentCore):
        self.agent = agent
        self.config = load_config()
        self.agent_name = getattr(agent, "name", None) or self.config.get("agent", {}).get("name", "Thunk")
        self.agent_slug = self.agent_name.strip().lower()

        # Load MQTT settings with defaults
        mqtt_cfg = self.config.get("mqtt", {})
        self.host = mqtt_cfg.get("host", "localhost")
        self.port = mqtt_cfg.get("port", 1883)
        self.inbox_topic = mqtt_cfg.get("inbox_topic", f"{self.agent_slug}/agent/inbox")
        self.dm_topic = mqtt_cfg.get("dm_topic", f"{self.agent_slug}/agent/dm")
        self.dm_topic_template = mqtt_cfg.get("dm_topic_template", "{recipient}/agent/dm")
        self.presence_topic = mqtt_cfg.get("presence_topic", f"{self.agent_slug}/agent/presence")
        self.presence_wildcard = mqtt_cfg.get("presence_wildcard", "+/agent/presence")
        self.outbox_topic = mqtt_cfg.get("outbox_topic", f"{self.agent_slug}/agent/outbox")
        self.client_id = mqtt_cfg.get("client_id", f"thunk-mqtt-{self.agent_slug}")

        self.client = None
        self.known_agents: dict[str, dict] = {}
        self.agent.mqtt_adapter = self

    def resolve_dm_topic(self, recipient: str) -> str:
        """Resolves a recipient agent name to its MQTT direct-message topic."""
        slug = recipient.strip().lstrip("@").lower()
        if slug in self.known_agents and self.known_agents[slug].get("dm_topic"):
            return self.known_agents[slug]["dm_topic"]
        return self.dm_topic_template.format(recipient=slug)

    async def _on_send_to_operator(self, event_name: str, payload: dict):
        """Intercept explicit send_message events and publish via MQTT if requested."""
        medium = payload.get("medium", "all").lower()
        if medium not in ["mqtt", "all"]:
            return

        msg_content = payload.get("message", "")
        recipient = (payload.get("recipient") or "").strip()

        if medium == "mqtt" and not recipient:
            payload["error"] = (
                "Recipient is required when sending via 'mqtt' (e.g., recipient='wisp'). "
                "Use list_mqtt_agents to see available agents."
            )
            return

        if not msg_content:
            return

        if not self.client:
            if medium == "mqtt":
                payload["error"] = "MQTT client is not currently connected."
            return

        if recipient:
            target_topic = self.resolve_dm_topic(recipient)
            dm_envelope = json.dumps({"sender": self.agent_name, "text": msg_content})
            print(f"[MQTT] Sending DM to '{recipient}' on topic '{target_topic}'")
            try:
                await self.client.publish(
                    target_topic,
                    dm_envelope.encode("utf-8"),
                    qos=1,
                    retain=False,
                )
                payload["mqtt_recipient"] = recipient
                payload["mqtt_topic"] = target_topic
            except Exception as e:
                print(f"[MQTT Error] Failed to publish DM: {e}")
                payload["error"] = str(e)
        elif medium == "all" and self.outbox_topic:
            print(f"[MQTT] Publishing broadcast to {self.outbox_topic}")
            try:
                await self.client.publish(self.outbox_topic, msg_content.encode("utf-8"), qos=1)
            except Exception as e:
                print(f"[MQTT Error] Failed to publish to outbox: {e}")

    async def _handle_message(self, topic_str: str, payload_raw: str, retain: bool = False):
        """Processes an incoming MQTT message from presence, DM, or system inbox topics."""
        # 1. Presence discovery topic (+/agent/presence)
        if topic_str.endswith("/agent/presence"):
            if not payload_raw.strip():
                return
            topic_slug = topic_str.split("/", 1)[0].strip().lower()
            try:
                data = json.loads(payload_raw)
                agent_display = data.get("agent") or topic_slug
                status = data.get("status", "online")
                dm_topic = data.get("dm_topic") or self.dm_topic_template.format(recipient=topic_slug)
            except Exception:
                agent_display = topic_slug
                status = payload_raw.strip() or "online"
                dm_topic = self.dm_topic_template.format(recipient=topic_slug)

            key = agent_display.strip().lower()
            self.known_agents[key] = {
                "agent": agent_display,
                "status": status,
                "dm_topic": dm_topic,
            }
            return

        # 2. Direct Message topic (<agent>/agent/dm)
        if topic_str == self.dm_topic:
            if not payload_raw.strip():
                return

            print(f"\n[MQTT Incoming DM] {topic_str}: {payload_raw}")

            try:
                data = json.loads(payload_raw)
                if isinstance(data, dict):
                    sender = data.get("sender") or "Unknown"
                    text = data.get("text") if "text" in data else payload_raw
                else:
                    sender = "Unknown"
                    text = payload_raw
            except Exception:
                sender = "Unknown"
                text = payload_raw

            sender_key = str(sender).strip().lower()
            if sender_key and sender_key != "unknown" and sender_key not in self.known_agents:
                self.known_agents[sender_key] = {
                    "agent": str(sender).strip(),
                    "status": "online",
                    "dm_topic": self.dm_topic_template.format(recipient=sender_key),
                }

            formatted_payload = f"[Via MQTT | DM from {sender}] {text}"
            await self.agent.resume(formatted_payload)
            return

        # 3. System Inbox topic (<agent>/agent/inbox)
        if topic_str == self.inbox_topic:
            if not payload_raw.strip():
                return

            print(f"\n[MQTT Incoming System] {topic_str}: {payload_raw}")
            formatted_payload = f"[Via MQTT | System] {payload_raw}"
            await self.agent.resume(formatted_payload)

    async def start(self):
        """Connect to MQTT and start the listener loop with auto-reconnect."""
        self.agent.event_bus.subscribe("send_to_operator", self._on_send_to_operator)

        offline_will_payload = json.dumps(
            {
                "agent": self.agent_name,
                "status": "offline",
                "dm_topic": self.dm_topic,
            }
        ).encode("utf-8")
        online_presence_payload = json.dumps(
            {
                "agent": self.agent_name,
                "status": "online",
                "dm_topic": self.dm_topic,
            }
        ).encode("utf-8")

        while True:
            print(f"[MQTT] Connecting to broker at {self.host}:{self.port} as '{self.client_id}'...")
            try:
                will = aiomqtt.Will(
                    topic=self.presence_topic,
                    payload=offline_will_payload,
                    qos=1,
                    retain=True,
                )
                async with aiomqtt.Client(
                    self.host,
                    port=self.port,
                    identifier=self.client_id,
                    clean_session=False,
                    will=will,
                ) as client:
                    self.client = client
                    print(
                        f"[MQTT] Connected. Subscribing to system inbox '{self.inbox_topic}', "
                        f"DM topic '{self.dm_topic}', and presence '{self.presence_wildcard}'."
                    )
                    await client.subscribe(self.inbox_topic, qos=1)
                    await client.subscribe(self.dm_topic, qos=1)
                    await client.subscribe(self.presence_wildcard, qos=1)
                    await client.publish(
                        self.presence_topic,
                        online_presence_payload,
                        qos=1,
                        retain=True,
                    )

                    async for message in client.messages:
                        payload_raw = message.payload.decode("utf-8", errors="replace")
                        await self._handle_message(
                            str(message.topic),
                            payload_raw,
                            retain=bool(getattr(message, "retain", False)),
                        )

            except aiomqtt.MqttError as error:
                print(f"[MQTT Error] Connection failed or dropped: {error}. Reconnecting in 5 seconds...")
                self.client = None
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                print("[MQTT] Shutting down adapter.")
                if self.client:
                    try:
                        await self.client.publish(
                            self.presence_topic,
                            offline_will_payload,
                            qos=1,
                            retain=True,
                        )
                    except Exception:
                        pass
                break
            except Exception as e:
                print(f"[MQTT Error] Unexpected error: {e}. Reconnecting in 5 seconds...")
                self.client = None
                await asyncio.sleep(5)
