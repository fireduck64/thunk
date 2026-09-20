import asyncio
import json
import aiomqtt
from src.core.agent import AgentCore
from src.utils.config import load_config

class MQTTAdapter:
    """
    Connects the AgentCore to an MQTT broker.
    Listens for messages on an inbox topic to wake the agent.
    Publishes the agent's explicit `send_to_operator` events to an outbox topic.
    """
    def __init__(self, agent: AgentCore):
        self.agent = agent
        self.config = load_config()
        
        # Load MQTT settings with defaults
        mqtt_cfg = self.config.get("mqtt", {})
        self.host = mqtt_cfg.get("host", "localhost")
        self.port = mqtt_cfg.get("port", 1883)
        self.inbox_topic = mqtt_cfg.get("inbox_topic", "thunk/agent/inbox")
        self.outbox_topic = mqtt_cfg.get("outbox_topic", "thunk/agent/outbox")
        
        self.client = None

    async def _on_send_to_operator(self, event_name: str, payload: dict):
        """Intercept explicit operator messages and publish via MQTT."""
        msg_content = payload.get("message", "")
        if msg_content and self.client:
            print(f"[MQTT] Publishing to {self.outbox_topic}")
            try:
                await self.client.publish(self.outbox_topic, msg_content.encode('utf-8'))
            except Exception as e:
                print(f"[MQTT Error] Failed to publish: {e}")

    async def start(self):
        """Connect to MQTT and start the listener loop."""
        self.agent.event_bus.subscribe("send_to_operator", self._on_send_to_operator)
        
        print(f"[MQTT] Connecting to broker at {self.host}:{self.port}...")
        try:
            async with aiomqtt.Client(self.host, port=self.port) as client:
                self.client = client
                print(f"[MQTT] Connected. Listening on '{self.inbox_topic}'")
                await client.subscribe(self.inbox_topic)
                
                async for message in client.messages:
                    payload = message.payload.decode('utf-8')
                    print(f"\n[MQTT Incoming] {message.topic}: {payload}")
                    
                    # Wake up the agent with the new message
                    await self.agent.resume(payload)
                    
        except aiomqtt.MqttError as error:
            print(f"[MQTT Error] Connection failed: {error}")
