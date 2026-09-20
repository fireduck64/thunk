import asyncio
import sys
import os
import aiomqtt

# Add project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config

async def chat_loop():
    config = load_config()
    host = config.get("mqtt", {}).get("host", "localhost")
    port = config.get("mqtt", {}).get("port", 1883)
    inbox = config.get("mqtt", {}).get("inbox_topic", "thunk/agent/inbox")
    outbox = config.get("mqtt", {}).get("outbox_topic", "thunk/agent/outbox")

    print(f"Connecting to MQTT broker at {host}:{port}")
    print("Type your messages and press Enter to send. Type '/quit' to exit.\n")

    try:
        async with aiomqtt.Client(host, port=port) as client:
            await client.subscribe(outbox)
            
            async def listen():
                async for message in client.messages:
                    payload = message.payload.decode('utf-8')
                    # Clear the current line, print the agent message, and redraw the prompt
                    print(f"\r\033[K\n[Agent] {payload}")
                    print("> ", end="", flush=True)

            async def get_input():
                loop = asyncio.get_event_loop()
                print("> ", end="", flush=True)
                while True:
                    # Run input in a thread so it doesn't block the asyncio event loop
                    msg = await loop.run_in_executor(None, sys.stdin.readline)
                    msg = msg.strip()
                    if msg == "/quit":
                        # We use sys.exit to kill both tasks
                        sys.exit(0)
                    if msg:
                        await client.publish(inbox, msg)
                    # The prompt will be redrawn by listen() when the agent replies, 
                    # but we also redraw it here if the user just pressed enter
                    print("> ", end="", flush=True)

            await asyncio.gather(listen(), get_input())
            
    except aiomqtt.MqttError as e:
        print(f"\nMQTT Connection Error: {e}")
        print("Make sure you have an MQTT broker (like mosquitto) running.")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        sys.exit(0)
