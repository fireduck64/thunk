#!./venv/bin/python3
import argparse
import sys
import os
import asyncio

# Ensure the script can import src modules when run from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config, set_config_path

try:
    import aiomqtt
except ImportError:
    print("Error: aiomqtt not found. Ensure you are running this script within the correct virtual environment.")
    sys.exit(1)

async def send_notification():
    # Read all content from stdin
    content = sys.stdin.read().strip()
    if not content:
        print("No content provided on stdin. Exiting.")
        return

    # Load MQTT settings from config.toml
    try:
        config = load_config()
        mqtt_config = config.get("mqtt", {})
        host = mqtt_config.get("host", "localhost")
        port = mqtt_config.get("port", 1883)
        topic = mqtt_config.get("inbox_topic", "thunk/agent/inbox")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Publish message to MQTT broker
    try:
        async with aiomqtt.Client(hostname=host, port=port) as client:
            await client.publish(topic, payload=content.encode('utf-8'))
            print(f"Successfully sent notification to '{topic}' via MQTT.")
    except Exception as e:
        print(f"Failed to send notification via MQTT: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send stdin notification to the agent via MQTT")
    parser.add_argument("-c", "--config", type=str, help="Path to the TOML configuration file")
    args = parser.parse_args()
    if args.config:
        set_config_path(args.config)

    asyncio.run(send_notification())
