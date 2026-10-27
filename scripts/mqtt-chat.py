#!./venv/bin/python3
import argparse
import asyncio
import json
import os
import sys
import aiomqtt

# Add project root to sys.path so we can import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import load_config, set_config_path


def build_presence_payload(user_display: str, status: str, dm_topic: str) -> bytes:
    return json.dumps(
        {
            "agent": user_display,
            "status": status,
            "dm_topic": dm_topic,
        }
    ).encode("utf-8")


def build_dm_payload(sender_display: str, text: str) -> bytes:
    return json.dumps(
        {
            "sender": sender_display,
            "text": text,
        }
    ).encode("utf-8")


def resolve_recipient_dm_topic(recipient: str, known_agents: dict, dm_template: str = "{recipient}/agent/dm") -> str:
    slug = recipient.strip().lstrip("@").lower()
    if slug in known_agents and known_agents[slug].get("dm_topic"):
        return known_agents[slug]["dm_topic"]
    return dm_template.format(recipient=slug)


async def chat_loop(username: str, initial_recipient: str = None):
    config = load_config()
    mqtt_cfg = config.get("mqtt", {})
    host = mqtt_cfg.get("host", "localhost")
    port = int(mqtt_cfg.get("port", 1883))

    user_display = username.strip()
    user_slug = user_display.lower()
    if not user_slug:
        raise ValueError("Username cannot be empty.")

    default_target = (
        initial_recipient
        or config.get("agent", {}).get("name", "thunk")
    ).strip().lstrip("@").lower()

    dm_topic = f"{user_slug}/agent/dm"
    presence_topic = f"{user_slug}/agent/presence"
    presence_wildcard = mqtt_cfg.get("presence_wildcard", "+/agent/presence")
    dm_template = mqtt_cfg.get("dm_topic_template", "{recipient}/agent/dm")
    client_id = f"thunk-mqtt-{user_slug}"

    known_agents: dict[str, dict] = {}
    state = {"recipient": default_target}

    offline_payload = build_presence_payload(user_display, "offline", dm_topic)
    online_payload = build_presence_payload(user_display, "online", dm_topic)

    def prompt_str() -> str:
        return f"[{user_display} -> {state['recipient']}]> "

    def print_agents():
        if not known_agents:
            print("\nNo MQTT agents discovered yet.")
            return
        print("\nDiscovered MQTT Agents:")
        for key in sorted(known_agents.keys()):
            info = known_agents[key]
            marker = " (you)" if key == user_slug else ""
            print(
                f"  - {info.get('agent', key)}{marker}: status={info.get('status', 'unknown')}, "
                f"dm_topic='{info.get('dm_topic')}'"
            )

    print(f"Connecting to MQTT broker at {host}:{port} as agent '{user_display}' (client_id='{client_id}')...")
    print(f"Listening for DMs on '{dm_topic}' and announcing presence on '{presence_topic}'.")
    print("Commands: '/agents' to list peers, '/to <agent>' to switch recipient, '@<agent> <msg>' for one-off DM, '/quit' to exit.\n")

    will = aiomqtt.Will(
        topic=presence_topic,
        payload=offline_payload,
        qos=1,
        retain=True,
    )

    try:
        async with aiomqtt.Client(
            host,
            port=port,
            identifier=client_id,
            clean_session=False,
            will=will,
        ) as client:
            await client.subscribe(dm_topic, qos=1)
            await client.subscribe(presence_wildcard, qos=1)
            await client.publish(presence_topic, online_payload, qos=1, retain=True)

            stop_event = asyncio.Event()

            async def listen():
                async for message in client.messages:
                    topic_str = str(message.topic)
                    payload_raw = message.payload.decode("utf-8", errors="replace")

                    if topic_str.endswith("/agent/presence"):
                        if not payload_raw.strip():
                            continue
                        topic_slug = topic_str.split("/", 1)[0].strip().lower()
                        try:
                            data = json.loads(payload_raw)
                            agent_name = data.get("agent") or topic_slug
                            status = data.get("status", "online")
                            peer_dm = data.get("dm_topic") or dm_template.format(recipient=topic_slug)
                        except Exception:
                            agent_name = topic_slug
                            status = payload_raw.strip() or "online"
                            peer_dm = dm_template.format(recipient=topic_slug)

                        key = agent_name.strip().lower()
                        prev_status = known_agents.get(key, {}).get("status")
                        known_agents[key] = {
                            "agent": agent_name,
                            "status": status,
                            "dm_topic": peer_dm,
                        }
                        if key != user_slug and prev_status is not None and prev_status != status:
                            print(f"\r\033[K\n[Presence] {agent_name} is now {status}.")
                            print(prompt_str(), end="", flush=True)
                        continue

                    if topic_str == dm_topic:
                        if not payload_raw.strip():
                            continue

                        if getattr(message, "retain", False):
                            try:
                                await client.publish(dm_topic, b"", qos=1, retain=True)
                            except Exception:
                                pass

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

                        print(f"\r\033[K\n[{sender}] {text}")
                        print(prompt_str(), end="", flush=True)

            async def get_input():
                loop = asyncio.get_event_loop()
                # Brief pause to allow initial retained presence messages to arrive before first prompt
                await asyncio.sleep(0.15)
                print_agents()
                print(prompt_str(), end="", flush=True)

                while not stop_event.is_set():
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                    if not line:
                        stop_event.set()
                        break

                    msg = line.strip()
                    if not msg:
                        print(prompt_str(), end="", flush=True)
                        continue

                    if msg in ("/quit", "/exit"):
                        stop_event.set()
                        break

                    if msg == "/agents":
                        print_agents()
                        print(prompt_str(), end="", flush=True)
                        continue

                    if msg.startswith("/to "):
                        new_target = msg[4:].strip().lstrip("@").lower()
                        if new_target:
                            state["recipient"] = new_target
                            print(f"Active recipient set to '{new_target}'.")
                        else:
                            print("Usage: /to <agent_name>")
                        print(prompt_str(), end="", flush=True)
                        continue

                    target = state["recipient"]
                    body = msg
                    if msg.startswith("@") and " " in msg:
                        target_part, remainder = msg[1:].split(" ", 1)
                        if target_part.strip() and remainder.strip():
                            target = target_part.strip().lower()
                            body = remainder.strip()

                    target_topic = resolve_recipient_dm_topic(target, known_agents, dm_template)
                    await client.publish(
                        target_topic,
                        build_dm_payload(user_display, body),
                        qos=1,
                        retain=False,
                    )
                    print(prompt_str(), end="", flush=True)

            listen_task = asyncio.create_task(listen())
            input_task = asyncio.create_task(get_input())
            await asyncio.wait([listen_task, input_task], return_when=asyncio.FIRST_COMPLETED)

            for t in (listen_task, input_task):
                if not t.done():
                    t.cancel()

            # Publish offline status cleanly before disconnecting
            try:
                await client.publish(presence_topic, offline_payload, qos=1, retain=True)
            except Exception:
                pass

    except aiomqtt.MqttError as e:
        print(f"\nMQTT Connection Error: {e}")
        print("Make sure you have an MQTT broker (like mosquitto) running.")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive peer-agent MQTT chat client following mqtt-standards.md")
    parser.add_argument(
        "-u",
        "--user",
        "--username",
        dest="username",
        type=str,
        required=True,
        help="Your agent username on MQTT (e.g., Sombus)",
    )
    parser.add_argument(
        "-r",
        "--recipient",
        type=str,
        default=None,
        help="Initial recipient agent name (defaults to [agent].name from config.toml)",
    )
    parser.add_argument("-c", "--config", type=str, help="Path to the TOML configuration file")
    args = parser.parse_args()
    if args.config:
        set_config_path(args.config)

    try:
        asyncio.run(chat_loop(username=args.username, initial_recipient=args.recipient))
    except KeyboardInterrupt:
        sys.exit(0)

