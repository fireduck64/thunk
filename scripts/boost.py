#!./venv/bin/python3
import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure the script can import src modules when run from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config, set_config_path


def _read_boost_status(config: Dict[str, Any]) -> int:
    db_path = config.get("memory", {}).get("db_path")
    agent_name = config.get("agent", {}).get("name", "Agent")
    boost_cfg = config.get("boost", {})
    print(
        f"Agent: {agent_name} | [boost] enabled={bool(boost_cfg.get('enabled', bool(boost_cfg)))} "
        f"model={boost_cfg.get('model_name') or boost_cfg.get('model') or 'gemini-4-neon'} "
        f"(provider={boost_cfg.get('provider', 'gemini')})"
    )
    if not db_path or not os.path.exists(db_path):
        print(f"Memory DB not found at '{db_path}'.")
        return 0

    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        rows = conn.execute(
            "SELECT key, value FROM memory_settings WHERE key = 'boost_state' OR key LIKE '%:boost_state' ORDER BY key ASC"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("Active Boost Threads: none (all threads on base model)")
        return 0

    for key, raw_val in rows:
        tid = "main" if key == "boost_state" else key.split(":boost_state", 1)[0]
        try:
            state = json.loads(raw_val)
        except Exception:
            state = {"raw": raw_val}
        print(f"  - thread='{tid}': {json.dumps(state)}")
    return 0


def build_boost_payload(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "control": "boost",
        "action": "off" if args.off else "on",
        "thread": (args.thread or "all").strip(),
        "wake": not bool(args.no_wake),
    }
    if args.reason:
        payload["reason"] = args.reason.strip()
    if not args.off:
        if args.compactions is not None:
            payload["compactions"] = int(args.compactions)
        if args.turns is not None:
            payload["turns"] = int(args.turns)
    return payload


async def publish_boost_command(config: Dict[str, Any], payload: Dict[str, Any]) -> None:
    try:
        import aiomqtt
    except ImportError:
        print(
            "Error: aiomqtt not found. Run this script using ./venv/bin/python3 or ./scripts/boost.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    mqtt_cfg = config.get("mqtt", {})
    agent_slug = str(config.get("agent", {}).get("name", "thunk")).strip().lower()
    host = mqtt_cfg.get("host", "localhost")
    port = int(mqtt_cfg.get("port", 1883))
    topic = mqtt_cfg.get("inbox_topic", f"{agent_slug}/agent/inbox")

    encoded = json.dumps(payload).encode("utf-8")
    async with aiomqtt.Client(hostname=host, port=port) as client:
        await client.publish(topic, payload=encoded, qos=1, retain=False)
    print(f"Published boost command to '{topic}' ({host}:{port}): {json.dumps(payload)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trigger, cancel, or inspect Boost Mode on a running Thunk/Wisp agent via MQTT."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to the agent's config.toml")
    parser.add_argument(
        "-t",
        "--thread",
        type=str,
        default="all",
        help="Target thread ('all', 'mind', 'social', or 'main'; default: 'all')",
    )
    parser.add_argument(
        "--compactions",
        type=int,
        default=None,
        help="Override compaction sweep budget (omit to use [boost].default_compactions unless --turns is set)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Override turn budget (omit to use [boost].default_turns unless --compactions is set)",
    )
    parser.add_argument(
        "-r",
        "--reason",
        type=str,
        default="",
        help="Optional reason recorded with the boost activation",
    )
    parser.add_argument(
        "--off",
        action="store_true",
        help="Deactivate Boost Mode and return target thread(s) to their base model",
    )
    parser.add_argument(
        "--no-wake",
        action="store_true",
        help="Do not immediately wake suspended threads (apply on their next natural wake)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Read-only check of persisted boost status from memory.db without publishing to MQTT",
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.config:
        set_config_path(args.config)

    if not args.off and not args.status:
        if args.compactions is not None and args.compactions <= 0 and (args.turns is None or args.turns <= 0):
            parser.error("--compactions must be >= 1 unless a positive --turns is also provided.")
        if args.turns is not None and args.turns <= 0 and (args.compactions is None or args.compactions <= 0):
            parser.error("--turns must be >= 1 unless a positive --compactions is also provided.")

    config = load_config()
    if args.status:
        return _read_boost_status(config)

    payload = build_boost_payload(args)
    asyncio.run(publish_boost_command(config, payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
