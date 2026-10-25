#!./venv/bin/python3
import argparse
import sys
import json

def format_log(filepath, thread_filter=None):
    print("="*80)
    print(f"📖 THUNK LOG EXTRACTOR")
    print(f"📄 File: {filepath}")
    if thread_filter:
        print(f"🧵 Thread Filter: {thread_filter}")
    print("="*80)

    records = []
    threads_seen = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            tid = data.get("thread_id", "main")
            threads_seen.add(tid)
            records.append(data)

    show_thread_badge = bool(thread_filter) or len(threads_seen) > 1 or (threads_seen and threads_seen != {"main"})

    for data in records:
        tid = data.get("thread_id", "main")
        if thread_filter and tid != thread_filter:
            continue

        badge = f"[{tid}] " if show_thread_badge else ""
        event = data.get("event")
        payload = data.get("payload", {})
        
        if event == "message_added":
            msg = payload.get("message", {})
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            
            if role == "system" and content and content.startswith("[Internal Feeling]:"):
                print(f"\n🧠 {badge}[CRITIC] {content[20:]}")
            elif role == "user" and content:
                print(f"\n👤 {badge}[USER / EVENT]\n{content}")
            elif role == "assistant":
                if content:
                    print(f"\n🤖 {badge}[AGENT MONOLOGUE]\n{content}")
                if tool_calls:
                    for tc in tool_calls:
                        # Handle different serialization formats (nested function or flat)
                        if "function" in tc:
                            name = tc["function"].get("name")
                            args = tc["function"].get("arguments")
                        else:
                            name = tc.get("name")
                            args = tc.get("arguments")
                        print(f"\n🛠️  {badge}[TOOL CALL] {name}({args})")
            elif role == "tool":
                pass # Handled by tool_executed to get better names/results
        
        elif event == "tool_executed":
            name = payload.get("name")
            args = payload.get("args", {})
            result = str(payload.get("result", ""))
            
            if name in ("send_message", "send_message_to_operator"):
                medium = args.get("medium", "all")
                message = args.get("message", "")
                print(f"\n📢 {badge}[OUTBOUND -> {medium.upper()}]\n{message}")
            else:
                if len(result) > 250:
                    result = result[:250] + "... [TRUNCATED]"
                print(f"   ↳ {badge}[RESULT] {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract readable monologue from a Thunk JSONL session log")
    parser.add_argument("-t", "--thread", type=str, default=None, help="Filter output to a specific thread_id (e.g. social, reader, worker)")
    parser.add_argument("logfile", type=str, help="Path to the session JSONL log file")
    args = parser.parse_args()
    format_log(args.logfile, thread_filter=args.thread)

