#!/usr/bin/env python3
import sys
import json

def format_log(filepath):
    print("="*80)
    print(f"📖 THUNK LOG EXTRACTOR")
    print(f"📄 File: {filepath}")
    print("="*80)
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except:
                continue
                
            event = data.get("event")
            payload = data.get("payload", {})
            
            if event == "message_added":
                msg = payload.get("message", {})
                role = msg.get("role")
                content = msg.get("content")
                tool_calls = msg.get("tool_calls")
                
                if role == "system" and content and content.startswith("[Internal Feeling]:"):
                    print(f"\n🧠 [CRITIC] {content[20:]}")
                elif role == "user" and content:
                    print(f"\n👤 [USER / EVENT]\n{content}")
                elif role == "assistant":
                    if content:
                        print(f"\n🤖 [AGENT MONOLOGUE]\n{content}")
                    if tool_calls:
                        for tc in tool_calls:
                            # Handle different serialization formats (nested function or flat)
                            if "function" in tc:
                                name = tc["function"].get("name")
                                args = tc["function"].get("arguments")
                            else:
                                name = tc.get("name")
                                args = tc.get("arguments")
                            print(f"\n🛠️  [TOOL CALL] {name}({args})")
                elif role == "tool":
                    pass # Handled by tool_executed to get better names/results
            
            elif event == "tool_executed":
                name = payload.get("name")
                args = payload.get("args", {})
                result = str(payload.get("result", ""))
                
                if name == "send_message_to_operator":
                    medium = args.get("medium", "all")
                    message = args.get("message", "")
                    print(f"\n📢 [OUTBOUND -> {medium.upper()}]\n{message}")
                else:
                    if len(result) > 250:
                        result = result[:250] + "... [TRUNCATED]"
                    print(f"   ↳ [RESULT] {result}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./extract-monolog.sh <logfile>")
        sys.exit(1)
    format_log(sys.argv[1])
