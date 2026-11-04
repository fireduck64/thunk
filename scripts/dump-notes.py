#!./venv/bin/python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add the project root to the path so we can import src.*
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.components.notes import StructuredNotesComponent
from src.utils.config import load_config, set_config_path


_PREFIX_ALIASES = {
    "prompt": "prompts",
    "prompt/": "prompts/",
    "assistent": "assistant",
    "assistent/": "assistant/",
}


def matches_prefix(key: str, raw_prefix: Optional[str]) -> bool:
    if not raw_prefix or not raw_prefix.strip():
        return True
    p = raw_prefix.strip().lstrip("/").lower()
    p = _PREFIX_ALIASES.get(p, p)
    k = (key or "").strip().lstrip("/").lower()
    if k.startswith(p):
        return True
    # Also match if user passed singular without trailing slash (e.g. 'prompt' matching 'prompts/...')
    if not p.endswith("s") and k.startswith(f"{p}s/"):
        return True
    return False


def dump_notes(db_path: str, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
    notes_comp = StructuredNotesComponent(db_path=db_path)
    all_notes = notes_comp.get_all_notes()
    results: List[Dict[str, Any]] = []
    for n in all_notes:
        key = n["key"]
        if not matches_prefix(key, prefix):
            continue
        content = n.get("content") or ""
        results.append(
            {
                "key": key,
                "summary": n.get("summary") or "",
                "content": content,
                "size_chars": len(content),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Dump structured notes from notes.db as JSON (with optional key prefix filtering)"
    )
    parser.add_argument("-c", "--config", type=str, help="Path to the TOML configuration file")
    parser.add_argument("-d", "--db", type=str, help="Direct path to notes.db SQLite file (overrides -c)")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default=None,
        help="Optional key prefix filter (e.g. 'assistant', 'prompts')",
    )
    parser.add_argument(
        "positional_prefix",
        nargs="?",
        default=None,
        help="Optional key prefix filter (e.g. 'assistant', 'prompt')",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Output one JSON record per line (JSONL) instead of a pretty-printed JSON array",
    )
    args = parser.parse_args()

    effective_prefix = args.prefix if args.prefix is not None else args.positional_prefix

    if args.db:
        db_path = args.db
    else:
        if args.config:
            set_config_path(args.config)
        config = load_config()
        db_path = config.get("notes", {}).get("db_path")
        if not db_path:
            print("Error: [notes].db_path is not defined in config.", file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(db_path):
        print(f"Error: Notes database '{db_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        records = dump_notes(db_path=db_path, prefix=effective_prefix)
        if args.jsonl:
            for rec in records:
                sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            sys.stdout.write(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
    except Exception as e:
        print(f"Error dumping notes from '{db_path}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
