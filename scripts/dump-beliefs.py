#!./venv/bin/python3
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Set, Tuple

# Add the project root to the path so we can import src.*
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config, set_config_path


ALL_BELIEF_TABLES: Tuple[str, ...] = ("beliefs", "belief_edges", "belief_inbox")

_TABLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "all": ALL_BELIEF_TABLES,
    "belief": ("beliefs",),
    "beliefs": ("beliefs",),
    "edge": ("belief_edges",),
    "edges": ("belief_edges",),
    "belief_edge": ("belief_edges",),
    "belief_edges": ("belief_edges",),
    "inbox": ("belief_inbox",),
    "belief_inbox": ("belief_inbox",),
}


def resolve_tables(raw_table: Optional[str]) -> Tuple[str, ...]:
    if not raw_table or not raw_table.strip():
        return ALL_BELIEF_TABLES
    key = raw_table.strip().lower().replace("-", "_")
    if key not in _TABLE_ALIASES:
        valid = ", ".join(["all", "beliefs", "edges", "inbox"])
        raise ValueError(f"Unknown table filter '{raw_table}'. Choose from: {valid}")
    return _TABLE_ALIASES[key]


def _connect_ro(db_path: str) -> sqlite3.Connection:
    uri = f"{Path(os.path.abspath(db_path)).as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def _decode_evidence_refs(raw: Any) -> Any:
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def iter_belief_records(
    db_path: str,
    tables: Sequence[str] = ALL_BELIEF_TABLES,
) -> Iterator[Dict[str, Any]]:
    conn = _connect_ro(db_path)
    try:
        existing: Set[str] = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('beliefs', 'belief_edges', 'belief_inbox')"
            ).fetchall()
        }

        for tbl in tables:
            if tbl not in existing:
                continue

            if tbl == "beliefs":
                cur = conn.execute(
                    """
                    SELECT id, claim, epistemic_bucket, status, confidence,
                           revision_note, evidence_refs, activation_count,
                           created_at, updated_at, last_accessed_at
                    FROM beliefs
                    ORDER BY id ASC
                    """
                )
                for row in cur:
                    yield {
                        "table": "beliefs",
                        "id": row["id"],
                        "claim": row["claim"],
                        "epistemic_bucket": row["epistemic_bucket"],
                        "status": row["status"],
                        "confidence": row["confidence"],
                        "revision_note": row["revision_note"],
                        "evidence_refs": _decode_evidence_refs(row["evidence_refs"]),
                        "activation_count": row["activation_count"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "last_accessed_at": row["last_accessed_at"],
                    }

            elif tbl == "belief_edges":
                cur = conn.execute(
                    """
                    SELECT source_id, target_id, relation, weight,
                           created_at, updated_at
                    FROM belief_edges
                    ORDER BY source_id ASC, target_id ASC, relation ASC
                    """
                )
                for row in cur:
                    yield {
                        "table": "belief_edges",
                        "source_id": row["source_id"],
                        "target_id": row["target_id"],
                        "relation": row["relation"],
                        "weight": row["weight"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }

            elif tbl == "belief_inbox":
                cur = conn.execute(
                    """
                    SELECT id, thread_id, claim, stance, reason, processed, created_at
                    FROM belief_inbox
                    ORDER BY id ASC
                    """
                )
                for row in cur:
                    yield {
                        "table": "belief_inbox",
                        "id": row["id"],
                        "thread_id": row["thread_id"],
                        "claim": row["claim"],
                        "stance": row["stance"],
                        "reason": row["reason"],
                        "processed": bool(row["processed"]),
                        "created_at": row["created_at"],
                    }
    finally:
        conn.close()


def dump_beliefs_jsonl(
    db_path: str,
    tables: Sequence[str] = ALL_BELIEF_TABLES,
    out_stream=sys.stdout,
) -> int:
    count = 0
    for rec in iter_belief_records(db_path=db_path, tables=tables):
        out_stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
        count += 1
    out_stream.flush()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump belief tables (beliefs, belief_edges, belief_inbox) from memory.db as JSONL"
    )
    parser.add_argument("-c", "--config", type=str, help="Path to the TOML configuration file")
    parser.add_argument("-d", "--db", type=str, help="Direct path to memory.db SQLite file (overrides -c)")
    parser.add_argument(
        "-t",
        "--table",
        type=str,
        default=None,
        help="Optional table filter: all (default), beliefs, edges, inbox",
    )
    parser.add_argument(
        "positional_table",
        nargs="?",
        default=None,
        help="Optional table filter: all (default), beliefs, edges, inbox",
    )
    args = parser.parse_args()

    raw_table = args.table if args.table is not None else args.positional_table
    try:
        selected_tables = resolve_tables(raw_table)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    if args.db:
        db_path = args.db
    else:
        if args.config:
            set_config_path(args.config)
        config = load_config()
        db_path = (
            config.get("belief_graph", {}).get("db_path")
            or config.get("memory", {}).get("db_path")
        )
        if not db_path:
            print(
                "Error: Neither [belief_graph].db_path nor [memory].db_path is defined in config.",
                file=sys.stderr,
            )
            sys.exit(1)

    if not os.path.exists(db_path):
        print(f"Error: Belief database '{db_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        dump_beliefs_jsonl(db_path=db_path, tables=selected_tables, out_stream=sys.stdout)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
    except Exception as e:
        print(f"Error dumping beliefs from '{db_path}': {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
