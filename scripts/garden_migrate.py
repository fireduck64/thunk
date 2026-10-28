#!./venv/bin/python3
import argparse
from datetime import datetime
import json
import math
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
from src.utils.config import load_config, set_config_path


def _normalize_vec(vec: Any) -> Optional[List[float]]:
    if vec is None:
        return None
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if not isinstance(vec, list) or not vec:
        return None
    floats = [float(x) for x in vec]
    norm = math.sqrt(sum(x * x for x in floats))
    if norm <= 1e-12:
        return floats
    return [x / norm for x in floats]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def migrate_agent_memory(
    client: QdrantClient,
    collection_name: str,
    sqlite_db_path: Optional[str] = None,
    backup_path: Optional[str] = None,
    dedup_threshold: float = 0.92,
    dry_run: bool = False,
) -> Dict[str, int]:
    # 1. Scroll all points with payloads and vectors
    all_points = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break
        all_points.extend(records)
        if next_offset is None:
            break
        offset = next_offset

    # 2. Write atomic JSONL backup if requested
    if backup_path:
        os.makedirs(os.path.dirname(os.path.abspath(backup_path)), exist_ok=True)
        tmp_backup = f"{backup_path}.tmp"
        with open(tmp_backup, "w", encoding="utf-8") as f:
            for pt in all_points:
                payload = dict(pt.payload or {})
                vec = getattr(pt, "vector", None)
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                rec = {
                    "id": str(pt.id),
                    "collection": collection_name,
                    "timestamp": payload.get("timestamp"),
                    "text": payload.get("text", ""),
                    "payload": payload,
                    "vector": vec,
                }
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp_backup, backup_path)

    # 3. Separate Consolidated Memory Block vectors from deliberate memories
    consolidated_ids: List[str] = []
    deliberate_points: List[Any] = []
    for pt in all_points:
        payload = dict(pt.payload or {})
        text = str(payload.get("text") or "").strip()
        if text.startswith("Consolidated Memory Block"):
            consolidated_ids.append(str(pt.id))
        else:
            deliberate_points.append(pt)

    # 4. Greedy cosine deduplication (sim >= dedup_threshold) on deliberate memories
    # Sort from longest/newest to shortest/oldest so the richest record is kept
    def _sort_key(pt: Any):
        payload = dict(pt.payload or {})
        text = str(payload.get("text") or "")
        ts = str(payload.get("timestamp") or "")
        return (len(text), ts)

    deliberate_points.sort(key=_sort_key, reverse=True)

    kept_ids: List[str] = []
    kept_vecs: List[List[float]] = []
    dedup_deleted_ids: List[str] = []

    for pt in deliberate_points:
        pid = str(pt.id)
        vec = _normalize_vec(getattr(pt, "vector", None))
        if vec is None:
            kept_ids.append(pid)
            continue

        is_dup = False
        for kvec in kept_vecs:
            if len(kvec) == len(vec) and _dot(vec, kvec) >= dedup_threshold:
                is_dup = True
                break
        if is_dup:
            dedup_deleted_ids.append(pid)
        else:
            kept_ids.append(pid)
            kept_vecs.append(vec)

    total_to_delete = consolidated_ids + dedup_deleted_ids

    if not dry_run:
        # Delete in batches of 500
        for i in range(0, len(total_to_delete), 500):
            batch = total_to_delete[i : i + 500]
            client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=batch),
            )

        # Tag surviving deliberate memories with {"memory_type": "deliberate"}
        for i in range(0, len(kept_ids), 500):
            batch = kept_ids[i : i + 500]
            client.set_payload(
                collection_name=collection_name,
                payload={"memory_type": "deliberate"},
                points=batch,
            )

        # Clear SQLite layer_1_pool so spreading activation refills with clean deliberate memories
        if sqlite_db_path and os.path.exists(sqlite_db_path):
            conn = sqlite3.connect(sqlite_db_path, timeout=5.0)
            try:
                with conn:
                    table_exists = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='layer_1_pool'"
                    ).fetchone()
                    if table_exists:
                        conn.execute("DELETE FROM layer_1_pool")
            finally:
                conn.close()

    return {
        "total_before": len(all_points),
        "consolidated_purged": len(consolidated_ids),
        "deliberate_duplicates_purged": len(dedup_deleted_ids),
        "surviving_deliberate": len(kept_ids),
    }


def main():
    parser = argparse.ArgumentParser(
        description="One-time Qdrant & SQLite memory gardening migration (purges Consolidated Memory Blocks and deduplicates deliberate memories)"
    )
    parser.add_argument("-c", "--config", type=str, help="Path to agent TOML config file")
    parser.add_argument("--collection", type=str, default=None, help="Override Qdrant collection name")
    parser.add_argument("--backup-file", type=str, default=None, help="Path for JSONL backup file")
    parser.add_argument("--threshold", type=float, default=0.92, help="Cosine similarity dedup threshold (default: 0.92)")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without modifying Qdrant or SQLite")
    args = parser.parse_args()

    if args.config:
        set_config_path(args.config)

    config = load_config()
    host = config.get("vectordb", {}).get("host", "localhost")
    port = config.get("vectordb", {}).get("port", 6333)
    collection = args.collection or config.get("vectordb", {}).get("memory_collection", "agent_memory")
    sqlite_db = config.get("memory", {}).get("db_path")

    if args.backup_file:
        backup_path = args.backup_file
    elif sqlite_db:
        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(sqlite_db))
        backup_path = os.path.join(base_dir, f"qdrant_backup_{collection}_{ts_str}.jsonl")
    else:
        backup_path = None

    client = QdrantClient(host=host, port=port)
    stats = migrate_agent_memory(
        client=client,
        collection_name=collection,
        sqlite_db_path=sqlite_db,
        backup_path=backup_path,
        dedup_threshold=args.threshold,
        dry_run=args.dry_run,
    )
    mode_str = "DRY RUN" if args.dry_run else "MIGRATED"
    print(f"[{mode_str}] Collection '{collection}' (backup: {backup_path}):")
    print(f"  Total vectors before:         {stats['total_before']}")
    print(f"  Consolidated blocks purged:   {stats['consolidated_purged']}")
    print(f"  Deliberate duplicates purged: {stats['deliberate_duplicates_purged']}")
    print(f"  Surviving deliberate vectors: {stats['surviving_deliberate']}")


if __name__ == "__main__":
    main()
