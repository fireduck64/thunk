#!./venv/bin/python3
import argparse
import json
import os
import sys

# Add the project root to the path so we can import src.utils.config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import QdrantClient
from src.utils.config import load_config, set_config_path


def _normalize_vector(vec):
    if vec is None:
        return None
    if isinstance(vec, list):
        return [float(x) if isinstance(x, (int, float)) else x for x in vec]
    if isinstance(vec, dict):
        return {k: _normalize_vector(v) for k, v in vec.items()}
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return vec


def dump_collection_jsonl(client: QdrantClient, collection_name: str, out_stream=sys.stdout, batch_size: int = 100) -> int:
    offset = None
    total_records = 0

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not records:
            break

        for point in records:
            total_records += 1
            payload = dict(point.payload or {})
            record_obj = {
                "id": point.id,
                "collection": collection_name,
                "timestamp": payload.get("timestamp"),
                "text": payload.get("text", ""),
                "payload": payload,
                "vector": _normalize_vector(getattr(point, "vector", None)),
            }
            if getattr(point, "shard_key", None) is not None:
                record_obj["shard_key"] = point.shard_key
            if getattr(point, "order_value", None) is not None:
                record_obj["order_value"] = point.order_value

            out_stream.write(json.dumps(record_obj, ensure_ascii=False, default=str) + "\n")

        out_stream.flush()
        if next_offset is None:
            break
        offset = next_offset

    return total_records


def main():
    parser = argparse.ArgumentParser(
        description="Dump records (including vectors and metadata) from a Qdrant collection as JSONL to stdout"
    )
    parser.add_argument("-c", "--config", type=str, help="Path to the TOML configuration file")
    parser.add_argument(
        "collection",
        nargs="?",
        default=None,
        help="Collection name (defaults to memory_collection in config)",
    )
    args = parser.parse_args()
    if args.config:
        set_config_path(args.config)

    config = load_config()
    host = config.get("vectordb", {}).get("host", "localhost")
    port = config.get("vectordb", {}).get("port", 6333)
    collection_name = (
        args.collection
        if args.collection
        else config.get("vectordb", {}).get("memory_collection", "agent_memory")
    )

    try:
        client = QdrantClient(host=host, port=port)
        dump_collection_jsonl(client, collection_name, out_stream=sys.stdout)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
    except Exception as e:
        print(f"Error dumping collection '{collection_name}' from Qdrant ({host}:{port}): {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
