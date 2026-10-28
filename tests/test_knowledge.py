import uuid
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.utils.config import load_config
from src.components.vector_memory import VectorMemoryComponent
from src.components.knowledge import KnowledgeBaseComponent


def _delete_test_collection(collection_name: str):
    assert collection_name.startswith("test_"), f"Refusing to delete non-test collection: {collection_name}"
    cfg = load_config()
    client = QdrantClient(host=cfg["vectordb"]["host"], port=cfg["vectordb"]["port"])
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception:
        pass


def test_embeddings_and_vector_memory():
    """Tests the embeddings model endpoint and VectorMemoryComponent against isolated test_agent_memory."""
    cfg = load_config()
    collection = cfg["vectordb"]["memory_collection"]
    assert collection.startswith("test_")
    _delete_test_collection(collection)

    vm = VectorMemoryComponent()
    try:
        # 1. Directly test embedding generation
        vec = vm._get_embedding("Semantic embedding test vector for rig migration.")
        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(x, float) for x in vec[:10])

        # 2. Test saving semantic memories into Qdrant
        res1 = vm.save_semantic_memory(
            "The Copper Key is hidden inside the Tomb of Horrors on the world of Ludus."
        )
        assert "successfully" in res1.lower()

        res2 = vm.save_semantic_memory(
            "Brennen builds deterministic distributed systems with Bazel and SQLite."
        )
        assert "successfully" in res2.lower()

        # 3. Test semantic search retrieval via embeddings
        search_res = vm.search_semantic_memory("Where is the Copper Key located?", num_results=2)
        assert "Tomb of Horrors" in search_res
        assert "Ludus" in search_res

        # 4. Test dumping raw records
        dump_res = vm.dump_semantic_memory(limit=10, offset=0)
        assert "Tomb of Horrors" in dump_res
        assert "Bazel" in dump_res

        # 5. Test JSONL dump script (with vectors and metadata)
        import importlib.util
        import io
        import json

        spec = importlib.util.spec_from_file_location(
            "dump_vector_db_json", "scripts/dump-vector-db-json.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        buf = io.StringIO()
        q_client = QdrantClient(host=cfg["vectordb"]["host"], port=cfg["vectordb"]["port"])
        count = mod.dump_collection_jsonl(q_client, collection, out_stream=buf)
        assert count == 2
        lines = [json.loads(line) for line in buf.getvalue().strip().splitlines()]
        assert len(lines) == 2
        for rec in lines:
            assert rec["collection"] == collection
            assert isinstance(rec["id"], str)
            assert isinstance(rec["timestamp"], str)
            assert isinstance(rec["text"], str) and len(rec["text"]) > 0
            assert isinstance(rec["payload"], dict)
            assert isinstance(rec["vector"], list) and len(rec["vector"]) == 768
    finally:
        _delete_test_collection(collection)


def test_knowledge_base_embeddings_and_search():
    """Tests KnowledgeBaseComponent embedding queries and Qdrant search against isolated test_knowledge_base."""
    cfg = load_config()
    collection = cfg["vectordb"]["knowledge_collection"]
    assert collection.startswith("test_")
    _delete_test_collection(collection)

    kb = KnowledgeBaseComponent()
    client = QdrantClient(host=cfg["vectordb"]["host"], port=cfg["vectordb"]["port"])
    try:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        passages = [
            (
                "Pride and Prejudice",
                "Mr. Bingley is a wealthy, amiable young gentleman from the north of England who rents Netherfield Park.",
            ),
            (
                "Dungeon Crawler Carl",
                "Princess Donut the Queen Anne Chonk is a tortoiseshell Persian cat who gained sapience in the dungeon.",
            ),
        ]

        points = []
        for source, text in passages:
            vector = kb._get_embedding(text)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"source": source, "text": text},
                )
            )
        client.upsert(collection_name=collection, points=points)

        result = kb.search_knowledge_base("Who is Mr. Bingley and what estate did he rent?", num_results=1)
        assert "Pride and Prejudice" in result
        assert "Netherfield Park" in result
    finally:
        _delete_test_collection(collection)

