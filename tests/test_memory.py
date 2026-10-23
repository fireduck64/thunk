import os
import sqlite3
import pytest
from transformers import AutoTokenizer
from src.utils.config import load_config
from src.components.tiered import TieredMemoryComponent


def _cleanup_test_memory_db(db_path: str):
    for p in (db_path, f"{db_path}.lock"):
        if os.path.exists(p):
            os.remove(p)


def test_tokenizer_loads_and_encodes():
    """Verifies that the configured HuggingFace tokenizer loads directly without falling back to gpt2."""
    cfg = load_config()
    model_name = cfg["tokenizer"]["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=100000)
    sample_text = "Thunk and Wisp are testing the tokenizer before migrating to a new rig."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    assert isinstance(encoded, list)
    assert len(encoded) > 5
    assert "Thunk" in decoded


def test_tiered_memory_tokenizer_and_compaction_state():
    """Tests TieredMemoryComponent token counting, summary truncation, history table, and file locking."""
    cfg = load_config()
    db_path = cfg["memory"]["db_path"]
    _cleanup_test_memory_db(db_path)

    mem = TieredMemoryComponent()
    try:
        # Ensure the primary configured tokenizer was loaded (not the gpt2 fallback)
        assert cfg["tokenizer"]["model_name"] in mem.tokenizer.name_or_path

        # Verify exclusive OS lockfile prevents a second instance from opening the same DB
        with pytest.raises(RuntimeError, match="already locked"):
            TieredMemoryComponent(db_path=db_path)

        # Verify token counting
        msgs = [
            {"role": "system", "content": "You are TestAgent."},
            {"role": "user", "content": "My name is John and I like apples."},
            {"role": "assistant", "content": "Hello John! I will remember that you like apples."},
        ]
        token_count = mem._count_tokens(msgs)
        assert token_count > 15

        # Verify empty summary guard
        with pytest.raises(ValueError, match="Refusing to overwrite core_summary"):
            mem._update_core_summary("   ", max_evicted_id=5)

        # Verify atomic core_summary + core_summary_history + last_compacted_id update
        mem._update_core_summary("User's name is John and he likes apples.", max_evicted_id=12)
        assert "John" in mem._get_core_summary()
        assert mem._get_last_compacted_id() == 12

        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT summary, compacted_up_to_id FROM core_summary_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
            assert row is not None
            assert "John" in row[0]
            assert row[1] == 12
        finally:
            conn.close()

        # Verify summary truncation when exceeding max_summary_tokens
        huge_summary = "word " * (mem.max_summary_tokens + 100)
        mem._update_core_summary(huge_summary, max_evicted_id=20)
        truncated = mem._get_core_summary()
        assert truncated.endswith("... [TRUNCATED]")
    finally:
        mem._lock_file.close()
        _cleanup_test_memory_db(db_path)


@pytest.mark.asyncio
async def test_subconscious_recall_synthesis_and_filtering():
    """Verifies SubconsciousRecallComponent queries top_k, synthesizes relevant memories, and skips NONE / clock messages."""
    from src.components.subconscious import SubconsciousRecallComponent

    class DummyVectorMemory:
        def __init__(self):
            self.last_num_results = None

        def search_semantic_memory(self, query: str, num_results: int = 3) -> str:
            self.last_num_results = num_results
            return "--- Memory 1 ---\nCarl and Donut formed the Princess Posse on Floor 9."

    sub = SubconsciousRecallComponent()
    dummy_vm = DummyVectorMemory()
    sub.internal_memory_components = [dummy_vm]

    # 1. Verify [System Clock] messages are skipped without searching
    clock_payload = {"content": "[System Clock] The current time is 12:00."}
    await sub.on_event("before_user_message_added", clock_payload)
    assert dummy_vm.last_num_results is None
    assert clock_payload["content"] == "[System Clock] The current time is 12:00."

    # 2. Verify relevant synthesis is injected and top_k is passed
    async def mock_synthesize_relevant(msg: str, raw: str) -> str:
        return "* Carl and Donut lead the Princess Posse on Floor 9."

    sub._synthesize_memories = mock_synthesize_relevant
    msg_payload = {"content": "What faction did Carl and Donut form?"}
    await sub.on_event("before_user_message_added", msg_payload)
    assert dummy_vm.last_num_results == sub.top_k
    assert msg_payload["content"].startswith("[Subconscious Recall Triggered by below message:]")
    assert "Princess Posse" in msg_payload["content"]
    assert msg_payload["content"].endswith("What faction did Carl and Donut form?")

    # 3. Verify NONE filtering leaves the payload untouched
    async def mock_synthesize_none(msg: str, raw: str) -> str:
        return ""

    sub._synthesize_memories = mock_synthesize_none
    unrelated_payload = {"content": "Hello there!"}
    await sub.on_event("before_user_message_added", unrelated_payload)
    assert unrelated_payload["content"] == "Hello there!"


def test_vector_memory_chunking_retries_and_fatal_exit(monkeypatch):
    """Verifies VectorMemoryComponent auto-chunks large texts, retries transient errors, and calls sys.exit(1) on unrecoverable errors."""
    from src.components.vector_memory import VectorMemoryComponent

    monkeypatch.setattr(VectorMemoryComponent, "_ensure_collection", lambda self: None)
    vm = VectorMemoryComponent(collection_name="test_collection")
    vm.retry_base_delay = 0.0
    vm.max_retries = 3

    upserted_batches = []

    class DummyQdrantClient:
        def upsert(self, collection_name, points):
            upserted_batches.append((collection_name, points))

    vm._client = DummyQdrantClient()

    embedded_texts = []
    vm._get_embedding = lambda text: (embedded_texts.append(text) or [0.1] * 768)

    # 1. Large consolidated block (> 4000 chars) should split into multiple overlapping chunks
    large_consolidated = "Consolidated Memory Block:\nSummary: Summary text\nRaw Events:\n" + ("A" * 9000)
    res = vm.save_semantic_memory(large_consolidated)
    assert res == "Memory saved successfully."
    assert len(upserted_batches) == 1
    _, points = upserted_batches[0]
    assert len(points) == 3
    assert points[0].payload["text"].startswith("Consolidated Memory Block:")
    assert points[1].payload["text"].startswith("Consolidated Memory Block (Part 2/3):")
    assert points[2].payload["text"].startswith("Consolidated Memory Block (Part 3/3):")
    assert all(len(t) <= 4100 for t in embedded_texts)

    # 2. Transient failure retries and succeeds before max_retries
    attempts = {"count": 0}

    def flaky_embedding(text):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("Transient Ollama timeout")
        return [0.2] * 768

    vm._get_embedding = flaky_embedding
    assert vm.save_semantic_memory("Short memory") == "Memory saved successfully."
    assert attempts["count"] == 3

    # 3. High token density ("exceeds the context length") auto-splits chunk in half
    import requests

    def density_embedding(text):
        if len(text) > 1000:
            resp = requests.Response()
            resp.status_code = 500
            resp._content = b'{"error":"the input length exceeds the context length"}'
            raise requests.exceptions.HTTPError("500 Server Error", response=resp)
        return [0.3] * 768

    vm._get_embedding = density_embedding
    upserted_batches.clear()
    assert vm.save_semantic_memory("X" * 1600) == "Memory saved successfully."
    assert len(upserted_batches) == 1
    assert len(upserted_batches[0][1]) == 2

    # 4. Persistent failure exhausts retries and calls sys.exit(1)
    def failing_embedding(text):
        raise RuntimeError("Ollama down permanently")

    vm._get_embedding = failing_embedding
    with pytest.raises(SystemExit) as exc_info:
        vm.save_semantic_memory("Critical memory")
    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_tiered_memory_transactional_compaction_and_fatal_exit():
    """Verifies TieredMemoryComponent does not evict messages or advance last_compacted_id if summarization or vector consolidation fails."""
    cfg = load_config()
    db_path = cfg["memory"]["db_path"]
    _cleanup_test_memory_db(db_path)

    mem = TieredMemoryComponent()
    mem.retry_base_delay = 0.0
    mem.max_retries = 2
    mem.low_watermark = 10

    try:
        class DummyMessage:
            def __init__(self, content):
                self.content = content

        class DummyCompletions:
            def __init__(self, should_fail=False, summary_text="Updated summary."):
                self.should_fail = should_fail
                self.summary_text = summary_text

            async def create(self, **kwargs):
                if self.should_fail:
                    raise RuntimeError("LLM endpoint unreachable")
                return self.summary_text

        class DummyChat:
            def __init__(self, completions):
                self.completions = completions

        class DummyClient:
            def __init__(self, completions):
                self.chat = DummyChat(completions)

        class VectorMemoryComponent:
            def __init__(self, should_fail=False):
                self.should_fail = should_fail
                self.saved = []

            def save_semantic_memory(self, content: str) -> str:
                if self.should_fail:
                    raise SystemExit(1)
                self.saved.append(content)
                return "Memory saved successfully."

        class DummyAgent:
            def __init__(self, completions, vm_comp):
                self.model = "test-model"
                self.client = DummyClient(completions)
                self.components = [vm_comp]
                self.messages = [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Message 1 " * 20, "_db_id": 101},
                    {"role": "assistant", "content": "Message 2 " * 20, "_db_id": 102},
                ]
                self.rebuilt = False

            async def _accumulate_stream(self, stream_val):
                return DummyMessage(stream_val)

            def rebuild_system_prompt(self):
                self.rebuilt = True

        # 1. Summarization failure -> exits with code 1, messages and last_compacted_id untouched
        failing_agent = DummyAgent(DummyCompletions(should_fail=True), VectorMemoryComponent())
        with pytest.raises(SystemExit) as exc_info:
            await mem._compress_memory(failing_agent)
        assert exc_info.value.code == 1
        assert len(failing_agent.messages) == 3
        assert mem._get_last_compacted_id() == 0

        # 2. VectorMemory consolidation failure -> exits with code 1, messages and last_compacted_id untouched
        failing_vm_agent = DummyAgent(DummyCompletions(should_fail=False), VectorMemoryComponent(should_fail=True))
        with pytest.raises(SystemExit) as exc_info:
            await mem._compress_memory(failing_vm_agent)
        assert exc_info.value.code == 1
        assert len(failing_vm_agent.messages) == 3
        assert mem._get_last_compacted_id() == 0

        # 3. Successful compaction -> saves to VectorMemory, updates SQLite summary + last_compacted_id, then evicts messages
        good_vm = VectorMemoryComponent(should_fail=False)
        good_agent = DummyAgent(DummyCompletions(should_fail=False, summary_text="Compacted summary!"), good_vm)
        await mem._compress_memory(good_agent)
        assert len(good_vm.saved) == 1
        assert good_vm.saved[0].startswith("Consolidated Memory Block:\nSummary: Compacted summary!")
        assert mem._get_last_compacted_id() == 102
        assert "Compacted summary!" in mem._get_core_summary()
        assert len(good_agent.messages) == 1
        assert good_agent.rebuilt is True
    finally:
        mem._lock_file.close()
        _cleanup_test_memory_db(db_path)




