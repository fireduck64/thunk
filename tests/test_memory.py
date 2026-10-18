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



