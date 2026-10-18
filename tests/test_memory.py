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


