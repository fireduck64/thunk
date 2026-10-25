import json
import os
import sqlite3
import subprocess
import sys
import pytest

from src.core.agent import AgentCore
from src.components.tiered import TieredMemoryComponent
from src.components.spreading_activation import SpreadingActivationComponent
from src.components.thread_presence import ThreadRegistry, ThreadPresenceComponent
from src.components.logger import AuditLogComponent


class FastDummyTokenizer:
    def encode(self, text: str):
        return (text or "").split()

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.mark.asyncio
async def test_sqlite_thread_id_isolation_tiered_and_spreading_activation(tmp_path):
    """
    1. SQLite thread_id Isolation:
    Two TieredMemoryComponent instances (thread_id='social' and thread_id='reader') and two
    SpreadingActivationComponent instances sharing the exact same memory.db file maintain
    completely isolated working memory messages, core summaries, last_compacted_id, and
    Layer-1 pools without cross-contaminating or locking each other.
    """
    db_path = str(tmp_path / "shared_memory.db")
    tok = FastDummyTokenizer()

    social_mem = TieredMemoryComponent(db_path=db_path, thread_id="social", tokenizer=tok)
    reader_mem = TieredMemoryComponent(db_path=db_path, thread_id="reader", tokenizer=tok)

    try:
        # Duplicate instance on the SAME thread_id should still fail with lock error
        with pytest.raises(RuntimeError, match="already locked"):
            TieredMemoryComponent(db_path=db_path, thread_id="social", tokenizer=tok)

        # Log messages to each thread
        social_msg = {"role": "user", "content": "Hello from Discord on social thread"}
        reader_msg = {"role": "user", "content": "Reading Chapter 4 of Guards! Guards! on reader thread"}
        await social_mem.on_event("message_added", {"message": social_msg})
        await reader_mem.on_event("message_added", {"message": reader_msg})

        # Update core summaries independently
        social_mem._update_core_summary("Social thread summary: chatting with Wisp.", max_evicted_id=social_msg["_db_id"])
        reader_mem._update_core_summary("Reader thread summary: analyzing Sam Vimes.", max_evicted_id=reader_msg["_db_id"])

        assert social_mem._get_core_summary() == "Social thread summary: chatting with Wisp."
        assert reader_mem._get_core_summary() == "Reader thread summary: analyzing Sam Vimes."
        assert social_mem._get_last_compacted_id() == social_msg["_db_id"]
        assert reader_mem._get_last_compacted_id() == reader_msg["_db_id"]

        # Append one more post-compaction message to each thread and verify _restore_working_memory isolates them
        social_msg_2 = {"role": "assistant", "content": "Social reply after compaction"}
        reader_msg_2 = {"role": "assistant", "content": "Reader note after compaction"}
        await social_mem.on_event("message_added", {"message": social_msg_2})
        await reader_mem.on_event("message_added", {"message": reader_msg_2})

        class DummyRestoreAgent:
            def __init__(self):
                self.messages = [{"role": "system", "content": "sys"}]

        social_agent = DummyRestoreAgent()
        reader_agent = DummyRestoreAgent()
        social_mem._restore_working_memory(social_agent)
        reader_mem._restore_working_memory(reader_agent)

        assert [m["content"] for m in social_agent.messages[1:]] == ["Social reply after compaction"]
        assert [m["content"] for m in reader_agent.messages[1:]] == ["Reader note note" if False else "Reader note after compaction"]

        # Verify VectorMemory consolidation receives metadata={"thread_id": self.thread_id}
        captured_meta = []

        class MockVectorMemory:
            def save_semantic_memory(self, content: str, **kwargs) -> str:
                captured_meta.append(kwargs.get("metadata"))
                return "Memory saved successfully."

        MockVectorMemory.__name__ = "VectorMemoryComponent"

        class DummyCompressAgent:
            def __init__(self):
                self.name = "Thunk"
                self.model = "test-model"
                self.components = [MockVectorMemory()]
                self.messages = [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": " ".join(["word"] * 30), "_db_id": 99},
                ]

                class DummyClient:
                    class chat:
                        class completions:
                            @staticmethod
                            async def create(**kwargs):
                                return "New reader summary"

                self.client = DummyClient()

            async def _accumulate_stream(self, s):
                class Msg:
                    content = s
                return Msg()

            def rebuild_system_prompt(self):
                pass

        reader_mem.low_watermark = 2
        await reader_mem._compress_memory(DummyCompressAgent())
        assert captured_meta == [{"thread_id": "reader"}]

        # SpreadingActivationComponent isolation on the same DB file
        sa_social = SpreadingActivationComponent(db_path=db_path, persist=True, thread_id="social")
        sa_reader = SpreadingActivationComponent(db_path=db_path, persist=True, thread_id="reader")

        # Same item ID in both threads with different hit counts / contents
        sa_social._merge_and_reinforce([{"id": "shared_id", "content": "Social memory item", "score": 0.9}])
        sa_social._merge_and_reinforce([{"id": "shared_id", "content": "Social memory item", "score": 0.95}])
        sa_social._save_pool_to_db()

        sa_reader._merge_and_reinforce([{"id": "shared_id", "content": "Reader memory item", "score": 0.7}])
        sa_reader._merge_and_reinforce([{"id": "reader_only", "content": "Only in reader pool", "score": 0.8}])
        sa_reader._save_pool_to_db()

        # Reload fresh instances from DB and verify no cross-contamination
        sa_social_reloaded = SpreadingActivationComponent(db_path=db_path, persist=True, thread_id="social")
        sa_reader_reloaded = SpreadingActivationComponent(db_path=db_path, persist=True, thread_id="reader")

        assert set(sa_social_reloaded.layer_1_pool.keys()) == {"shared_id"}
        assert sa_social_reloaded.layer_1_pool["shared_id"].hits == 2
        assert sa_social_reloaded.layer_1_pool["shared_id"].content == "Social memory item"

        assert set(sa_reader_reloaded.layer_1_pool.keys()) == {"shared_id", "reader_only"}
        assert sa_reader_reloaded.layer_1_pool["shared_id"].hits == 1
        assert sa_reader_reloaded.layer_1_pool["shared_id"].content == "Reader memory item"
    finally:
        social_mem._lock_file.close()
        reader_mem._lock_file.close()


def test_schema_auto_migration_from_legacy_memory_db(tmp_path):
    """
    2. Schema Auto-Migration:
    Existing memory.db tables created without a thread_id column cleanly auto-migrate on startup
    and default existing rows to 'main'.
    """
    db_path = str(tmp_path / "legacy_memory.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            "CREATE TABLE archival_log (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute("INSERT INTO archival_log (role, content) VALUES ('user', 'Legacy message 1')")
        conn.execute("CREATE TABLE core_summary (id INTEGER PRIMARY KEY, summary TEXT)")
        conn.execute("INSERT INTO core_summary (id, summary) VALUES (1, 'Legacy summary text')")
        conn.execute(
            "CREATE TABLE core_summary_history (id INTEGER PRIMARY KEY AUTOINCREMENT, summary TEXT, compacted_up_to_id INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute("INSERT INTO core_summary_history (summary, compacted_up_to_id) VALUES ('Legacy summary text', 1)")
        conn.execute("CREATE TABLE memory_settings (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO memory_settings (key, value) VALUES ('last_compacted_id', '0')")
        conn.execute(
            """
            CREATE TABLE layer_1_pool (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                hits INTEGER,
                age INTEGER,
                last_similarity REAL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "INSERT INTO layer_1_pool (id, content, metadata, hits, age, last_similarity) VALUES ('legacy_m1', 'Legacy pool item', '{}', 3, 1, 0.88)"
        )
    conn.close()

    tok = FastDummyTokenizer()
    main_mem = TieredMemoryComponent(db_path=db_path, thread_id="main", tokenizer=tok)
    sa_main = SpreadingActivationComponent(db_path=db_path, persist=True, thread_id="main")
    try:
        assert main_mem._get_core_summary() == "Legacy summary text"
        assert "legacy_m1" in sa_main.layer_1_pool
        assert sa_main.layer_1_pool["legacy_m1"].hits == 3

        # Verify a second thread can now use the migrated tables alongside 'main'
        worker_mem = TieredMemoryComponent(db_path=db_path, thread_id="worker", tokenizer=tok)
        try:
            assert worker_mem._get_core_summary() == "No summary yet."
            worker_mem._update_core_summary("Worker summary", max_evicted_id=5)
            assert main_mem._get_core_summary() == "Legacy summary text"
            assert worker_mem._get_core_summary() == "Worker summary"
        finally:
            worker_mem._lock_file.close()
    finally:
        main_mem._lock_file.close()


@pytest.mark.asyncio
async def test_transient_presence_footer_and_cross_thread_pings(tmp_path):
    """
    3 & 4. Transient Presence Footer & Cross-Thread Pings + Loop Protection:
    - ThreadPresenceComponent injects sibling statuses transiently on before_llm_call without polluting SQLite.
    - ping_thread delivers formatted internal ping to target_thread's inbox and wakes target_thread if suspended.
    - Rapid back-and-forth ping-ponging is rate-limited, and external user messages clear the rate limit.
    """
    db_path = str(tmp_path / "presence_memory.db")
    tok = FastDummyTokenizer()

    registry = ThreadRegistry(max_pings_per_window=2, ping_window_sec=60.0)
    social_core = AgentCore(thread_id="social")
    reader_core = AgentCore(thread_id="reader")

    social_presence = ThreadPresenceComponent(
        registry=registry,
        thread_id="social",
        role="External communication & social presence",
        auto_status_turns=2,
    )
    reader_presence = ThreadPresenceComponent(
        registry=registry,
        thread_id="reader",
        role="Deep sequential reading",
        auto_status_turns=2,
    )

    social_mem = TieredMemoryComponent(db_path=db_path, thread_id="social", tokenizer=tok)
    try:
        social_core.register_component(social_presence)
        social_core.register_component(social_mem)
        reader_core.register_component(reader_presence)

        await social_core.event_bus.publish("agent_started", {"agent": social_core})
        await reader_core.event_bus.publish("agent_started", {"agent": reader_core})

        # Static system prompt includes current thread and sibling thread
        prompt_block = reader_presence.get_system_prompt()
        assert "`reader` (Deep sequential reading)" in prompt_block
        assert "`social`: External communication & social presence" in prompt_block

        # Update social status and verify reader sees it transiently on before_llm_call
        social_presence.set_thread_status("Debating ontology with Wisp.")
        await reader_core.add_message("user", "Read next chapter", publish_event=False)
        await reader_core.event_bus.publish(
            "before_llm_call", {"agent": reader_core, "messages": reader_core.messages}
        )

        assert reader_core.messages[-1].get("_transient_presence") is True
        assert "[Parallel Streams of Mind (Live Status)]" in reader_core.messages[-1]["content"]
        assert "Debating ontology with Wisp." in reader_core.messages[-1]["content"]

        # Wire messages include the footer without underscore keys
        wire_msgs = reader_core._prepare_llm_messages()
        assert "[Parallel Streams of Mind (Live Status)]" in wire_msgs[-1]["content"]
        assert "_transient_presence" not in wire_msgs[-1]

        # after_llm_call strips the transient footer from working memory
        await reader_core.event_bus.publish(
            "after_llm_call",
            {"agent": reader_core, "response": {"content": "Reading chunk 1"}},
        )
        assert all(not m.get("_transient_presence") for m in reader_core.messages)

        # Verify nothing transient was written to SQLite archival_log
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT content FROM archival_log").fetchall()
            assert all("[Parallel Streams of Mind" not in r[0] for r in rows)
        finally:
            conn.close()

        # Verify suspension updates status to sleeping
        social_core.suspended.clear()
        await social_core.event_bus.publish("agent_suspended", {"agent": social_core})
        assert "sleeping — last: Debating ontology with Wisp." in registry.statuses["social"]["status"]

        # Cross-thread ping 1: reader -> social wakes social and enqueues ping in social.inbox
        res1 = await reader_presence.ping_thread("social", "Found pink crocs in Chapter 7!")
        assert "Internal ping delivered to thread 'social'" in res1
        assert social_core.suspended.is_set() is True
        ping_msg_1 = await social_core.inbox.get()
        assert ping_msg_1 == "[Internal Ping from thread 'reader']: Found pink crocs in Chapter 7!"

        # Cross-thread ping 2: social -> reader succeeds (2nd ping in 60s window)
        res2 = await social_presence.ping_thread("reader", "Awesome, posting that now!")
        assert "Internal ping delivered to thread 'reader'" in res2
        ping_msg_2 = await reader_core.inbox.get()
        assert ping_msg_2 == "[Internal Ping from thread 'social']: Awesome, posting that now!"

        # Cross-thread ping 3 within 60s without external message -> suppressed by loop protection
        res3 = await reader_presence.ping_thread("social", "Thanks for posting it!")
        assert "suppressed to prevent an internal echo loop" in res3
        assert social_core.inbox.empty()
        assert "[Note for social]: Thanks for posting it!" in registry.statuses["reader"]["status"]

        # External user message arrives on social -> clears ping rate limit
        await social_core.event_bus.publish(
            "before_user_message_added",
            {"content": "[Via Discord | Channel: #general | Author: Sombus [Operator]] Ask reader what page that was on"},
        )
        res4 = await social_presence.ping_thread("reader", "Sombus wants to know what page that was on.")
        assert "Internal ping delivered to thread 'reader'" in res4
    finally:
        social_mem._lock_file.close()


@pytest.mark.asyncio
async def test_combined_audit_log_and_extract_monolog_filtering(tmp_path):
    """
    5. Combined Audit Log & extract-monolog.py:
    Events from multiple threads write to the same JSONL file with their respective 'thread_id' fields intact,
    and scripts/extract-monolog.py displays [thread_id] badges and filters by -t / --thread.
    """
    shared_log = str(tmp_path / "session_shared.jsonl")
    social_logger = AuditLogComponent(log_dir=str(tmp_path), thread_id="social", log_file=shared_log)
    reader_logger = AuditLogComponent(log_dir=str(tmp_path), thread_id="reader", log_file=shared_log)

    await social_logger.on_event(
        "message_added",
        {"message": {"role": "assistant", "content": "Social thread thought"}},
    )
    await reader_logger.on_event(
        "message_added",
        {"message": {"role": "assistant", "content": "Reader thread thought"}},
    )

    with open(shared_log, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    assert len(entries) == 2
    assert entries[0]["thread_id"] == "social"
    assert entries[0]["payload"]["message"]["content"] == "Social thread thought"
    assert entries[1]["thread_id"] == "reader"
    assert entries[1]["payload"]["message"]["content"] == "Reader thread thought"

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "extract-monolog.py"))

    # Combined output shows both [social] and [reader] badges
    combined_out = subprocess.check_output([sys.executable, script_path, shared_log], text=True)
    assert "[social] [AGENT MONOLOGUE]" in combined_out
    assert "Social thread thought" in combined_out
    assert "[reader] [AGENT MONOLOGUE]" in combined_out
    assert "Reader thread thought" in combined_out

    # Filtered output (-t reader) shows only reader thread
    reader_out = subprocess.check_output([sys.executable, script_path, "-t", "reader", shared_log], text=True)
    assert "[reader] [AGENT MONOLOGUE]" in reader_out
    assert "Reader thread thought" in reader_out
    assert "Social thread thought" not in reader_out
