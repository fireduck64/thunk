import os
import sqlite3
import pytest
from src.utils.config import load_config
from src.core.agent import AgentCore
from src.components.subconscious import SubconsciousRecallComponent
from src.components.spreading_activation import SpreadingActivationComponent, Layer1Item


@pytest.fixture(autouse=True)
def cleanup_layer_1_pool_table():
    db_path = load_config()["memory"]["db_path"]
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            with conn:
                conn.execute("DROP TABLE IF EXISTS layer_1_pool")
        finally:
            conn.close()
    yield
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            with conn:
                conn.execute("DROP TABLE IF EXISTS layer_1_pool")
        finally:
            conn.close()


class DummyVectorMemory:
    """Mock VectorMemoryComponent returning programmable raw search hits and recording fetch_count."""
    def __init__(self, batches=None):
        self.batches = list(batches or [])
        self.requested_limits = []

    def search_semantic_memory_raw(self, query: str, num_results: int = 8):
        self.requested_limits.append(num_results)
        if self.batches:
            batch = self.batches.pop(0)
            return batch[:num_results]
        return []


@pytest.mark.asyncio
async def test_cold_start_vacuum_filling_and_steady_state_fetch():
    """
    1. Cold-start vacuum filling: when pool is empty (0 items), fetch_count == layer_1_pool_size + layer_0_promotion_count.
    2. Steady-state fetch: when pool is full (32 items), fetch_count == layer_0_promotion_count.
    """
    sa = SpreadingActivationComponent(layer_1_pool_size=32, layer_0_promotion_count=8)
    assert sa.compute_fetch_count() == 40

    # Fill 32 items on turn 1
    initial_batch = [
        {"id": f"mem-{i}", "content": f"Memory content {i}", "metadata": {}, "score": 0.9 - (i * 0.01)}
        for i in range(40)
    ]
    second_batch = [
        {"id": f"mem-new-{i}", "content": f"New memory {i}", "metadata": {}, "score": 0.85}
        for i in range(8)
    ]
    dummy_vm = DummyVectorMemory(batches=[initial_batch, second_batch])
    sa.internal_memory_components = [dummy_vm]

    async def mock_synthesize(msg: str, raw: str) -> str:
        return "* Distilled point."

    sa._synthesize_memories = mock_synthesize

    # Turn 1: Cold start (0 items in pool -> requests 32 + 8 = 40)
    payload_1 = {"content": "First user turn"}
    await sa.on_event("before_user_message_added", payload_1)
    assert dummy_vm.requested_limits[0] == 40
    assert len(sa.layer_1_pool) == 32
    assert payload_1["content"].startswith("[Subconscious Recall Triggered by below message:]")

    # Turn 2: Pool is full (32 items -> requests 0 + 8 = 8)
    assert sa.compute_fetch_count() == 8
    payload_2 = {"content": "Second user turn"}
    await sa.on_event("before_user_message_added", payload_2)
    assert dummy_vm.requested_limits[1] == 8
    assert len(sa.layer_1_pool) == 32


def test_spreading_activation_reinforcement():
    """
    3. Spreading activation reinforcement:
    When an item already in Layer-1 is returned by a subsequent vector query,
    its hits increments, its age resets to 0, and its priority (hits / (age + 1))
    increases above single-hit items.
    """
    sa = SpreadingActivationComponent(layer_1_pool_size=8, layer_0_promotion_count=4)

    # Turn 1: Insert mem-a and mem-b
    sa._age_pool()
    sa._merge_and_reinforce([
        {"id": "mem-a", "content": "Alpha concept", "metadata": {}, "score": 0.70},
        {"id": "mem-b", "content": "Beta concept", "metadata": {}, "score": 0.80},
    ])
    assert sa.layer_1_pool["mem-a"].hits == 1
    assert sa.layer_1_pool["mem-a"].age == 0
    assert sa.layer_1_pool["mem-a"].priority == 1.0

    # Turn 2: Age pool, then hit mem-a again along with brand-new mem-c
    sa._age_pool()
    assert sa.layer_1_pool["mem-a"].age == 1
    assert sa.layer_1_pool["mem-b"].age == 1
    assert sa.layer_1_pool["mem-b"].priority == 0.5

    sa._merge_and_reinforce([
        {"id": "mem-a", "content": "Alpha concept", "metadata": {}, "score": 0.75},
        {"id": "mem-c", "content": "Gamma concept", "metadata": {}, "score": 0.90},
    ])

    item_a = sa.layer_1_pool["mem-a"]
    item_b = sa.layer_1_pool["mem-b"]
    item_c = sa.layer_1_pool["mem-c"]

    assert item_a.hits == 2
    assert item_a.age == 0
    assert item_a.last_similarity == 0.75
    assert item_a.priority == 2.0

    assert item_c.hits == 1
    assert item_c.age == 0
    assert item_c.priority == 1.0

    # Reinforced item_a has higher priority than single-hit item_c and aged item_b
    assert item_a.priority > item_c.priority > item_b.priority


def test_aging_and_competitive_eviction():
    """
    4. Aging and competitive eviction:
    Unhit items increment age each turn, lowering their priority, and are evicted
    when new items push the pool past layer_1_pool_size.
    """
    sa = SpreadingActivationComponent(layer_1_pool_size=4, layer_0_promotion_count=2)

    # Seed pool with 4 items
    sa._merge_and_reinforce([
        {"id": "cold-1", "content": "Cold 1", "metadata": {}, "score": 0.60},
        {"id": "cold-2", "content": "Cold 2", "metadata": {}, "score": 0.65},
        {"id": "warm-1", "content": "Warm 1", "metadata": {}, "score": 0.70},
        {"id": "warm-2", "content": "Warm 2", "metadata": {}, "score": 0.75},
    ])
    assert len(sa.layer_1_pool) == 4

    # Next turn: age pool (all age=1, priority=0.5), reinforce warm-1 and add 2 fresh items (fresh-1, fresh-2)
    sa._age_pool()
    sa._merge_and_reinforce([
        {"id": "warm-1", "content": "Warm 1", "metadata": {}, "score": 0.72},  # hits=2, age=0, priority=2.0
        {"id": "fresh-1", "content": "Fresh 1", "metadata": {}, "score": 0.80},  # hits=1, age=0, priority=1.0
        {"id": "fresh-2", "content": "Fresh 2", "metadata": {}, "score": 0.85},  # hits=1, age=0, priority=1.0
    ])
    sa._evict_pool()

    assert len(sa.layer_1_pool) == 4
    # Retained: warm-1 (prio 2.0), fresh-2 (prio 1.0, sim 0.85), fresh-1 (prio 1.0, sim 0.80),
    # and warm-2 (prio 0.5, sim 0.75) beats cold-1 (0.60) and cold-2 (0.65)
    assert set(sa.layer_1_pool.keys()) == {"warm-1", "fresh-2", "fresh-1", "warm-2"}
    assert "cold-1" not in sa.layer_1_pool
    assert "cold-2" not in sa.layer_1_pool


@pytest.mark.asyncio
async def test_split_slate_promotion_and_refractory_reset():
    """
    5. Split-slate promotion & refractory reset:
    Promoted items include both multi-hit items (hits > 1) and immediate similarity matches,
    and promoted items have their hits reset to 1 and age reset to 0 after promotion.
    """
    sa = SpreadingActivationComponent(layer_1_pool_size=10, layer_0_promotion_count=4)

    # Populate pool directly with 2 recurring multi-hit items (lower similarity, higher hits)
    # and 4 single-hit items (higher immediate similarity)
    sa.layer_1_pool = {
        "rec-1": Layer1Item(id="rec-1", content="Recurring memory 1", hits=3, age=0, last_similarity=0.55),
        "rec-2": Layer1Item(id="rec-2", content="Recurring memory 2", hits=2, age=0, last_similarity=0.50),
        "imm-1": Layer1Item(id="imm-1", content="Immediate memory 1", hits=1, age=0, last_similarity=0.95),
        "imm-2": Layer1Item(id="imm-2", content="Immediate memory 2", hits=1, age=0, last_similarity=0.92),
        "imm-3": Layer1Item(id="imm-3", content="Immediate memory 3", hits=1, age=0, last_similarity=0.89),
        "imm-4": Layer1Item(id="imm-4", content="Immediate memory 4", hits=1, age=0, last_similarity=0.40),
    }

    promoted, recurring_ids = sa._select_promotion_slate()
    promoted_ids = [item.id for item in promoted]

    # Group A takes up to 4 // 2 = 2 recurring items: rec-1, rec-2
    # Group B takes remaining 2 highest-similarity items: imm-1, imm-2
    assert promoted_ids == ["rec-1", "rec-2", "imm-1", "imm-2"]
    assert recurring_ids == {"rec-1", "rec-2"}

    # Verify formatting notes recurring vs immediate
    formatted = sa._format_promoted_candidates(promoted, recurring_ids)
    assert "[Recurring Cross-Turn Connection | hits=3" in formatted
    assert "[Immediate Associative Match | hits=1" in formatted

    # Apply refractory cooldown and verify hits reset to 1 and age reset to 0
    sa._apply_refractory_cooldown(promoted)
    assert sa.layer_1_pool["rec-1"].hits == 1
    assert sa.layer_1_pool["rec-1"].age == 0
    assert sa.layer_1_pool["rec-2"].hits == 1
    assert sa.layer_1_pool["rec-2"].age == 0


def test_mutual_exclusion_between_subconscious_and_spreading_activation():
    """Verifies AgentCore refuses to register both SubconsciousRecallComponent and SpreadingActivationComponent."""
    agent = AgentCore()
    agent.register_component(SpreadingActivationComponent())
    with pytest.raises(ValueError, match="mutually exclusive"):
        agent.register_component(SubconsciousRecallComponent())


@pytest.mark.asyncio
async def test_layer_1_pool_sqlite_persistence_and_reload():
    """Verifies layer_1_pool is saved to SQLite after turns and restored on component initialization."""
    sa1 = SpreadingActivationComponent(layer_1_pool_size=8, layer_0_promotion_count=2)
    batch = [
        {"id": "sql-1", "content": "Persisted memory alpha", "metadata": {"timestamp": "2026-10-20T00:00:00Z"}, "score": 0.91},
        {"id": "sql-2", "content": "Persisted memory beta", "metadata": {"timestamp": "2026-10-20T00:00:00Z"}, "score": 0.82},
        {"id": "sql-3", "content": "Persisted memory gamma", "metadata": {"timestamp": "2026-10-20T00:00:00Z"}, "score": 0.73},
    ]
    sa1.internal_memory_components = [DummyVectorMemory(batches=[batch])]

    async def mock_synthesize(msg: str, raw: str) -> str:
        return "* Alpha and Beta recalled."

    sa1._synthesize_memories = mock_synthesize
    await sa1.on_event("before_user_message_added", {"content": "Check SQLite persistence"})

    # Query SQLite directly
    conn = sqlite3.connect(sa1.db_path)
    try:
        rows = conn.execute(
            "SELECT id, hits, age, round(last_similarity, 2), content FROM layer_1_pool ORDER BY last_similarity DESC"
        ).fetchall()
        assert len(rows) == 3
        assert rows[0][0] == "sql-1"
        assert rows[0][4] == "Persisted memory alpha"
    finally:
        conn.close()

    # Re-instantiate component and verify it restores all 3 items from SQLite
    sa2 = SpreadingActivationComponent(layer_1_pool_size=8, layer_0_promotion_count=2)
    assert set(sa2.layer_1_pool.keys()) == {"sql-1", "sql-2", "sql-3"}
    assert sa2.layer_1_pool["sql-1"].content == "Persisted memory alpha"

