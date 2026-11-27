import json
import sqlite3
from types import SimpleNamespace
import pytest

from src.core.agent import AgentCore
from src.components.belief_graph import BeliefArchivist, BeliefGraphComponent
from src.components.executive_assistant import ExecutiveAssistantComponent


def _mock_completion(text: str, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=text),
            )
        ]
    )


def test_record_belief_tool_and_inbox_staging(tmp_path):
    """
    1. record_belief tool:
    Queues into belief_inbox with processed=0, validates stance and non-empty claim,
    and does not touch beliefs.
    """
    db_path = str(tmp_path / "memory.db")
    bg = BeliefGraphComponent(db_path=db_path, thread_id="mind")

    assert "record_belief" in bg.get_system_prompt_addition()
    tools = bg.get_tools()
    assert len(tools) == 1 and tools[0].__name__ == "record_belief"

    err_empty = bg.record_belief("   ", stance="held", reason="test")
    assert err_empty.startswith("Error:")

    err_stance = bg.record_belief("RXJ1131 is elliptical", stance="certain", reason="test")
    assert "Invalid stance" in err_stance

    ok = bg.record_belief(
        "RXJ1131 requires an elliptical mass profile",
        stance="held",
        reason="Spherical assumption failed chi2 check",
    )
    assert ok == "Noted. Belief declaration queued for epistemic review."

    conn = sqlite3.connect(db_path)
    try:
        belief_count = conn.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0]
        assert belief_count == 0

        inbox_rows = conn.execute(
            "SELECT thread_id, claim, stance, reason, processed FROM belief_inbox"
        ).fetchall()
        assert len(inbox_rows) == 1
        assert inbox_rows[0] == (
            "mind",
            "RXJ1131 requires an elliptical mass profile",
            "held",
            "Spherical assumption failed chi2 check",
            0,
        )
    finally:
        conn.close()


def test_bootstrap_cursor_respects_existing_archival_log(tmp_path):
    """
    Verifies first-time init on an existing memory.db seeds belief_archivist_last_log_id
    to MAX(archival_log.id) so live agents do not replay 100k+ ancient rows from id=0,
    while a fresh empty DB seeds '0'.
    """
    fresh_db = str(tmp_path / "fresh.db")
    fresh_bg = BeliefGraphComponent(db_path=fresh_db)
    conn = sqlite3.connect(fresh_db)
    try:
        cur_val = conn.execute(
            "SELECT value FROM memory_settings WHERE key = 'belief_archivist_last_log_id'"
        ).fetchone()[0]
        assert cur_val == "0"
    finally:
        conn.close()

    existing_db = str(tmp_path / "existing.db")
    conn = sqlite3.connect(existing_db)
    try:
        with conn:
            conn.execute(
                "CREATE TABLE archival_log (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT)"
            )
            for i in range(15):
                conn.execute("INSERT INTO archival_log (role, content) VALUES ('user', ?)", (f"msg {i}",))
    finally:
        conn.close()

    BeliefGraphComponent(db_path=existing_db)
    conn = sqlite3.connect(existing_db)
    try:
        cur_val = conn.execute(
            "SELECT value FROM memory_settings WHERE key = 'belief_archivist_last_log_id'"
        ).fetchone()[0]
        assert cur_val == "15"
    finally:
        conn.close()


def test_retraction_tombstone_fts_sync_and_cascading_doubt(tmp_path):
    """
    2. Retraction tombstone & cascading doubt:
    retract_belief preserves the original claim + revision_note, writes a 'supersedes' edge
    when replaced, keeps external-content FTS5 synchronized, and flips incoming 'depends_on'
    active/tentative nodes to 'needs_review'.
    """
    db_path = str(tmp_path / "memory.db")
    bg = BeliefGraphComponent(db_path=db_path)

    old_id = bg.upsert_belief(
        claim="Spherical isotropic model fits RXJ1131 kinematics",
        epistemic_bucket="working_hypothesis",
        status="active",
        confidence=0.8,
    )
    dep_active = bg.upsert_belief(
        claim="H0 inference from RXJ1131 spherical run is unbiased",
        epistemic_bucket="working_hypothesis",
        status="active",
        confidence=0.75,
    )
    dep_tentative = bg.upsert_belief(
        claim="Cluster alpha shares RXJ1131 spherical calibration",
        epistemic_bucket="working_hypothesis",
        status="tentative",
        confidence=0.5,
    )
    unrelated_related = bg.upsert_belief(
        claim="TDCOSMO notebooks document lens modeling conventions",
        epistemic_bucket="verified_fact",
        status="active",
        confidence=0.9,
    )

    bg.upsert_edge(dep_active, old_id, "depends_on", weight=1.0)
    bg.upsert_edge(dep_tentative, old_id, "depends_on", weight=0.9)
    bg.upsert_edge(unrelated_related, old_id, "related_to", weight=0.7)

    replacement_id = bg.upsert_belief(
        claim="RXJ1131 requires an axisymmetric JAM elliptical mass profile",
        epistemic_bucket="verified_fact",
        status="active",
        confidence=0.92,
    )

    affected = bg.retract_belief(
        belief_id=old_id,
        status="superseded",
        revision_note="Debunked after removing hardcoded qobs_lum=0.8 constraint",
        superseded_by_id=replacement_id,
    )
    assert affected == sorted([dep_active, dep_tentative])

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT claim, status, revision_note FROM beliefs WHERE id = ?",
            (old_id,),
        ).fetchone()
        assert row == (
            "Spherical isotropic model fits RXJ1131 kinematics",
            "superseded",
            "Debunked after removing hardcoded qobs_lum=0.8 constraint",
        )

        statuses = dict(
            conn.execute(
                "SELECT id, status FROM beliefs WHERE id IN (?, ?, ?)",
                (dep_active, dep_tentative, unrelated_related),
            ).fetchall()
        )
        assert statuses[dep_active] == "needs_review"
        assert statuses[dep_tentative] == "needs_review"
        # 'related_to' edges do NOT trigger cascading doubt
        assert statuses[unrelated_related] == "active"

        sup_edge = conn.execute(
            "SELECT weight FROM belief_edges WHERE source_id = ? AND target_id = ? AND relation = 'supersedes'",
            (replacement_id, old_id),
        ).fetchone()
        assert sup_edge is not None and sup_edge[0] == pytest.approx(1.0)

        # Verify external-content FTS5 triggers kept revision_note searchable without corruption
        fts_hits = bg.find_seed_ids("qobs_lum constraint")
        assert old_id in fts_hits
    finally:
        conn.close()


def test_adaptive_spidering_and_boilerplate_filtering(tmp_path):
    """
    3. Adaptive spidering (spider_beliefs):
    Respects max_depth, min_weight, and node_budget for related_to edges, while ALWAYS
    traversing supersedes and contradicts edges even below min_weight. Also strips
    MQTT/NVR attachment boilerplate so camera alerts don't match technical beliefs.
    """
    db_path = str(tmp_path / "memory.db")
    bg = BeliefGraphComponent(db_path=db_path)

    seed_tombstone = bg.upsert_belief(
        claim="Cassowary pipeline uses legacy spherical solver",
        epistemic_bucket="working_hypothesis",
        status="superseded",
        confidence=0.2,
        revision_note="Replaced by elliptical solver",
    )
    replacement = bg.upsert_belief(
        claim="Axisymmetric solver replaced Cassowary spherical pipeline",
        epistemic_bucket="verified_fact",
        status="active",
        confidence=0.95,
    )
    contradicting = bg.upsert_belief(
        claim="Legacy solver diverges on high-ellipticity lenses",
        epistemic_bucket="verified_fact",
        status="active",
        confidence=0.85,
    )
    strong_hop1 = bg.upsert_belief(
        claim="Emcee sampler requires 64 walkers for convergence",
        epistemic_bucket="working_hypothesis",
        status="active",
        confidence=0.7,
    )
    weak_hop1 = bg.upsert_belief(
        claim="Unrelated Weak Neighbor Below Threshold",
        epistemic_bucket="working_hypothesis",
        status="active",
        confidence=0.6,
    )
    hop2 = bg.upsert_belief(
        claim="Autocorrelation time check validates walker burn-in",
        epistemic_bucket="working_hypothesis",
        status="active",
        confidence=0.65,
    )
    system_image_belief = bg.upsert_belief(
        claim="System image library pipeline uses Docker inspect",
        epistemic_bucket="verified_fact",
        status="active",
        confidence=0.9,
    )

    # Even if supersedes/contradicts weights were stored low (0.1), spider MUST traverse them without decay
    bg.upsert_edge(replacement, seed_tombstone, "supersedes", weight=0.1)
    bg.upsert_edge(contradicting, seed_tombstone, "contradicts", weight=0.1)
    bg.upsert_edge(seed_tombstone, strong_hop1, "related_to", weight=0.8)
    bg.upsert_edge(seed_tombstone, weak_hop1, "related_to", weight=0.2)  # below min_weight=0.35
    bg.upsert_edge(strong_hop1, hop2, "related_to", weight=0.8)

    # Depth 1 spidering on "Cassowary"
    d1_nodes = bg.spider_beliefs("How does Cassowary work?", max_depth=1, min_weight=0.35, node_budget=8)
    d1_ids = {n["id"] for n in d1_nodes}
    assert seed_tombstone in d1_ids
    assert replacement in d1_ids
    assert contradicting in d1_ids
    assert strong_hop1 in d1_ids
    assert weak_hop1 not in d1_ids
    assert hop2 not in d1_ids

    # Relevance of supersedes/contradicts does not decay (stays 1.0)
    rel_by_id = {n["id"]: n["relevance"] for n in d1_nodes}
    assert rel_by_id[replacement] == pytest.approx(1.0)
    assert rel_by_id[contradicting] == pytest.approx(1.0)
    assert rel_by_id[strong_hop1] == pytest.approx(0.8)

    # Depth 2 spidering reaches hop2
    d2_nodes = bg.spider_beliefs("How does Cassowary work?", max_depth=2, min_weight=0.35, node_budget=8)
    d2_ids = {n["id"] for n in d2_nodes}
    assert hop2 in d2_ids

    # Node budget cap works
    capped = bg.spider_beliefs("How does Cassowary work?", max_depth=2, min_weight=0.35, node_budget=2)
    assert len(capped) == 2

    # NVR camera alert with [Via MQTT | System] and [Attached Image: ... library ... inspect]
    # must NOT spuriously match `system_image_belief`
    nvr_msg = (
        "[Via MQTT | System] NVR:person saw a face a person\n"
        "[Attached Image: 'Gate_1795286329318.jpg' (1024x576px, type: image/jpeg, "
        "saved in library 'discord_file' — call view_image('Gate_1795286329318.jpg') to inspect)]"
    )
    nvr_hits = bg.spider_beliefs(nvr_msg, max_depth=1)
    assert all(n["id"] != system_image_belief for n in nvr_hits)


def test_hebbian_reinforcement_and_selective_edge_decay(tmp_path):
    """
    4. Mechanical Hebbian weighting & decay:
    reinforce_coactivated_edges increments related_to weight (capped at 1.0), and
    decay_related_edges decays/prunes stale related_to edges without decaying
    depends_on, supersedes, or contradicts.
    """
    db_path = str(tmp_path / "memory.db")
    bg = BeliefGraphComponent(db_path=db_path)

    b1 = bg.upsert_belief("Claim A")
    b2 = bg.upsert_belief("Claim B")
    b3 = bg.upsert_belief("Claim C")
    b4 = bg.upsert_belief("Claim D")

    bg.upsert_edge(b1, b2, "related_to", weight=0.97)
    bg.upsert_edge(b2, b3, "related_to", weight=0.15)
    bg.upsert_edge(b1, b3, "depends_on", weight=0.20)
    bg.upsert_edge(b4, b1, "supersedes", weight=0.20)
    bg.upsert_edge(b3, b4, "contradicts", weight=0.20)

    # Reinforce co-activated pair (b1, b2) twice -> capped at 1.0; b2->b3 untouched since b3 not in list
    bg.reinforce_coactivated_edges([b1, b2], step=0.05)
    conn = sqlite3.connect(db_path)
    try:
        w_12 = conn.execute(
            "SELECT weight FROM belief_edges WHERE source_id=? AND target_id=? AND relation='related_to'",
            (b1, b2),
        ).fetchone()[0]
        assert w_12 == pytest.approx(1.0)
    finally:
        conn.close()

    # Decay: b2->b3 (0.15 * 0.98 = 0.147 < 0.15) gets pruned; structural edges at 0.20 stay untouched!
    pruned = bg.decay_related_edges(factor=0.98, prune_floor=0.15)
    assert pruned == 1

    conn = sqlite3.connect(db_path)
    try:
        edges = {
            (s, t, r): w
            for s, t, r, w in conn.execute(
                "SELECT source_id, target_id, relation, weight FROM belief_edges"
            ).fetchall()
        }
        assert (b2, b3, "related_to") not in edges
        assert edges[(b1, b2, "related_to")] == pytest.approx(0.98)
        # Structural edges must NEVER decay even if weight <= 0.20
        assert edges[(b1, b3, "depends_on")] == pytest.approx(0.20)
        assert edges[(b4, b1, "supersedes")] == pytest.approx(0.20)
        assert edges[(b3, b4, "contradicts")] == pytest.approx(0.20)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_archivist_trigger_gate_batching_and_crash_safety(tmp_path):
    """
    5. Archivist trigger gate, batching, & crash-safe cursor:
    - Does not invoke LLM below archival_row_threshold when belief_inbox is empty.
    - Triggers immediately when flush_inbox_immediately=True and belief_inbox has a pending row.
    - Applies mocked JSON output (add_beliefs, revise_beliefs, add_edges), resolves temp_id
      references, cascades needs_review, marks inbox rows processed, and advances cursor.
    - On mocked LLM/JSON failure, leaves belief_archivist_last_log_id and belief_inbox.processed unchanged.
    """
    db_path = str(tmp_path / "memory.db")
    audit_log_path = str(tmp_path / "session.jsonl")
    bg = BeliefGraphComponent(db_path=db_path, thread_id="mind")

    old_belief_id = bg.upsert_belief(
        claim="everybody-codes-data.git is our writable shared repository",
        epistemic_bucket="working_hypothesis",
        status="active",
        confidence=0.7,
    )
    dependent_id = bg.upsert_belief(
        claim="Push daily solver scripts to everybody-codes-data.git origin/main",
        epistemic_bucket="working_hypothesis",
        status="active",
        confidence=0.65,
    )
    bg.upsert_edge(dependent_id, old_belief_id, "depends_on", weight=1.0)

    llm_calls = []
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=None))
    )

    archivist = BeliefArchivist(
        db_path=db_path,
        graph=bg,
        archival_row_threshold=5,
        max_rows_per_batch=10,
        max_chars_per_row=200,
        flush_inbox_immediately=True,
        provider="openai",
        api_url="http://localhost:11435/v1",
        model="gemma4:31b",
        client=mock_client,
        log_file=audit_log_path,
    )

    def _unexpected_llm(**kwargs):
        llm_calls.append(kwargs)
        raise AssertionError("LLM should not be called below archival_row_threshold")

    mock_client.chat.completions.create = _unexpected_llm

    # Insert 3 archival_log rows (< threshold=5) and 0 inbox rows -> must NOT call LLM
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            for i in range(3):
                conn.execute(
                    "INSERT INTO archival_log (thread_id, role, content) VALUES ('mind', 'user', ?)",
                    (f"Observation {i} with padding " + ("x" * 300),),
                )
    finally:
        conn.close()

    res_below = await archivist.run_once()
    assert res_below["triggered"] is False
    assert len(llm_calls) == 0
    assert archivist.get_last_log_id() == 0

    # Queue an explicit Boss declaration in belief_inbox -> triggers immediately even with 3 log rows!
    bg.record_belief(
        claim="broken.git on spark-git is the only writable shared git repository",
        stance="held",
        reason="Confirmed by fireduck; everybody-codes-data.git is read-only reference",
    )

    # First simulate a malformed JSON failure -> cursor and inbox must remain untouched!
    def _failing_llm(**kwargs):
        llm_calls.append(kwargs)
        return _mock_completion("NOT VALID JSON {{{")

    mock_client.chat.completions.create = _failing_llm
    res_fail = await archivist.run_once()
    assert res_fail["triggered"] is False
    assert "error" in res_fail
    assert archivist.get_last_log_id() == 0

    conn = sqlite3.connect(db_path)
    try:
        unprocessed = conn.execute(
            "SELECT COUNT(*) FROM belief_inbox WHERE processed = 0"
        ).fetchone()[0]
        assert unprocessed == 1
    finally:
        conn.close()

    # Now return valid epistemic review JSON using temp_id references and superseding old_belief_id
    valid_payload = {
        "add_beliefs": [
            {
                "temp_id": "new_1",
                "claim": "broken.git on spark-git is the only writable shared git repository",
                "epistemic_bucket": "verified_fact",
                "status": "active",
                "confidence": 0.98,
                "evidence_refs": [1, 2],
            },
            {
                "temp_id": "new_2",
                "claim": "The Forgotten Repo of Floor 5 guards ancient solver commits",
                "epistemic_bucket": "lore_mythology",
                "status": "active",
                "confidence": 0.5,
                "evidence_refs": [],
            },
        ],
        "revise_beliefs": [
            {
                "belief_id": old_belief_id,
                "status": "superseded",
                "epistemic_bucket": "working_hypothesis",
                "confidence": 0.05,
                "revision_note": "everybody-codes-data.git is a read-only reference clone; use broken.git",
                "superseded_by_ref": "new_1",
            }
        ],
        "add_edges": [
            {
                "source_ref": "new_2",
                "target_ref": "new_1",
                "relation": "related_to",
                "weight": 0.6,
            }
        ],
    }

    captured_prompts = []

    def _succeeding_llm(**kwargs):
        captured_prompts.append(kwargs["messages"][1]["content"])
        return _mock_completion(json.dumps(valid_payload))

    mock_client.chat.completions.create = _succeeding_llm
    res_ok = await archivist.run_once()

    assert res_ok["triggered"] is True
    assert len(res_ok["added_belief_ids"]) == 2
    assert res_ok["revised_belief_ids"] == [old_belief_id]
    assert res_ok["cascaded_review_ids"] == [dependent_id]
    assert res_ok["inbox_processed"] == 1
    assert res_ok["last_log_id"] == 3
    assert archivist.get_last_log_id() == 3

    # Verify per-row archival_log truncation protected prompt size
    assert "... [truncated]" in captured_prompts[0]

    conn = sqlite3.connect(db_path)
    try:
        unprocessed = conn.execute(
            "SELECT COUNT(*) FROM belief_inbox WHERE processed = 0"
        ).fetchone()[0]
        assert unprocessed == 0

        dep_status = conn.execute(
            "SELECT status FROM beliefs WHERE id = ?", (dependent_id,)
        ).fetchone()[0]
        assert dep_status == "needs_review"

        new_1_id, new_2_id = res_ok["added_belief_ids"]
        sup_edge = conn.execute(
            "SELECT 1 FROM belief_edges WHERE source_id=? AND target_id=? AND relation='supersedes'",
            (new_1_id, old_belief_id),
        ).fetchone()
        assert sup_edge is not None

        rel_edge = conn.execute(
            "SELECT weight FROM belief_edges WHERE source_id=? AND target_id=? AND relation='related_to'",
            (new_2_id, new_1_id),
        ).fetchone()
        # Decayed once after commit (0.6 * 0.98)
        assert rel_edge is not None and rel_edge[0] == pytest.approx(0.6 * 0.98)
    finally:
        conn.close()

    with open(audit_log_path, "r", encoding="utf-8") as f:
        audit_events = [json.loads(line) for line in f if line.strip()]
    assert any(ev["event"] == "belief_graph_updated" for ev in audit_events)


@pytest.mark.asyncio
async def test_executive_assistant_transient_worldview_injection_and_tension_depth(tmp_path):
    """
    6. ExecutiveAssistant transient injection:
    Verifies [Worldview / Active Stances] is injected transiently on before_llm_call
    (before _transient_presence), stays at default_spider_depth=1 on calm social turns,
    escalates to deep_spider_depth=2 when has_tension is true, reinforces co-activated
    related_to edges, and strips _transient_worldview cleanly on after_llm_call / _clear_ephemeral_images.
    """
    db_path = str(tmp_path / "memory.db")
    bg = BeliefGraphComponent(db_path=db_path, thread_id="social")
    bg.default_spider_depth = 1
    bg.deep_spider_depth = 2

    # Calm component chain: A --related_to--> B --related_to--> C (no tension)
    a_id = bg.upsert_belief("Quasar modeling uses axisymmetric profiles", epistemic_bucket="verified_fact", confidence=0.9)
    b_id = bg.upsert_belief("Axisymmetric profiles pair well with emcee walkers", epistemic_bucket="working_hypothesis", confidence=0.75)
    c_id = bg.upsert_belief("Corner plots visualize posterior degeneracy", epistemic_bucket="lore_mythology", confidence=0.6)
    bg.upsert_edge(a_id, b_id, "related_to", weight=0.60)
    bg.upsert_edge(b_id, c_id, "related_to", weight=0.60)

    agent = AgentCore(thread_id="social")
    ea = ExecutiveAssistantComponent(thread_id="social", enabled=True, belief_comp=bg)
    agent.register_component(bg)
    agent.register_component(ea)

    agent.messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Tell me about Quasar modeling."},
        {"role": "system", "content": "[Threads: mind active]", "_transient_presence": True},
    ]

    await agent.event_bus.publish("before_llm_call", {"agent": agent, "messages": agent.messages})

    worldview_msgs = [m for m in agent.messages if m.get("_transient_worldview")]
    assert len(worldview_msgs) == 1
    wv_text = worldview_msgs[0]["content"]
    assert wv_text.startswith("[Worldview / Active Stances]")
    assert "Quasar modeling uses axisymmetric profiles" in wv_text
    assert "Axisymmetric profiles pair well with emcee walkers" in wv_text
    # Calm social turn stays at depth=1, so 2-hop C is NOT included yet
    assert "Corner plots visualize posterior degeneracy" not in wv_text
    # Inserted BEFORE _transient_presence footer
    assert agent.messages[-1].get("_transient_presence") is True
    assert agent.messages[-2].get("_transient_worldview") is True

    # Co-activated edge a_id -> b_id reinforced from 0.60 to 0.65
    conn = sqlite3.connect(db_path)
    try:
        w_ab = conn.execute(
            "SELECT weight FROM belief_edges WHERE source_id=? AND target_id=? AND relation='related_to'",
            (a_id, b_id),
        ).fetchone()[0]
        assert w_ab == pytest.approx(0.65)
    finally:
        conn.close()

    # Cleanup on after_llm_call removes _transient_worldview
    await agent.event_bus.publish("after_llm_call", {"agent": agent, "messages": agent.messages})
    assert not any(m.get("_transient_worldview") for m in agent.messages)

    # Now introduce tension on b_id (e.g. an obsolete belief it supersedes) -> social turn escalates to depth=2!
    obsolete_id = bg.upsert_belief(
        "Grid search beats MCMC on 12-parameter lenses",
        epistemic_bucket="working_hypothesis",
        status="active",
    )
    bg.retract_belief(
        belief_id=obsolete_id,
        status="superseded",
        revision_note="Cursed dimensionality above 4 parameters",
        superseded_by_id=b_id,
    )
    assert bg.has_tension([a_id, b_id]) is True

    agent.messages.append({"role": "assistant", "content": "Sure!"})
    agent.messages.append({"role": "user", "content": "Tell me more about Quasar modeling."})

    await agent.event_bus.publish("before_llm_call", {"agent": agent, "messages": agent.messages})
    worldview_msgs_2 = [m for m in agent.messages if m.get("_transient_worldview")]
    assert len(worldview_msgs_2) == 1
    wv_text_2 = worldview_msgs_2[0]["content"]
    # Escalated to depth=2: 2-hop lore C AND superseded tombstone with revision_note are both included!
    assert "Corner plots visualize posterior degeneracy" in wv_text_2
    assert "Lore & Mythology:" in wv_text_2
    assert "Refuted / Superseded (Do Not Resurrect):" in wv_text_2
    assert "Cursed dimensionality above 4 parameters" in wv_text_2

    # Also verify AgentCore._clear_ephemeral_images() strips _transient_worldview
    agent._clear_ephemeral_images()
    assert not any(m.get("_transient_worldview") for m in agent.messages)


def test_main_registration_provider_routing_and_workspace_docs(tmp_path):
    """
    Verifies _register_thread_components wires BeliefGraphComponent when [belief_graph].enabled
    or explicitly listed in components, BeliefArchivist supports GeminiOpenAIClientAdapter when
    provider='gemini', and MIND_README_CONTENT documents the belief tables.
    """
    from src.main import _register_thread_components
    from src.components.workspace import MIND_README_CONTENT
    from src.core.gemini_client import GeminiOpenAIClientAdapter
    from src.utils.config import load_config

    for tbl in ("beliefs", "beliefs_fts", "belief_edges", "belief_inbox"):
        assert tbl in MIND_README_CONTENT

    base_cfg = load_config()
    cfg = dict(base_cfg)
    cfg["memory"] = dict(base_cfg.get("memory", {}))
    cfg["memory"]["db_path"] = str(tmp_path / "reg_memory.db")
    cfg["belief_graph"] = {"enabled": True}

    agent = AgentCore(thread_id="mind")
    _register_thread_components(
        agent,
        cfg,
        thread_id="mind",
        enabled_components=["notes"],  # even if not explicitly listed, enabled=True attaches it
    )
    # Note: BeliefGraphComponent(thread_id="mind") loads global config db_path unless passed,
    # check component type registration:
    assert any(isinstance(c, BeliefGraphComponent) for c in agent.components)

    # Verify Gemini provider client initialization on BeliefArchivist
    gem_archivist = BeliefArchivist(
        db_path=str(tmp_path / "gem_memory.db"),
        provider="gemini",
        api_key="test-gemini-key",
        model="gemini-4-carbon",
    )
    assert isinstance(gem_archivist.client, GeminiOpenAIClientAdapter)
    assert gem_archivist.exec_location == "cloud"


@pytest.mark.asyncio
async def test_archivist_16k_thinking_headroom_inbox_batch_cap_and_ephemeral_guard(tmp_path):
    """
    Verifies:
    - BeliefArchivist defaults max_output_tokens to 16384 (>= 16k) with thinking enabled (no think=False),
      cleanly parsing <think>...</think> blocks ahead of JSON.
    - max_inbox_per_batch caps how many belief_inbox items are processed per run_once() call so large
      inbox backlogs drain in bounded chunks.
    - Tool instructions and Archivist system prompt explicitly discourage ephemeral status/traceback beliefs.
    """
    from src.components.belief_graph import DEFAULT_ARCHIVIST_SYSTEM_PROMPT

    db_path = str(tmp_path / "chunked_inbox.db")
    bg = BeliefGraphComponent(db_path=db_path, thread_id="mind")

    assert "ephemeral" in bg.get_system_prompt_addition().lower()
    assert "ephemeral" in (bg.record_belief.__doc__ or "").lower()
    assert "ephemeral" in DEFAULT_ARCHIVIST_SYSTEM_PROMPT.lower()

    mock_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=None))
    )
    archivist = BeliefArchivist(
        db_path=db_path,
        graph=bg,
        max_inbox_per_batch=3,
        flush_inbox_immediately=True,
        provider="openai",
        api_url="http://localhost:11435/v1",
        model="gemma4:31b",
        client=mock_client,
    )
    assert archivist.max_output_tokens >= 16384

    for idx in range(7):
        bg.record_belief(
            claim=f"Durable hypothesis #{idx}",
            stance="held",
            reason=f"Evidence item {idx}",
        )

    observed_kwargs = []

    def _thinking_llm(**kwargs):
        observed_kwargs.append(kwargs)
        user_text = kwargs["messages"][1]["content"]
        # Count how many inbox items were included in this batch prompt
        inbox_lines = [
            ln for ln in user_text.splitlines() if ln.startswith("- [inbox_id=")
        ]
        assert len(inbox_lines) <= 3
        long_thought = "<think>" + ("evaluating provenance and durability... " * 200) + "</think>\n"
        payload = {
            "add_beliefs": [],
            "revise_beliefs": [],
            "add_edges": [],
        }
        return _mock_completion(long_thought + json.dumps(payload))

    mock_client.chat.completions.create = _thinking_llm

    # Pass 1 processes 3 of 7
    r1 = await archivist.run_once()
    assert r1["triggered"] is True
    assert r1["inbox_processed"] == 3
    assert observed_kwargs[-1]["max_tokens"] >= 16384
    assert "extra_body" not in observed_kwargs[-1]  # thinking stays enabled

    # Pass 2 processes next 3 of 7
    r2 = await archivist.run_once()
    assert r2["triggered"] is True
    assert r2["inbox_processed"] == 3

    # Pass 3 processes final 1 of 7
    r3 = await archivist.run_once()
    assert r3["triggered"] is True
    assert r3["inbox_processed"] == 1

    # Pass 4 sees 0 remaining inbox rows and 0 archival_log rows -> does not trigger
    r4 = await archivist.run_once()
    assert r4["triggered"] is False
    assert len(observed_kwargs) == 3


def test_dump_beliefs_script_jsonl_table_filters_and_missing_tables(tmp_path):
    """
    Verifies scripts/dump-beliefs.py:
    - Dumps beliefs, belief_edges, and belief_inbox as pure JSONL with decoded evidence_refs
      and boolean processed flags.
    - Supports table filters (beliefs, edges, inbox) via CLI.
    - Handles an existing SQLite DB that lacks belief tables cleanly without mutating it.
    """
    import subprocess
    import sys

    db_path = tmp_path / "dump_test.db"
    bg = BeliefGraphComponent(db_path=str(db_path), thread_id="mind")

    b1 = bg.upsert_belief(
        claim="Spine order preserves EPUB frontmatter and appendices",
        epistemic_bucket="verified_fact",
        status="active",
        confidence=0.95,
        evidence_refs=["log:101", "file_reader.py"],
    )
    b2 = bg.upsert_belief(
        claim="Lobster-watch 2x2 storyboards fit cleanly in 1024px JPEG frames",
        epistemic_bucket="working_hypothesis",
        status="tentative",
        confidence=0.8,
    )
    bg.upsert_edge(b2, b1, "related_to", weight=0.75)
    bg.record_belief(
        claim="Garak is more than a simple tailor",
        stance="held",
        reason="Observed in DS9 S01E02 Past Prologue",
    )

    script_path = "scripts/dump-beliefs.py"
    res_all = subprocess.run(
        [sys.executable, script_path, "-d", str(db_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [json.loads(ln) for ln in res_all.stdout.strip().splitlines()]
    assert [r["table"] for r in lines] == ["beliefs", "beliefs", "belief_edges", "belief_inbox"]

    belief_rows = [r for r in lines if r["table"] == "beliefs"]
    assert belief_rows[0]["id"] == b1
    assert belief_rows[0]["evidence_refs"] == ["log:101", "file_reader.py"]
    assert belief_rows[1]["id"] == b2
    assert belief_rows[1]["evidence_refs"] == []

    edge_row = next(r for r in lines if r["table"] == "belief_edges")
    assert edge_row["source_id"] == b2
    assert edge_row["target_id"] == b1
    assert edge_row["relation"] == "related_to"
    assert edge_row["weight"] == pytest.approx(0.75)

    inbox_row = next(r for r in lines if r["table"] == "belief_inbox")
    assert inbox_row["thread_id"] == "mind"
    assert inbox_row["claim"] == "Garak is more than a simple tailor"
    assert inbox_row["stance"] == "held"
    assert inbox_row["processed"] is False

    # Filter to edges only via positional arg
    res_edges = subprocess.run(
        [sys.executable, script_path, "-d", str(db_path), "edges"],
        capture_output=True,
        text=True,
        check=True,
    )
    edge_only_lines = [json.loads(ln) for ln in res_edges.stdout.strip().splitlines()]
    assert len(edge_only_lines) == 1
    assert edge_only_lines[0]["table"] == "belief_edges"

    # Config resolution (-c) fallback to [memory].db_path
    cfg_file = tmp_path / "mini_config.toml"
    cfg_file.write_text(f'[memory]\ndb_path = "{db_path}"\n', encoding="utf-8")
    res_cfg = subprocess.run(
        [sys.executable, script_path, "-c", str(cfg_file), "inbox"],
        capture_output=True,
        text=True,
        check=True,
    )
    cfg_lines = [json.loads(ln) for ln in res_cfg.stdout.strip().splitlines()]
    assert len(cfg_lines) == 1
    assert cfg_lines[0]["table"] == "belief_inbox"

    # Legacy SQLite DB with no belief tables outputs 0 lines cleanly and does not create tables
    bare_db = tmp_path / "bare.db"
    conn = sqlite3.connect(str(bare_db))
    conn.execute("CREATE TABLE dummy (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    res_bare = subprocess.run(
        [sys.executable, script_path, "-d", str(bare_db)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res_bare.stdout == ""



