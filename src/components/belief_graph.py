import asyncio
from datetime import datetime, timezone
import inspect
import json
import os
import re
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import httpx
from openai import AsyncOpenAI

from src.components.base import BaseComponent
from src.core.gemini_client import GeminiOpenAIClientAdapter
from src.utils.config import load_config
from src.utils.metrics import (
    MetricSpan,
    count_tokens,
    extract_chunk_usage,
    infer_exec_location,
)


VALID_EPISTEMIC_BUCKETS = (
    "verified_fact",
    "working_hypothesis",
    "value_preference",
    "lore_mythology",
)
VALID_BELIEF_STATUSES = (
    "active",
    "tentative",
    "needs_review",
    "refuted",
    "superseded",
)
VALID_INBOX_STANCES = ("held", "tentative", "refuted", "lore")
VALID_EDGE_RELATIONS = ("depends_on", "supersedes", "contradicts", "related_to")

_SEARCH_STOPWORDS: Set[str] = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "and",
    "any",
    "are",
    "attached",
    "assistant",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "call",
    "can",
    "could",
    "current",
    "currently",
    "did",
    "discord",
    "discord_file",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "event",
    "face",
    "few",
    "file",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "image",
    "incoming",
    "inspect",
    "into",
    "its",
    "jpeg",
    "jpg",
    "just",
    "library",
    "like",
    "message",
    "more",
    "most",
    "mqtt",
    "must",
    "new",
    "not",
    "now",
    "nvr",
    "off",
    "once",
    "only",
    "other",
    "our",
    "out",
    "over",
    "own",
    "person",
    "png",
    "result",
    "same",
    "saved",
    "saw",
    "see",
    "she",
    "should",
    "some",
    "such",
    "system",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "tool",
    "too",
    "type",
    "under",
    "until",
    "user",
    "vehicle",
    "very",
    "via",
    "view",
    "view_image",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}

DEFAULT_ARCHIVIST_SYSTEM_PROMPT = """You are the epistemic Belief Archivist for an autonomous AI agent.
Your job is to review new Boss belief declarations (`belief_inbox`) and recent transcript batches (`archival_log`) alongside existing candidate beliefs, and emit strict JSON graph updates.

### Epistemic Discipline & Provenance Rules
1. `epistemic_bucket` values:
   - `"verified_fact"`: RESERVED STRICTLY for claims directly confirmed by successful (exit-0) tool execution or explicitly stated as ground truth by the human operator (`fireduck`). NEVER classify agent deductions, unverified peer claims, or speculations about failed commands, HTTP 403/proxy blocks, or container firewall errors as `"verified_fact"` — those MUST be `"working_hypothesis"` at most.
   - `"working_hypothesis"`: Testable technical theories, explanations, ongoing research assumptions, or inferred causes.
   - `"value_preference"`: Explicit agent or operator preferences, collaborative norms, or aesthetic priorities.
   - `"lore_mythology"`: Playful stories, fictional worldbuilding, metaphors, nicknames, or symbolic lore (`stance="lore"`).
2. `status` values:
   - For `add_beliefs`: `"active"` or `"tentative"`.
   - For `revise_beliefs`: `"active"`, `"tentative"`, `"refuted"`, or `"superseded"`.
3. Tombstones & Retractions:
   - Never ignore an existing belief when new evidence contradicts or replaces it. Emit a `revise_beliefs` entry marking the old belief `"refuted"` or `"superseded"` with a concise `revision_note` (and `superseded_by_ref` when replaced by a newer belief).
   - If an existing belief is in `"needs_review"`, either reaffirm it (`"active"`/`"tentative"`), update it, or mark it `"refuted"`/`"superseded"` if the batch addresses it.
4. High Signal Only:
   - Ignore routine greetings, NVR camera motion alerts, clock ticks, raw book chunks, and transient CLI/network hiccups.
   - If nothing durable or belief-worthy happened, return empty arrays: `{"add_beliefs": [], "revise_beliefs": [], "add_edges": []}`.
5. Edges (`relation`):
   - `"depends_on"`: `source_ref` relies on `target_ref` being true (if `target_ref` is later refuted, `source_ref` will automatically flip to `needs_review`).
   - `"supersedes"`: `source_ref` replaces obsolete/retracted `target_ref`.
   - `"contradicts"`: `source_ref` and `target_ref` are in direct tension.
   - `"related_to"`: Associative conceptual link (`weight` between `0.2` and `1.0`).

### Output Format
Output STRICT JSON ONLY (no markdown prose, no commentary) matching this exact schema:
{
  "add_beliefs": [
    {
      "temp_id": "new_1",
      "claim": "Concise, self-contained statement of the belief",
      "epistemic_bucket": "verified_fact|working_hypothesis|value_preference|lore_mythology",
      "status": "active|tentative",
      "confidence": 0.75,
      "evidence_refs": [1042]
    }
  ],
  "revise_beliefs": [
    {
      "belief_id": 12,
      "status": "active|tentative|refuted|superseded",
      "epistemic_bucket": "verified_fact|working_hypothesis|value_preference|lore_mythology",
      "confidence": 0.2,
      "revision_note": "Why this changed or was debunked",
      "superseded_by_ref": "new_1"
    }
  ],
  "add_edges": [
    {
      "source_ref": "new_1",
      "target_ref": 12,
      "relation": "depends_on|supersedes|contradicts|related_to",
      "weight": 0.8
    }
  ]
}"""


class BeliefGraphComponent(BaseComponent):
    """
    SQLite-backed epistemic belief graph stored in `memory.db`.
    Provides a zero-overhead `record_belief` declaration inbox for the Boss and
    deterministic graph operations (tombstone retractions, cascading doubt,
    FTS5-seeded recursive CTE spidering, and Hebbian edge weighting/decay).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        thread_id: str = "main",
        enabled: Optional[bool] = None,
        bootstrap_from_latest: Optional[bool] = None,
    ):
        self.config = load_config()
        self.thread_id = (thread_id or "main").strip() or "main"

        bg_cfg = self.config.get("belief_graph", {})
        mem_cfg = self.config.get("memory", {})

        resolved_db = db_path or bg_cfg.get("db_path") or mem_cfg.get("db_path")
        if not resolved_db:
            raise ValueError(
                "BeliefGraphComponent requires 'db_path' under [belief_graph].db_path or [memory].db_path"
            )
        self.db_path = resolved_db

        self.enabled = True if enabled is None else bool(enabled)
        self.bootstrap_from_latest = (
            bool(bg_cfg.get("bootstrap_from_latest", True))
            if bootstrap_from_latest is None
            else bool(bootstrap_from_latest)
        )

        self.default_spider_depth = int(bg_cfg.get("default_spider_depth", 1))
        self.deep_spider_depth = int(bg_cfg.get("deep_spider_depth", 2))
        self.min_edge_weight = float(bg_cfg.get("min_edge_weight", 0.35))
        self.node_budget = int(bg_cfg.get("node_budget", 8))

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS archival_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        role TEXT,
                        content TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        thread_id TEXT NOT NULL DEFAULT 'main',
                        tool_calls TEXT,
                        tool_call_id TEXT,
                        name TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                    """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS beliefs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        claim TEXT NOT NULL,
                        epistemic_bucket TEXT NOT NULL DEFAULT 'working_hypothesis',
                        status TEXT NOT NULL DEFAULT 'active',
                        confidence REAL NOT NULL DEFAULT 0.6,
                        revision_note TEXT NOT NULL DEFAULT '',
                        evidence_refs TEXT NOT NULL DEFAULT '[]',
                        activation_count INTEGER NOT NULL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_beliefs_bucket ON beliefs(epistemic_bucket)"
                )

                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS beliefs_fts USING fts5(
                        claim,
                        revision_note,
                        content='beliefs',
                        content_rowid='id'
                    )
                    """
                )

                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS beliefs_ai AFTER INSERT ON beliefs BEGIN
                        INSERT INTO beliefs_fts(rowid, claim, revision_note)
                        VALUES (new.id, new.claim, new.revision_note);
                    END
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS beliefs_ad AFTER DELETE ON beliefs BEGIN
                        INSERT INTO beliefs_fts(beliefs_fts, rowid, claim, revision_note)
                        VALUES ('delete', old.id, old.claim, old.revision_note);
                    END
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS beliefs_au AFTER UPDATE OF claim, revision_note ON beliefs BEGIN
                        INSERT INTO beliefs_fts(beliefs_fts, rowid, claim, revision_note)
                        VALUES ('delete', old.id, old.claim, old.revision_note);
                        INSERT INTO beliefs_fts(rowid, claim, revision_note)
                        VALUES (new.id, new.claim, new.revision_note);
                    END
                    """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS belief_edges (
                        source_id INTEGER NOT NULL,
                        target_id INTEGER NOT NULL,
                        relation TEXT NOT NULL,
                        weight REAL NOT NULL DEFAULT 1.0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (source_id, target_id, relation),
                        FOREIGN KEY (source_id) REFERENCES beliefs(id) ON DELETE CASCADE,
                        FOREIGN KEY (target_id) REFERENCES beliefs(id) ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_belief_edges_source "
                    "ON belief_edges(source_id, relation, weight DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_belief_edges_target "
                    "ON belief_edges(target_id, relation, weight DESC)"
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS belief_inbox (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT NOT NULL DEFAULT 'main',
                        claim TEXT NOT NULL,
                        stance TEXT NOT NULL,
                        reason TEXT NOT NULL DEFAULT '',
                        processed INTEGER NOT NULL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_belief_inbox_unprocessed "
                    "ON belief_inbox(processed, id)"
                )

                if self.bootstrap_from_latest:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO memory_settings (key, value)
                        VALUES (
                            'belief_archivist_last_log_id',
                            CAST((SELECT COALESCE(MAX(id), 0) FROM archival_log) AS TEXT)
                        )
                        """
                    )
                else:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO memory_settings (key, value)
                        VALUES ('belief_archivist_last_log_id', '0')
                        """
                    )
        finally:
            conn.close()

    def get_system_prompt_addition(self) -> str:
        if not self.enabled:
            return ""
        return (
            "You maintain an epistemic belief graph separately from raw memories. "
            "Use `record_belief(claim, stance, reason)` whenever you deliberately adopt a new stance, "
            "revise an old assumption, mark something `refuted`, or reclassify an idea as `lore`. "
            "A background archivist maintains links and revisions automatically."
        )

    def get_tools(self) -> list:
        if not self.enabled:
            return []
        return [self.record_belief]

    def record_belief(self, claim: str, stance: str = "held", reason: str = "") -> str:
        """
        Records a deliberate epistemic stance, hypothesis revision, refutation, or piece of lore
        into your belief inbox for background graph indexing.

        Args:
            claim: Clear, self-contained statement of the belief or retracted assumption.
            stance: One of 'held', 'tentative', 'refuted', or 'lore'.
            reason: Brief evidence, context, or reason for adopting or retracting this belief.
        """
        clean_claim = (claim or "").strip()
        if not clean_claim:
            return "Error: 'claim' cannot be empty."

        clean_stance = (stance or "").strip().lower()
        if clean_stance not in VALID_INBOX_STANCES:
            allowed = ", ".join(VALID_INBOX_STANCES)
            return f"Error: Invalid stance '{stance}'. Must be one of: {allowed}."

        clean_reason = (reason or "").strip()
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO belief_inbox (thread_id, claim, stance, reason, processed)
                    VALUES (?, ?, ?, ?, 0)
                    """,
                    (self.thread_id, clean_claim, clean_stance, clean_reason),
                )
        finally:
            conn.close()

        return "Noted. Belief declaration queued for epistemic review."

    @staticmethod
    def _clamp_float(val: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.6) -> float:
        try:
            f = float(val)
        except (TypeError, ValueError):
            f = default
        return max(lo, min(hi, f))

    @staticmethod
    def _normalize_evidence_refs(refs: Optional[List[Any]]) -> List[Any]:
        if not refs or not isinstance(refs, list):
            return []
        out: List[Any] = []
        seen: Set[str] = set()
        for item in refs:
            if item is None:
                continue
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                norm: Any = int(item)
            else:
                s = str(item).strip()
                if not s:
                    continue
                norm = int(s) if s.isdigit() else s
            key = f"{type(norm).__name__}:{norm}"
            if key not in seen:
                seen.add(key)
                out.append(norm)
        return out

    def upsert_belief(
        self,
        claim: str,
        epistemic_bucket: str = "working_hypothesis",
        status: str = "active",
        confidence: float = 0.6,
        revision_note: str = "",
        evidence_refs: Optional[List[Any]] = None,
        belief_id: Optional[int] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        clean_claim = (claim or "").strip()
        if not clean_claim and belief_id is None:
            raise ValueError("Belief claim cannot be empty.")

        bucket = (epistemic_bucket or "working_hypothesis").strip().lower()
        if bucket not in VALID_EPISTEMIC_BUCKETS:
            raise ValueError(f"Invalid epistemic_bucket '{epistemic_bucket}'.")

        clean_status = (status or "active").strip().lower()
        if clean_status not in VALID_BELIEF_STATUSES:
            raise ValueError(f"Invalid belief status '{status}'.")

        conf = self._clamp_float(confidence, 0.0, 1.0, default=0.6)
        clean_rev = (revision_note or "").strip()
        norm_refs = self._normalize_evidence_refs(evidence_refs)

        owns_conn = conn is None
        db = self._connect_db() if owns_conn else conn
        assert db is not None
        try:
            def _do_upsert() -> int:
                target_id = int(belief_id) if belief_id is not None else None
                existing_row = None
                if target_id is not None:
                    existing_row = db.execute(
                        "SELECT id, claim, evidence_refs FROM beliefs WHERE id = ?",
                        (target_id,),
                    ).fetchone()
                    if existing_row is None:
                        raise ValueError(f"Belief id {target_id} does not exist.")
                else:
                    existing_row = db.execute(
                        "SELECT id, claim, evidence_refs FROM beliefs WHERE LOWER(TRIM(claim)) = LOWER(?) LIMIT 1",
                        (clean_claim,),
                    ).fetchone()

                if existing_row is not None:
                    bid = int(existing_row[0])
                    final_claim = clean_claim or str(existing_row[1])
                    try:
                        prev_refs = json.loads(existing_row[2] or "[]")
                    except Exception:
                        prev_refs = []
                    merged_refs = (
                        self._normalize_evidence_refs(list(prev_refs) + norm_refs)
                        if evidence_refs is not None
                        else self._normalize_evidence_refs(prev_refs)
                    )
                    db.execute(
                        """
                        UPDATE beliefs
                        SET claim = ?,
                            epistemic_bucket = ?,
                            status = ?,
                            confidence = ?,
                            revision_note = ?,
                            evidence_refs = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (
                            final_claim,
                            bucket,
                            clean_status,
                            conf,
                            clean_rev,
                            json.dumps(merged_refs),
                            bid,
                        ),
                    )
                    return bid

                cur = db.execute(
                    """
                    INSERT INTO beliefs (
                        claim, epistemic_bucket, status, confidence, revision_note, evidence_refs
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clean_claim,
                        bucket,
                        clean_status,
                        conf,
                        clean_rev,
                        json.dumps(norm_refs),
                    ),
                )
                return int(cur.lastrowid)

            if owns_conn:
                with db:
                    return _do_upsert()
            return _do_upsert()
        finally:
            if owns_conn:
                db.close()

    def upsert_edge(
        self,
        source_id: int,
        target_id: int,
        relation: str,
        weight: float = 1.0,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        clean_rel = (relation or "").strip().lower()
        if clean_rel not in VALID_EDGE_RELATIONS:
            raise ValueError(f"Invalid edge relation '{relation}'.")
        src = int(source_id)
        tgt = int(target_id)
        if src == tgt:
            return
        w = self._clamp_float(weight, 0.0, 1.0, default=1.0)

        owns_conn = conn is None
        db = self._connect_db() if owns_conn else conn
        assert db is not None
        try:
            def _do_edge() -> None:
                db.execute(
                    """
                    INSERT INTO belief_edges (source_id, target_id, relation, weight)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, relation) DO UPDATE SET
                        weight = excluded.weight,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (src, tgt, clean_rel, w),
                )

            if owns_conn:
                with db:
                    _do_edge()
            else:
                _do_edge()
        finally:
            if owns_conn:
                db.close()

    def retract_belief(
        self,
        belief_id: int,
        status: str = "refuted",
        revision_note: str = "",
        superseded_by_id: Optional[int] = None,
        epistemic_bucket: Optional[str] = None,
        confidence: Optional[float] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[int]:
        clean_status = (status or "").strip().lower()
        if clean_status not in ("refuted", "superseded"):
            raise ValueError(
                f"retract_belief requires status in ('refuted', 'superseded'), got '{status}'."
            )

        bid = int(belief_id)
        clean_rev = (revision_note or "").strip()

        owns_conn = conn is None
        db = self._connect_db() if owns_conn else conn
        assert db is not None
        try:
            def _do_retract() -> List[int]:
                row = db.execute(
                    "SELECT epistemic_bucket, confidence FROM beliefs WHERE id = ?",
                    (bid,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Belief id {bid} does not exist.")

                final_bucket = (
                    epistemic_bucket.strip().lower()
                    if epistemic_bucket and epistemic_bucket.strip().lower() in VALID_EPISTEMIC_BUCKETS
                    else str(row[0])
                )
                final_conf = (
                    self._clamp_float(confidence, 0.0, 1.0, default=float(row[1]))
                    if confidence is not None
                    else float(row[1])
                )

                db.execute(
                    """
                    UPDATE beliefs
                    SET status = ?,
                        epistemic_bucket = ?,
                        confidence = ?,
                        revision_note = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (clean_status, final_bucket, final_conf, clean_rev, bid),
                )

                if superseded_by_id is not None and int(superseded_by_id) != bid:
                    self.upsert_edge(
                        source_id=int(superseded_by_id),
                        target_id=bid,
                        relation="supersedes",
                        weight=1.0,
                        conn=db,
                    )

                dep_rows = db.execute(
                    """
                    SELECT b.id
                    FROM beliefs b
                    JOIN belief_edges e ON e.source_id = b.id
                    WHERE e.target_id = ?
                      AND e.relation = 'depends_on'
                      AND b.status IN ('active', 'tentative')
                    ORDER BY b.id ASC
                    """,
                    (bid,),
                ).fetchall()
                affected_ids = [int(r[0]) for r in dep_rows]

                if affected_ids:
                    db.execute(
                        """
                        UPDATE beliefs
                        SET status = 'needs_review',
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id IN (
                            SELECT source_id FROM belief_edges
                            WHERE target_id = ? AND relation = 'depends_on'
                        ) AND status IN ('active', 'tentative')
                        """,
                        (bid,),
                    )

                return affected_ids

            if owns_conn:
                with db:
                    return _do_retract()
            return _do_retract()
        finally:
            if owns_conn:
                db.close()

    @classmethod
    def extract_search_tokens(cls, query: str, max_tokens: int = 24) -> List[str]:
        raw = query or ""
        if raw.startswith("[Subconscious Recall Triggered"):
            parts = raw.split("--- End Recall ---\n\n", 1)
            if len(parts) == 2:
                raw = parts[1]

        # Strip transport headers like "[Via MQTT | System]" and image attachment footers
        cleaned = re.sub(r"\[Via [^\]]*\]", " ", raw)
        cleaned = re.sub(r"\[Attached Image:[^\]]*\]", " ", cleaned)

        tokens: List[str] = []
        seen: Set[str] = set()
        for tok in re.findall(r"[A-Za-z0-9_]+", cleaned):
            low = tok.lower()
            if low in _SEARCH_STOPWORDS or low in seen:
                continue
            has_digit = any(ch.isdigit() for ch in low)
            # Skip pure numbers (timestamps/image dimensions) while keeping mixed IDs like q11, b1, rxj1131
            if low.isdigit():
                continue
            if len(low) < 3 and not (len(low) == 2 and has_digit):
                continue
            seen.add(low)
            tokens.append(low)
            if len(tokens) >= max_tokens:
                break
        return tokens

    def find_seed_ids(
        self,
        query: str,
        limit: int = 3,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[int]:
        if limit <= 0:
            return []
        tokens = self.extract_search_tokens(query)
        if not tokens:
            return []

        fts_expr = " OR ".join(f'"{tok}"*' for tok in tokens)
        owns_conn = conn is None
        db = self._connect_db() if owns_conn else conn
        assert db is not None
        try:
            rows = db.execute(
                """
                SELECT rowid, bm25(beliefs_fts) AS rank
                FROM beliefs_fts
                WHERE beliefs_fts MATCH ?
                ORDER BY rank ASC, rowid DESC
                LIMIT ?
                """,
                (fts_expr, int(limit)),
            ).fetchall()
            return [int(r[0]) for r in rows]
        except Exception:
            return []
        finally:
            if owns_conn:
                db.close()

    def record_belief_access(
        self,
        belief_ids: List[int],
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        unique_ids = sorted({int(i) for i in (belief_ids or []) if i is not None})
        if not unique_ids:
            return
        placeholders = ",".join("?" for _ in unique_ids)
        owns_conn = conn is None
        db = self._connect_db() if owns_conn else conn
        assert db is not None
        try:
            def _do_update() -> None:
                db.execute(
                    f"""
                    UPDATE beliefs
                    SET activation_count = activation_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id IN ({placeholders})
                    """,
                    tuple(unique_ids),
                )

            if owns_conn:
                with db:
                    _do_update()
            else:
                _do_update()
        finally:
            if owns_conn:
                db.close()

    def spider_beliefs(
        self,
        query: str,
        max_depth: int = 1,
        min_weight: float = 0.35,
        node_budget: int = 8,
        record_access: bool = True,
    ) -> List[Dict[str, Any]]:
        if node_budget <= 0:
            return []
        depth_cap = max(0, int(max_depth))
        weight_floor = float(min_weight)
        budget = int(node_budget)

        conn = self._connect_db()
        try:
            with conn:
                seed_ids = self.find_seed_ids(query, limit=3, conn=conn)
                if not seed_ids:
                    return []

                seed_placeholders = ",".join(f":seed_{i}" for i in range(len(seed_ids)))
                params: Dict[str, Any] = {
                    "max_depth": depth_cap,
                    "min_weight": weight_floor,
                    "node_budget": budget,
                }
                for i, sid in enumerate(seed_ids):
                    params[f"seed_{i}"] = int(sid)

                sql = f"""
                WITH RECURSIVE spider(node_id, depth, score, path) AS (
                    SELECT id, 0, 1.0, ',' || id || ','
                    FROM beliefs
                    WHERE id IN ({seed_placeholders})
                    UNION ALL
                    SELECT
                        CASE WHEN e.source_id = s.node_id THEN e.target_id ELSE e.source_id END AS next_id,
                        s.depth + 1,
                        CASE
                            WHEN e.relation IN ('supersedes', 'contradicts') THEN s.score
                            ELSE s.score * e.weight
                        END,
                        s.path || (CASE WHEN e.source_id = s.node_id THEN e.target_id ELSE e.source_id END) || ','
                    FROM spider s
                    JOIN belief_edges e
                      ON (e.source_id = s.node_id OR e.target_id = s.node_id)
                    WHERE s.depth < :max_depth
                      AND (e.weight >= :min_weight OR e.relation IN ('supersedes', 'contradicts'))
                      AND instr(
                          s.path,
                          ',' || (CASE WHEN e.source_id = s.node_id THEN e.target_id ELSE e.source_id END) || ','
                      ) = 0
                )
                SELECT
                    b.id,
                    b.claim,
                    b.epistemic_bucket,
                    b.status,
                    b.confidence,
                    b.revision_note,
                    MAX(sp.score) AS relevance
                FROM spider sp
                JOIN beliefs b ON b.id = sp.node_id
                GROUP BY b.id
                ORDER BY relevance DESC, b.confidence DESC, b.id ASC
                LIMIT :node_budget
                """
                rows = conn.execute(sql, params).fetchall()
                results: List[Dict[str, Any]] = [
                    {
                        "id": int(r[0]),
                        "claim": str(r[1]),
                        "epistemic_bucket": str(r[2]),
                        "status": str(r[3]),
                        "confidence": float(r[4]),
                        "revision_note": str(r[5] or ""),
                        "relevance": float(r[6]),
                    }
                    for r in rows
                ]

                if record_access and results:
                    self.record_belief_access([r["id"] for r in results], conn=conn)

                return results
        finally:
            conn.close()

    def has_tension(self, belief_ids: List[int]) -> bool:
        unique_ids = sorted({int(i) for i in (belief_ids or []) if i is not None})
        if not unique_ids:
            return False

        placeholders = ",".join("?" for _ in unique_ids)
        conn = self._connect_db()
        try:
            row = conn.execute(
                f"SELECT 1 FROM beliefs WHERE id IN ({placeholders}) AND status = 'needs_review' LIMIT 1",
                tuple(unique_ids),
            ).fetchone()
            if row is not None:
                return True

            edge_row = conn.execute(
                f"""
                SELECT 1
                FROM belief_edges
                WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders}))
                  AND relation IN ('contradicts', 'supersedes')
                LIMIT 1
                """,
                tuple(unique_ids) + tuple(unique_ids),
            ).fetchone()
            return edge_row is not None
        finally:
            conn.close()

    def reinforce_coactivated_edges(self, belief_ids: List[int], step: float = 0.05) -> None:
        unique_ids = sorted({int(i) for i in (belief_ids or []) if i is not None})
        if len(unique_ids) < 2:
            return

        inc = max(0.0, float(step))
        placeholders = ",".join("?" for _ in unique_ids)
        params = (inc,) + tuple(unique_ids) + tuple(unique_ids)

        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    f"""
                    UPDATE belief_edges
                    SET weight = MIN(1.0, weight + ?),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE relation = 'related_to'
                      AND source_id IN ({placeholders})
                      AND target_id IN ({placeholders})
                      AND source_id != target_id
                    """,
                    params,
                )
        finally:
            conn.close()

    def decay_related_edges(self, factor: float = 0.98, prune_floor: float = 0.15) -> int:
        f_val = max(0.0, float(factor))
        floor_val = float(prune_floor)

        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE belief_edges
                    SET weight = weight * ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE relation = 'related_to'
                    """,
                    (f_val,),
                )
                cur = conn.execute(
                    """
                    DELETE FROM belief_edges
                    WHERE relation = 'related_to'
                      AND weight < ?
                    """,
                    (floor_val,),
                )
                return int(cur.rowcount or 0)
        finally:
            conn.close()

    @staticmethod
    def format_worldview_block(nodes: List[Dict[str, Any]]) -> str:
        if not nodes:
            return ""

        active_lines: List[str] = []
        review_lines: List[str] = []
        lore_lines: List[str] = []
        retracted_lines: List[str] = []

        for node in nodes:
            claim = str(node.get("claim") or "").strip()
            if not claim:
                continue
            bucket = str(node.get("epistemic_bucket") or "working_hypothesis")
            status = str(node.get("status") or "active")
            conf = float(node.get("confidence") or 0.0)
            rev = str(node.get("revision_note") or "").strip()

            if status in ("refuted", "superseded"):
                tag = status.upper()
                suffix = f" — Reason: {rev}" if rev else ""
                retracted_lines.append(f"- [{tag}] {claim}{suffix}")
            elif status == "needs_review":
                suffix = f" (Note: {rev})" if rev else " (Dependency recently retracted; verify before relying on this)"
                review_lines.append(f"- [NEEDS REVIEW | {bucket}, conf={conf:.2f}] {claim}{suffix}")
            elif bucket == "lore_mythology":
                lore_lines.append(f"- [LORE] {claim}")
            else:
                state_prefix = "TENTATIVE | " if status == "tentative" else ""
                active_lines.append(f"- [{state_prefix}{bucket}, conf={conf:.2f}] {claim}")

        sections: List[str] = []
        if active_lines:
            sections.append("Active Stances:\n" + "\n".join(active_lines))
        if review_lines:
            sections.append("Needs Review:\n" + "\n".join(review_lines))
        if lore_lines:
            sections.append("Lore & Mythology:\n" + "\n".join(lore_lines))
        if retracted_lines:
            sections.append("Refuted / Superseded (Do Not Resurrect):\n" + "\n".join(retracted_lines))

        if not sections:
            return ""
        return "[Worldview / Active Stances]\n" + "\n\n".join(sections)


class BeliefArchivist:
    """
    Process-wide asynchronous background worker that periodically reviews
    `belief_inbox` declarations and `archival_log` batches to maintain the
    epistemic belief graph in `memory.db`.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        graph: Optional[BeliefGraphComponent] = None,
        archival_row_threshold: Optional[int] = None,
        max_rows_per_batch: Optional[int] = None,
        max_chars_per_row: Optional[int] = None,
        poll_interval_sec: Optional[float] = None,
        min_run_interval_sec: Optional[float] = None,
        flush_inbox_immediately: Optional[bool] = None,
        provider: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        max_output_tokens: Optional[int] = None,
        client: Optional[Any] = None,
        log_file: Optional[str] = None,
        audit_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        bootstrap_from_latest: Optional[bool] = None,
    ):
        self.config = load_config()
        self.agent_name = self.config.get("agent", {}).get("name", "Thunk")

        bg_cfg = self.config.get("belief_graph", {})
        llm_cfg = self.config.get("llm", {})
        mem_cfg = self.config.get("memory", {})

        resolved_db = (
            db_path
            or (graph.db_path if graph is not None else None)
            or bg_cfg.get("db_path")
            or mem_cfg.get("db_path")
        )
        if not resolved_db:
            raise ValueError(
                "BeliefArchivist requires 'db_path' under [belief_graph].db_path or [memory].db_path"
            )
        self.db_path = resolved_db
        self.graph = graph or BeliefGraphComponent(
            db_path=self.db_path,
            thread_id="belief_archivist",
            bootstrap_from_latest=bootstrap_from_latest,
        )

        self.archival_row_threshold = int(
            archival_row_threshold
            if archival_row_threshold is not None
            else bg_cfg.get("archival_row_threshold", 30)
        )
        self.max_rows_per_batch = int(
            max_rows_per_batch
            if max_rows_per_batch is not None
            else bg_cfg.get("max_rows_per_batch", 60)
        )
        self.max_chars_per_row = int(
            max_chars_per_row
            if max_chars_per_row is not None
            else bg_cfg.get("max_chars_per_row", 1500)
        )
        self.poll_interval_sec = float(
            poll_interval_sec
            if poll_interval_sec is not None
            else bg_cfg.get("poll_interval_sec", 60.0)
        )
        self.min_run_interval_sec = float(
            min_run_interval_sec
            if min_run_interval_sec is not None
            else bg_cfg.get("min_run_interval_sec", 120.0)
        )
        self.flush_inbox_immediately = bool(
            flush_inbox_immediately
            if flush_inbox_immediately is not None
            else bg_cfg.get("flush_inbox_immediately", True)
        )
        self.max_output_tokens = int(
            max_output_tokens
            if max_output_tokens is not None
            else bg_cfg.get("max_output_tokens", 4096)
        )

        cfg_api_url = str(api_url if api_url is not None else (bg_cfg.get("api_url") or "")).strip()
        cfg_provider = str(provider if provider is not None else (bg_cfg.get("provider") or "")).strip().lower()
        if not cfg_provider:
            # If an explicit OpenAI-compatible api_url override is configured, default to openai;
            # otherwise inherit [llm].provider so Gemini Cloud configs work out of the box.
            cfg_provider = "openai" if cfg_api_url else str(llm_cfg.get("provider", "openai")).strip().lower()
        self.provider = cfg_provider

        self.api_url = cfg_api_url or str(llm_cfg.get("api_url") or "http://localhost:11434/v1")
        self.api_key = str(
            api_key
            if api_key is not None
            else (bg_cfg.get("api_key") or llm_cfg.get("api_key") or "ollama")
        )

        cfg_model = str(model if model is not None else (bg_cfg.get("model") or "")).strip()
        if not cfg_model:
            cfg_model = str(llm_cfg.get("model_name") or "gemma4:31b").strip()
        self.model = cfg_model

        default_headers = {"X-Agent-ID": f"{self.agent_name.lower()}-belief-archivist"}
        cfg_headers = extra_headers if extra_headers is not None else bg_cfg.get("extra_headers")
        if isinstance(cfg_headers, dict):
            for hk, hv in cfg_headers.items():
                if hk and hv is not None:
                    default_headers[str(hk)] = str(hv)
        self.extra_headers = default_headers

        if client is not None:
            self.client = client
            self.exec_location = (
                "cloud" if self.provider == "gemini" else infer_exec_location(self.api_url)
            )
        elif self.provider == "gemini":
            gemini_key = (
                bg_cfg.get("gemini_api_key")
                or (bg_cfg.get("api_key") if bg_cfg.get("api_key") else None)
                or llm_cfg.get("gemini_api_key")
                or self.config.get("oracle", {}).get("gemini_api_key")
                or llm_cfg.get("api_key")
                or "gemini"
            )
            self.client = GeminiOpenAIClientAdapter(
                api_key=gemini_key,
                default_headers=self.extra_headers,
            )
            self.exec_location = "cloud"
        else:
            self.client = AsyncOpenAI(
                base_url=self.api_url,
                api_key=self.api_key or "ollama",
                default_headers=self.extra_headers,
                timeout=httpx.Timeout(None),
            )
            self.exec_location = infer_exec_location(self.api_url)

        self.log_file = log_file
        self.audit_callback = audit_callback
        self.last_run_ts: float = 0.0
        self.last_finish_reason: Optional[str] = None
        self.consecutive_failures: int = 0

    def get_last_log_id(self, conn: Optional[sqlite3.Connection] = None) -> int:
        owns_conn = conn is None
        db = self.graph._connect_db() if owns_conn else conn
        assert db is not None
        try:
            row = db.execute(
                "SELECT value FROM memory_settings WHERE key = 'belief_archivist_last_log_id'"
            ).fetchone()
            if row is None or row[0] is None:
                return 0
            return int(row[0])
        except (TypeError, ValueError):
            return 0
        finally:
            if owns_conn:
                db.close()

    def should_run(self) -> Tuple[bool, int, int, int]:
        conn = self.graph._connect_db()
        try:
            last_log_id = self.get_last_log_id(conn=conn)
            new_log_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM archival_log WHERE id > ?",
                    (last_log_id,),
                ).fetchone()[0]
            )
            pending_inbox_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM belief_inbox WHERE processed = 0"
                ).fetchone()[0]
            )
        finally:
            conn.close()

        trigger = (new_log_count >= self.archival_row_threshold) or (
            self.flush_inbox_immediately and pending_inbox_count > 0
        )
        return trigger, last_log_id, new_log_count, pending_inbox_count

    @staticmethod
    def _strip_thinking_and_fences(raw_text: str) -> str:
        clean = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL).strip()
        if "<think>" in clean:
            clean = clean.split("<think>", 1)[0].strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)
        return clean.strip()

    @classmethod
    def _parse_strict_json(cls, raw_text: str) -> Dict[str, Any]:
        cleaned = cls._strip_thinking_and_fences(raw_text)
        if not cleaned:
            raise ValueError("Empty response from BeliefArchivist LLM.")
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("BeliefArchivist JSON root must be an object.")
        for list_key in ("add_beliefs", "revise_beliefs", "add_edges"):
            if list_key in parsed and parsed[list_key] is not None and not isinstance(parsed[list_key], list):
                raise ValueError(f"Field '{list_key}' must be a JSON array.")
        return parsed

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        req_text = f"{system_prompt}\n{user_prompt}"

        with MetricSpan(
            operation="llm_call",
            agent=self.agent_name,
            thread="belief_archivist",
            component="belief_archivist",
            type="epistemic_review",
            model=self.model,
            exec_location=self.exec_location,
        ) as span:
            span.inc("request_tokens", count_tokens(req_text))
            span.inc("request_bytes", len(req_text.encode("utf-8")))

            with span.subtimer("llm_call_ms"):
                create_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": self.max_output_tokens,
                    "stream": True,
                }
                if self.provider != "gemini":
                    create_kwargs["stream_options"] = {"include_usage": True}

                create_call = self.client.chat.completions.create(**create_kwargs)
                resp = await create_call if inspect.isawaitable(create_call) else create_call

                content = ""
                usage = None
                finish_reason: Optional[str] = None
                self.last_finish_reason = None

                if hasattr(resp, "__aiter__"):
                    async for chunk in resp:
                        parsed_usage = extract_chunk_usage(chunk, prev_usage=usage)
                        if parsed_usage is not None:
                            usage = parsed_usage
                        if getattr(chunk, "choices", None) and chunk.choices:
                            ch0 = chunk.choices[0]
                            fr = getattr(ch0, "finish_reason", None)
                            if fr:
                                finish_reason = str(fr)
                            if getattr(ch0, "delta", None) and getattr(ch0.delta, "content", None):
                                content += ch0.delta.content
                elif hasattr(resp, "choices") and resp.choices:
                    ch0 = resp.choices[0]
                    if isinstance(ch0, dict):
                        fr = ch0.get("finish_reason")
                        content = str(ch0.get("message", {}).get("content") or "")
                    else:
                        fr = getattr(ch0, "finish_reason", None)
                        msg = getattr(ch0, "message", None)
                        if msg is not None and getattr(msg, "content", None):
                            content = str(msg.content)
                    if fr:
                        finish_reason = str(fr)
                elif isinstance(resp, str):
                    content = resp

            self.last_finish_reason = finish_reason
            if finish_reason:
                span.set_field("finish_reason", finish_reason)

            clean_out = self._strip_thinking_and_fences(content)
            span.inc("response_bytes", len(clean_out.encode("utf-8")))
            if usage:
                if usage.get("prompt_tokens") is not None:
                    span.set_counter("request_tokens", int(usage["prompt_tokens"]))
                if usage.get("completion_tokens") is not None:
                    span.set_counter("response_tokens", int(usage["completion_tokens"]))
                if usage.get("cached_tokens") is not None:
                    span.set_counter("cached_tokens", int(usage["cached_tokens"]))
                if usage.get("thinking_tokens") is not None:
                    span.set_counter("thinking_tokens", int(usage["thinking_tokens"]))
            else:
                span.inc("response_tokens", count_tokens(clean_out))

            if finish_reason == "length":
                span.fail()
                raise ValueError("BeliefArchivist response truncated at max_output_tokens (finish_reason='length').")

            return content

    def _gather_candidate_beliefs(
        self,
        conn: sqlite3.Connection,
        inbox_rows: List[Tuple[Any, ...]],
        log_rows: List[Tuple[Any, ...]],
        max_candidates: int = 35,
    ) -> List[Dict[str, Any]]:
        candidates_by_id: Dict[int, Dict[str, Any]] = {}

        review_rows = conn.execute(
            """
            SELECT id, claim, epistemic_bucket, status, confidence, revision_note
            FROM beliefs
            WHERE status = 'needs_review'
            ORDER BY updated_at DESC, id DESC
            LIMIT 15
            """
        ).fetchall()
        for r in review_rows:
            candidates_by_id[int(r[0])] = {
                "id": int(r[0]),
                "claim": str(r[1]),
                "epistemic_bucket": str(r[2]),
                "status": str(r[3]),
                "confidence": float(r[4]),
                "revision_note": str(r[5] or ""),
            }

        combined_text_parts: List[str] = []
        for _, _, claim, _, reason, _ in inbox_rows:
            combined_text_parts.append(f"{claim} {reason}")
        for _, _, role, content, _ in log_rows:
            c_str = str(content or "")
            if c_str.startswith("[System Clock]"):
                continue
            if role in ("user", "assistant"):
                combined_text_parts.append(c_str[:400])

        tokens = self.graph.extract_search_tokens("\n".join(combined_text_parts), max_tokens=30)
        if tokens and len(candidates_by_id) < max_candidates:
            fts_expr = " OR ".join(f'"{tok}"*' for tok in tokens)
            remaining = max_candidates - len(candidates_by_id)
            try:
                fts_rows = conn.execute(
                    """
                    SELECT b.id, b.claim, b.epistemic_bucket, b.status, b.confidence, b.revision_note
                    FROM beliefs_fts f
                    JOIN beliefs b ON b.id = f.rowid
                    WHERE beliefs_fts MATCH ?
                    ORDER BY bm25(beliefs_fts) ASC, b.id DESC
                    LIMIT ?
                    """,
                    (fts_expr, remaining + 10),
                ).fetchall()
                for r in fts_rows:
                    bid = int(r[0])
                    if bid not in candidates_by_id:
                        candidates_by_id[bid] = {
                            "id": bid,
                            "claim": str(r[1]),
                            "epistemic_bucket": str(r[2]),
                            "status": str(r[3]),
                            "confidence": float(r[4]),
                            "revision_note": str(r[5] or ""),
                        }
                        if len(candidates_by_id) >= max_candidates:
                            break
            except Exception:
                pass

        return list(candidates_by_id.values())

    def _build_user_prompt(
        self,
        inbox_rows: List[Tuple[Any, ...]],
        log_rows: List[Tuple[Any, ...]],
        candidates: List[Dict[str, Any]],
    ) -> str:
        inbox_lines: List[str] = []
        for row_id, thread_id, claim, stance, reason, created_at in inbox_rows:
            reason_part = f" | reason: {reason}" if reason else ""
            inbox_lines.append(
                f"- [inbox_id={row_id}, thread={thread_id}, stance={stance}, at={created_at}] {claim}{reason_part}"
            )
        inbox_section = "\n".join(inbox_lines) if inbox_lines else "(none)"

        log_lines: List[str] = []
        for log_id, thread_id, role, content, ts in log_rows:
            raw_content = str(content or "").strip()
            if not raw_content or raw_content.startswith("[System Clock]"):
                continue
            if len(raw_content) > self.max_chars_per_row:
                raw_content = raw_content[: self.max_chars_per_row] + "... [truncated]"
            log_lines.append(f"[log_id={log_id} | thread={thread_id} | role={role} | ts={ts}]\n{raw_content}")
        log_section = "\n\n".join(log_lines) if log_lines else "(no substantive transcript rows)"

        cand_lines: List[str] = []
        for c in candidates:
            rev_part = f" | revision_note={c['revision_note']}" if c["revision_note"] else ""
            cand_lines.append(
                f"- [belief_id={c['id']} | status={c['status']} | bucket={c['epistemic_bucket']} "
                f"| conf={c['confidence']:.2f}{rev_part}] {c['claim']}"
            )
        cand_section = "\n".join(cand_lines) if cand_lines else "(none)"

        return (
            f"### Explicit Boss Declarations (`belief_inbox`)\n{inbox_section}\n\n"
            f"### Existing Candidate & Needs-Review Beliefs\n{cand_section}\n\n"
            f"### Recent Archival Transcript Batch (`archival_log`)\n{log_section}\n"
        )

    def _emit_audit_event(self, payload: Dict[str, Any]) -> None:
        if self.log_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
                entry = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "thread_id": "belief_archivist",
                    "event": "belief_graph_updated",
                    "payload": payload,
                }
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                print(f"[BeliefArchivist] Warning writing audit log: {e}")

        if self.audit_callback is not None:
            try:
                self.audit_callback("belief_graph_updated", payload)
            except Exception as e:
                print(f"[BeliefArchivist] Warning in audit_callback: {e}")

    async def run_once(self, force: bool = False) -> Dict[str, Any]:
        should_trigger, last_log_id, new_log_count, pending_inbox_count = self.should_run()
        if not force and not should_trigger:
            return {
                "triggered": False,
                "last_log_id": last_log_id,
                "new_log_count": new_log_count,
                "pending_inbox_count": pending_inbox_count,
            }
        if force and new_log_count == 0 and pending_inbox_count == 0:
            return {
                "triggered": False,
                "last_log_id": last_log_id,
                "new_log_count": 0,
                "pending_inbox_count": 0,
            }

        conn = self.graph._connect_db()
        try:
            inbox_rows = conn.execute(
                """
                SELECT id, thread_id, claim, stance, reason, created_at
                FROM belief_inbox
                WHERE processed = 0
                ORDER BY id ASC
                """
            ).fetchall()
            log_rows = conn.execute(
                """
                SELECT id, thread_id, role, content, timestamp
                FROM archival_log
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (last_log_id, self.max_rows_per_batch),
            ).fetchall()
            candidates = self._gather_candidate_beliefs(conn, inbox_rows, log_rows)
        finally:
            conn.close()

        batch_max_log_id = max((int(r[0]) for r in log_rows), default=last_log_id)
        fetched_inbox_ids = [int(r[0]) for r in inbox_rows]

        user_prompt = self._build_user_prompt(inbox_rows, log_rows, candidates)

        try:
            raw_response = await self._call_llm(
                system_prompt=DEFAULT_ARCHIVIST_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            parsed = self._parse_strict_json(raw_response)
        except Exception as e:
            self.consecutive_failures += 1
            print(
                f"[BeliefArchivist] Error during epistemic review (cursor kept at {last_log_id}): {e}"
            )
            return {
                "triggered": False,
                "error": str(e),
                "last_log_id": last_log_id,
            }

        add_beliefs = parsed.get("add_beliefs") or []
        revise_beliefs = parsed.get("revise_beliefs") or []
        add_edges = parsed.get("add_edges") or []

        added_ids: List[int] = []
        revised_ids: List[int] = []
        cascaded_review_ids: List[int] = []
        edges_upserted = 0

        conn = self.graph._connect_db()
        try:
            with conn:
                ref_map: Dict[str, int] = {}

                for item in add_beliefs:
                    if not isinstance(item, dict):
                        continue
                    claim = str(item.get("claim") or "").strip()
                    if not claim:
                        continue
                    bucket = str(item.get("epistemic_bucket") or "working_hypothesis").strip().lower()
                    if bucket not in VALID_EPISTEMIC_BUCKETS:
                        bucket = "working_hypothesis"
                    status = str(item.get("status") or "active").strip().lower()
                    if status not in ("active", "tentative"):
                        status = "active"
                    conf = self.graph._clamp_float(item.get("confidence"), 0.0, 1.0, default=0.6)
                    ev_refs = item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else []

                    new_id = self.graph.upsert_belief(
                        claim=claim,
                        epistemic_bucket=bucket,
                        status=status,
                        confidence=conf,
                        evidence_refs=ev_refs,
                        conn=conn,
                    )
                    added_ids.append(new_id)
                    temp_id = item.get("temp_id")
                    if temp_id is not None and str(temp_id).strip():
                        ref_map[str(temp_id).strip()] = new_id

                def _resolve_ref(ref_val: Any) -> Optional[int]:
                    if ref_val is None or isinstance(ref_val, bool):
                        return None
                    if isinstance(ref_val, int):
                        cand_id = ref_val
                    else:
                        s_ref = str(ref_val).strip()
                        if not s_ref:
                            return None
                        if s_ref in ref_map:
                            cand_id = ref_map[s_ref]
                        elif s_ref.isdigit():
                            cand_id = int(s_ref)
                        else:
                            return None
                    exists = conn.execute(
                        "SELECT 1 FROM beliefs WHERE id = ? LIMIT 1",
                        (cand_id,),
                    ).fetchone()
                    return cand_id if exists is not None else None

                for item in revise_beliefs:
                    if not isinstance(item, dict):
                        continue
                    target_bid = _resolve_ref(item.get("belief_id"))
                    if target_bid is None:
                        continue
                    new_status = str(item.get("status") or "active").strip().lower()
                    if new_status not in VALID_BELIEF_STATUSES:
                        continue
                    raw_bucket = item.get("epistemic_bucket")
                    clean_bucket = (
                        str(raw_bucket).strip().lower()
                        if raw_bucket and str(raw_bucket).strip().lower() in VALID_EPISTEMIC_BUCKETS
                        else None
                    )
                    raw_conf = item.get("confidence")
                    clean_conf = (
                        self.graph._clamp_float(raw_conf, 0.0, 1.0, default=0.6)
                        if raw_conf is not None
                        else None
                    )
                    rev_note = str(item.get("revision_note") or "").strip()

                    if new_status in ("refuted", "superseded"):
                        sup_id = _resolve_ref(item.get("superseded_by_ref"))
                        cascaded = self.graph.retract_belief(
                            belief_id=target_bid,
                            status=new_status,
                            revision_note=rev_note,
                            superseded_by_id=sup_id,
                            epistemic_bucket=clean_bucket,
                            confidence=clean_conf,
                            conn=conn,
                        )
                        cascaded_review_ids.extend(cascaded)
                    else:
                        existing = conn.execute(
                            "SELECT claim, epistemic_bucket, confidence FROM beliefs WHERE id = ?",
                            (target_bid,),
                        ).fetchone()
                        if existing is None:
                            continue
                        self.graph.upsert_belief(
                            claim=str(existing[0]),
                            epistemic_bucket=clean_bucket or str(existing[1]),
                            status=new_status,
                            confidence=clean_conf if clean_conf is not None else float(existing[2]),
                            revision_note=rev_note,
                            belief_id=target_bid,
                            conn=conn,
                        )
                    revised_ids.append(target_bid)

                for edge in add_edges:
                    if not isinstance(edge, dict):
                        continue
                    src_id = _resolve_ref(edge.get("source_ref"))
                    tgt_id = _resolve_ref(edge.get("target_ref"))
                    rel = str(edge.get("relation") or "").strip().lower()
                    if src_id is None or tgt_id is None or src_id == tgt_id:
                        continue
                    if rel not in VALID_EDGE_RELATIONS:
                        continue
                    w = self.graph._clamp_float(edge.get("weight"), 0.0, 1.0, default=1.0)
                    self.graph.upsert_edge(
                        source_id=src_id,
                        target_id=tgt_id,
                        relation=rel,
                        weight=w,
                        conn=conn,
                    )
                    edges_upserted += 1

                if fetched_inbox_ids:
                    placeholders = ",".join("?" for _ in fetched_inbox_ids)
                    conn.execute(
                        f"UPDATE belief_inbox SET processed = 1 WHERE id IN ({placeholders})",
                        tuple(fetched_inbox_ids),
                    )

                conn.execute(
                    """
                    INSERT INTO memory_settings (key, value)
                    VALUES ('belief_archivist_last_log_id', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (str(batch_max_log_id),),
                )
        except Exception as e:
            self.consecutive_failures += 1
            print(f"[BeliefArchivist] Database transaction error (rolled back): {e}")
            return {
                "triggered": False,
                "error": str(e),
                "last_log_id": last_log_id,
            }
        finally:
            conn.close()

        self.consecutive_failures = 0
        self.last_run_ts = time.time()
        pruned_edges = self.graph.decay_related_edges()

        summary = {
            "triggered": True,
            "added_belief_ids": added_ids,
            "revised_belief_ids": revised_ids,
            "cascaded_review_ids": sorted(set(cascaded_review_ids)),
            "edges_upserted": edges_upserted,
            "edges_pruned": pruned_edges,
            "inbox_processed": len(fetched_inbox_ids),
            "last_log_id": batch_max_log_id,
        }
        self._emit_audit_event(summary)
        return summary

    async def poll_once(self) -> Dict[str, Any]:
        now = time.time()
        effective_cooldown = self.min_run_interval_sec
        if self.consecutive_failures > 0:
            effective_cooldown = min(
                3600.0,
                max(self.min_run_interval_sec, 30.0) * (2 ** min(self.consecutive_failures - 1, 5)),
            )
        if self.last_run_ts > 0 and (now - self.last_run_ts) < effective_cooldown:
            return {"triggered": False, "cooldown": True}

        should_trigger, last_log_id, new_log_count, pending_inbox_count = self.should_run()
        if not should_trigger:
            return {
                "triggered": False,
                "last_log_id": last_log_id,
                "new_log_count": new_log_count,
                "pending_inbox_count": pending_inbox_count,
            }

        self.last_run_ts = now
        return await self.run_once(force=True)

    async def run(self) -> None:
        print(
            f"[BeliefArchivist] Started (db={self.db_path}, model={self.model}, "
            f"threshold={self.archival_row_threshold}, poll={self.poll_interval_sec}s)"
        )
        try:
            while True:
                await asyncio.sleep(self.poll_interval_sec)
                try:
                    await self.poll_once()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[BeliefArchivist] Unexpected error in poll loop: {e}")
        except asyncio.CancelledError:
            print("[BeliefArchivist] Shutting down cleanly.")
            raise
