import asyncio
import hashlib
import inspect
import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set
import httpx
from openai import AsyncOpenAI
from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.metrics import MetricSpan, count_tokens, extract_chunk_usage, infer_exec_location


@dataclass
class Layer1Item:
    """Represents a candidate memory held in the stateful Layer-1 Working Pool."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    hits: int = 1
    age: int = 0
    last_similarity: float = 0.0
    groomed_at: int = 0

    @property
    def priority(self) -> float:
        return self.hits / (self.age + 1)


class SpreadingActivationComponent(BaseComponent):
    """
    Stateful 'Layer-1 Mind Gardener' drop-in alternative to SubconsciousRecallComponent.
    Maintains an in-memory Layer-1 Working Pool of candidate memories that warm up when
    repeatedly hit across turns and naturally drift out via competitive eviction when cold.
    """
    def __init__(
        self,
        layer_1_pool_size: int = None,
        layer_0_promotion_count: int = None,
        db_path: str = None,
        persist: bool = True,
        thread_id: str = "main",
    ):
        self.config = load_config()
        self.agent_name = self.config.get("agent", {}).get("name", "Thunk")
        self.thread_id = (thread_id or "main").strip() or "main"
        sub_cfg = self.config.get("subconscious", {})
        sa_cfg = self.config.get("spreading_activation", {})
        llm_cfg = self.config.get("llm", {})

        self.api_url = sa_cfg.get("api_url") or sub_cfg.get("api_url") or llm_cfg.get("api_url")
        self.api_key = sa_cfg.get("api_key") or sub_cfg.get("api_key") or llm_cfg.get("api_key") or "ollama"
        self.model_name = sa_cfg.get("model_name") or sub_cfg.get("model_name", "gemma4:e2b")
        self.max_output_tokens = int(
            sa_cfg.get("max_output_tokens") or sub_cfg.get("max_output_tokens", 2048)
        )

        if layer_1_pool_size is not None:
            self.layer_1_pool_size = int(layer_1_pool_size)
        else:
            self.layer_1_pool_size = int(
                sa_cfg.get("layer_1_pool_size", sub_cfg.get("layer_1_pool_size", 32))
            )

        if layer_0_promotion_count is not None:
            self.layer_0_promotion_count = int(layer_0_promotion_count)
        else:
            self.layer_0_promotion_count = int(
                sa_cfg.get("layer_0_promotion_count", sub_cfg.get("layer_0_promotion_count", 8))
            )

        self.db_path = (
            db_path
            if db_path is not None
            else sa_cfg.get("db_path")
            or sub_cfg.get("db_path")
            or self.config.get("memory", {}).get("db_path")
        )
        self.persist = bool(persist and self.db_path)

        self.client = AsyncOpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
            default_headers={"X-Agent-ID": f"{self.agent_name}-{self.thread_id}-spreading_activation"},
            timeout=httpx.Timeout(None),
        )
        self.internal_memory_components: List[Any] = []
        self.layer_1_pool: Dict[str, Layer1Item] = {}

        if self.persist:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            self._init_db()
            self._load_pool_from_db()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        if not self.persist:
            return
        conn = self._connect_db()
        try:
            with conn:
                table_exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='layer_1_pool'"
                ).fetchone()
                if table_exists:
                    cols = [row[1] for row in conn.execute("PRAGMA table_info(layer_1_pool)").fetchall()]
                    if "thread_id" not in cols:
                        conn.execute("ALTER TABLE layer_1_pool RENAME TO layer_1_pool_legacy")
                        conn.execute(
                            """
                            CREATE TABLE layer_1_pool (
                                thread_id TEXT NOT NULL DEFAULT 'main',
                                id TEXT NOT NULL,
                                content TEXT,
                                metadata TEXT,
                                hits INTEGER,
                                age INTEGER,
                                last_similarity REAL,
                                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                                PRIMARY KEY (thread_id, id)
                            )
                            """
                        )
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO layer_1_pool (thread_id, id, content, metadata, hits, age, last_similarity, updated_at)
                            SELECT 'main', id, content, metadata, hits, age, last_similarity, updated_at
                            FROM layer_1_pool_legacy
                            """
                        )
                        conn.execute("DROP TABLE layer_1_pool_legacy")
                else:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS layer_1_pool (
                            thread_id TEXT NOT NULL DEFAULT 'main',
                            id TEXT NOT NULL,
                            content TEXT,
                            metadata TEXT,
                            hits INTEGER,
                            age INTEGER,
                            last_similarity REAL,
                            groomed_at INTEGER NOT NULL DEFAULT 0,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (thread_id, id)
                        )
                        """
                    )
                cols_now = [row[1] for row in conn.execute("PRAGMA table_info(layer_1_pool)").fetchall()]
                if "groomed_at" not in cols_now:
                    conn.execute("ALTER TABLE layer_1_pool ADD COLUMN groomed_at INTEGER NOT NULL DEFAULT 0")
        finally:
            conn.close()

    def _load_pool_from_db(self) -> None:
        if not self.persist:
            return
        conn = self._connect_db()
        try:
            rows = conn.execute(
                "SELECT id, content, metadata, hits, age, last_similarity, groomed_at FROM layer_1_pool WHERE thread_id = ?",
                (self.thread_id,),
            ).fetchall()
            for row_id, content, meta_json, hits, age, last_sim, groomed_at in rows:
                try:
                    metadata = json.loads(meta_json) if meta_json else {}
                except Exception:
                    metadata = {}
                self.layer_1_pool[str(row_id)] = Layer1Item(
                    id=str(row_id),
                    content=str(content or ""),
                    metadata=metadata,
                    hits=int(hits or 1),
                    age=int(age or 0),
                    last_similarity=float(last_sim or 0.0),
                    groomed_at=int(groomed_at or 0),
                )
            self._evict_pool()
        finally:
            conn.close()

    def _save_pool_to_db(self) -> None:
        if not self.persist:
            return
        conn = self._connect_db()
        try:
            with conn:
                conn.execute("DELETE FROM layer_1_pool WHERE thread_id = ?", (self.thread_id,))
                conn.executemany(
                    """
                    INSERT INTO layer_1_pool (thread_id, id, content, metadata, hits, age, last_similarity, groomed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            self.thread_id,
                            item.id,
                            item.content,
                            json.dumps(item.metadata),
                            item.hits,
                            item.age,
                            item.last_similarity,
                            int(getattr(item, "groomed_at", 0) or 0),
                        )
                        for item in self.layer_1_pool.values()
                    ],
                )
        finally:
            conn.close()

    def compute_fetch_count(self) -> int:
        """
        Calculates how many items to fetch from Qdrant (Layer-2) based on current Layer-1 pool vacuum:
        fetch_count = max(missing + layer_0_promotion_count, layer_0_promotion_count * 2)
        """
        missing = max(0, self.layer_1_pool_size - len(self.layer_1_pool))
        return max(missing + self.layer_0_promotion_count, self.layer_0_promotion_count * 2)

    def _age_pool(self) -> None:
        """1. Age Existing Pool: increment age by 1 for every item currently in Layer-1."""
        for item in self.layer_1_pool.values():
            item.age += 1

    def _merge_and_reinforce(self, matches: List[Any]) -> None:
        """
        3. Merge & Reinforce (Spreading Activation):
        - If match.id is already in layer_1_pool: increment hits, reset age = 0, update last_similarity.
        - If match.id is not in layer_1_pool: insert new Layer1Item(hits=1, age=0, last_similarity=score).
        """
        for match in matches:
            if isinstance(match, dict):
                content = str(match.get("content") or match.get("text") or "").strip()
                if not content:
                    continue
                match_id = str(
                    match.get("id")
                    or hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
                )
                metadata = dict(match.get("metadata") or {})
                score = float(match.get("score", match.get("last_similarity", 0.0)))
            else:
                content = str(getattr(match, "content", None) or getattr(match, "text", "") or "").strip()
                if not content:
                    continue
                match_id = str(
                    getattr(match, "id", None)
                    or hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
                )
                metadata = dict(getattr(match, "metadata", None) or getattr(match, "payload", None) or {})
                score = float(getattr(match, "score", getattr(match, "last_similarity", 0.0)))

            if match_id in self.layer_1_pool:
                item = self.layer_1_pool[match_id]
                item.hits += 1
                item.age = 0
                item.last_similarity = score
                item.content = content
                if metadata:
                    item.metadata = metadata
            else:
                self.layer_1_pool[match_id] = Layer1Item(
                    id=match_id,
                    content=content,
                    metadata=metadata,
                    hits=1,
                    age=0,
                    last_similarity=score,
                )

    def _evict_pool(self) -> None:
        """
        4. Competitive Eviction:
        If len(layer_1_pool) > layer_1_pool_size, sort descending by (priority, last_similarity)
        and retain only the top layer_1_pool_size items.
        """
        if len(self.layer_1_pool) <= self.layer_1_pool_size:
            return

        sorted_items = sorted(
            self.layer_1_pool.values(),
            key=lambda item: (item.priority, item.last_similarity),
            reverse=True,
        )
        retained = sorted_items[: self.layer_1_pool_size]
        self.layer_1_pool = {item.id: item for item in retained}

    def _select_promotion_slate(self) -> Tuple[List[Layer1Item], Set[str]]:
        """
        5. Split-Slate Selection for Layer-0 Promotion (up to layer_0_promotion_count):
        - Slot Group A (up to layer_0_promotion_count // 2): items with hits > 1,
          sorted descending by (priority, last_similarity).
        - Slot Group B (remaining slots up to layer_0_promotion_count): remaining items
          sorted descending by (last_similarity, priority).
        Returns (promoted_items, recurring_ids).
        """
        if not self.layer_1_pool or self.layer_0_promotion_count <= 0:
            return [], set()

        group_a_cap = self.layer_0_promotion_count // 2
        recurring_candidates = [
            item for item in self.layer_1_pool.values() if item.hits > 1
        ]
        recurring_candidates.sort(
            key=lambda item: (item.priority, item.last_similarity),
            reverse=True,
        )
        group_a = recurring_candidates[:group_a_cap]
        selected_ids = {item.id for item in group_a}

        remaining_slots = self.layer_0_promotion_count - len(group_a)
        remaining_candidates = [
            item for item in self.layer_1_pool.values() if item.id not in selected_ids
        ]
        remaining_candidates.sort(
            key=lambda item: (item.last_similarity, item.priority),
            reverse=True,
        )
        group_b = remaining_candidates[:remaining_slots]

        promoted = group_a + group_b
        recurring_ids = {item.id for item in promoted if item.hits > 1}
        return promoted, recurring_ids

    def _apply_refractory_cooldown(self, promoted_items: List[Layer1Item]) -> None:
        """
        6. Refractory Cooldown (Prevent Echo Loops):
        Reset hits = 1 and age = 0 for all promoted items in layer_1_pool.
        """
        for item in promoted_items:
            pool_item = self.layer_1_pool.get(item.id)
            if pool_item is not None:
                pool_item.hits = 1
                pool_item.age = 0

    def _format_promoted_candidates(
        self, promoted_items: List[Layer1Item], recurring_ids: Set[str]
    ) -> str:
        """Formats promoted Layer-1 items with recurring vs immediate annotations for the synthesis model."""
        blocks = []
        for idx, item in enumerate(promoted_items, start=1):
            timestamp = item.metadata.get("timestamp", "Unknown Time")
            sim_str = f"{item.last_similarity:.3f}"
            prio_str = f"{item.priority:.3f}"
            if item.id in recurring_ids:
                tag = (
                    f"[Recurring Cross-Turn Connection | hits={item.hits}, "
                    f"priority={prio_str}, similarity={sim_str}, saved={timestamp}]"
                )
            else:
                tag = (
                    f"[Immediate Associative Match | hits={item.hits}, "
                    f"priority={prio_str}, similarity={sim_str}, saved={timestamp}]"
                )
            snippet = (
                item.content[:1200] + "\n...[truncated]..."
                if len(item.content) > 1200
                else item.content
            )
            blocks.append(f"--- Candidate {idx} {tag} ---\n{snippet}")
        return "\n\n".join(blocks)

    async def _synthesize_memories(self, incoming_message: str, raw_memories: str) -> str:
        """Uses the lightweight subconscious model to filter and synthesize promoted Layer-1 candidate memories."""
        prompt = (
            "You are the high-fidelity distillation bridge for an AI agent's subconscious memory.\n"
            "Given the incoming message and a set of candidate memories promoted from the Layer-1 Spreading Activation pool, "
            "extract the raw essence, idiosyncrasies, and specific facts relevant to the message. "
            "Candidates marked as [Recurring Cross-Turn Connection] have been activated across multiple turns and represent "
            "warming thematic threads. Do NOT synthesize or smooth the data into a coherent narrative.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. PRESERVE THE EDGES: Keep specific, bizarre, or grotesque details exactly as they appear. Do not normalize idiosyncratic language into generalities.\n"
            "2. EMBRACE DISSONANCE: If candidate memories contain conflicting information or contradictions, do NOT resolve them. List them as competing data points. Friction is a signal.\n"
            "3. ANTI-SMOOTHING: Avoid narrative fillers, corporate phrasing, or 'averaging' the tone. No phrases like 'it appears that,' 'overall,' or 'the party encountered.'\n"
            "4. ZERO INVENTION: Use only the provided memories. If no memories are genuinely relevant, reply with the single word: NONE\n\n"
            f"<incoming_message>\n{incoming_message}\n</incoming_message>\n\n"
            f"<candidate_memories>\n{raw_memories}\n</candidate_memories>\n\n"
            "OUTPUT: A high-contrast list of 3-6 raw, distilled points of recall."
        )

        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=self.max_output_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        content = ""
        self.last_usage = None
        async for chunk in stream:
            parsed_usage = extract_chunk_usage(chunk)
            if parsed_usage is not None:
                self.last_usage = parsed_usage
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content

        synthesized = content.strip()
        if synthesized.upper() == "NONE" or synthesized.upper().startswith("NONE\n"):
            return ""
        return synthesized

    async def _fetch_matches(self, query: str, fetch_count: int) -> List[Dict[str, Any]]:
        """Queries registered internal memory components for up to fetch_count structured matches."""
        tasks = []
        loop = asyncio.get_event_loop()

        for comp in self.internal_memory_components:
            if hasattr(comp, "search_semantic_memory_raw"):
                if inspect.iscoroutinefunction(comp.search_semantic_memory_raw):
                    tasks.append(comp.search_semantic_memory_raw(query, fetch_count))
                else:
                    tasks.append(
                        loop.run_in_executor(None, comp.search_semantic_memory_raw, query, fetch_count)
                    )
            elif hasattr(comp, "search_semantic_memory"):
                if inspect.iscoroutinefunction(comp.search_semantic_memory):
                    tasks.append(comp.search_semantic_memory(query, fetch_count))
                else:
                    tasks.append(
                        loop.run_in_executor(None, comp.search_semantic_memory, query, fetch_count)
                    )
            elif hasattr(comp, "execute_search"):
                if inspect.iscoroutinefunction(comp.execute_search):
                    tasks.append(comp.execute_search(query))
                else:
                    tasks.append(loop.run_in_executor(None, comp.execute_search, query))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        matches: List[Dict[str, Any]] = []

        for result in results:
            if isinstance(result, Exception) or not result:
                continue
            if isinstance(result, list):
                matches.extend(result)
            elif isinstance(result, str):
                text = result.strip()
                if text and not text.startswith("No relevant memories found") and not text.startswith("Error "):
                    matches.append(
                        {
                            "id": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                            "content": text,
                            "metadata": {"source": "notes_or_text"},
                            "score": 0.5,
                        }
                    )

        return matches

    async def on_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == "agent_started":
            self.internal_memory_components = []
            for comp in self.agent.components:
                if comp.__class__.__name__ in ["VectorMemoryComponent", "StructuredNotesComponent"]:
                    if hasattr(comp, "search_semantic_memory_raw") or hasattr(comp, "execute_search"):
                        self.internal_memory_components.append(comp)
            print(
                f"[SpreadingActivation:{self.thread_id}] Wired up to {len(self.internal_memory_components)} internal memory stores "
                f"(model: {self.model_name}, pool_size: {self.layer_1_pool_size}, promotion_count: {self.layer_0_promotion_count})."
            )

        elif event_name == "before_user_message_added":
            original_text = payload.get("content", "")
            if not original_text or not self.internal_memory_components:
                return

            # Skip routine clock ticks so they don't age the pool or trigger synthesis calls
            if original_text.startswith("[System Clock]"):
                return

            agent_obj = getattr(self, "agent", None)
            agent_name = getattr(agent_obj, "name", None) or self.agent_name
            thread_id = getattr(agent_obj, "thread_id", None) or self.thread_id or "main"
            exec_loc = infer_exec_location(self.api_url)

            with MetricSpan(
                operation="llm_call",
                agent=agent_name,
                thread=thread_id,
                component="spreading_activation",
                type="subconscious_recall",
                model=self.model_name,
                exec_location=exec_loc,
            ) as span:
                # 1. Age Existing Pool
                self._age_pool()

                # 2. Dynamic Intake (Fetch from Layer-2 Qdrant DB)
                fetch_count = self.compute_fetch_count()
                print(
                    f"[SpreadingActivation:{thread_id}] Pool size={len(self.layer_1_pool)}/{self.layer_1_pool_size}; "
                    f"fetching top-{fetch_count} candidates from Layer-2..."
                )
                with span.subtimer("db_lookup_ms"):
                    matches = await self._fetch_matches(original_text, fetch_count)
                span.inc("db_lookup", len(self.internal_memory_components))
                span.inc("db_lookup_records", len(matches))

                # 3. Merge & Reinforce (Spreading Activation)
                self._merge_and_reinforce(matches)

                # 4. Competitive Eviction (Keep Pool at layer_1_pool_size)
                self._evict_pool()

                # 5. Split-Slate Selection for Layer-0 Promotion
                promoted_items, recurring_ids = self._select_promotion_slate()
                span.inc("promoted_records", len(promoted_items))
                span.inc("recurring_records", len(recurring_ids))
                if not promoted_items:
                    with span.subtimer("db_save_ms"):
                        self._save_pool_to_db()
                    span.inc("db_save", 1)
                    span.inc("db_save_records", len(self.layer_1_pool))
                    return

                formatted_candidates = self._format_promoted_candidates(promoted_items, recurring_ids)

                # 6. Refractory Cooldown (Prevent Echo Loops)
                self._apply_refractory_cooldown(promoted_items)
                with span.subtimer("db_save_ms"):
                    self._save_pool_to_db()
                span.inc("db_save", 1)
                span.inc("db_save_records", len(self.layer_1_pool))

                req_str = f"{original_text}\n{formatted_candidates}"
                span.inc("request_tokens", count_tokens(req_str, agent=agent_obj))
                span.inc("request_bytes", len(req_str.encode("utf-8")))

                try:
                    print(
                        f"[SpreadingActivation:{thread_id}] Synthesizing {len(promoted_items)} promoted candidates "
                        f"({len(recurring_ids)} recurring) with {self.model_name}..."
                    )
                    with span.subtimer("llm_call_ms"):
                        subconscious_thoughts = await self._synthesize_memories(
                            original_text, formatted_candidates
                        )
                except Exception as e:
                    span.fail()
                    print(f"[SpreadingActivation Warning:{thread_id}] Synthesis failed ({e}); falling back to raw recall.")
                    subconscious_thoughts = formatted_candidates

                usage = getattr(self, "last_usage", None) or {}
                if usage.get("prompt_tokens"):
                    span.set_counter("request_tokens", int(usage["prompt_tokens"]))
                if usage.get("cached_tokens") is not None:
                    span.set_counter("cached_tokens", int(usage["cached_tokens"]))
                if usage.get("thinking_tokens") is not None:
                    span.set_counter("thinking_tokens", int(usage["thinking_tokens"]))
                span.inc(
                    "response_tokens",
                    usage.get("completion_tokens") or count_tokens(subconscious_thoughts, agent=agent_obj),
                )
                span.inc("response_bytes", len((subconscious_thoughts or "").encode("utf-8")))

                if subconscious_thoughts.strip():
                    print(f"[SpreadingActivation:{thread_id}] Found relevant associative memories. Injecting into Layer-0.")
                    new_content = (
                        f"[Subconscious Recall Triggered by below message:]\n"
                        f"{subconscious_thoughts.strip()}\n"
                        f"--- End Recall ---\n\n"
                        f"{original_text}"
                    )
                    payload["content"] = new_content
