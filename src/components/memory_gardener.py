import json
import re
import time
from typing import Any, Dict, List, Optional
import httpx
from openai import AsyncOpenAI

from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.metrics import MetricSpan, count_tokens, infer_exec_location


class MemoryGardenerComponent(BaseComponent):
    """
    Homeostatic Background Gardener (Daemon 1: 'The Clutter Merger').
    Driven by `layer_1_pool` in SpreadingActivationComponent (zero O(N^2) scans).
    Uses the lightweight subconscious model (gemma4:e2b) to consolidate near-duplicate
    semantic memories in Qdrant and heal split activation votes in Layer-1.
    """

    def __init__(
        self,
        tool_interval: int = 20,
        min_interval_sec: float = 900.0,
        thread_id: str = "mind",
        spreading_comp: Any = None,
        vector_comp: Any = None,
    ):
        self.config = load_config()
        self.agent_name = self.config.get("agent", {}).get("name", "Thunk")
        self.thread_id = (thread_id or "mind").strip() or "mind"

        sub_cfg = self.config.get("subconscious", {})
        gardener_cfg = self.config.get("memory_gardener", {})
        llm_cfg = self.config.get("llm", {})

        self.api_url = (
            gardener_cfg.get("api_url")
            or sub_cfg.get("api_url")
            or llm_cfg.get("api_url")
        )
        self.api_key = (
            gardener_cfg.get("api_key")
            or sub_cfg.get("api_key")
            or llm_cfg.get("api_key")
            or "ollama"
        )
        self.model_name = (
            gardener_cfg.get("model_name")
            or sub_cfg.get("model_name", "gemma4:e2b")
        )
        self.max_output_tokens = int(
            gardener_cfg.get("max_output_tokens")
            or sub_cfg.get("max_output_tokens", 1024)
        )
        self.tool_interval = int(
            gardener_cfg.get("tool_interval", tool_interval)
        )
        self.min_interval_sec = float(
            gardener_cfg.get("min_interval_sec", min_interval_sec)
        )

        self.client = AsyncOpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
            default_headers={
                "X-Agent-ID": f"{self.agent_name}-{self.thread_id}-memory_gardener"
            },
            timeout=httpx.Timeout(None),
        )

        self.spreading_comp = spreading_comp
        self.vector_comp = vector_comp
        self.tool_counter = 0
        self.last_run_ts = 0.0
        self.pending_garden = False

    def _discover_components(self) -> None:
        agent_obj = getattr(self, "agent", None)
        if not agent_obj:
            return
        for comp in getattr(agent_obj, "components", []):
            cname = comp.__class__.__name__
            if cname == "SpreadingActivationComponent" and self.spreading_comp is None:
                self.spreading_comp = comp
            elif cname == "VectorMemoryComponent" and self.vector_comp is None:
                self.vector_comp = comp

    async def on_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == "agent_started":
            self._discover_components()
        elif event_name == "tool_executed":
            self.tool_counter += 1
            now = time.time()
            if (
                self.tool_counter >= self.tool_interval
                and (now - self.last_run_ts) >= self.min_interval_sec
            ):
                self.tool_counter = 0
                self.pending_garden = True
        elif event_name == "before_llm_call":
            if self.pending_garden:
                self.pending_garden = False
                self.last_run_ts = time.time()
                try:
                    await self.run_gardening_pass()
                except Exception as e:
                    print(f"[MemoryGardener:{self.thread_id}] Warning during gardening pass: {e}")

    def _select_candidate(self, now_ts: int) -> Optional[Any]:
        if not self.spreading_comp or not self.spreading_comp.layer_1_pool:
            return None

        pool_items = list(self.spreading_comp.layer_1_pool.values())
        ungroomed = [
            item for item in pool_items if int(getattr(item, "groomed_at", 0) or 0) == 0
        ]
        if ungroomed:
            ungroomed.sort(
                key=lambda item: (item.priority, item.last_similarity),
                reverse=True,
            )
            return ungroomed[0]

        stale_cutoff = now_ts - 86400
        stale = [
            item
            for item in pool_items
            if int(getattr(item, "groomed_at", 0) or 0) < stale_cutoff
        ]
        if stale:
            stale.sort(key=lambda item: int(getattr(item, "groomed_at", 0) or 0))
            return stale[0]
        return None

    def _collapse_in_layer1(
        self,
        target: Any,
        absorbed_ids: List[str],
        merged_content: str,
        now_ts: int,
        fast_path: bool = False,
    ) -> None:
        pool = self.spreading_comp.layer_1_pool
        absorbed_set = {str(aid) for aid in absorbed_ids if str(aid) != str(target.id)}

        if fast_path:
            all_hits = [target.hits] + [
                pool[aid].hits for aid in absorbed_set if aid in pool
            ]
            target.hits = max(all_hits) + len(absorbed_set)
        else:
            extra_hits = sum(
                pool[aid].hits if aid in pool else 1 for aid in absorbed_set
            )
            target.hits = target.hits + extra_hits

        all_ages = [target.age] + [
            pool[aid].age for aid in absorbed_set if aid in pool
        ]
        target.age = min(all_ages)
        target.content = merged_content
        target.groomed_at = now_ts

        for aid in absorbed_set:
            pool.pop(aid, None)

        if getattr(self.spreading_comp, "persist", False):
            self.spreading_comp._save_pool_to_db()
            # Also remove absorbed IDs from any other thread rows in SQLite layer_1_pool
            try:
                conn = self.spreading_comp._connect_db()
                try:
                    with conn:
                        placeholders = ",".join("?" for _ in absorbed_set)
                        if placeholders:
                            conn.execute(
                                f"DELETE FROM layer_1_pool WHERE id IN ({placeholders})",
                                tuple(absorbed_set),
                            )
                finally:
                    conn.close()
            except Exception:
                pass

    @staticmethod
    def _parse_merge_json(raw_text: str) -> Dict[str, Any]:
        clean = (raw_text or "").strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)
        try:
            return json.loads(clean)
        except Exception:
            m = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return {"should_merge": False}

    async def _ask_llm_to_merge(
        self, target: Any, neighbors: List[Dict[str, Any]], span: MetricSpan
    ) -> Dict[str, Any]:
        entries_text = f"[Memory Primary (ID: {target.id})]:\n{target.content}\n\n"
        for idx, n in enumerate(neighbors, start=1):
            entries_text += (
                f"[Neighbor {idx} (ID: {n['id']}, similarity={n.get('score', 0.0):.3f})]:\n"
                f"{n.get('content', '')}\n\n"
            )

        prompt = (
            "You are a memory consolidation daemon for an autonomous AI agent.\n"
            "Inspect the following closely related semantic memories (cosine similarity >= 0.88).\n"
            "If they express overlapping or redundant facts/insights that can be losslessly consolidated "
            "into a single self-contained memory block (preserving all specific names, dates, numbers, "
            "book/chunk references, and conclusions), return JSON:\n"
            '{"should_merge": true, "merged_content": "..."}\n'
            "If they represent genuinely distinct facts that should remain separate, return JSON:\n"
            '{"should_merge": false}\n'
            "Return ONLY valid JSON.\n\n"
            f"{entries_text}"
        )

        span.inc("request_tokens", count_tokens(prompt))
        span.inc("request_bytes", len(prompt.encode("utf-8")))

        with span.subtimer("llm_call_ms"):
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=self.max_output_tokens,
                stream=True,
            )
            resp_text = ""
            async for chunk in stream:
                if (
                    chunk.choices
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.content
                ):
                    resp_text += chunk.choices[0].delta.content

        span.inc("response_tokens", count_tokens(resp_text))
        span.inc("response_bytes", len(resp_text.encode("utf-8")))
        return self._parse_merge_json(resp_text)

    async def run_gardening_pass(self) -> Dict[str, Any]:
        """Executes a single Layer-1-driven semantic deduplication pass (Daemon 1: Clutter Merger)."""
        self._discover_components()
        if not self.spreading_comp or not self.vector_comp:
            return {"status": "skipped_missing_components"}

        now_ts = int(time.time())
        target = self._select_candidate(now_ts)
        if target is None:
            return {"status": "no_ungroomed_candidates"}

        with MetricSpan(
            operation="memory_garden",
            agent=self.agent_name,
            thread=self.thread_id,
            component="memory_gardener",
            type="clutter_merger",
            model=self.model_name,
            exec_location=infer_exec_location(self.api_url),
        ) as span:
            with span.subtimer("db_lookup_ms"):
                neighbors = self.vector_comp.find_neighbors_by_id(
                    target.id, top_k=5, score_threshold=0.88
                )
            span.inc("db_lookup", 1)
            span.inc("db_lookup_records", len(neighbors))

            if not neighbors:
                target.groomed_at = now_ts
                if getattr(self.spreading_comp, "persist", False):
                    self.spreading_comp._save_pool_to_db()
                return {"status": "clean", "target_id": target.id}

            fast_neighbors = [
                n for n in neighbors if float(n.get("score", 0.0)) >= 0.95
            ]
            if fast_neighbors:
                candidates = [(len(target.content or ""), target.content)] + [
                    (len(n.get("content") or ""), n.get("content") or "")
                    for n in fast_neighbors
                ]
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_content = candidates[0][1]
                redundant_ids = [str(n["id"]) for n in fast_neighbors]

                with span.subtimer("db_save_ms"):
                    if best_content != target.content:
                        self.vector_comp.upsert_point(
                            target.id, best_content, target.metadata
                        )
                    self.vector_comp.delete_points(redundant_ids)
                    self._collapse_in_layer1(
                        target,
                        redundant_ids,
                        best_content,
                        now_ts,
                        fast_path=True,
                    )
                span.inc("db_save", 1)
                span.inc("merged_records", len(redundant_ids))
                return {
                    "status": "fast_merged",
                    "target_id": target.id,
                    "deleted_ids": redundant_ids,
                }

            # LLM Synthesis Path (0.88 <= sim < 0.95)
            decision = await self._ask_llm_to_merge(target, neighbors, span)
            merged_content = str(decision.get("merged_content") or "").strip()
            if decision.get("should_merge") and merged_content:
                neighbor_ids = [str(n["id"]) for n in neighbors]
                with span.subtimer("db_save_ms"):
                    self.vector_comp.upsert_point(
                        target.id, merged_content, target.metadata
                    )
                    self.vector_comp.delete_points(neighbor_ids)
                    self._collapse_in_layer1(
                        target,
                        neighbor_ids,
                        merged_content,
                        now_ts,
                        fast_path=False,
                    )
                span.inc("db_save", 1)
                span.inc("merged_records", len(neighbor_ids))
                return {
                    "status": "llm_merged",
                    "target_id": target.id,
                    "deleted_ids": neighbor_ids,
                }

            target.groomed_at = now_ts
            if getattr(self.spreading_comp, "persist", False):
                self.spreading_comp._save_pool_to_db()
            return {"status": "kept_separate", "target_id": target.id}
