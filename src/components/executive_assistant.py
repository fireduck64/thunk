import inspect
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx
from openai import AsyncOpenAI

from src.components.base import BaseComponent
from src.components.notes import (
    DEFAULT_ASSISTANT_BRIEFING_PROMPT,
    DEFAULT_ASSISTANT_COMPACTION_PROMPT,
    DEFAULT_ASSISTANT_FILING_PROMPT,
    StructuredNotesComponent,
)
from src.utils.config import load_config
from src.utils.metrics import (
    MetricSpan,
    count_tokens,
    extract_chunk_usage,
    infer_exec_location,
    resolve_metrics_dir,
)


_FILING_JSON_FORMAT_APPENDIX = (
    "\n\nOUTPUT FORMAT (STRICT JSON ONLY):\n"
    "You MUST respond with a single valid JSON object matching this exact schema:\n"
    '{"updates": [{"key": "assistant/people/...", "summary": "Updated one-line summary", "new_points": ["- Specific new fact 1", "- Specific new fact 2"]}]}\n'
    "Do NOT repeat facts that are already written in the existing note content.\n"
    "If nothing in the evicted messages warrants saving or updating, respond with:\n"
    '{"updates": []}\n'
    "Do not include any prose outside the JSON object."
)


class ExecutiveAssistantComponent(BaseComponent):
    """
    Automated Chief of Staff / Filing Clerk component.
    Operates strictly in the background (exposing zero tools to the main agent):
    1. Pre-Turn Briefing (before_llm_call): Uses a Two-Pass Query + FTS5 + Qdrant Fallback
       pipeline to identify relevant 'assistant/*' dossiers, general notes, and semantic memories,
       then injects a transient '[Assistant Briefing]' block at the bottom of the prompt for
       entities/topics missing from active Layer-0 context.
    2. Background Filing Clerk (messages_evicted): Extracts net-new bullet points from messages
       evicted during TieredMemoryComponent compaction and appends them to 'assistant/*' dossiers.
    3. Background Note Compaction: Periodically condenses and deduplicates oversized 'assistant/*'
       dossiers exceeding max_note_chars using 'prompts/assistant_compaction'.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_candidate_notes: Optional[int] = None,
        max_briefing_tokens: Optional[int] = None,
        max_note_chars: Optional[int] = None,
        compaction_tool_interval: Optional[int] = None,
        compaction_min_interval_sec: Optional[float] = None,
        qdrant_score_threshold: Optional[float] = None,
        audit_log_path: Optional[str] = None,
        thread_id: str = "main",
        notes_comp: Optional[StructuredNotesComponent] = None,
        memory_comp: Optional[Any] = None,
        vector_comp: Optional[Any] = None,
    ):
        self.config = load_config()
        self.agent_name = self.config.get("agent", {}).get("name", "Thunk")
        self.thread_id = (thread_id or "main").strip() or "main"

        ea_cfg = self.config.get("executive_assistant", {})
        sub_cfg = self.config.get("subconscious", {})
        llm_cfg = self.config.get("llm", {})

        if enabled is not None:
            self.enabled = bool(enabled)
        elif "enabled" in ea_cfg:
            self.enabled = bool(ea_cfg["enabled"])
        else:
            self.enabled = True

        self.api_base = (
            api_base
            or ea_cfg.get("api_base")
            or ea_cfg.get("api_url")
            or sub_cfg.get("api_base")
            or sub_cfg.get("api_url")
            or llm_cfg.get("api_base")
            or llm_cfg.get("api_url")
            or "http://localhost:11434/v1"
        )
        self.api_url = self.api_base
        self.api_key = (
            api_key
            or ea_cfg.get("api_key")
            or sub_cfg.get("api_key")
            or llm_cfg.get("api_key")
            or "ollama"
        )
        self.model = (
            model
            or ea_cfg.get("model")
            or ea_cfg.get("model_name")
            or sub_cfg.get("model")
            or sub_cfg.get("model_name")
            or llm_cfg.get("model")
            or llm_cfg.get("model_name")
            or "gemma4:e2b"
        )
        self.model_name = self.model
        self.max_candidate_notes = int(
            max_candidate_notes
            if max_candidate_notes is not None
            else ea_cfg.get("max_candidate_notes", 10)
        )
        self.max_briefing_tokens = int(
            max_briefing_tokens
            if max_briefing_tokens is not None
            else ea_cfg.get("max_briefing_tokens", 500)
        )
        self.max_filing_tokens = int(ea_cfg.get("max_filing_tokens", 2048))
        self.max_note_chars = int(
            max_note_chars
            if max_note_chars is not None
            else ea_cfg.get("max_note_chars", 2000)
        )
        self.compaction_tool_interval = int(
            compaction_tool_interval
            if compaction_tool_interval is not None
            else ea_cfg.get("compaction_tool_interval", 20)
        )
        self.compaction_min_interval_sec = float(
            compaction_min_interval_sec
            if compaction_min_interval_sec is not None
            else ea_cfg.get("compaction_min_interval_sec", 600.0)
        )
        self.qdrant_score_threshold = float(
            qdrant_score_threshold
            if qdrant_score_threshold is not None
            else ea_cfg.get("qdrant_score_threshold", 0.72)
        )
        self.audit_log_path = (
            audit_log_path
            or ea_cfg.get("audit_log_path")
            or os.path.join(resolve_metrics_dir(), "ea_briefings.jsonl")
        )

        self.client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            default_headers={
                "X-Agent-ID": f"{self.agent_name}-{self.thread_id}-executive_assistant"
            },
            timeout=httpx.Timeout(None),
        )

        self.notes_comp: Optional[StructuredNotesComponent] = notes_comp
        self.memory_comp: Optional[Any] = memory_comp
        self.vector_comp: Optional[Any] = vector_comp
        self.last_usage: Optional[Dict[str, Any]] = None
        self._last_briefed_sig: Optional[Tuple[int, str]] = None

        self.tool_counter: int = 0
        self.last_compaction_ts: float = 0.0
        self.pending_note_compaction: bool = False
        self._pending_audit_record: Optional[Dict[str, Any]] = None
        self._pending_probe_tokens: Set[str] = set()

        if self.notes_comp is not None:
            self.notes_comp.ensure_seed_prompts()

    def get_system_prompt_addition(self) -> str:
        if not self.enabled:
            return ""
        return (
            "You have an automated Executive Assistant that maintains reference dossiers under 'assistant/*' "
            "(read-only to you) and injects a transient '[Assistant Briefing]' before your turn when incoming "
            "messages reference people or topics missing from your active context. You control how your assistant "
            "operates by editing 'prompts/assistant_briefing', 'prompts/assistant_filing', and "
            "'prompts/assistant_compaction' via put_note."
        )

    def get_tools(self) -> list:
        return []

    def _discover_components(self, agent: Any = None) -> None:
        agent_obj = agent or getattr(self, "agent", None)
        if not agent_obj:
            return
        for comp in getattr(agent_obj, "components", []):
            cname = comp.__class__.__name__
            if cname == "StructuredNotesComponent" and self.notes_comp is None:
                self.notes_comp = comp
            elif cname == "TieredMemoryComponent" and self.memory_comp is None:
                self.memory_comp = comp
            elif cname in ("VectorMemoryComponent", "SemanticMemoryComponent") and self.vector_comp is None:
                self.vector_comp = comp
        if self.notes_comp is not None:
            self.notes_comp.ensure_seed_prompts()

    def _get_prompt(self, key: str, default_text: str) -> str:
        if self.notes_comp is None:
            return default_text
        val = self.notes_comp.assistant_get_note(key)
        if val and val.strip():
            return val.strip()
        self.notes_comp.ensure_seed_prompts()
        val = self.notes_comp.assistant_get_note(key)
        return val.strip() if (val and val.strip()) else default_text

    @staticmethod
    def _strip_subconscious_prefix(text: str) -> str:
        raw = text or ""
        if raw.startswith("[Subconscious Recall Triggered"):
            parts = raw.split("--- End Recall ---\n\n", 1)
            if len(parts) == 2:
                return parts[1]
        return raw

    @staticmethod
    def _strip_transient_briefing(messages: List[Dict[str, Any]]) -> None:
        if not messages:
            return
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], dict) and messages[idx].get("_transient_briefing"):
                del messages[idx]

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
    def _strip_code_fences(cls, raw_text: str) -> str:
        return cls._strip_thinking_and_fences(raw_text)

    def _write_audit_record(self, record: Dict[str, Any]) -> None:
        if not self.audit_log_path:
            return
        try:
            parent = os.path.dirname(os.path.abspath(self.audit_log_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass
        except Exception as e:
            print(f"[ExecutiveAssistant:{self.thread_id}] Warning writing audit log: {e}")

    async def _select_relevant_note_keys(
        self,
        context_text: str,
        candidate_notes: List[Dict[str, str]],
        purpose: str = "briefing",
    ) -> List[str]:
        """
        Uses a fast LLM selector pass over the compact directory of (key: summary) pairs
        to choose up to self.max_candidate_notes relevant note keys (used by compaction filing).
        """
        if not candidate_notes:
            return []

        directory_lines = [
            f"- {n['key']}: {n.get('summary') or ''}" for n in candidate_notes
        ]
        directory_str = "\n".join(directory_lines)

        system_prompt = (
            "You are an indexing assistant. Given a target text and a directory of available notes (key + summary), "
            f"select up to {self.max_candidate_notes} note keys that are directly or indirectly relevant to the people, "
            "entities, projects, or concepts in the target text. "
            'Return ONLY a JSON object: {"keys": ["key1", "key2"]} (or {"keys": []} if none are relevant).'
        )
        user_prompt = (
            f"### Purpose: {purpose}\n\n"
            f"### Available Notes Directory\n{directory_str}\n\n"
            f"### Target Text\n{context_text}\n"
        )

        try:
            raw_resp = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=256,
                span_type="assistant_note_select",
            )
        except Exception as e:
            print(f"[ExecutiveAssistant:{self.thread_id}] Warning during note selection ({purpose}): {e}")
            return []

        clean = self._strip_thinking_and_fences(raw_resp)
        if not clean or clean.upper() == "NONE":
            return []

        parsed: Any = None
        try:
            parsed = json.loads(clean)
        except Exception:
            m = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

        raw_keys: List[Any] = []
        if isinstance(parsed, dict):
            val = parsed.get("keys")
            if isinstance(val, list):
                raw_keys = val
            elif isinstance(parsed.get("assistant_keys"), list):
                raw_keys = parsed.get("assistant_keys")
        elif isinstance(parsed, list):
            raw_keys = parsed

        if not raw_keys:
            return []

        key_lookup: Dict[str, str] = {}
        for n in candidate_notes:
            k = n["key"]
            k_low = k.lower()
            key_lookup[k_low] = k
            if k_low.startswith("assistant/"):
                key_lookup[k_low[len("assistant/"):]] = k

        selected: List[str] = []
        seen = set()
        for rk in raw_keys:
            rk_clean = str(rk or "").strip().lstrip("/").lower()
            if not rk_clean:
                continue
            canonical = key_lookup.get(rk_clean)
            if canonical and canonical not in seen:
                seen.add(canonical)
                selected.append(canonical)
                if len(selected) >= self.max_candidate_notes:
                    break

        return selected

    async def _select_briefing_targets(
        self,
        incoming_event: str,
        active_context_str: str,
        assistant_notes: List[Dict[str, str]],
        searchable_notes: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Pass 1 of Pre-Turn Briefing: Gap Detection & Query Generation (~1k-2k prompt tokens).
        Inspects Active Context, Incoming Event, and the compact 'assistant/*' dossier directory,
        returning (selected_assistant_keys, search_queries).
        """
        directory_lines = [
            f"- {n['key']}: {n.get('summary') or ''}" for n in assistant_notes
        ]
        directory_str = (
            "\n".join(directory_lines)
            if directory_lines
            else "(No 'assistant/*' dossiers exist yet)"
        )

        system_prompt = (
            "You are an indexing assistant. Given my Active Context (Core Summary + Recent Messages), "
            "an Incoming Event, and a directory of available 'assistant/*' dossiers, identify people, "
            "projects, terms, or history mentioned in the Incoming Event that are missing or under-explained "
            "in my Active Context.\n"
            "Return ONLY a JSON object with up to 5 relevant 'assistant/*' keys and up to 3 short keyword "
            "search queries for general notes/memories:\n"
            '{"assistant_keys": ["assistant/people/fireduck"], "queries": ["pinkcroc", "taravangian"]}\n'
            'If nothing is missing from my active context, return: {"assistant_keys": [], "queries": []}'
        )
        user_prompt = (
            f"### Active Context (Core Summary + Recent Messages)\n{active_context_str}\n\n"
            f"### Incoming Event\n{incoming_event}\n\n"
            f"### Available 'assistant/*' Dossiers\n{directory_str}\n"
        )

        try:
            raw_resp = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=256,
                span_type="assistant_note_select",
            )
        except Exception as e:
            print(f"[ExecutiveAssistant:{self.thread_id}] Warning during briefing target selection: {e}")
            return [], []

        clean = self._strip_thinking_and_fences(raw_resp)
        if not clean or clean.upper() == "NONE":
            return [], []

        parsed: Any = None
        try:
            parsed = json.loads(clean)
        except Exception:
            m = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

        assistant_lookup: Dict[str, str] = {}
        for n in assistant_notes:
            k = n["key"]
            k_low = k.lower()
            assistant_lookup[k_low] = k
            if k_low.startswith("assistant/"):
                assistant_lookup[k_low[len("assistant/"):]] = k

        searchable_lookup: Dict[str, str] = {}
        for n in searchable_notes or []:
            k = n["key"]
            searchable_lookup[k.lower()] = k

        raw_asst_keys: List[Any] = []
        raw_queries: List[Any] = []

        if isinstance(parsed, dict):
            if isinstance(parsed.get("assistant_keys"), list):
                raw_asst_keys.extend(parsed["assistant_keys"])
            if isinstance(parsed.get("queries"), list):
                raw_queries.extend(parsed["queries"])
            # Backward-compatible support for {"keys": [...]}
            if isinstance(parsed.get("keys"), list):
                for rk in parsed["keys"]:
                    rk_str = str(rk or "").strip().lstrip("/")
                    rk_low = rk_str.lower()
                    if rk_low in assistant_lookup:
                        raw_asst_keys.append(rk_str)
                    elif rk_low in searchable_lookup:
                        raw_asst_keys.append(searchable_lookup[rk_low])
                    elif rk_str:
                        raw_queries.append(rk_str)
        elif isinstance(parsed, list):
            for rk in parsed:
                rk_str = str(rk or "").strip().lstrip("/")
                if rk_str.lower() in assistant_lookup:
                    raw_asst_keys.append(rk_str)
                elif rk_str:
                    raw_queries.append(rk_str)

        selected_asst_keys: List[str] = []
        seen_keys: Set[str] = set()
        for rk in raw_asst_keys:
            rk_clean = str(rk or "").strip().lstrip("/").lower()
            if not rk_clean:
                continue
            canonical = assistant_lookup.get(rk_clean) or searchable_lookup.get(rk_clean)
            if canonical and canonical not in seen_keys:
                seen_keys.add(canonical)
                selected_asst_keys.append(canonical)
                if len(selected_asst_keys) >= 5:
                    break

        selected_queries: List[str] = []
        seen_queries: Set[str] = set()
        for rq in raw_queries:
            q_clean = str(rq or "").strip()
            q_low = q_clean.lower()
            if q_clean and q_low not in seen_queries:
                seen_queries.add(q_low)
                selected_queries.append(q_clean)
                if len(selected_queries) >= 3:
                    break

        return selected_asst_keys, selected_queries

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        span_type: str,
    ) -> str:
        agent_obj = getattr(self, "agent", None)
        agent_name = getattr(agent_obj, "name", None) or self.agent_name
        thread_id = getattr(agent_obj, "thread_id", None) or self.thread_id or "main"
        exec_loc = infer_exec_location(self.api_base)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        req_text = f"{system_prompt}\n{user_prompt}"

        with MetricSpan(
            operation="llm_call",
            agent=agent_name,
            thread=thread_id,
            component="executive_assistant",
            type=span_type,
            model=self.model,
            exec_location=exec_loc,
        ) as span:
            span.inc("request_tokens", count_tokens(req_text, agent=agent_obj))
            span.inc("request_bytes", len(req_text.encode("utf-8")))

            with span.subtimer("llm_call_ms"):
                create_call = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    extra_body={"think": False},
                    stream=True,
                    stream_options={"include_usage": True},
                )
                resp = await create_call if inspect.isawaitable(create_call) else create_call

                content = ""
                usage = None
                if hasattr(resp, "__aiter__"):
                    async for chunk in resp:
                        parsed_usage = extract_chunk_usage(chunk, prev_usage=usage)
                        if parsed_usage is not None:
                            usage = parsed_usage
                        if (
                            getattr(chunk, "choices", None)
                            and chunk.choices[0].delta
                            and getattr(chunk.choices[0].delta, "content", None)
                        ):
                            content += chunk.choices[0].delta.content
                elif hasattr(resp, "choices") and resp.choices:
                    msg = getattr(resp.choices[0], "message", None)
                    if msg is not None and getattr(msg, "content", None):
                        content = str(msg.content)
                    elif isinstance(resp.choices[0], dict):
                        content = str(resp.choices[0].get("message", {}).get("content") or "")
                elif isinstance(resp, str):
                    content = resp

            content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
            if "<think>" in content:
                content = content.split("<think>", 1)[0].strip()

            self.last_usage = usage
            if usage:
                if usage.get("prompt_tokens"):
                    span.set_counter("request_tokens", int(usage["prompt_tokens"]))
                if usage.get("cached_tokens") is not None:
                    span.set_counter("cached_tokens", int(usage["cached_tokens"]))
                if usage.get("thinking_tokens") is not None:
                    span.set_counter("thinking_tokens", int(usage["thinking_tokens"]))
            span.inc(
                "response_tokens",
                (usage or {}).get("completion_tokens") or count_tokens(content, agent=agent_obj),
            )
            span.inc("response_bytes", len((content or "").encode("utf-8")))
            return content

    def _get_core_summary(self) -> str:
        if self.memory_comp is not None and hasattr(self.memory_comp, "_get_core_summary"):
            try:
                return self.memory_comp._get_core_summary()
            except Exception:
                pass
        return "No summary yet."

    @staticmethod
    def _format_recent_messages(messages: List[Dict[str, Any]], max_messages: int = 5) -> str:
        lines: List[str] = []
        for msg in messages[-max_messages:]:
            role = str(msg.get("role") or "user").upper()
            content = ExecutiveAssistantComponent._strip_subconscious_prefix(
                str(msg.get("content") or "")
            ).strip()
            if not content and msg.get("tool_calls"):
                content = "[Called tools]"
            if len(content) > 500:
                content = content[:500] + "...[truncated]"
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines) if lines else "(No prior messages in active window)"

    @classmethod
    def _parse_filing_json(cls, raw_text: str) -> List[Dict[str, Any]]:
        clean = cls._strip_thinking_and_fences(raw_text)
        if not clean or clean.upper() == "NONE":
            return []

        parsed: Any = None
        try:
            parsed = json.loads(clean)
        except Exception:
            m = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
            if parsed is None:
                # Repair truncated trailing JSON array/object if cut off mid-stream
                start_idx = clean.find("{")
                if start_idx != -1:
                    fragment = clean[start_idx:]
                    for suffix in ("]}", '"]}', '"}]}', "}]}", "]"):
                        try:
                            parsed = json.loads(fragment + suffix)
                            break
                        except Exception:
                            continue

        if isinstance(parsed, dict):
            updates = parsed.get("updates", [])
        elif isinstance(parsed, list):
            updates = parsed
        else:
            return []

        if not isinstance(updates, list):
            return []

        valid_updates: List[Dict[str, Any]] = []
        for item in updates:
            if not isinstance(item, dict):
                continue
            raw_key = str(item.get("key") or "").strip().lstrip("/")
            if not raw_key:
                continue
            if not raw_key.lower().startswith("assistant/"):
                raw_key = f"assistant/{raw_key}"
            else:
                raw_key = "assistant/" + raw_key[len("assistant/"):]

            summary = str(item.get("summary") or "").strip()

            points: List[str] = []
            raw_new_points = item.get("new_points")
            if isinstance(raw_new_points, list):
                for pt in raw_new_points:
                    for line in str(pt or "").splitlines():
                        if line.strip():
                            points.append(line.strip())
            elif isinstance(raw_new_points, str) and raw_new_points.strip():
                for line in raw_new_points.splitlines():
                    if line.strip():
                        points.append(line.strip())
            else:
                # Fallback if model returned "content" or "value" string
                fallback_content = str(
                    item.get("content") if "content" in item else item.get("value") or ""
                ).strip()
                if fallback_content:
                    for line in fallback_content.splitlines():
                        if line.strip():
                            points.append(line.strip())

            if not points:
                continue

            valid_updates.append(
                {
                    "key": raw_key,
                    "summary": summary,
                    "new_points": points,
                }
            )
        return valid_updates

    async def run_note_compaction_pass(self) -> Optional[str]:
        """
        Finds the largest 'assistant/*' note exceeding self.max_note_chars and compacts/deduplicates
        it using 'prompts/assistant_compaction'. Returns the compacted note key, or None if skipped.
        """
        self._discover_components()
        if not self.enabled or self.notes_comp is None:
            return None

        all_notes = self.notes_comp.get_all_notes()
        oversized = [
            n
            for n in all_notes
            if n["key"].startswith("assistant/")
            and len(n.get("content") or "") > self.max_note_chars
        ]
        if not oversized:
            return None

        oversized.sort(key=lambda n: len(n.get("content") or ""), reverse=True)
        target_note = oversized[0]
        old_content = target_note.get("content") or ""
        old_chars = len(old_content)

        compaction_instructions = self._get_prompt(
            "prompts/assistant_compaction", DEFAULT_ASSISTANT_COMPACTION_PROMPT
        )
        user_prompt = (
            f"### Note Key: {target_note['key']}\n"
            f"### Current Summary: {target_note.get('summary') or ''}\n\n"
            f"### Current Note Content ({old_chars} chars)\n{old_content}\n"
        )

        try:
            raw_compacted = await self._call_llm(
                system_prompt=compaction_instructions,
                user_prompt=user_prompt,
                max_tokens=self.max_filing_tokens,
                span_type="assistant_note_compaction",
            )
        except Exception as e:
            print(f"[ExecutiveAssistant:{self.thread_id}] Warning during note compaction pass: {e}")
            return None

        compacted_content = self._strip_thinking_and_fences(raw_compacted)
        if (
            not compacted_content
            or compacted_content.upper() == "NONE"
            or len(compacted_content) < 100
        ):
            print(
                f"[ExecutiveAssistant:{self.thread_id}] Refusing to overwrite '{target_note['key']}' "
                f"with invalid/too-short compacted content ({len(compacted_content)} chars)."
            )
            return None

        self.notes_comp.assistant_put_note(
            target_note["key"],
            target_note.get("summary") or "",
            compacted_content,
        )

        agent = getattr(self, "agent", None)
        if agent is not None and hasattr(agent, "event_bus"):
            try:
                await agent.event_bus.publish(
                    "assistant_note_compacted",
                    {
                        "thread_id": self.thread_id,
                        "key": target_note["key"],
                        "old_chars": old_chars,
                        "new_chars": len(compacted_content),
                    },
                )
            except Exception:
                pass

        return target_note["key"]

    async def _handle_before_llm_call(self, payload: Dict[str, Any]) -> None:
        agent = payload.get("agent") or getattr(self, "agent", None)
        self._discover_components(agent)
        if not self.enabled or self.notes_comp is None:
            return

        if self.pending_note_compaction:
            self.pending_note_compaction = False
            self.last_compaction_ts = time.time()
            try:
                await self.run_note_compaction_pass()
            except Exception as e:
                print(f"[ExecutiveAssistant:{self.thread_id}] Warning during scheduled note compaction: {e}")

        messages = payload.get("messages")
        if messages is None and agent is not None:
            messages = getattr(agent, "messages", None)
        if not isinstance(messages, list):
            return

        self._strip_transient_briefing(messages)

        non_transient = [
            m
            for m in messages
            if isinstance(m, dict)
            and not m.get("_transient_presence")
            and not m.get("_transient_briefing")
            and not m.get("_transient_nudge")
        ]
        if not non_transient:
            return

        last_msg = non_transient[-1]
        if last_msg.get("role") != "user":
            return

        raw_incoming = str(last_msg.get("content") or "").strip()
        incoming_event = self._strip_subconscious_prefix(raw_incoming).strip()
        if (
            not incoming_event
            or incoming_event.startswith("[System Clock]")
            or incoming_event.startswith("[System Startup]")
        ):
            return

        sig = (len(non_transient), incoming_event)
        if self._last_briefed_sig == sig:
            return
        self._last_briefed_sig = sig

        all_notes = self.notes_comp.get_all_notes()
        searchable_notes = [
            n
            for n in all_notes
            if n["key"] != "sticky" and not n["key"].startswith("prompts/")
        ]
        assistant_notes = [n for n in all_notes if n["key"].startswith("assistant/")]
        if not searchable_notes and self.vector_comp is None:
            return

        prior_messages = [
            m for m in non_transient[1:-1] if m.get("role") != "system"
        ]
        core_summary = self._get_core_summary()
        recent_messages_str = self._format_recent_messages(prior_messages, max_messages=5)
        active_context_str = f"Core Summary:\n{core_summary}\n\nRecent Messages:\n{recent_messages_str}"

        selected_assistant_keys, selected_queries = await self._select_briefing_targets(
            incoming_event=incoming_event,
            active_context_str=active_context_str,
            assistant_notes=assistant_notes,
            searchable_notes=searchable_notes,
        )

        fts5_hit_keys: List[str] = []
        qdrant_hits: List[Dict[str, Any]] = []

        audit_record: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "agent": getattr(agent, "name", None) or self.agent_name,
            "thread": getattr(agent, "thread_id", None) or self.thread_id,
            "incoming_event_preview": incoming_event[:200],
            "assistant_keys": selected_assistant_keys,
            "queries": selected_queries,
            "fts5_hit_keys": fts5_hit_keys,
            "qdrant_fallback_hits": [],
            "briefing_injected": False,
            "briefing_bytes": 0,
            "briefing_preview": "",
            "referenced_by_boss": None,
        }

        if not selected_assistant_keys and not selected_queries:
            self._write_audit_record(audit_record)
            return

        notes_by_key = {n["key"]: n for n in searchable_notes}
        candidates: List[Dict[str, str]] = []
        seen_candidate_keys: Set[str] = set()

        for ak in selected_assistant_keys:
            if ak in notes_by_key and ak not in seen_candidate_keys:
                seen_candidate_keys.add(ak)
                candidates.append(notes_by_key[ak])

        seen_qdrant_ids: Set[str] = set()
        qdrant_blocks: List[str] = []

        for q in selected_queries[:3]:
            fts_hits: List[Dict[str, Any]] = []
            if hasattr(self.notes_comp, "search_notes_structured"):
                try:
                    fts_hits = self.notes_comp.search_notes_structured(q, limit=3)
                except Exception:
                    fts_hits = []

            for hit in fts_hits:
                hk = hit.get("key")
                if hk and hk not in fts5_hit_keys:
                    fts5_hit_keys.append(hk)
                if hk and hk not in seen_candidate_keys and len(candidates) < self.max_candidate_notes:
                    seen_candidate_keys.add(hk)
                    candidates.append(
                        {
                            "key": hk,
                            "summary": hit.get("summary") or "",
                            "content": hit.get("content") or "",
                        }
                    )

            if (
                len(fts_hits) < 2
                and self.vector_comp is not None
                and hasattr(self.vector_comp, "search_semantic_memory_raw")
            ):
                try:
                    raw_vec = self.vector_comp.search_semantic_memory_raw(q, num_results=3)
                    if inspect.isawaitable(raw_vec):
                        raw_vec = await raw_vec
                except Exception:
                    raw_vec = []

                for vhit in raw_vec or []:
                    if not isinstance(vhit, dict):
                        continue
                    vscore = float(vhit.get("score") or 0.0)
                    vid = str(vhit.get("id") or "")
                    vcontent = str(vhit.get("content") or vhit.get("text") or "").strip()
                    if vscore < self.qdrant_score_threshold or not vcontent:
                        continue
                    if vid and vid in seen_qdrant_ids:
                        continue
                    if vid:
                        seen_qdrant_ids.add(vid)
                    vmeta = vhit.get("metadata") or {}
                    vts = str(vhit.get("timestamp") or vmeta.get("timestamp") or "unknown")
                    qdrant_hits.append(
                        {
                            "id": vid,
                            "score": vscore,
                            "content": vcontent,
                            "timestamp": vts,
                            "query": q,
                        }
                    )
                    qdrant_blocks.append(
                        f"--- [Semantic Memory Fallback] (query='{q}', score={vscore:.2f}, saved={vts}) ---\n{vcontent}"
                    )

        audit_record["fts5_hit_keys"] = fts5_hit_keys
        audit_record["qdrant_fallback_hits"] = [
            {"id": h["id"], "score": round(h["score"], 3), "preview": h["content"][:120]}
            for h in qdrant_hits
        ]

        if not candidates and not qdrant_blocks:
            self._write_audit_record(audit_record)
            return

        briefing_instructions = self._get_prompt(
            "prompts/assistant_briefing", DEFAULT_ASSISTANT_BRIEFING_PROMPT
        )

        cand_blocks: List[str] = []
        for c in candidates:
            cand_blocks.append(
                f"--- Note: {c['key']} (Summary: {c['summary']}) ---\n{c['content']}"
            )
        cand_blocks.extend(qdrant_blocks)
        candidates_section = "\n\n".join(cand_blocks)

        assistant_index_lines = [
            f"- {n['key']}: {n['summary']}" for n in assistant_notes
        ]
        assistant_index_section = (
            "\n".join(assistant_index_lines)
            if assistant_index_lines
            else "(No 'assistant/*' dossiers exist yet)"
        )

        user_prompt = (
            f"### Active Context (What the Boss already sees)\n{active_context_str}\n\n"
            f"### Incoming Event\n{incoming_event}\n\n"
            f"### Existing 'assistant/*' Dossier Index\n{assistant_index_section}\n\n"
            f"### Candidate Notes (from 'assistant/*' and general notes)\n{candidates_section}\n"
        )

        try:
            response_text = await self._call_llm(
                system_prompt=briefing_instructions,
                user_prompt=user_prompt,
                max_tokens=self.max_briefing_tokens,
                span_type="assistant_briefing",
            )
        except Exception as e:
            print(f"[ExecutiveAssistant:{self.thread_id}] Warning during pre-turn briefing: {e}")
            self._write_audit_record(audit_record)
            return

        cleaned = (response_text or "").strip()
        if not cleaned or cleaned.upper() == "NONE" or cleaned.upper().startswith("NONE\n"):
            self._write_audit_record(audit_record)
            return

        if cleaned.startswith("[Assistant Briefing]"):
            briefing_block = cleaned
        else:
            briefing_block = f"[Assistant Briefing]\n{cleaned}"

        # Insert before any _transient_presence footer if present, else at the very bottom
        insert_idx = len(messages)
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], dict) and messages[idx].get("_transient_presence"):
                insert_idx = idx
            else:
                break

        messages.insert(
            insert_idx,
            {
                "role": "user",
                "content": briefing_block,
                "_transient_briefing": True,
            },
        )

        probe_tokens: Set[str] = set()
        for k in selected_assistant_keys + fts5_hit_keys:
            for seg in re.split(r"[/_\-\s]+", k.lower()):
                if len(seg) >= 3 and seg not in ("assistant", "people", "topics", "notes", "general"):
                    probe_tokens.add(seg)
        for q in selected_queries:
            for tok in re.findall(r"[A-Za-z0-9_]{3,}", q.lower()):
                probe_tokens.add(tok)

        audit_record["briefing_injected"] = True
        audit_record["briefing_bytes"] = len(briefing_block.encode("utf-8"))
        audit_record["briefing_preview"] = briefing_block[:200]
        self._pending_audit_record = audit_record
        self._pending_probe_tokens = probe_tokens

        if agent is not None and hasattr(agent, "event_bus"):
            try:
                await agent.event_bus.publish(
                    "assistant_briefing_injected",
                    {
                        "thread_id": self.thread_id,
                        "briefing": briefing_block,
                        "candidate_keys": [c["key"] for c in candidates],
                    },
                )
            except Exception:
                pass

    def _flush_pending_audit_on_after_llm(self, payload: Dict[str, Any]) -> None:
        if self._pending_audit_record is None:
            return
        record = self._pending_audit_record
        probe_tokens = self._pending_probe_tokens
        self._pending_audit_record = None
        self._pending_probe_tokens = set()

        resp = payload.get("response")
        agent = payload.get("agent") or getattr(self, "agent", None)
        out_parts: List[str] = []

        if resp is not None:
            if isinstance(resp, dict):
                if resp.get("content"):
                    out_parts.append(str(resp["content"]))
                if resp.get("tool_calls"):
                    out_parts.append(str(resp["tool_calls"]))
            else:
                if getattr(resp, "content", None):
                    out_parts.append(str(resp.content))
                if getattr(resp, "tool_calls", None):
                    out_parts.append(str(resp.tool_calls))
        elif agent is not None and isinstance(getattr(agent, "messages", None), list):
            for m in reversed(agent.messages):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    if m.get("content"):
                        out_parts.append(str(m["content"]))
                    if m.get("tool_calls"):
                        out_parts.append(str(m["tool_calls"]))
                    break

        assistant_output_lower = "\n".join(out_parts).lower()
        record["referenced_by_boss"] = bool(
            probe_tokens
            and any(tok.lower() in assistant_output_lower for tok in probe_tokens if len(tok) >= 3)
        )
        self._write_audit_record(record)

    async def _handle_messages_evicted(self, payload: Dict[str, Any]) -> List[str]:
        agent = payload.get("agent") or getattr(self, "agent", None)
        self._discover_components(agent)
        if not self.enabled or self.notes_comp is None:
            return []

        evicted_messages = payload.get("messages") or []
        if not evicted_messages:
            return []

        transcript_lines: List[str] = []
        for msg in evicted_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "unknown").upper()
            content = self._strip_subconscious_prefix(str(msg.get("content") or "")).strip()
            if content.startswith("[System Clock]") or content.startswith("[System Startup]"):
                continue
            if not content and msg.get("tool_calls"):
                content = "[Called tools]"
            if role == "TOOL" and len(content) > 600:
                content = content[:600] + "\n...[truncated]..."
            if content:
                transcript_lines.append(f"[{role}]: {content}")

        if not transcript_lines:
            return []

        transcript_text = "\n".join(transcript_lines)
        filing_instructions = (
            self._get_prompt("prompts/assistant_filing", DEFAULT_ASSISTANT_FILING_PROMPT)
            + _FILING_JSON_FORMAT_APPENDIX
        )

        all_notes = self.notes_comp.get_all_notes()
        assistant_notes = [n for n in all_notes if n["key"].startswith("assistant/")]

        matched_existing: List[Dict[str, str]] = []
        if assistant_notes:
            matched_keys = await self._select_relevant_note_keys(
                context_text=transcript_text,
                candidate_notes=assistant_notes,
                purpose="filing",
            )
            assistant_by_key = {n["key"]: n for n in assistant_notes}
            matched_existing = [assistant_by_key[k] for k in matched_keys if k in assistant_by_key]

        index_section = (
            "\n".join(f"- {n['key']}: {n['summary']}" for n in assistant_notes)
            if assistant_notes
            else "(No 'assistant/*' notes exist yet)"
        )
        matched_section = (
            "\n\n".join(
                f"--- Existing Note: {n['key']} (Summary: {n['summary']}) ---\n{n['content']}"
                for n in matched_existing[: self.max_candidate_notes]
            )
            if matched_existing
            else "(No existing 'assistant/*' notes selected for this batch)"
        )

        user_prompt = (
            f"### Existing 'assistant/*' Notes Directory\n{index_section}\n\n"
            f"### Relevant Existing 'assistant/*' Notes (Full Content)\n{matched_section}\n\n"
            f"### Evicted Messages\n{transcript_text}\n"
        )

        try:
            raw_response = await self._call_llm(
                system_prompt=filing_instructions,
                user_prompt=user_prompt,
                max_tokens=self.max_filing_tokens,
                span_type="assistant_filing",
            )
        except Exception as e:
            print(f"[ExecutiveAssistant:{self.thread_id}] Warning during compaction filing: {e}")
            return []

        updates = self._parse_filing_json(raw_response)
        updated_keys: List[str] = []
        for item in updates:
            try:
                canonical_key = self.notes_comp.assistant_append_note(
                    key=item["key"],
                    summary=item["summary"],
                    new_points=item["new_points"],
                )
                if canonical_key not in updated_keys:
                    updated_keys.append(canonical_key)
            except Exception as e:
                print(f"[ExecutiveAssistant:{self.thread_id}] Failed saving note '{item.get('key')}': {e}")

        if updated_keys:
            now = time.time()
            if (now - self.last_compaction_ts) >= self.compaction_min_interval_sec:
                latest_notes = self.notes_comp.get_all_notes()
                if any(
                    n["key"].startswith("assistant/")
                    and len(n.get("content") or "") > self.max_note_chars
                    for n in latest_notes
                ):
                    self.pending_note_compaction = True

            if agent is not None and hasattr(agent, "event_bus"):
                try:
                    await agent.event_bus.publish(
                        "assistant_notes_updated",
                        {
                            "thread_id": self.thread_id,
                            "updated_keys": updated_keys,
                        },
                    )
                except Exception:
                    pass

        return updated_keys

    async def on_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == "agent_started":
            self._discover_components(payload.get("agent"))
        elif event_name == "tool_executed":
            self.tool_counter += 1
            now = time.time()
            if (
                self.tool_counter >= self.compaction_tool_interval
                and (now - self.last_compaction_ts) >= self.compaction_min_interval_sec
            ):
                self.tool_counter = 0
                self.pending_note_compaction = True
        elif event_name == "before_llm_call":
            await self._handle_before_llm_call(payload)
        elif event_name == "after_llm_call":
            agent = payload.get("agent") or getattr(self, "agent", None)
            if agent is not None and isinstance(getattr(agent, "messages", None), list):
                self._strip_transient_briefing(agent.messages)
            messages = payload.get("messages")
            if isinstance(messages, list):
                self._strip_transient_briefing(messages)
            self._flush_pending_audit_on_after_llm(payload)
        elif event_name == "messages_evicted":
            await self._handle_messages_evicted(payload)
