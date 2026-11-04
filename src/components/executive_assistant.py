import inspect
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx
from openai import AsyncOpenAI

from src.components.base import BaseComponent
from src.components.notes import (
    DEFAULT_ASSISTANT_BRIEFING_PROMPT,
    DEFAULT_ASSISTANT_FILING_PROMPT,
    StructuredNotesComponent,
)
from src.utils.config import load_config
from src.utils.metrics import MetricSpan, count_tokens, extract_chunk_usage, infer_exec_location


_STOPWORDS: Set[str] = {
    "about", "after", "again", "also", "always", "another", "because", "been",
    "before", "being", "between", "both", "could", "does", "doing", "down",
    "during", "each", "even", "every", "from", "further", "have", "having",
    "here", "hers", "herself", "himself", "into", "itself", "just", "know",
    "like", "make", "many", "more", "most", "much", "must", "myself", "never",
    "only", "other", "ours", "ourselves", "over", "please", "same", "should",
    "some", "such", "than", "that", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "under",
    "until", "very", "want", "were", "what", "when", "where", "which", "while",
    "whom", "will", "with", "would", "your", "yours", "yourself", "yourselves",
    "channel", "author", "discord", "mqtt", "message", "system", "thread",
    "internal", "ping", "operator", "general",
}

_GENERIC_KEY_SEGMENTS: Set[str] = {
    "assistant", "people", "person", "topics", "topic", "notes", "note",
    "projects", "project", "general", "dossiers", "dossier",
}

_FILING_JSON_FORMAT_APPENDIX = (
    "\n\nOUTPUT FORMAT (STRICT JSON ONLY):\n"
    "You MUST respond with a single valid JSON object matching this exact schema:\n"
    '{"updates": [{"key": "assistant/people/...", "summary": "One-line summary", "content": "Full updated markdown note content"}]}\n'
    "If nothing in the evicted messages warrants saving or updating, respond with:\n"
    '{"updates": []}\n'
    "Do not include any prose outside the JSON object."
)


class ExecutiveAssistantComponent(BaseComponent):
    """
    Automated Chief of Staff / Filing Clerk component.
    Operates strictly in the background (exposing zero tools to the main agent):
    1. Pre-Turn Briefing (before_llm_call): Scans StructuredNotesComponent ('assistant/*' and general notes)
       for people/topics referenced in the incoming event that are missing from active Layer-0 context,
       and injects a transient '[Assistant Briefing]' block at the bottom of the prompt.
    2. Background Filing Clerk (messages_evicted): Processes messages evicted during TieredMemoryComponent
       compaction and updates structured reference dossiers under 'assistant/*'.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_candidate_notes: Optional[int] = None,
        max_briefing_tokens: Optional[int] = None,
        thread_id: str = "main",
        notes_comp: Optional[StructuredNotesComponent] = None,
        memory_comp: Optional[Any] = None,
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
            else ea_cfg.get("max_briefing_tokens", 300)
        )
        self.max_filing_tokens = int(ea_cfg.get("max_filing_tokens", 2048))

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
        self.last_usage: Optional[Dict[str, Any]] = None
        self._last_briefed_sig: Optional[Tuple[int, str]] = None

        if self.notes_comp is not None:
            self.notes_comp.ensure_seed_prompts()

    def get_system_prompt_addition(self) -> str:
        if not self.enabled:
            return ""
        return (
            "You have an automated Executive Assistant that maintains reference dossiers under 'assistant/*' "
            "(read-only to you) and injects a transient '[Assistant Briefing]' before your turn when incoming "
            "messages reference people or topics missing from your active context. You control how your assistant "
            "operates by editing 'prompts/assistant_briefing' and 'prompts/assistant_filing' via put_note."
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
    def _extract_tokens_and_entities(text: str) -> Tuple[List[str], Set[str]]:
        clean = ExecutiveAssistantComponent._strip_subconscious_prefix(text or "")
        raw_words = re.findall(r"[A-Za-z0-9_]+", clean)
        tokens: List[str] = []
        seen: Set[str] = set()
        entities: Set[str] = set()

        for w in raw_words:
            low = w.lower()
            if len(w) >= 2 and w[0].isupper() and low not in _STOPWORDS:
                entities.add(low)
            if len(low) >= 3 and low not in _STOPWORDS and low not in seen:
                seen.add(low)
                tokens.append(low)
            elif len(low) >= 2 and low in entities and low not in seen:
                seen.add(low)
                tokens.append(low)

        return tokens, entities

    def _gather_candidate_notes(
        self,
        incoming_text: str,
        active_context_text: str,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Step A: Fast deterministic candidate note gather.
        Returns (top_candidate_notes, all_assistant_notes).
        """
        if self.notes_comp is None:
            return [], []

        all_notes = self.notes_comp.get_all_notes()
        assistant_notes = [
            n for n in all_notes if n["key"].startswith("assistant/")
        ]
        searchable_notes = [
            n
            for n in all_notes
            if n["key"] != "sticky" and not n["key"].startswith("prompts/")
        ]
        if not searchable_notes:
            return [], assistant_notes

        tokens, entities = self._extract_tokens_and_entities(incoming_text)
        if not tokens:
            return [], assistant_notes

        incoming_low = self._strip_subconscious_prefix(incoming_text).lower()
        active_low = (active_context_text or "").lower()

        scored: List[Tuple[float, Dict[str, str]]] = []
        for note in searchable_notes:
            key = note["key"]
            summary = note.get("summary") or ""
            content = note.get("content") or ""
            k_low = key.lower()
            s_low = summary.lower()
            c_low = content.lower()

            segments = [
                seg.lower()
                for seg in re.split(r"[/_\-\s]+", k_low)
                if seg and seg.lower() not in _GENERIC_KEY_SEGMENTS
            ]
            leaf = segments[-1] if segments else k_low

            score = 0.0
            if leaf and len(leaf) >= 2 and re.search(rf"\b{re.escape(leaf)}\b", incoming_low):
                score += 8.0

            for seg in segments:
                if len(seg) >= 3 and seg in tokens:
                    score += 5.0

            for tok in tokens:
                is_entity = tok in entities
                if tok in k_low:
                    score += 5.0 if is_entity else 3.0
                if tok in s_low:
                    score += 3.0 if is_entity else 2.0
                if tok in c_low:
                    score += 2.0 if is_entity else 1.0

            if score > 0:
                # Prioritize notes whose leaf/topic is not already discussed verbatim in active context
                leaf_in_active = bool(
                    leaf
                    and len(leaf) >= 3
                    and re.search(rf"\b{re.escape(leaf)}\b", active_low)
                )
                if not leaf_in_active:
                    score = score * 1.5 + 3.0
                scored.append((score, note))

        scored.sort(key=lambda item: (item[0], item[1]["key"]), reverse=True)
        top_candidates = [n for _, n in scored[: self.max_candidate_notes]]
        return top_candidates, assistant_notes

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
                    stream=True,
                    stream_options={"include_usage": True},
                )
                resp = await create_call if inspect.isawaitable(create_call) else create_call

                content = ""
                usage = None
                if hasattr(resp, "__aiter__"):
                    async for chunk in resp:
                        parsed_usage = extract_chunk_usage(chunk)
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
            return (content or "").strip()

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

    @staticmethod
    def _parse_filing_json(raw_text: str) -> List[Dict[str, str]]:
        clean = (raw_text or "").strip()
        if not clean or clean.upper() == "NONE":
            return []
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)

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

        if isinstance(parsed, dict):
            updates = parsed.get("updates", [])
        elif isinstance(parsed, list):
            updates = parsed
        else:
            return []

        if not isinstance(updates, list):
            return []

        valid_updates: List[Dict[str, str]] = []
        for item in updates:
            if not isinstance(item, dict):
                continue
            raw_key = str(item.get("key") or "").strip().lstrip("/")
            content = str(item.get("content") if "content" in item else item.get("value") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if not raw_key or not content:
                continue
            if not raw_key.startswith("assistant/"):
                raw_key = f"assistant/{raw_key}"
            valid_updates.append(
                {
                    "key": raw_key,
                    "summary": summary,
                    "content": content,
                }
            )
        return valid_updates

    async def _handle_before_llm_call(self, payload: Dict[str, Any]) -> None:
        agent = payload.get("agent") or getattr(self, "agent", None)
        self._discover_components(agent)
        if not self.enabled or self.notes_comp is None:
            return

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
        ]
        if not non_transient:
            return

        last_msg = non_transient[-1]
        if last_msg.get("role") != "user":
            return

        raw_incoming = str(last_msg.get("content") or "").strip()
        incoming_event = self._strip_subconscious_prefix(raw_incoming).strip()
        if not incoming_event or incoming_event.startswith("[System Clock]"):
            return

        sig = (len(non_transient), incoming_event)
        if self._last_briefed_sig == sig:
            return
        self._last_briefed_sig = sig

        prior_messages = [
            m for m in non_transient[1:-1] if m.get("role") != "system"
        ]
        core_summary = self._get_core_summary()
        recent_messages_str = self._format_recent_messages(prior_messages, max_messages=5)
        active_context_str = f"Core Summary:\n{core_summary}\n\nRecent Messages:\n{recent_messages_str}"

        candidates, assistant_notes = self._gather_candidate_notes(
            incoming_text=incoming_event,
            active_context_text=active_context_str,
        )
        if not candidates:
            return

        briefing_instructions = self._get_prompt(
            "prompts/assistant_briefing", DEFAULT_ASSISTANT_BRIEFING_PROMPT
        )

        cand_blocks: List[str] = []
        for c in candidates:
            cand_blocks.append(
                f"--- Note: {c['key']} (Summary: {c['summary']}) ---\n{c['content']}"
            )
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
            return

        cleaned = (response_text or "").strip()
        if not cleaned or cleaned.upper() == "NONE" or cleaned.upper().startswith("NONE\n"):
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
            if content.startswith("[System Clock]"):
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
        tokens, entities = self._extract_tokens_and_entities(transcript_text)

        matched_existing: List[Dict[str, str]] = []
        for note in assistant_notes:
            k_low = note["key"].lower()
            s_low = (note.get("summary") or "").lower()
            c_low = (note.get("content") or "").lower()
            if any(t in k_low or t in s_low or t in c_low for t in (entities or tokens)):
                matched_existing.append(note)

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
            else "(No existing 'assistant/*' notes matched tokens in the evicted messages)"
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
                self.notes_comp.assistant_put_note(
                    key=item["key"],
                    summary=item["summary"],
                    content=item["content"],
                )
                updated_keys.append(item["key"])
            except Exception as e:
                print(f"[ExecutiveAssistant:{self.thread_id}] Failed saving note '{item.get('key')}': {e}")

        if updated_keys and agent is not None and hasattr(agent, "event_bus"):
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
        elif event_name == "before_llm_call":
            await self._handle_before_llm_call(payload)
        elif event_name == "after_llm_call":
            agent = payload.get("agent") or getattr(self, "agent", None)
            if agent is not None and isinstance(getattr(agent, "messages", None), list):
                self._strip_transient_briefing(agent.messages)
            messages = payload.get("messages")
            if isinstance(messages, list):
                self._strip_transient_briefing(messages)
        elif event_name == "messages_evicted":
            await self._handle_messages_evicted(payload)
