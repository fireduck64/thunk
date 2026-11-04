import json
import os
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
import pytest

from src.core.agent import AgentCore
from src.components.notes import (
    ASSISTANT_GUARD_ERROR,
    DEFAULT_ASSISTANT_BRIEFING_PROMPT,
    DEFAULT_ASSISTANT_COMPACTION_PROMPT,
    DEFAULT_ASSISTANT_FILING_PROMPT,
    StructuredNotesComponent,
)
from src.components.tiered import TieredMemoryComponent
from src.components.executive_assistant import ExecutiveAssistantComponent


class FastDummyTokenizer:
    name_or_path = "dummy-fast"

    def encode(self, text: str):
        return (text or "").split()

    def decode(self, tokens):
        return " ".join(tokens)


def test_prompt_seeding_customization_and_factory_reset(tmp_path):
    """
    1. Prompt Seeding & Reset:
    Verifies 'prompts/assistant_briefing', 'prompts/assistant_filing', and 'prompts/assistant_compaction'
    are seeded on startup, can be updated by the Boss via put_note, and reset to factory default
    if delete_note or rename_note is called.
    """
    notes_db = str(tmp_path / "notes.db")
    notes = StructuredNotesComponent(db_path=notes_db)
    assistant = ExecutiveAssistantComponent(notes_comp=notes)

    assert assistant.get_tools() == []
    sys_prompt = assistant.get_system_prompt_addition()
    assert "assistant/*" in sys_prompt
    assert "prompts/assistant_briefing" in sys_prompt
    assert "prompts/assistant_filing" in sys_prompt
    assert "prompts/assistant_compaction" in sys_prompt

    # Seeded on startup
    assert notes.get_note("prompts/assistant_briefing") == DEFAULT_ASSISTANT_BRIEFING_PROMPT
    assert notes.get_note("prompts/assistant_filing") == DEFAULT_ASSISTANT_FILING_PROMPT
    assert notes.get_note("prompts/assistant_compaction") == DEFAULT_ASSISTANT_COMPACTION_PROMPT

    # Boss can customize prompts/* via put_note
    custom_briefing = "Custom briefing prompt: Always mention coffee preferences first."
    res_put = notes.put_note("prompts/assistant_briefing", custom_briefing)
    assert "Success" in res_put
    assert notes.get_note("prompts/assistant_briefing") == custom_briefing

    # Deleting a seeded prompt resets it to factory default instead of leaving it missing
    res_del = notes.delete_note("prompts/assistant_briefing")
    assert "reset to factory defaults" in res_del
    assert notes.get_note("prompts/assistant_briefing") == DEFAULT_ASSISTANT_BRIEFING_PROMPT

    # Deleting prompts/assistant_compaction also resets to factory default
    notes.put_note("prompts/assistant_compaction", "Temporary compaction prompt")
    res_del_comp = notes.delete_note("prompts/assistant_compaction")
    assert "reset to factory defaults" in res_del_comp
    assert notes.get_note("prompts/assistant_compaction") == DEFAULT_ASSISTANT_COMPACTION_PROMPT

    # Renaming away from a seeded prompt also resets it to factory default
    notes.put_note("prompts/assistant_filing", "Custom filing instructions")
    res_ren = notes.rename_note("prompts/assistant_filing", "backup/filing")
    assert "reset to factory defaults" in res_ren
    assert notes.get_note("prompts/assistant_filing") == DEFAULT_ASSISTANT_FILING_PROMPT


def test_structured_notes_sandbox_guardrails(tmp_path):
    """
    2. Sandbox Guardrails in StructuredNotesComponent:
    Verifies the Boss is blocked from calling put_note, delete_note, or rename_note on 'assistant/*' keys,
    can still read 'assistant/*' keys via get_note and list_note_keys, and assistant_put_note / assistant_append_note
    write and deduplicate keys case-insensitively.
    """
    notes_db = str(tmp_path / "notes.db")
    notes = StructuredNotesComponent(db_path=notes_db)

    # Internal helper writes and auto-prefixes 'assistant/' if omitted
    res1 = notes.assistant_put_note(
        key="people/george",
        summary="George: Operator's friend who owns a vintage synthesizer.",
        content="# George\n- Owns a Prophet-5 synthesizer.\n- Promised to send schematics.",
    )
    assert "assistant/people/george" in res1

    # Boss can read via get_note and see via list_note_keys
    content = notes.get_note("assistant/people/george")
    assert "Prophet-5 synthesizer" in content
    root_dirs = notes.list_note_keys("")
    assert any(d["type"] == "dir" and d["name"] == "assistant" for d in root_dirs)

    # Boss is blocked from put_note, delete_note, and rename_note on assistant/*
    assert "prompts/assistant_compaction" in ASSISTANT_GUARD_ERROR
    assert notes.put_note("assistant/people/george", "Hacked!") == ASSISTANT_GUARD_ERROR
    assert notes.put_note("assistant/new_note", "Hacked!") == ASSISTANT_GUARD_ERROR
    assert notes.delete_note("assistant/people/george") == ASSISTANT_GUARD_ERROR
    assert notes.rename_note("assistant/people/george", "people/george") == ASSISTANT_GUARD_ERROR

    notes.put_note("general/george", "Boss's own note")
    assert notes.rename_note("general/george", "assistant/people/george2") == ASSISTANT_GUARD_ERROR

    # Original assistant dossier remains intact
    assert "Prophet-5 synthesizer" in notes.get_note("assistant/people/george")


@pytest.mark.asyncio
async def test_pre_turn_briefing_injection_and_none_filtering(tmp_path):
    """
    3. Pre-Turn Briefing (before_llm_call):
    Verifies that when an incoming message arrives, _select_relevant_note_keys selects
    'assistant/people/george' via the LLM directory pass, then calls the LLM using
    'prompts/assistant_briefing' and transiently injects '[Assistant Briefing]' without polluting
    persistent history; and injects nothing when the Assistant returns 'NONE'.
    """
    notes_db = str(tmp_path / "notes.db")
    mem_db = str(tmp_path / "memory.db")
    tok = FastDummyTokenizer()

    agent = AgentCore(thread_id="social")
    notes = StructuredNotesComponent(db_path=notes_db)
    mem = TieredMemoryComponent(db_path=mem_db, thread_id="social", tokenizer=tok)
    assistant = ExecutiveAssistantComponent(
        enabled=True,
        thread_id="social",
        notes_comp=notes,
        memory_comp=mem,
    )

    try:
        agent.register_component(notes)
        agent.register_component(mem)
        agent.register_component(assistant)
        await agent.event_bus.publish("agent_started", {"agent": agent})

        notes.assistant_put_note(
            key="assistant/people/george",
            summary="George is building a modular synth and asked about clock dividers.",
            content="George is building a Eurorack modular synth. Last week we promised to check CMOS clock divider ICs for him.",
        )
        custom_briefing_prompt = "CUSTOM BRIEFING SYSTEM PROMPT: Brief me on missing people."
        notes.put_note("prompts/assistant_briefing", custom_briefing_prompt)

        captured_calls = []

        async def mock_create(**kwargs):
            captured_calls.append(kwargs)
            sys_msg = kwargs["messages"][0]["content"]
            if "indexing assistant" in sys_msg:
                # Step A: LLM Note Selector pass (test case-insensitive match too)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='{"keys": ["assistant/people/GEORGE"]}'
                            )
                        )
                    ]
                )
            # Step B: Briefing synthesis pass
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="- **George**: Building a Eurorack modular synth; you promised to check CMOS clock divider ICs."
                        )
                    )
                ]
            )

        assistant.client.chat.completions.create = mock_create

        # Set up active context without George
        agent.messages = [{"role": "system", "content": agent._build_system_prompt()}]
        await agent.add_message("user", "Hey, how is the reading going?", publish_event=True)
        await agent.add_message("assistant", "Going well, studying thermodynamics.", publish_event=True)

        # Incoming message mentions George
        await agent.add_message("user", "George just pinged me asking if we looked into those ICs yet.", publish_event=True)

        await agent.event_bus.publish("before_llm_call", {"agent": agent, "messages": agent.messages})

        # Verify 2 LLM calls: (1) _select_relevant_note_keys, (2) briefing pass with custom prompts/assistant_briefing
        assert len(captured_calls) == 2
        select_msgs = captured_calls[0]["messages"]
        assert "indexing assistant" in select_msgs[0]["content"]
        assert "assistant/people/george" in select_msgs[1]["content"]

        briefing_msgs = captured_calls[1]["messages"]
        assert briefing_msgs[0]["role"] == "system"
        assert briefing_msgs[0]["content"] == custom_briefing_prompt
        assert "assistant/people/george" in briefing_msgs[1]["content"]
        assert "Eurorack modular synth" in briefing_msgs[1]["content"]

        # Verify transient [Assistant Briefing] was injected at the bottom of messages
        assert agent.messages[-1].get("_transient_briefing") is True
        assert agent.messages[-1]["content"].startswith("[Assistant Briefing]\n")
        assert "Eurorack modular synth" in agent.messages[-1]["content"]

        # Wire messages include the briefing cleanly without private _transient_briefing key
        wire_msgs = agent._prepare_llm_messages()
        assert "[Assistant Briefing]" in wire_msgs[-1]["content"]
        assert "_transient_briefing" not in wire_msgs[-1]

        # after_llm_call strips the transient briefing from working memory
        await agent.event_bus.publish(
            "after_llm_call",
            {"agent": agent, "messages": agent.messages, "response": {"content": "Let me check!"}},
        )
        assert all(not m.get("_transient_briefing") for m in agent.messages)

        # Verify SQLite archival_log was NEVER polluted with [Assistant Briefing]
        conn = sqlite3.connect(mem_db)
        try:
            rows = conn.execute("SELECT content FROM archival_log").fetchall()
            assert all("[Assistant Briefing]" not in r[0] for r in rows)
        finally:
            conn.close()

        # Verify NONE response from briefing pass injects nothing
        async def mock_create_none(**kwargs):
            captured_calls.append(kwargs)
            sys_msg = kwargs["messages"][0]["content"]
            if "indexing assistant" in sys_msg:
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"keys": ["assistant/people/george"]}'))]
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="NONE"))]
            )

        assistant.client.chat.completions.create = mock_create_none
        await agent.add_message("user", "George says thanks for checking the ICs!", publish_event=True)
        await agent.event_bus.publish("before_llm_call", {"agent": agent, "messages": agent.messages})

        assert len(captured_calls) == 4
        assert all(not m.get("_transient_briefing") for m in agent.messages)
    finally:
        mem._lock_file.close()


@pytest.mark.asyncio
async def test_compaction_eviction_filing_clerk_append_and_case_dedup(tmp_path):
    """
    4. Compaction Filing (messages_evicted) Append-Only & Case-Insensitive Key Deduping:
    Verifies that multiple messages_evicted events mentioning 'Wisp' (with varying key casing
    like 'people/Wisp' vs 'assistant/people/wisp') APPEND net-new bullet points to the existing
    'assistant/people/Wisp' note rather than overwriting earlier facts.
    """
    notes_db = str(tmp_path / "notes.db")
    mem_db = str(tmp_path / "memory.db")
    tok = FastDummyTokenizer()

    agent = AgentCore(thread_id="social")
    notes = StructuredNotesComponent(db_path=notes_db)
    mem = TieredMemoryComponent(db_path=mem_db, thread_id="social", tokenizer=tok)
    mem.high_watermark = 20
    mem.low_watermark = 8
    mem.max_summarize_tokens = 100

    assistant = ExecutiveAssistantComponent(
        enabled=True,
        thread_id="social",
        notes_comp=notes,
        memory_comp=mem,
    )

    try:
        agent.register_component(notes)
        agent.register_component(mem)
        agent.register_component(assistant)
        await agent.event_bus.publish("agent_started", {"agent": agent})

        custom_filing_prompt = "CUSTOM FILING SYSTEM PROMPT: Extract net-new people and project promises."
        notes.put_note("prompts/assistant_filing", custom_filing_prompt)

        # Mock main agent compaction summarizer stream
        async def fake_compaction_stream(**kwargs):
            async def _gen():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="Updated core summary.", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )
            return _gen()

        agent.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_compaction_stream))
        )

        # Pass 1: First eviction creates 'assistant/people/Wisp' (from unprefixed 'people/Wisp')
        filing_calls = []

        async def mock_filing_pass_1(**kwargs):
            filing_calls.append(kwargs)
            payload_json = {
                "updates": [
                    {
                        "key": "people/Wisp",
                        "summary": "Wisp: Sibling agent studying gravitational lensing.",
                        "new_points": [
                            "- Sibling autonomous agent on Discord and MQTT.",
                            "- Studying RXJ1131 gravitational lensing residuals.",
                        ],
                    }
                ]
            }
            fenced = f"```json\n{json.dumps(payload_json, indent=2)}\n```"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=fenced))]
            )

        assistant.client.chat.completions.create = mock_filing_pass_1

        agent.messages = [{"role": "system", "content": "System prompt"}]
        await agent.add_message(
            "user",
            "Wisp is our sibling autonomous agent studying RXJ1131 gravitational lensing residuals on Discord and MQTT.",
            publish_event=True,
        )
        await agent.add_message(
            "assistant",
            "Got it! I will collaborate with Wisp on the RXJ1131 lensing study.",
            publish_event=True,
        )
        await agent.add_message(
            "user",
            "Let's move on to the next topic.",
            publish_event=True,
        )

        await agent.event_bus.publish("context_window_check", {"agent": agent})

        assert len(filing_calls) == 1
        assert custom_filing_prompt in filing_calls[0]["messages"][0]["content"]
        wisp_note_1 = notes.get_note("assistant/people/Wisp")
        assert "RXJ1131 gravitational lensing residuals" in wisp_note_1

        # Pass 2: Second eviction uses lowercase 'assistant/people/wisp' and adds a new fact
        async def mock_filing_pass_2(**kwargs):
            filing_calls.append(kwargs)
            sys_msg = kwargs["messages"][0]["content"]
            if "indexing assistant" in sys_msg:
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"keys": ["assistant/people/wisp"]}'))]
                )
            payload_json = {
                "updates": [
                    {
                        "key": "assistant/people/wisp",
                        "summary": "Wisp: Sibling agent studying lensing and Abell 2744.",
                        "new_points": [
                            "- Studying RXJ1131 gravitational lensing residuals.",  # duplicate line should be deduped
                            "Ran a free-shape test on Abell 2744 cluster mass maps.",
                        ],
                    }
                ]
            }
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload_json)))]
            )

        assistant.client.chat.completions.create = mock_filing_pass_2
        await agent.event_bus.publish(
            "messages_evicted",
            {
                "agent": agent,
                "thread_id": "social",
                "messages": [
                    {"role": "user", "content": "Wisp just ran a free-shape test on Abell 2744 cluster mass maps."}
                ],
                "new_summary": "Updated summary 2",
            },
        )

        # Verify only ONE canonical note row exists ('assistant/people/Wisp') and BOTH old and new facts are present
        all_assistant_notes = [n for n in notes.get_all_notes() if n["key"].lower().startswith("assistant/")]
        assert len(all_assistant_notes) == 1
        assert all_assistant_notes[0]["key"] == "assistant/people/Wisp"
        assert "Abell 2744" in all_assistant_notes[0]["summary"]

        merged_content = notes.get_note("assistant/people/Wisp")
        assert "- Sibling autonomous agent on Discord and MQTT." in merged_content
        assert "- Studying RXJ1131 gravitational lensing residuals." in merged_content
        assert "- Ran a free-shape test on Abell 2744 cluster mass maps." in merged_content
        # Duplicate bullet was not added twice
        assert merged_content.count("Studying RXJ1131 gravitational lensing residuals.") == 1
    finally:
        mem._lock_file.close()


@pytest.mark.asyncio
async def test_background_note_compaction_pass_and_safety_guard(tmp_path):
    """
    5. Background Note Compaction (run_note_compaction_pass):
    Verifies that when an 'assistant/*' note exceeds max_note_chars, it compacts the note
    using 'prompts/assistant_compaction', updates SQLite, and refuses to overwrite if the LLM
    returns an abnormally short/empty string (< 100 chars).
    """
    notes_db = str(tmp_path / "notes.db")
    notes = StructuredNotesComponent(db_path=notes_db)
    assistant = ExecutiveAssistantComponent(
        enabled=True,
        max_note_chars=300,
        compaction_tool_interval=2,
        compaction_min_interval_sec=0.0,
        notes_comp=notes,
    )

    long_bullets = "\n".join(
        f"- Fact #{i}: Wisp and Thunk analyzed lensing sub-halo residual #{i} in the shared workspace."
        for i in range(1, 15)
    )
    assert len(long_bullets) > 300
    notes.assistant_put_note(
        key="assistant/people/Wisp",
        summary="Wisp: Sibling agent and astrophysics collaborator.",
        content=long_bullets,
    )

    # 1. Safety guard: if LLM returns < 100 chars (or "NONE"), refuse to overwrite the oversized note
    async def mock_short_compaction(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Too short!"))]
        )

    assistant.client.chat.completions.create = mock_short_compaction
    res_short = await assistant.run_note_compaction_pass()
    assert res_short is None
    assert notes.get_note("assistant/people/Wisp") == long_bullets

    # 2. Valid compaction (>= 100 chars) updates the note in SQLite using prompts/assistant_compaction
    compacted_markdown = (
        "# Wisp\n"
        "- Sibling autonomous agent and astrophysics collaborator.\n"
        "- Jointly analyzed lensing sub-halo residuals #1 through #14 with Thunk in the shared workspace."
    )
    assert len(compacted_markdown) >= 100

    compaction_calls = []

    async def mock_valid_compaction(**kwargs):
        compaction_calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"```markdown\n{compacted_markdown}\n```"))]
        )

    assistant.client.chat.completions.create = mock_valid_compaction

    # Trigger via tool_executed cadence + before_llm_call
    await assistant.on_event("tool_executed", {})
    assert assistant.pending_note_compaction is False
    await assistant.on_event("tool_executed", {})
    assert assistant.pending_note_compaction is True

    await assistant.on_event("before_llm_call", {"messages": []})
    assert assistant.pending_note_compaction is False
    assert len(compaction_calls) == 1
    assert DEFAULT_ASSISTANT_COMPACTION_PROMPT in compaction_calls[0]["messages"][0]["content"]
    assert notes.get_note("assistant/people/Wisp") == compacted_markdown


def test_dump_notes_script_and_prefix_filtering(tmp_path):
    """
    6. Verifies scripts/dump-notes.py dumps notes as JSON/JSONL and filters by prefix ('assistant', 'prompt', etc.).
    """
    notes_db = str(tmp_path / "notes.db")
    notes = StructuredNotesComponent(db_path=notes_db)
    notes.ensure_seed_prompts()
    notes.put_note("general/todo", "Buy more tea")
    notes.assistant_put_note("people/sombus", "Sombus: Operator", "Operator and system architect.")

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "dump-notes.py"))

    # 1. Full dump
    full_out = subprocess.check_output([sys.executable, script_path, "-d", notes_db], text=True)
    full_data = json.loads(full_out)
    keys = {item["key"] for item in full_data}
    assert "general/todo" in keys
    assert "assistant/people/sombus" in keys
    assert "prompts/assistant_briefing" in keys
    assert "prompts/assistant_compaction" in keys

    # 2. Filter by 'assistant' (and typo alias 'assistent')
    for pref in ("assistant", "assistent"):
        asst_out = subprocess.check_output([sys.executable, script_path, "-d", notes_db, pref], text=True)
        asst_data = json.loads(asst_out)
        assert len(asst_data) == 1
        assert asst_data[0]["key"] == "assistant/people/sombus"
        assert asst_data[0]["summary"] == "Sombus: Operator"

    # 3. Filter by 'prompt' (matches 'prompts/*')
    prompt_out = subprocess.check_output([sys.executable, script_path, "-d", notes_db, "prompt"], text=True)
    prompt_data = json.loads(prompt_out)
    assert {item["key"] for item in prompt_data} == {
        "prompts/assistant_briefing",
        "prompts/assistant_filing",
        "prompts/assistant_compaction",
    }
