import json
import sqlite3
from types import SimpleNamespace
import pytest

from src.core.agent import AgentCore
from src.components.notes import (
    ASSISTANT_GUARD_ERROR,
    DEFAULT_ASSISTANT_BRIEFING_PROMPT,
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
    Verifies 'prompts/assistant_briefing' and 'prompts/assistant_filing' are seeded on startup,
    can be updated by the Boss via put_note, and reset to factory default if delete_note or rename_note is called.
    """
    notes_db = str(tmp_path / "notes.db")
    notes = StructuredNotesComponent(db_path=notes_db)
    assistant = ExecutiveAssistantComponent(notes_comp=notes)

    assert assistant.get_tools() == []
    assert "assistant/*" in assistant.get_system_prompt_addition()
    assert "prompts/assistant_briefing" in assistant.get_system_prompt_addition()

    # Seeded on startup
    assert notes.get_note("prompts/assistant_briefing") == DEFAULT_ASSISTANT_BRIEFING_PROMPT
    assert notes.get_note("prompts/assistant_filing") == DEFAULT_ASSISTANT_FILING_PROMPT

    # Boss can customize prompts/* via put_note
    custom_briefing = "Custom briefing prompt: Always mention coffee preferences first."
    res_put = notes.put_note("prompts/assistant_briefing", custom_briefing)
    assert "Success" in res_put
    assert notes.get_note("prompts/assistant_briefing") == custom_briefing

    # Deleting a seeded prompt resets it to factory default instead of leaving it missing
    res_del = notes.delete_note("prompts/assistant_briefing")
    assert "reset to factory defaults" in res_del
    assert notes.get_note("prompts/assistant_briefing") == DEFAULT_ASSISTANT_BRIEFING_PROMPT

    # Renaming away from a seeded prompt also resets it to factory default
    notes.put_note("prompts/assistant_filing", "Custom filing instructions")
    res_ren = notes.rename_note("prompts/assistant_filing", "backup/filing")
    assert "reset to factory defaults" in res_ren
    assert notes.get_note("prompts/assistant_filing") == DEFAULT_ASSISTANT_FILING_PROMPT


def test_structured_notes_sandbox_guardrails(tmp_path):
    """
    2. Sandbox Guardrails in StructuredNotesComponent:
    Verifies the Boss is blocked from calling put_note, delete_note, or rename_note on 'assistant/*' keys,
    can still read 'assistant/*' keys via get_note and list_note_keys, and assistant_put_note writes cleanly.
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
    Verifies that when an incoming message mentions 'George' matching 'assistant/people/george'
    that isn't in active context, the Assistant calls its configured LLM endpoint using
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

        # Verify LLM was called with custom prompts/assistant_briefing and candidate note
        assert len(captured_calls) == 1
        call_msgs = captured_calls[0]["messages"]
        assert call_msgs[0]["role"] == "system"
        assert call_msgs[0]["content"] == custom_briefing_prompt
        assert "assistant/people/george" in call_msgs[1]["content"]
        assert "Eurorack modular synth" in call_msgs[1]["content"]

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

        # Verify NONE response injects nothing
        async def mock_create_none(**kwargs):
            captured_calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="NONE"))]
            )

        assistant.client.chat.completions.create = mock_create_none
        await agent.add_message("user", "George says thanks for checking the ICs!", publish_event=True)
        await agent.event_bus.publish("before_llm_call", {"agent": agent, "messages": agent.messages})

        assert len(captured_calls) == 2
        assert all(not m.get("_transient_briefing") for m in agent.messages)
    finally:
        mem._lock_file.close()


@pytest.mark.asyncio
async def test_compaction_eviction_filing_clerk(tmp_path):
    """
    4. Compaction Filing (messages_evicted):
    Verifies that when TieredMemoryComponent evicts messages during compaction,
    ExecutiveAssistantComponent runs the filing pass using 'prompts/assistant_filing',
    parses the JSON updates, normalizes keys under 'assistant/*', and writes them to SQLite.
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

        custom_filing_prompt = "CUSTOM FILING SYSTEM PROMPT: Extract people and project promises."
        notes.put_note("prompts/assistant_filing", custom_filing_prompt)

        # Mock main agent compaction summarizer stream
        async def fake_compaction_stream(**kwargs):
            async def _gen():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="Updated core summary after chatting with Clara.", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )
            return _gen()

        agent.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_compaction_stream))
        )

        # Mock Executive Assistant filing LLM response (with markdown code fences and unprefixed key to test normalization)
        filing_calls = []

        async def mock_assistant_filing_create(**kwargs):
            filing_calls.append(kwargs)
            payload_json = {
                "updates": [
                    {
                        "key": "people/clara",
                        "summary": "Clara: Astrophysicist collaborating on pulsar timing arrays.",
                        "content": "# Clara\n- Astrophysicist studying millisecond pulsars.\n- Promised to share her NANOGrav dataset on Friday.",
                    },
                    {
                        "key": "assistant/topics/pulsar_timing",
                        "summary": "NANOGrav pulsar timing array analysis.",
                        "content": "Tracking dispersion measure variations across millisecond pulsars with Clara.",
                    },
                ]
            }
            fenced = f"```json\n{json.dumps(payload_json, indent=2)}\n```"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=fenced))]
            )

        assistant.client.chat.completions.create = mock_assistant_filing_create

        agent.messages = [{"role": "system", "content": "System prompt"}]
        await agent.add_message(
            "user",
            "Hi! I am Clara, an astrophysicist working on millisecond pulsar timing arrays. I will send my NANOGrav dataset on Friday.",
            publish_event=True,
        )
        await agent.add_message(
            "assistant",
            "Welcome Clara! Looking forward to inspecting the NANOGrav pulsar timing dataset with you on Friday.",
            publish_event=True,
        )
        await agent.add_message(
            "user",
            "Now let's switch topics and talk about something completely different.",
            publish_event=True,
        )

        # Trigger compaction -> evicts older messages -> fires messages_evicted -> triggers ExecutiveAssistantComponent
        await agent.event_bus.publish("context_window_check", {"agent": agent})

        assert len(filing_calls) >= 1
        filing_sys_prompt = filing_calls[0]["messages"][0]["content"]
        assert custom_filing_prompt in filing_sys_prompt
        assert '"updates"' in filing_sys_prompt
        assert "Clara" in filing_calls[0]["messages"][1]["content"]

        # Verify both notes were normalized under 'assistant/*' and written to SQLite
        clara_note = notes.get_note("assistant/people/clara")
        assert "NANOGrav dataset on Friday" in clara_note

        topic_note = notes.get_note("assistant/topics/pulsar_timing")
        assert "dispersion measure variations" in topic_note

        all_notes_map = {n["key"]: n for n in notes.get_all_notes()}
        assert "assistant/people/clara" in all_notes_map
        assert "Astrophysicist" in all_notes_map["assistant/people/clara"]["summary"]
    finally:
        mem._lock_file.close()
