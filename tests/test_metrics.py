import json
import os
import re
from datetime import datetime, timezone
import pytest

from src.utils.metrics import (
    MetricSpan,
    count_tokens,
    format_rfc3339_ms,
    get_daily_metrics_filepath,
    infer_exec_location,
    set_shared_tokenizer,
)


def test_metrics_rfc3339_daily_append_and_accumulators(tmp_path):
    """Verifies RFC 3339 ms UTC timestamps, daily file naming/appending, timer/counter accumulators, and error=0/1."""
    fixed_dt = datetime(2026, 10, 26, 20, 24, 15, 123456, tzinfo=timezone.utc)
    assert format_rfc3339_ms(fixed_dt) == "2026-10-26T20:24:15.123Z"

    metrics_dir = str(tmp_path / "metrics")

    # 1. Successful span with accumulated timers and counters
    with MetricSpan(
        operation="llm_call",
        agent="Thunk",
        thread="mind",
        component="spreading_activation",
        type="subconscious_recall",
        model="gemma4:e2b",
        exec_location="local",
        metrics_dir=metrics_dir,
        now_utc=fixed_dt,
    ) as span:
        span.add_timer("db_lookup_ms", 4.25)
        span.add_timer("db_lookup_ms", 5.25)
        span.add_timer("db_save_ms", 3.0)
        span.inc("db_lookup", 1)
        span.inc("db_lookup", 1)
        span.inc("db_lookup_records", 8)
        span.inc("request_tokens", 120)
        span.inc("response_tokens", 45)

    # 2. Second span on the same UTC date appends to the same file and records error=1 on exception
    with pytest.raises(RuntimeError, match="boom"):
        with MetricSpan(
            operation="db_query",
            agent="Thunk",
            thread="mind",
            component="notes",
            metrics_dir=metrics_dir,
            now_utc=fixed_dt,
        ) as span2:
            span2.inc("db_lookup", 1)
            raise RuntimeError("boom")

    daily_path = get_daily_metrics_filepath(metrics_dir=metrics_dir, now_utc=fixed_dt)
    assert os.path.basename(daily_path) == "metrics-2026-10-26.jsonl"
    assert os.path.exists(daily_path)

    with open(daily_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 2

    first = lines[0]
    assert first["timestamp"] == "2026-10-26T20:24:15.123Z"
    assert first["operation"] == "llm_call"
    assert isinstance(first["duration_ms"], float)
    assert first["error"] == 0
    assert first["agent"] == "Thunk"
    assert first["thread"] == "mind"
    assert first["component"] == "spreading_activation"
    assert first["type"] == "subconscious_recall"
    assert first["model"] == "gemma4:e2b"
    assert first["exec_location"] == "local"
    assert first["timers"] == {"db_lookup_ms": 9.5, "db_save_ms": 3.0}
    assert first["counters"] == {
        "db_lookup": 2,
        "db_lookup_records": 8,
        "request_tokens": 120,
        "response_tokens": 45,
    }

    second = lines[1]
    assert second["operation"] == "db_query"
    assert second["error"] == 1
    assert second["counters"] == {"db_lookup": 1}


def test_tokenizer_fallback_and_exec_location():
    """Verifies count_tokens uses registered shared tokenizer and infer_exec_location distinguishes local vs cloud."""
    class DummyWordTokenizer:
        def encode(self, text: str):
            return text.split()

    set_shared_tokenizer(DummyWordTokenizer())
    assert count_tokens("one two three four five") == 5
    assert count_tokens([{"role": "user", "content": "hello world"}]) == 4

    assert infer_exec_location("http://localhost:11435/v1") == "local"
    assert infer_exec_location("http://potato.1209k.com:11434/v1") == "local"
    assert infer_exec_location("https://generativelanguage.googleapis.com/v1beta") == "cloud"
    assert infer_exec_location("https://api.openai.com/v1") == "cloud"


@pytest.mark.asyncio
async def test_component_llm_call_and_tool_call_metrics_emission(tmp_path, monkeypatch):
    """Verifies CriticComponent, OracleComponent, and AgentCore.execute_tool emit properly structured JSONL metrics."""
    from src.core.agent import AgentCore
    from src.components.critic import CriticComponent
    from src.components.oracle import OracleComponent
    from src.utils import metrics as metrics_mod

    metrics_dir = str(tmp_path / "live_metrics")
    monkeypatch.setattr(metrics_mod, "resolve_metrics_dir", lambda explicit=None: explicit or metrics_dir)

    agent = AgentCore(thread_id="social")

    # 1. CriticComponent emits operation='llm_call', type='critic', exec_location='local'
    class DummyMsg:
        content = "I notice steady progress."

    class DummyCompletions:
        async def create(self, **kwargs):
            return "stream"

    async def mock_acc(s):
        return DummyMsg()

    critic = CriticComponent(frequency=1)
    agent.register_component(critic)
    agent.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    monkeypatch.setattr(agent.client.chat, "completions", DummyCompletions())
    agent._accumulate_stream = mock_acc
    await critic._evaluate_progress()

    # 2. OracleComponent emits operation='llm_call', type='oracle', exec_location='cloud'
    oracle = OracleComponent()
    oracle.model_name = "gemini-3.8-flash"

    class DummyOracleResp:
        text = "Cloud oracle response text."
        usage_metadata = None

    class DummyAioModels:
        async def generate_content(self, **kwargs):
            return DummyOracleResp()

    class DummyOracleClient:
        class aio:
            models = DummyAioModels()

    oracle.client = DummyOracleClient()
    agent.register_component(oracle)
    agent._build_tools()

    # Execute ask_oracle via AgentCore.execute_tool so both tool_call and oracle llm_call are emitted
    tool_res = await agent.execute_tool("ask_oracle", json.dumps({"query": "What is the weather?"}))
    assert "Cloud oracle response text." in tool_res

    files = os.listdir(metrics_dir)
    assert len(files) == 1
    assert re.match(r"^metrics-\d{4}-\d{2}-\d{2}\.jsonl$", files[0])

    with open(os.path.join(metrics_dir, files[0]), "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    critic_rec = next(r for r in records if r["operation"] == "llm_call" and r.get("type") == "critic")
    assert critic_rec["exec_location"] == "local"
    assert critic_rec["thread"] == "social"
    assert critic_rec["error"] == 0
    assert "llm_call_ms" in critic_rec["timers"]
    assert critic_rec["counters"]["request_tokens"] > 0
    assert critic_rec["counters"]["response_tokens"] > 0

    oracle_rec = next(r for r in records if r["operation"] == "llm_call" and r.get("type") == "oracle")
    assert oracle_rec["exec_location"] == "cloud"
    assert oracle_rec["model"] == "gemini-3.8-flash"
    assert oracle_rec["error"] == 0
    assert "llm_call_ms" in oracle_rec["timers"]
    assert oracle_rec["counters"]["request_tokens"] > 0
    assert oracle_rec["counters"]["response_tokens"] > 0

    tool_rec = next(r for r in records if r["operation"] == "tool_call" and r.get("type") == "ask_oracle")
    assert tool_rec["error"] == 0
    assert tool_rec["counters"]["request_bytes"] > 0
    assert tool_rec["counters"]["response_bytes"] > 0


@pytest.mark.asyncio
async def test_gemini_provider_adapter_streaming_and_thought_signatures():
    """Verifies GeminiOpenAIClientAdapter translates messages/tools/images, streams chunks, and preserves thought_signature."""
    from google.genai import types
    from src.core.agent import AgentCore
    from src.core.gemini_client import GeminiOpenAIClientAdapter

    agent = AgentCore(thread_id="mind", provider="gemini", model_name="gemini-4-argon")
    assert agent.exec_location == "cloud"
    assert isinstance(agent.client, GeminiOpenAIClientAdapter)

    captured_calls = []

    class DummyUsageMeta:
        prompt_token_count = 321
        candidates_token_count = 45
        cached_content_token_count = 280
        thoughts_token_count = 112

    sig_part = types.Part(
        function_call=types.FunctionCall(name="read_file", args={"library": "ebook", "filepath": "MrDarcy.epub"}),
        thought_signature=b"secret_thought_sig_bytes",
    )
    text_part = types.Part.from_text(text="Let me open Pride and Prejudice.")

    class DummyGemChunk:
        def __init__(self, parts):
            self.candidates = [
                types.Candidate(
                    content=types.Content(role="model", parts=parts),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
            self.usage_metadata = DummyUsageMeta()

    class DummyAioModels:
        async def generate_content_stream(self, model, contents, config):
            captured_calls.append((model, contents, config))

            async def _gen():
                yield DummyGemChunk([text_part, sig_part])

            return _gen()

    class DummyGenaiClient:
        class aio:
            models = DummyAioModels()

    agent.client._genai_client = DummyGenaiClient()

    # Include a legacy non-Gemini tool call in history to verify safe plain-text fallback
    messages = [
        {"role": "system", "content": "You are TestAgent."},
        {"role": "user", "content": "Check the library."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "legacy_ollama_tc_1",
                    "function": {"name": "list_libraries", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "legacy_ollama_tc_1", "name": "list_libraries", "content": "- ebook"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Reads a file",
                "parameters": {"type": "object", "properties": {"library": {"type": "string"}}},
            },
        }
    ]

    stream = await agent.client.chat.completions.create(
        model="gemini-4-argon",
        messages=messages,
        tools=tools,
        max_tokens=4096,
        stream=True,
    )
    msg = await agent._accumulate_stream(stream)

    assert msg.content == "Let me open Pride and Prejudice."
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.function.name == "read_file"
    assert json.loads(tc.function.arguments) == {"library": "ebook", "filepath": "MrDarcy.epub"}
    assert agent.last_finish_reason == "tool_calls"
    assert agent.last_usage == {
        "prompt_tokens": 321,
        "completion_tokens": 45,
        "cached_tokens": 280,
        "thinking_tokens": 112,
    }

    # Verify next turn preserves the exact Gemini Part with thought_signature
    messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
    messages.append({"role": "tool", "tool_call_id": tc.id, "name": "read_file", "content": "Chapter 1 text"})

    _, converted_contents = agent.client._convert_messages(messages)
    model_turns = [c for c in converted_contents if c.role == "model"]
    assert any(
        p.thought_signature == b"secret_thought_sig_bytes"
        for turn in model_turns
        for p in (turn.parts or [])
    )


@pytest.mark.asyncio
async def test_agent_core_llm_retries_and_fail_fast_exit(tmp_path, monkeypatch):
    """Verifies AgentCore._call_llm_with_retries retries with backoff and exits with code 1 when quota/API fails."""
    from src.core.agent import AgentCore
    from src.utils import metrics as metrics_mod

    metrics_dir = str(tmp_path / "fail_fast_metrics")
    monkeypatch.setattr(metrics_mod, "resolve_metrics_dir", lambda explicit=None: explicit or metrics_dir)

    agent = AgentCore(thread_id="social", provider="gemini", model_name="gemini-4-argon")
    agent.max_retries = 3
    agent.retry_base_delay = 0.0

    attempts = {"count": 0}

    class FailingCompletions:
        async def create(self, **kwargs):
            attempts["count"] += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded for gemini-4-argon")

    monkeypatch.setattr(agent.client.chat, "completions", FailingCompletions())

    with pytest.raises(SystemExit) as exc_info:
        await agent._call_llm_with_retries([{"role": "user", "content": "Hello"}])

    assert exc_info.value.code == 1
    assert attempts["count"] == 3

    files = os.listdir(metrics_dir)
    assert len(files) == 1
    with open(os.path.join(metrics_dir, files[0]), "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())

    assert rec["operation"] == "llm_call"
    assert rec["type"] == "agent_turn"
    assert rec["exec_location"] == "cloud"
    assert rec["error"] == 1
    assert rec["finish_reason"] == "error"
    assert rec["counters"]["retries"] == 2


def test_gemini_convert_messages_post_compaction_function_call_and_multi_tool_grouping():
    """
    Verifies that when memory compaction leaves an assistant function_call turn at index 1
    (immediately after system prompt with no user turn before it), _convert_messages
    prepends a user turn and groups consecutive function_response parts into one user turn.
    """
    from google.genai import types
    from src.core.gemini_client import GeminiOpenAIClientAdapter

    adapter = GeminiOpenAIClientAdapter(api_key="test-key")
    fc1 = types.Part(
        function_call=types.FunctionCall(name="ping_thread", args={"target_thread": "mind", "message": "hi"}),
        thought_signature=b"sig1",
    )
    fc2 = types.Part(
        function_call=types.FunctionCall(name="suspend_until_next_event", args={}),
    )
    adapter._gemini_part_cache["gemini_tc_1"] = fc1
    adapter._gemini_part_cache["gemini_tc_2"] = fc2

    # Simulate Wisp's exact post-compaction state where [1] is role=assistant with 2 cached Gemini tool calls
    messages = [
        {"role": "system", "content": "You are Wisp."},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "gemini_tc_1", "function": {"name": "ping_thread", "arguments": '{"target_thread": "mind", "message": "hi"}'}},
                {"id": "gemini_tc_2", "function": {"name": "suspend_until_next_event", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "gemini_tc_1", "name": "ping_thread", "content": "Internal ping delivered."},
        {"role": "tool", "tool_call_id": "gemini_tc_2", "name": "suspend_until_next_event", "content": "Woken up."},
        {"role": "user", "content": "Hello Wisp!"},
        {"role": "system", "content": "[Parallel Streams of Mind (Live Status)]\n- [mind (1m ago)]: Working."},
    ]

    sys_inst, contents = adapter._convert_messages(messages)
    assert sys_inst == "You are Wisp."
    # contents[0] MUST be role='user' (synthetic anchor before leading model function_call turn)
    assert contents[0].role == "user"
    assert contents[1].role == "model"
    assert len(contents[1].parts) == 2
    assert contents[1].parts[0].function_call.name == "ping_thread"
    assert contents[1].parts[1].function_call.name == "suspend_until_next_event"

    # Both tool results MUST be grouped into a single Content(role='user') turn at contents[2]
    assert contents[2].role == "user"
    assert len(contents[2].parts) == 2
    assert contents[2].parts[0].function_response.name == "ping_thread"
    assert contents[2].parts[1].function_response.name == "suspend_until_next_event"



