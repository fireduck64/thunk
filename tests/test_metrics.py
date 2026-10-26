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
