import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.utils.config import load_config

_SHARED_TOKENIZER: Any = None


def set_shared_tokenizer(tokenizer: Any) -> None:
    """Registers a process-wide shared tokenizer for fallback token counting in metrics."""
    global _SHARED_TOKENIZER
    _SHARED_TOKENIZER = tokenizer


def get_shared_tokenizer(agent: Any = None) -> Any:
    """Returns the registered shared tokenizer, or looks one up from TieredMemoryComponent if available."""
    global _SHARED_TOKENIZER
    if _SHARED_TOKENIZER is not None:
        return _SHARED_TOKENIZER
    if agent is not None:
        for comp in getattr(agent, "components", []) or []:
            tok = getattr(comp, "tokenizer", None)
            if tok is not None:
                _SHARED_TOKENIZER = tok
                return tok
    return None


def count_tokens(text_or_messages: Any, agent: Any = None, tokenizer: Any = None) -> int:
    """
    Counts tokens using the shared AutoTokenizer if available, falling back to a UTF-8 byte/4 estimate.
    Accepts either a string or a list of OpenAI-format message dicts.
    """
    if not text_or_messages:
        return 0

    if isinstance(text_or_messages, list):
        parts = []
        for msg in text_or_messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content") or ""
                if isinstance(content, list):
                    text_parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content_str = "\n".join(text_parts)
                else:
                    content_str = str(content)
                parts.append(f"Role: {role}\n{content_str}")
                if msg.get("tool_calls"):
                    parts.append(str(msg["tool_calls"]))
            else:
                parts.append(str(msg))
        full_text = "\n".join(parts)
    else:
        full_text = str(text_or_messages)

    if not full_text:
        return 0

    tok = tokenizer or get_shared_tokenizer(agent)
    if tok is not None and hasattr(tok, "encode"):
        try:
            return len(tok.encode(full_text))
        except Exception:
            pass
    return max(1, len(full_text.encode("utf-8")) // 4)


def infer_exec_location(api_url: Optional[str] = None, default: str = "local") -> str:
    """Infers whether an LLM endpoint is 'local' or 'cloud' based on its URL."""
    if not api_url:
        return default
    lower = api_url.lower()
    cloud_domains = (
        "googleapis.com",
        "openai.com",
        "anthropic.com",
        "openrouter.ai",
        "together.xyz",
        "groq.com",
        "fireworks.ai",
    )
    if any(domain in lower for domain in cloud_domains):
        return "cloud"
    return default


def extract_chunk_usage(chunk: Any) -> Optional[Dict[str, int]]:
    """
    Extracts token usage (prompt_tokens, completion_tokens, cached_tokens, thinking_tokens)
    from an OpenAI/Ollama or Gemini streaming chunk if present.
    """
    chunk_usage = getattr(chunk, "usage", None)
    if chunk_usage is None:
        return None
    p_tok = getattr(chunk_usage, "prompt_tokens", None)
    c_tok = getattr(chunk_usage, "completion_tokens", None)
    cached_tok = getattr(chunk_usage, "cached_tokens", None)
    thought_tok = getattr(chunk_usage, "thinking_tokens", None)
    if isinstance(chunk_usage, dict):
        p_tok = chunk_usage.get("prompt_tokens")
        c_tok = chunk_usage.get("completion_tokens")
        cached_tok = chunk_usage.get("cached_tokens")
        thought_tok = chunk_usage.get("thinking_tokens")
        if cached_tok is None and isinstance(chunk_usage.get("prompt_tokens_details"), dict):
            cached_tok = chunk_usage["prompt_tokens_details"].get("cached_tokens")
    else:
        details = getattr(chunk_usage, "prompt_tokens_details", None)
        if details is not None and cached_tok is None:
            cached_tok = getattr(details, "cached_tokens", None)

    if p_tok is None and c_tok is None and cached_tok is None and thought_tok is None:
        return None

    out: Dict[str, int] = {
        "prompt_tokens": int(p_tok or 0),
        "completion_tokens": int(c_tok or 0),
    }
    if cached_tok is not None:
        out["cached_tokens"] = int(cached_tok)
    if thought_tok is not None:
        out["thinking_tokens"] = int(thought_tok)
    return out


def format_rfc3339_ms(dt: Optional[datetime] = None) -> str:
    """Formats a UTC datetime as RFC 3339 with millisecond precision (e.g. 2026-10-26T20:24:15.123Z)."""
    now = dt.astimezone(timezone.utc) if dt is not None else datetime.now(timezone.utc)
    ms = now.microsecond // 1000
    return f"{now.strftime('%Y-%m-%dT%H:%M:%S')}.{ms:03d}Z"


def resolve_metrics_dir(explicit_dir: Optional[str] = None) -> str:
    """Resolves the configured metrics directory path."""
    if explicit_dir:
        return explicit_dir
    try:
        cfg = load_config()
        logging_cfg = cfg.get("logging", {})
        if logging_cfg.get("metrics_dir"):
            return logging_cfg["metrics_dir"]
        log_dir = logging_cfg.get("log_dir")
        if log_dir:
            parent = os.path.dirname(os.path.abspath(log_dir).rstrip(os.sep))
            return os.path.join(parent, "metrics")
    except Exception:
        pass
    return os.path.join("data", "test", "metrics")


def get_daily_metrics_filepath(
    metrics_dir: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> str:
    """Returns the daily append-only metrics JSONL path for the given UTC date."""
    target_dir = resolve_metrics_dir(metrics_dir)
    os.makedirs(target_dir, exist_ok=True)
    now = now_utc.astimezone(timezone.utc) if now_utc is not None else datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    return os.path.join(target_dir, f"metrics-{date_str}.jsonl")


def write_metric_entry(
    entry: Dict[str, Any],
    metrics_dir: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> str:
    """Appends a single JSONL metric line to the daily UTC metrics file."""
    filepath = get_daily_metrics_filepath(metrics_dir=metrics_dir, now_utc=now_utc)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return filepath


class MetricSpan:
    """
    Context manager and accumulator for a single low-cardinality operation metric entry.
    """

    def __init__(
        self,
        operation: str,
        agent: Optional[str] = None,
        thread: Optional[str] = None,
        component: Optional[str] = None,
        type: Optional[str] = None,
        model: Optional[str] = None,
        exec_location: Optional[str] = None,
        finish_reason: Optional[str] = None,
        metrics_dir: Optional[str] = None,
        now_utc: Optional[datetime] = None,
        **extra_fields: Any,
    ):
        self.operation = operation
        self.agent = agent
        self.thread = thread
        self.component = component
        self.type = type
        self.model = model
        self.exec_location = exec_location
        self.finish_reason = finish_reason
        self.metrics_dir = metrics_dir
        self.now_utc = now_utc
        self.extra_fields: Dict[str, Any] = {k: v for k, v in extra_fields.items() if v is not None}

        self.error: int = 0
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self._start_perf: float = time.perf_counter()
        self._closed: bool = False

    def add_timer(self, name: str, duration_ms: float) -> None:
        """Accumulates milliseconds into a named sub-operation timer."""
        self.timers[name] = round(self.timers.get(name, 0.0) + float(duration_ms), 2)

    @contextmanager
    def subtimer(self, name: str):
        """Context manager that measures elapsed milliseconds and accumulates into self.timers[name]."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.add_timer(name, elapsed_ms)

    def inc(self, name: str, amount: int = 1) -> None:
        """Accumulates an integer count into a named counter."""
        self.counters[name] = int(self.counters.get(name, 0) + int(amount))

    def set_counter(self, name: str, value: int) -> None:
        """Sets a named counter to an exact integer value."""
        self.counters[name] = int(value)

    def set_field(self, key: str, value: Any) -> None:
        """Sets an optional field on the metric record."""
        if value is None:
            return
        if key in ("agent", "thread", "component", "type", "model", "exec_location", "finish_reason"):
            setattr(self, key, value)
        else:
            self.extra_fields[key] = value

    def fail(self) -> None:
        """Marks the operation as failed (error = 1)."""
        self.error = 1

    def finish(self) -> Dict[str, Any]:
        """Finalizes the metric span and appends it to the daily JSONL log."""
        if self._closed:
            return {}
        self._closed = True

        duration_ms = round((time.perf_counter() - self._start_perf) * 1000.0, 2)
        end_dt = self.now_utc.astimezone(timezone.utc) if self.now_utc is not None else datetime.now(timezone.utc)

        entry: Dict[str, Any] = {
            "timestamp": format_rfc3339_ms(end_dt),
            "operation": self.operation,
            "duration_ms": duration_ms,
            "error": int(bool(self.error)),
        }

        if self.agent:
            entry["agent"] = self.agent
        if self.thread:
            entry["thread"] = self.thread
        if self.component:
            entry["component"] = self.component
        if self.type:
            entry["type"] = self.type
        if self.model:
            entry["model"] = self.model
        if self.exec_location:
            entry["exec_location"] = self.exec_location
        if self.finish_reason:
            entry["finish_reason"] = self.finish_reason

        for k, v in self.extra_fields.items():
            if v is not None and k not in entry:
                entry[k] = v

        if self.timers:
            entry["timers"] = {k: round(v, 2) for k, v in self.timers.items()}
        if self.counters:
            entry["counters"] = {k: int(v) for k, v in self.counters.items()}

        try:
            write_metric_entry(entry, metrics_dir=self.metrics_dir, now_utc=end_dt)
        except Exception as e:
            print(f"[Metrics Warning] Failed to write metric entry for '{self.operation}': {e}")

        return entry

    def __enter__(self) -> "MetricSpan":
        self._start_perf = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.error = 1
        self.finish()
        return False

    async def __aenter__(self) -> "MetricSpan":
        self._start_perf = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.error = 1
        self.finish()
        return False
