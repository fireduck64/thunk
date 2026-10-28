import asyncio
from typing import List, Callable
from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.metrics import MetricSpan, count_tokens
from google import genai
from google.genai import types

class OracleComponent(BaseComponent):
    """
    Provides the agent with access to a cloud-based AI model (Oracle) 
    to look up current events, facts, or perform advanced reasoning.
    """
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.oracle_config = self.config.get("oracle", {})
        self.api_key = self.oracle_config.get("gemini_api_key")
        self.model_name = self.oracle_config.get("gemini_model", "gemini-2.5-flash")
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            print("[Oracle] Warning: No gemini_api_key found in config.toml. Oracle tool will be disabled.")

    def get_system_prompt_addition(self) -> str:
        return (
            "--- ORACLE (CLOUD AI) ---\n"
            "You have access to a cloud-based AI Oracle (Google Gemini). You can use the `ask_oracle` tool "
            "to look up current events, general knowledge, or ask questions that fall outside your training data or context."
        )

    def get_tools(self) -> List[Callable]:
        if not self.client:
            return []
        return [self.ask_oracle]

    async def ask_oracle(self, query: str) -> str:
        """
        Queries the Cloud AI Oracle (Google Gemini) with the provided text.
        Use this tool to look up current events, facts, or seek advice on complex topics.
        """
        print(f"[Oracle] Consulting the oracle for: '{query}'")
        agent_obj = getattr(self, "agent", None)
        agent_name = getattr(agent_obj, "name", None) or self.config.get("agent", {}).get("name", "Thunk")
        thread_id = getattr(agent_obj, "thread_id", None) or "main"

        with MetricSpan(
            operation="llm_call",
            agent=agent_name,
            thread=thread_id,
            component="oracle",
            type="oracle",
            model=self.model_name,
            exec_location="cloud",
        ) as span:
            span.inc("request_bytes", len((query or "").encode("utf-8")))
            try:
                with span.subtimer("llm_call_ms"):
                    response = await self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=query,
                        config=types.GenerateContentConfig(
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                        ),
                    )
                text = response.text or ""
                usage_meta = getattr(response, "usage_metadata", None)
                req_tok = getattr(usage_meta, "prompt_token_count", None) if usage_meta else None
                resp_tok = getattr(usage_meta, "candidates_token_count", None) if usage_meta else None
                cached_tok = getattr(usage_meta, "cached_content_token_count", None) if usage_meta else None
                thought_tok = getattr(usage_meta, "thoughts_token_count", None) if usage_meta else None
                if req_tok is None:
                    req_tok = count_tokens(query, agent=agent_obj)
                if resp_tok is None:
                    resp_tok = count_tokens(text, agent=agent_obj)

                span.inc("request_tokens", req_tok)
                span.inc("response_tokens", resp_tok)
                if cached_tok is not None:
                    span.inc("cached_tokens", int(cached_tok))
                if thought_tok is not None:
                    span.inc("thinking_tokens", int(thought_tok))
                span.inc("response_bytes", len(text.encode("utf-8")))
                return text
            except Exception as e:
                span.fail()
                error_msg = f"Failed to consult the oracle: {str(e)}"
                print(f"[Oracle] {error_msg}")
                return error_msg
