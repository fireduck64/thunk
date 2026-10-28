import base64
import json
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from google import genai
from google.genai import types


@dataclass
class _DeltaFunction:
    name: Optional[str] = None
    arguments: Optional[str] = None


@dataclass
class _DeltaToolCall:
    index: int
    id: str
    function: _DeltaFunction


@dataclass
class _StreamDelta:
    content: Optional[str] = None
    tool_calls: Optional[List[_DeltaToolCall]] = None


@dataclass
class _StreamChoice:
    delta: _StreamDelta
    finish_reason: Optional[str] = None


@dataclass
class _StreamChunk:
    choices: List[_StreamChoice]
    usage: Optional[Dict[str, int]] = None


class GeminiOpenAIClientAdapter:
    """
    Wraps google.genai.Client in an OpenAI-compatible async streaming interface
    (`client.chat.completions.create(...)`) so AgentCore, TieredMemoryComponent,
    and CriticComponent can switch seamlessly between local OpenAI/Ollama endpoints
    and Google Gemini Cloud via config.
    """

    def __init__(self, api_key: str, default_headers: Optional[Dict[str, str]] = None):
        if not api_key:
            raise ValueError(
                "Gemini provider requires 'api_key' under [llm] (or 'gemini_api_key' under [oracle]) in config.toml."
            )
        self.api_key = api_key
        self.default_headers: Dict[str, str] = dict(default_headers or {})
        self._genai_client = genai.Client(api_key=self.api_key)
        # Cache raw google.genai.types.Part objects (including thought_signature) by synthetic tool_call_id
        self._gemini_part_cache: Dict[str, types.Part] = {}
        self.chat = self._ChatNamespace(self)

    class _ChatNamespace:
        def __init__(self, adapter: "GeminiOpenAIClientAdapter"):
            self.completions = GeminiOpenAIClientAdapter._CompletionsNamespace(adapter)

    class _CompletionsNamespace:
        def __init__(self, adapter: "GeminiOpenAIClientAdapter"):
            self._adapter = adapter

        async def create(
            self,
            model: str,
            messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            extra_headers: Optional[Dict[str, str]] = None,
            stream: bool = True,
            **kwargs: Any,
        ) -> AsyncIterator[_StreamChunk]:
            return self._adapter._stream_completion(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    @staticmethod
    def _decode_data_url(url: str) -> Optional[Tuple[bytes, str]]:
        if not url or not url.startswith("data:"):
            return None
        try:
            header, b64_data = url.split(",", 1)
            mime_type = "image/jpeg"
            if ";" in header and ":" in header:
                mime_type = header.split(":", 1)[1].split(";", 1)[0].strip() or "image/jpeg"
            return base64.b64decode(b64_data), mime_type
        except Exception:
            return None

    def _extract_text_and_image_parts(self, raw_content: Any) -> Tuple[str, List[types.Part]]:
        if raw_content is None:
            return "", []
        if isinstance(raw_content, str):
            return raw_content, []
        if isinstance(raw_content, list):
            text_pieces: List[str] = []
            image_parts: List[types.Part] = []
            for item in raw_content:
                if not isinstance(item, dict):
                    text_pieces.append(str(item))
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    t = item.get("text")
                    if t:
                        text_pieces.append(str(t))
                elif item_type == "image_url":
                    url = (item.get("image_url") or {}).get("url", "")
                    decoded = self._decode_data_url(url)
                    if decoded is not None:
                        img_bytes, mime_type = decoded
                        image_parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
            return "\n".join(text_pieces), image_parts
        return str(raw_content), []

    @staticmethod
    def _extract_tc_fields(tc: Any) -> Tuple[str, str, Dict[str, Any]]:
        if isinstance(tc, dict):
            tc_id = str(tc.get("id") or "")
            fn = tc.get("function") or {}
            name = str(fn.get("name") or tc.get("name") or "")
            raw_args = fn.get("arguments") if "function" in tc else tc.get("arguments")
        else:
            tc_id = str(getattr(tc, "id", "") or "")
            fn = getattr(tc, "function", None)
            name = str(getattr(fn, "name", "") if fn else "")
            raw_args = getattr(fn, "arguments", "{}") if fn else "{}"

        if isinstance(raw_args, dict):
            args_dict = raw_args
        else:
            try:
                parsed = json.loads(raw_args or "{}")
                args_dict = parsed if isinstance(parsed, dict) else {"value": parsed}
            except Exception:
                args_dict = {"raw_arguments": str(raw_args or "")}
        return tc_id, name, args_dict

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[types.Content]]:
        """
        Converts OpenAI-format messages into (system_instruction, List[types.Content]).
        Preserves cached Gemini function_call parts (with thought_signature) when present,
        and safely serializes older non-Gemini tool calls as text so switching providers
        mid-conversation never fails Gemini schema/signature checks.
        """
        system_blocks: List[str] = []
        contents: List[types.Content] = []
        active_gemini_tc_ids: set[str] = set()

        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            raw_content = msg.get("content")
            text_str, image_parts = self._extract_text_and_image_parts(raw_content)

            if role == "system":
                if idx == 0 or not contents:
                    if text_str.strip():
                        system_blocks.append(text_str.strip())
                else:
                    # Mid-conversation system notice (e.g., Critic feeling or transient presence footer)
                    if text_str.strip():
                        contents.append(
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=f"[System Notice]\n{text_str}")],
                            )
                        )
                continue

            if role == "user":
                parts: List[types.Part] = []
                if text_str:
                    parts.append(types.Part.from_text(text=text_str))
                parts.extend(image_parts)
                if parts:
                    contents.append(types.Content(role="user", parts=parts))
                continue

            if role == "assistant":
                parts = []
                if text_str:
                    parts.append(types.Part.from_text(text=text_str))
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    tc_id, fn_name, args_dict = self._extract_tc_fields(tc)
                    cached_part = self._gemini_part_cache.get(tc_id)
                    if cached_part is not None:
                        parts.append(cached_part)
                        active_gemini_tc_ids.add(tc_id)
                    else:
                        # Historical tool call from a local OpenAI/Ollama turn (no Gemini thought_signature)
                        parts.append(
                            types.Part.from_text(
                                text=f"[Called tool {fn_name}({json.dumps(args_dict)})]"
                            )
                        )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue

            if role == "tool":
                tc_id = str(msg.get("tool_call_id") or "")
                fn_name = str(msg.get("name") or "tool")
                parts = []
                if tc_id in active_gemini_tc_ids:
                    parts.append(
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"result": text_str},
                        )
                    )
                else:
                    parts.append(
                        types.Part.from_text(
                            text=f"[Tool Result from {fn_name}]:\n{text_str}"
                        )
                    )
                parts.extend(image_parts)
                contents.append(types.Content(role="user", parts=parts))

        if not contents:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text="Continue.")]))

        system_instruction = "\n\n".join(system_blocks) if system_blocks else None
        return system_instruction, contents

    @staticmethod
    def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[types.Tool]]:
        if not tools:
            return None
        declarations: List[types.FunctionDeclaration] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            description = fn.get("description") or f"Execute {name}"
            parameters = fn.get("parameters") or {"type": "object", "properties": {}}
            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=description,
                    parameters_json_schema=parameters,
                )
            )
        if not declarations:
            return None
        return [types.Tool(function_declarations=declarations)]

    async def _stream_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[_StreamChunk]:
        system_instruction, contents = self._convert_messages(messages)
        gemini_tools = (
            self._convert_tools(tools)
            if (tools and tool_choice != "none")
            else None
        )

        config_kwargs: Dict[str, Any] = {
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = int(max_tokens)
        if temperature is not None:
            config_kwargs["temperature"] = float(temperature)

        gen_config = types.GenerateContentConfig(**config_kwargs)

        response_stream = await self._genai_client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=gen_config,
        )

        tc_index = 0
        saw_tool_call = False
        last_finish_reason = "stop"

        async for chunk in response_stream:
            usage_dict = None
            usage_meta = getattr(chunk, "usage_metadata", None)
            if usage_meta is not None:
                p_tok = getattr(usage_meta, "prompt_token_count", None)
                c_tok = getattr(usage_meta, "candidates_token_count", None)
                if p_tok is not None or c_tok is not None:
                    usage_dict = {
                        "prompt_tokens": int(p_tok or 0),
                        "completion_tokens": int(c_tok or 0),
                    }

            candidates = getattr(chunk, "candidates", None) or []
            if not candidates:
                if usage_dict is not None:
                    yield _StreamChunk(choices=[], usage=usage_dict)
                continue

            cand = candidates[0]
            cand_finish = getattr(cand, "finish_reason", None)
            if cand_finish is not None:
                finish_name = str(getattr(cand_finish, "name", cand_finish)).upper()
                if "MAX_TOKENS" in finish_name:
                    last_finish_reason = "length"
                else:
                    last_finish_reason = "stop"

            content_obj = getattr(cand, "content", None)
            parts = getattr(content_obj, "parts", None) or []

            for part in parts:
                if getattr(part, "thought", False) and not getattr(part, "function_call", None):
                    # Skip internal reasoning summary parts so only visible monologue is appended
                    continue

                fn_call = getattr(part, "function_call", None)
                if fn_call is not None and getattr(fn_call, "name", None):
                    saw_tool_call = True
                    tc_id = f"gemini_tc_{uuid.uuid4().hex[:12]}"
                    self._gemini_part_cache[tc_id] = part
                    args_dict = dict(getattr(fn_call, "args", None) or {})
                    delta = _StreamDelta(
                        content=None,
                        tool_calls=[
                            _DeltaToolCall(
                                index=tc_index,
                                id=tc_id,
                                function=_DeltaFunction(
                                    name=fn_call.name,
                                    arguments=json.dumps(args_dict),
                                ),
                            )
                        ],
                    )
                    tc_index += 1
                    yield _StreamChunk(
                        choices=[_StreamChoice(delta=delta, finish_reason=None)],
                        usage=usage_dict,
                    )
                    continue

                part_text = getattr(part, "text", None)
                if part_text:
                    delta = _StreamDelta(content=str(part_text), tool_calls=None)
                    yield _StreamChunk(
                        choices=[_StreamChoice(delta=delta, finish_reason=None)],
                        usage=usage_dict,
                    )

            if usage_dict is not None and not parts:
                yield _StreamChunk(choices=[], usage=usage_dict)

        final_reason = "tool_calls" if (saw_tool_call and last_finish_reason != "length") else last_finish_reason
        yield _StreamChunk(
            choices=[_StreamChoice(delta=_StreamDelta(), finish_reason=final_reason)],
            usage=None,
        )
