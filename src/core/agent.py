import asyncio
import json
import inspect
import sys
import time
import httpx
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from src.core.events import EventBus
from src.core.gemini_client import GeminiOpenAIClientAdapter
from src.utils.config import load_config
from src.utils.metrics import MetricSpan, count_tokens, infer_exec_location
from src.components.base import BaseComponent
from src.utils.schema import function_to_schema

class AgentCore:
    def __init__(self, thread_id: str = "main", provider: str = None, model_name: str = None):
        self.config = load_config()
        self.thread_id = (thread_id or "main").strip() or "main"
        agent_cfg = self.config.get("agent", {})
        if "name" not in agent_cfg or not agent_cfg["name"]:
            raise ValueError("AgentCore requires 'name' to be explicitly defined in config under [agent].name")
        self.name = agent_cfg["name"]

        self.event_bus = EventBus()
        self.components: list[BaseComponent] = []

        llm_cfg = self.config.get("llm", {})
        self.provider = (provider or llm_cfg.get("provider", "openai")).strip().lower()
        self.model = (model_name or llm_cfg.get("model_name", "gemma4:31b")).strip()
        default_headers = {"X-Agent-ID": f"{self.name}-{self.thread_id}-core"}

        if self.provider == "gemini":
            gemini_key = (
                llm_cfg.get("gemini_api_key")
                or self.config.get("oracle", {}).get("gemini_api_key")
                or llm_cfg.get("api_key")
            )
            self.client = GeminiOpenAIClientAdapter(
                api_key=gemini_key,
                default_headers=default_headers,
            )
            self.exec_location = "cloud"
        else:
            # Initialize OpenAI Client (Compatible with standard endpoints and Ollama)
            # We disable the local client timeout entirely (timeout=None)
            # and rely on the streaming keep-alive to bypass Nginx timeouts.
            self.client = AsyncOpenAI(
                base_url=llm_cfg["api_url"],
                api_key=llm_cfg["api_key"],
                default_headers=default_headers,
                timeout=httpx.Timeout(None),
            )
            self.exec_location = infer_exec_location(llm_cfg.get("api_url"))

        self.max_output_tokens = int(llm_cfg.get("max_output_tokens", 16384))
        self.min_turn_interval_sec = float(llm_cfg.get("min_turn_interval_sec", 0.0))
        self.max_retries = int(llm_cfg.get("max_retries", 5))
        self.retry_base_delay = float(llm_cfg.get("retry_base_delay", 2.0))
        self.debug_stream = self.config.get("debug", {}).get("stream_output", False)
        self.last_finish_reason = None
        self.last_ttft_ms = None
        self.last_usage = None
        
        # State
        self.messages = []
        self.tools = []
        self.tool_map = {}
        
        # Concurrency / Lifecycle
        self.suspended = asyncio.Event()
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.pending_tool_images: list[str] = []

    def register_component(self, component: BaseComponent):
        """Attaches a component to the agent."""
        subconscious_types = {"SubconsciousRecallComponent", "SpreadingActivationComponent"}
        comp_name = component.__class__.__name__
        if comp_name in subconscious_types:
            for existing in self.components:
                if existing.__class__.__name__ in subconscious_types:
                    raise ValueError(
                        f"Cannot register {comp_name}: {existing.__class__.__name__} is already registered. "
                        "SubconsciousRecallComponent and SpreadingActivationComponent are mutually exclusive."
                    )

        component.agent = self
        self.components.append(component)
        # Bind the component to the event bus
        self.event_bus.subscribe("agent_started", component.on_event)
        self.event_bus.subscribe("tool_executed", component.on_event)
        self.event_bus.subscribe("message_added", component.on_event)
        self.event_bus.subscribe("before_user_message_added", component.on_event)
        self.event_bus.subscribe("context_window_check", component.on_event)
        self.event_bus.subscribe("before_llm_call", component.on_event)
        self.event_bus.subscribe("after_llm_call", component.on_event)
        self.event_bus.subscribe("agent_suspended", component.on_event)

    def _build_system_prompt(self) -> str:
        """Aggregates all component system instructions."""
        prompt = f"You are {self.name}, an autonomous, long-running AI agent.\n\n"
        for comp in self.components:
            addition = comp.get_system_prompt_addition()
            if addition:
                prompt += f"{addition}\n\n"
        return prompt

    def _build_tools(self):
        """Gathers all tools from components and builds their schemas."""
        self.tools = []
        self.tool_map = {}
        
        for comp in self.components:
            for func in comp.get_tools():
                schema = function_to_schema(func)
                self.tools.append(schema)
                self.tool_map[func.__name__] = func

    @staticmethod
    def _extract_tc_tuple(tc: dict | object) -> tuple[str, str, str]:
        if isinstance(tc, dict):
            tc_id = str(tc.get("id") or "")
            fn = tc.get("function") or {}
            name = str(fn.get("name") or tc.get("name") or "tool")
            args = str(fn.get("arguments") if "function" in tc else tc.get("arguments") or "{}")
        else:
            tc_id = str(getattr(tc, "id", "") or "")
            fn = getattr(tc, "function", None)
            name = str(getattr(fn, "name", "tool") if fn else "tool")
            args = str(getattr(fn, "arguments", "{}") if fn else "{}")
        return tc_id, name, args

    @classmethod
    def normalize_wire_messages(cls, raw_messages: list[dict]) -> list[dict]:
        """
        Normalizes working memory messages into a clean, strictly ordered wire format
        for both OpenAI/Ollama and Gemini:
        1. Expands transient `_images` into image_url content parts and strips private `_*` keys.
        2. Converts mid-conversation `system` notices into `user` context blocks.
        3. Validates `assistant (tool_calls)` <-> `tool` pairing, converting any legacy
           orphaned tool calls or tool results into clean text blocks.
        4. Guarantees the first turn after `system` is `role='user'` and prevents
           consecutive `assistant` turns without an intervening `user` or `tool` turn.
        """
        if not raw_messages:
            return []

        valid_tool_call_ids: set[str] = set()
        valid_assistant_indices: set[int] = set()

        for idx, msg in enumerate(raw_messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                following_ids: set[str] = set()
                look = idx + 1
                while look < len(raw_messages) and raw_messages[look].get("role") == "tool":
                    tid = str(raw_messages[look].get("tool_call_id") or "")
                    if tid:
                        following_ids.add(tid)
                    look += 1
                tc_ids = [cls._extract_tc_tuple(tc)[0] for tc in msg.get("tool_calls") or []]
                if tc_ids and all(tid and tid in following_ids for tid in tc_ids):
                    valid_assistant_indices.add(idx)
                    valid_tool_call_ids.update(tc_ids)

        wire_messages: list[dict] = []
        for idx, msg in enumerate(raw_messages):
            role = msg.get("role", "user")
            content = msg.get("content")
            images = msg.get("_images")

            # 1. Keep primary system prompt at index 0; convert mid-stream system notices to user turns
            if role == "system":
                if idx == 0 and not wire_messages:
                    wire_messages.append({"role": "system", "content": content or ""})
                elif content and str(content).strip():
                    text_val = str(content).strip()
                    if not text_val.startswith("[System Notice]"):
                        text_val = f"[System Notice]\n{text_val}"
                    wire_messages.append({"role": "user", "content": text_val})
                continue

            # 2. Assistant turn
            if role == "assistant":
                tcs = msg.get("tool_calls") or []
                if tcs and idx not in valid_assistant_indices:
                    tc_lines = [
                        f"[Called tool {name}({args})]"
                        for _, name, args in (cls._extract_tc_tuple(tc) for tc in tcs)
                    ]
                    combined_text = "\n".join(([str(content)] if content else []) + tc_lines)
                    out_msg = {"role": "assistant", "content": combined_text}
                else:
                    out_msg = {k: v for k, v in msg.items() if not k.startswith("_")}

                # Guarantee every assistant turn follows a user or tool turn
                has_non_system = any(m.get("role") != "system" for m in wire_messages)
                if not has_non_system:
                    wire_messages.append({"role": "user", "content": "[Continuing from Core Summary]"})
                elif wire_messages[-1].get("role") == "assistant":
                    wire_messages.append({"role": "user", "content": "[Continue]"})

                wire_messages.append(out_msg)
                continue

            # 3. Tool result turn
            if role == "tool":
                tc_id = str(msg.get("tool_call_id") or "")
                fn_name = str(msg.get("name") or "tool")
                if tc_id and tc_id in valid_tool_call_ids:
                    out_msg = {k: v for k, v in msg.items() if not k.startswith("_")}
                    if images:
                        content_parts = [{"type": "text", "text": str(content or "")}]
                        for img_url in images:
                            content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                        out_msg["content"] = content_parts
                    wire_messages.append(out_msg)
                else:
                    # Orphaned tool result from a legacy SQLite restore without tool_call_id
                    fallback_text = f"[Tool Result from {fn_name}]:\n{content or ''}"
                    if images:
                        content_parts = [{"type": "text", "text": fallback_text}]
                        for img_url in images:
                            content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                        wire_messages.append({"role": "user", "content": content_parts})
                    else:
                        wire_messages.append({"role": "user", "content": fallback_text})
                continue

            # 4. Standard user turn
            if images:
                content_parts = [{"type": "text", "text": str(content or "")}]
                for img_url in images:
                    content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                out_msg = {k: v for k, v in msg.items() if not k.startswith("_")}
                out_msg["content"] = content_parts
                wire_messages.append(out_msg)
            else:
                wire_messages.append({k: v for k, v in msg.items() if not k.startswith("_")})

        return wire_messages

    def _prepare_llm_messages(self) -> list[dict]:
        """Builds the normalized wire-format messages list for the LLM."""
        return self.normalize_wire_messages(self.messages)

    def _clear_ephemeral_images(self) -> None:
        """Strips transient base64 `_images` and transient presence footers from working memory after they have been sent to the LLM once."""
        for idx in range(len(self.messages) - 1, -1, -1):
            if self.messages[idx].get("_transient_presence"):
                del self.messages[idx]
            else:
                self.messages[idx].pop("_images", None)

    def rebuild_system_prompt(self):
        """Rebuilds the system prompt (Index 0 of messages) using all current component states."""
        system_prompt = self._build_system_prompt()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = system_prompt
            
    async def _append_message(self, message: dict, publish_event: bool = True):
        """Internal helper to safely append a message and notify the event bus."""
        self.messages.append(message)
        if publish_event:
            await self.event_bus.publish("message_added", {"message": message})

    async def add_message(self, role: str, content: str, publish_event: bool = True):
        """Adds a message to the agent's working memory context."""
        await self._append_message({"role": role, "content": content}, publish_event)

    async def execute_tool(self, name: str, args_str: str) -> str:
        """Safely executes a Python tool function and returns the string result."""
        with MetricSpan(
            operation="tool_call",
            agent=self.name,
            thread=self.thread_id,
            component="core",
            type=name,
        ) as span:
            span.inc("request_bytes", len((args_str or "").encode("utf-8")))
            if name not in self.tool_map:
                span.fail()
                err = f"Error: Tool '{name}' not found."
                span.inc("response_bytes", len(err.encode("utf-8")))
                return err
                
            args = {}
            valid_params = []
            try:
                args = json.loads(args_str) if args_str else {}
                if not isinstance(args, dict):
                    args = {}
                func = self.tool_map[name]
                
                valid_params = [p for p in inspect.signature(func).parameters.keys() if p != "self"]
                normalized_args = dict(args)

                # Normalize common path, message, and chunk/page parameter aliases
                alias_map = {
                    "path": ("filepath", "file_path", "filename", "subdir"),
                    "filepath": ("path", "file_path", "filename"),
                    "subdir": ("path", "filepath", "file_path"),
                    "chunk": ("page_number", "page"),
                    "message": ("msg", "text"),
                }
                for target_param, aliases in alias_map.items():
                    if target_param in valid_params and target_param not in normalized_args:
                        for alias in aliases:
                            if alias in normalized_args:
                                normalized_args[target_param] = normalized_args[alias]
                                break

                # Filter out any hallucinated arguments not in the function signature
                safe_args = {k: v for k, v in normalized_args.items() if k in valid_params}
                
                # If the tool is an async function, await it, otherwise just call it.
                if asyncio.iscoroutinefunction(func):
                    result = await func(**safe_args)
                else:
                    result = func(**safe_args)
                    
                await self.event_bus.publish("tool_executed", {"name": name, "args": safe_args, "result": result})
                res_str = str(result)
                span.inc("response_bytes", len(res_str.encode("utf-8")))
                return res_str
                
            except TypeError as e:
                span.fail()
                err = (
                    f"Error executing tool '{name}': {str(e)} "
                    f"(received args: {list(args.keys())}, expected params: {valid_params})"
                )
                span.inc("response_bytes", len(err.encode("utf-8")))
                return err
            except Exception as e:
                span.fail()
                err = f"Error executing tool '{name}': {str(e)}"
                span.inc("response_bytes", len(err.encode("utf-8")))
                return err

    async def _accumulate_stream(self, stream) -> ChatCompletionMessage:
        """
        Consumes an OpenAI async stream chunk-by-chunk to prevent proxy timeouts,
        then reconstructs and returns the final parsed ChatCompletionMessage.
        """
        content = ""
        tool_calls_dict = {}
        finish_reason = None
        t0 = time.perf_counter()
        ttft_ms = None
        usage_dict = None
        
        async for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                p_tok = getattr(chunk_usage, "prompt_tokens", None)
                c_tok = getattr(chunk_usage, "completion_tokens", None)
                if isinstance(chunk_usage, dict):
                    p_tok = chunk_usage.get("prompt_tokens")
                    c_tok = chunk_usage.get("completion_tokens")
                if p_tok is not None or c_tok is not None:
                    usage_dict = {"prompt_tokens": p_tok, "completion_tokens": c_tok}

            if not chunk.choices:
                continue
                
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            delta = chunk.choices[0].delta
            
            if delta.content:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                content += delta.content
                if self.debug_stream:
                    print(delta.content, end="", flush=True)
                
            if delta.tool_calls:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_dict:
                        tool_calls_dict[idx] = {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": ""}
                        }
                    if tc.function.arguments:
                        tool_calls_dict[idx]["function"]["arguments"] += tc.function.arguments

        self.last_finish_reason = finish_reason
        self.last_ttft_ms = ttft_ms
        self.last_usage = usage_dict

        tool_calls = []
        if tool_calls_dict:
            # Sort by index to ensure order is preserved
            for idx in sorted(tool_calls_dict.keys()):
                tc_data = tool_calls_dict[idx]
                if finish_reason == "length":
                    # Discard truncated tool calls with incomplete JSON arguments
                    try:
                        if not tc_data["function"]["name"]:
                            continue
                        json.loads(tc_data["function"]["arguments"] or "{}")
                    except Exception:
                        print(f"\n[Core Warning] Dropping truncated tool call '{tc_data['function']['name']}' due to max_output_tokens limit.")
                        continue
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tc_data["id"],
                        type="function",
                        function=Function(
                            name=tc_data["function"]["name"], 
                            arguments=tc_data["function"]["arguments"]
                        )
                    )
                )

        if self.debug_stream and content:
            print() # Print a final newline after the stream finishes

        return ChatCompletionMessage(
            role="assistant",
            content=content if content else None,
            tool_calls=tool_calls if tool_calls else None
        )

    async def _call_llm_with_retries(self, wire_messages: list[dict]) -> ChatCompletionMessage:
        """
        Calls the configured LLM with exponential backoff up to self.max_retries.
        Records metrics via MetricSpan and exits the process (sys.exit(1)) if all attempts fail.
        """
        exec_loc = getattr(self, "exec_location", None) or infer_exec_location(
            self.config.get("llm", {}).get("api_url")
        )
        with MetricSpan(
            operation="llm_call",
            agent=self.name,
            thread=self.thread_id,
            component="core",
            type="agent_turn",
            model=self.model,
            exec_location=exec_loc,
        ) as llm_span:
            req_bytes = len(json.dumps(wire_messages, default=str).encode("utf-8"))
            llm_span.inc("request_messages", len(wire_messages))
            llm_span.inc("request_bytes", req_bytes)

            message = None
            for attempt in range(self.max_retries):
                try:
                    with llm_span.subtimer("llm_call_ms"):
                        response_stream = await self.client.chat.completions.create(
                            model=self.model,
                            messages=wire_messages,
                            tools=self.tools if self.tools else None,
                            tool_choice="auto" if self.tools else "none",
                            max_tokens=self.max_output_tokens,
                            stream=True,
                        )
                        message = await self._accumulate_stream(response_stream)
                    break
                except Exception as e:
                    print(
                        f"[Core ERROR] LLM call attempt {attempt + 1}/{self.max_retries} failed: {e}",
                        flush=True,
                    )
                    if attempt + 1 < self.max_retries:
                        llm_span.inc("retries", 1)
                        delay = self.retry_base_delay * (2 ** attempt)
                        if delay > 0:
                            await asyncio.sleep(delay)

            if message is None:
                llm_span.fail()
                llm_span.set_field("finish_reason", "error")
                llm_span.finish()
                print(
                    f"[Core FATAL] All {self.max_retries} LLM call attempts failed. Exiting process.",
                    flush=True,
                )
                sys.exit(1)

            if self.last_ttft_ms is not None:
                llm_span.add_timer("ttft_ms", self.last_ttft_ms)

            usage = self.last_usage or {}
            req_tok = usage.get("prompt_tokens")
            if req_tok is None:
                req_tok = count_tokens(wire_messages, agent=self)
            resp_text_for_tok = (message.content or "") + (
                str(message.tool_calls) if message.tool_calls else ""
            )
            resp_tok = usage.get("completion_tokens")
            if resp_tok is None:
                resp_tok = count_tokens(resp_text_for_tok, agent=self)

            llm_span.inc("request_tokens", req_tok)
            llm_span.inc("response_tokens", resp_tok)
            llm_span.inc("response_bytes", len(resp_text_for_tok.encode("utf-8")))
            llm_span.inc("tool_calls", len(message.tool_calls or []))
            llm_span.set_field("finish_reason", self.last_finish_reason or "stop")
            return message

    async def start(self):
        """The main continuous thinking loop."""
        self._build_tools()
        system_prompt = self._build_system_prompt()
        
        # Ensure system prompt is always at the top of working memory
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(0, {"role": "system", "content": system_prompt})
            
        # Tell components the agent is starting (e.g. to load their own state)
        await self.event_bus.publish("agent_started", {"agent": self})
        print(f"[Core:{self.thread_id}] Agent loop started. (Model: {self.model})")

        # Automatically wake the agent up on boot so it can process background tasks
        self.suspended.set()

        while True:
            print(f"  [Core:{self.thread_id}] Agent loop top")
            # Wait until there's an explicit reason to run (like a user message or un-suspended state)
            await self.suspended.wait()
            # self.suspended.clear() # Clear the flag so it suspends again after processing

            print(f"  [Core:{self.thread_id}] Doing things")
                
            try:
                # 0. Drain any external messages that arrived while we were mid-turn
                await self._drain_inbox()

                # 4. Context Window Check
                # We moved this to the TOP of the loop so it guarantees the context 
                # is pruned before the LLM is called, preventing timeout death spirals.
                await self.event_bus.publish("context_window_check", {"agent": self})

                # Tell the logger exactly what we are sending to the LLM
                await self.event_bus.publish("before_llm_call", {"agent": self, "messages": self.messages})
                
                wire_messages = self._prepare_llm_messages()
                message = await self._call_llm_with_retries(wire_messages)

                # Clear transient base64 images now that the LLM has seen them on this call
                self._clear_ephemeral_images()
                
                # Tell the logger exactly what the LLM returned
                # (We pass the reconstructed message instead of the raw stream)
                await self.event_bus.publish("after_llm_call", {"agent": self, "response": message})
                
                # Convert the object to a dict safely to store in our messages list
                msg_dict = {"role": "assistant"}
                if message.content:
                    msg_dict["content"] = message.content
                if message.tool_calls:
                    # We have to keep the exact pydantic objects for tool calls for OpenAI compatibility
                    msg_dict["tool_calls"] = message.tool_calls
                    
                # OpenAI strictly forbids assistant messages with both null content AND no tool_calls,
                # though it shouldn't happen, we provide a fallback empty string just in case
                if "content" not in msg_dict and "tool_calls" not in msg_dict:
                    msg_dict["content"] = ""
                    
                await self._append_message(msg_dict)
                
                # 2. Check for Tool Calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        print(f"\n[Tool Execution:{self.thread_id}] -> {tool_call.function.name}({tool_call.function.arguments})")
                        
                        result = await self.execute_tool(
                            name=tool_call.function.name, 
                            args_str=tool_call.function.arguments
                        )
                        
                        print(f"[Tool Result:{self.thread_id}] -> {result}\n")
                        
                        # Add tool result back to the context
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": result
                        }
                        if self.pending_tool_images:
                            tool_msg["_images"] = list(self.pending_tool_images)
                            self.pending_tool_images.clear()
                        await self._append_message(tool_msg)
                        
                    # Set the flag so it loops immediately to see tool results
                    self.suspended.set()
                    # We do NOT 'continue' here because we still need to run 
                    # the context window checks and trigger other event hooks!

                # 3. Check for standard text output
                if message.content:
                    print(f"[Agent Monologue:{self.thread_id}]: {message.content}")

                if self.last_finish_reason == "length":
                    print(f"[Core Warning:{self.thread_id}] Generation hit max_output_tokens ({self.max_output_tokens}) and was truncated.")
                    await self.add_message(
                        "system",
                        f"[System Notice: Your previous generation hit the {self.max_output_tokens}-token output limit and was cut off. "
                        "Keep your internal reasoning concise and execute one small step at a time.]"
                    )
                    self.suspended.set()

                # Note: We removed the auto-suspension logic here. 
                # The agent is now expected to explicitly use the `suspend_until_next_event` tool
                # from the SystemComponent when it is finished working.

            except Exception as e:
                print(f"[Core Error] LLM Call Failed: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5) # Prevent tight error loops

    async def _drain_inbox(self):
        """Drains all queued external messages at a safe boundary between turns, batching subconscious recall."""
        if self.inbox.empty():
            return

        batch = []
        while not self.inbox.empty():
            item = await self.inbox.get()
            if isinstance(item, dict):
                batch.append((item.get("content", ""), list(item.get("images") or [])))
            else:
                batch.append((str(item), []))

        non_clock_messages = [
            msg for msg, _ in batch if msg and not msg.startswith("[System Clock]")
        ]

        recall_prefix = ""
        if non_clock_messages:
            combined_text = "\n".join(non_clock_messages)
            msg_context = {"content": combined_text}
            await self.event_bus.publish("before_user_message_added", msg_context)

            modified = msg_context["content"]
            if modified != combined_text:
                if "--- End Recall ---\n\n" in modified:
                    recall_prefix = modified.split("--- End Recall ---\n\n", 1)[0] + "--- End Recall ---\n\n"
                elif modified.endswith(combined_text):
                    recall_prefix = modified[: -len(combined_text)]

        # Drain any straggler messages that arrived while subconscious recall was running
        # without triggering a second synthesis cycle.
        while not self.inbox.empty():
            item = await self.inbox.get()
            if isinstance(item, dict):
                batch.append((item.get("content", ""), list(item.get("images") or [])))
            else:
                batch.append((str(item), []))

        for user_message, images in batch:
            # Save the clean message to the archival log
            await self.add_message("user", user_message, publish_event=True)

            # Attach the batched subconscious recall prefix once to the first non-clock message in working memory
            if recall_prefix and user_message and not user_message.startswith("[System Clock]"):
                self.messages[-1]["content"] = f"{recall_prefix}{user_message}"
                recall_prefix = ""

            if images:
                self.messages[-1]["_images"] = list(images)

    async def resume(self, user_message: str = None, images: list[str] = None):
        """Wakes the agent up, optionally enqueueing new external information (and transient images) for the next turn boundary."""
        if user_message:
            if images:
                await self.inbox.put({"content": user_message, "images": list(images)})
            else:
                await self.inbox.put(user_message)
        self.suspended.set()
