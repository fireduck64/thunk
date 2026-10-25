import asyncio
import json
import inspect
import httpx
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from src.core.events import EventBus
from src.utils.config import load_config
from src.components.base import BaseComponent
from src.utils.schema import function_to_schema

class AgentCore:
    def __init__(self, thread_id: str = "main"):
        self.config = load_config()
        self.thread_id = (thread_id or "main").strip() or "main"
        agent_cfg = self.config.get("agent", {})
        if "name" not in agent_cfg or not agent_cfg["name"]:
            raise ValueError("AgentCore requires 'name' to be explicitly defined in config under [agent].name")
        self.name = agent_cfg["name"]

        self.event_bus = EventBus()
        self.components: list[BaseComponent] = []
        
        # Initialize OpenAI Client (Compatible with standard endpoints and Ollama)
        # We disable the local client timeout entirely (timeout=None) 
        # and rely on the streaming keep-alive to bypass Nginx timeouts.
        self.client = AsyncOpenAI(
            base_url=self.config["llm"]["api_url"],
            api_key=self.config["llm"]["api_key"],
            default_headers={"X-Agent-ID": f"{self.name}-{self.thread_id}-core"},
            timeout=httpx.Timeout(None)
        )
        self.model = self.config["llm"]["model_name"]
        self.max_output_tokens = int(self.config.get("llm", {}).get("max_output_tokens", 16384))
        self.debug_stream = self.config.get("debug", {}).get("stream_output", False)
        self.last_finish_reason = None
        
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

    def _prepare_llm_messages(self) -> list[dict]:
        """Builds the wire-format messages list, expanding any transient `_images` into OpenAI image_url parts."""
        wire_messages = []
        for msg in self.messages:
            images = msg.get("_images")
            if images:
                content_parts = [{"type": "text", "text": msg.get("content") or ""}]
                for img_url in images:
                    content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
                wire_msg = {k: v for k, v in msg.items() if not k.startswith("_")}
                wire_msg["content"] = content_parts
                wire_messages.append(wire_msg)
            else:
                wire_messages.append({k: v for k, v in msg.items() if not k.startswith("_")})
        return wire_messages

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
        if name not in self.tool_map:
            return f"Error: Tool '{name}' not found."
            
        args = {}
        valid_params = []
        try:
            args = json.loads(args_str) if args_str else {}
            if not isinstance(args, dict):
                args = {}
            func = self.tool_map[name]
            
            valid_params = [p for p in inspect.signature(func).parameters.keys() if p != "self"]
            normalized_args = dict(args)

            # Normalize common path parameter aliases across FileReaderComponent and WorkspaceComponent
            alias_map = {
                "path": ("filepath", "file_path", "subdir"),
                "filepath": ("path", "file_path"),
                "subdir": ("path", "filepath", "file_path"),
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
            return str(result)
            
        except TypeError as e:
            return (
                f"Error executing tool '{name}': {str(e)} "
                f"(received args: {list(args.keys())}, expected params: {valid_params})"
            )
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    async def _accumulate_stream(self, stream) -> ChatCompletionMessage:
        """
        Consumes an OpenAI async stream chunk-by-chunk to prevent proxy timeouts,
        then reconstructs and returns the final parsed ChatCompletionMessage.
        """
        content = ""
        tool_calls_dict = {}
        finish_reason = None
        
        async for chunk in stream:
            if not chunk.choices:
                continue
                
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            delta = chunk.choices[0].delta
            
            if delta.content:
                content += delta.content
                if self.debug_stream:
                    print(delta.content, end="", flush=True)
                
            if delta.tool_calls:
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

    async def start(self):
        """The main continuous thinking loop."""
        self._build_tools()
        system_prompt = self._build_system_prompt()
        
        # Ensure system prompt is always at the top of working memory
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(0, {"role": "system", "content": system_prompt})
            
        # Tell components the agent is starting (e.g. to load their own state)
        await self.event_bus.publish("agent_started", {"agent": self})
        print(f"[Core] Agent loop started. (Model: {self.model})")

        # Automatically wake the agent up on boot so it can process background tasks
        self.suspended.set()

        while True:
            print("  Agent loop top")
            # Wait until there's an explicit reason to run (like a user message or un-suspended state)
            await self.suspended.wait()
            # self.suspended.clear() # Clear the flag so it suspends again after processing

            print("  Doing things")
                
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

                # 1. Prompt the LLM using a stream to keep Nginx alive
                response_stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=wire_messages,
                    tools=self.tools if self.tools else None,
                    # Tool choice auto ensures it can choose to use tools or output text
                    tool_choice="auto" if self.tools else "none",
                    max_tokens=self.max_output_tokens,
                    stream=True
                )
                
                # Reconstruct the message object silently
                message = await self._accumulate_stream(response_stream)

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
                        print(f"\n[Tool Execution] -> {tool_call.function.name}({tool_call.function.arguments})")
                        
                        result = await self.execute_tool(
                            name=tool_call.function.name, 
                            args_str=tool_call.function.arguments
                        )
                        
                        print(f"[Tool Result] -> {result}\n")
                        
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
                    print(f"[Agent Monologue]: {message.content}")

                if self.last_finish_reason == "length":
                    print(f"[Core Warning] Generation hit max_output_tokens ({self.max_output_tokens}) and was truncated.")
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
