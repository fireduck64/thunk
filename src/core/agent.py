import asyncio
import json
from openai import AsyncOpenAI

from src.core.events import EventBus
from src.utils.config import load_config
from src.components.base import BaseComponent
from src.utils.schema import function_to_schema

class AgentCore:
    def __init__(self):
        self.config = load_config()
        self.event_bus = EventBus()
        self.components: list[BaseComponent] = []
        
        # Initialize OpenAI Client (Compatible with standard endpoints and Ollama)
        # We set a high timeout (5 minutes) because massive context windows or 
        # complex compression tasks can take a long time on local models.
        self.client = AsyncOpenAI(
            base_url=self.config["llm"]["api_url"],
            api_key=self.config["llm"]["api_key"],
            timeout=300.0
        )
        self.model = self.config["llm"]["model_name"]
        
        # State
        self.messages = []
        self.tools = []
        self.tool_map = {}
        
        # Concurrency / Lifecycle
        self.suspended = asyncio.Event()

    def register_component(self, component: BaseComponent):
        """Attaches a component to the agent."""
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

    def _build_system_prompt(self) -> str:
        """Aggregates all component system instructions."""
        prompt = "You are Thunk, an autonomous, long-running AI agent.\n\n"
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
            
        try:
            args = json.loads(args_str)
            func = self.tool_map[name]
            
            # If the tool is an async function, await it, otherwise just call it.
            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
                
            await self.event_bus.publish("tool_executed", {"name": name, "args": args, "result": result})
            return str(result)
            
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

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
            # Wait until there's an explicit reason to run (like a user message or un-suspended state)
            await self.suspended.wait()
            self.suspended.clear() # Clear the flag so it suspends again after processing
                
            try:
                # Tell the logger exactly what we are sending to the LLM
                await self.event_bus.publish("before_llm_call", {"agent": self, "messages": self.messages})
                
                # 1. Prompt the LLM
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools if self.tools else None,
                    # Tool choice auto ensures it can choose to use tools or output text
                    tool_choice="auto" if self.tools else "none",
                    frequency_penalty=0.2, # Light penalty to prevent "<|channel>thought" infinite loops
                    presence_penalty=0.2
                )
                
                # Tell the logger exactly what the LLM returned (useful for debugging token usage and full responses)
                await self.event_bus.publish("after_llm_call", {"agent": self, "response": response})
                
                message = response.choices[0].message
                
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
                        await self._append_message({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": result
                        })
                        
                    # Set the flag so it loops immediately to see tool results
                    self.suspended.set()
                    # We do NOT 'continue' here because we still need to run 
                    # the context window checks and trigger other event hooks!

                # 3. Check for standard text output
                if message.content:
                    print(f"[Agent Monologue]: {message.content}")

                # 4. Context Window Check
                await self.event_bus.publish("context_window_check", {"agent": self})
                
                # 5. Suspend if no tools were called (to prevent infinite monologue loops)
                # We do this AFTER the context window check so memory compression can run
                if message.content and not message.tool_calls:
                    print("[Core] Agent produced no tool calls. Auto-suspending...")
                    self.suspended.clear()
                    await self.event_bus.publish("agent_suspended", {"agent": self})

            except Exception as e:
                print(f"[Core Error] LLM Call Failed: {e}")
                await asyncio.sleep(5) # Prevent tight error loops

    async def resume(self, user_message: str = None):
        """Wakes the agent up, optionally with new information."""
        if user_message:
            # We intercept User messages here. 
            # If the user sends a message, we do a subconscious associative search
            # across internal memory and inject the results transparently.
            # We don't want to save this injected text to the archival log though!
            
            # Fire an event that components can hook into to modify the message BEFORE it gets saved
            # or to append context. We use a dict so it's mutable by reference.
            msg_context = {"content": user_message}
            await self.event_bus.publish("before_user_message_added", msg_context)
            
            # Save the clean message to the archival log
            await self.add_message("user", user_message, publish_event=True)
            
            # If a component added subconscious context, we replace the working memory content
            if msg_context["content"] != user_message:
                self.messages[-1]["content"] = msg_context["content"]
            
        self.suspended.set()
