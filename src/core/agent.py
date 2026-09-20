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
        self.client = AsyncOpenAI(
            base_url=self.config["llm"]["api_url"],
            api_key=self.config["llm"]["api_key"]
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
        self.components.append(component)
        # Assuming components might need to hook into the event bus in the future
        # component.bind_events(self.event_bus) 

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

    def add_message(self, role: str, content: str):
        """Adds a message to the agent's working memory context."""
        self.messages.append({"role": role, "content": content})

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

        await self.event_bus.publish("agent_started")
        print(f"[Core] Agent loop started. (Model: {self.model})")

        while True:
            # Wait until there's an explicit reason to run (like a user message or un-suspended state)
            await self.suspended.wait()
            self.suspended.clear() # Clear the flag so it suspends again after processing
                
            try:
                # 1. Prompt the LLM
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools if self.tools else None,
                    # Tool choice auto ensures it can choose to use tools or output text
                    tool_choice="auto" if self.tools else "none" 
                )
                
                message = response.choices[0].message
                
                # We must append the exact Assistant message back to context for OpenAI format rules
                self.messages.append(message)
                
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
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": result
                        })
                        
                    # Set the flag so it loops immediately to see tool results
                    self.suspended.set()
                    continue 

                # 3. Check for standard text output
                if message.content:
                    print(f"[Agent]: {message.content}")
                    # If it just outputted text and no tools, we leave the suspended flag cleared
                    if not message.tool_calls:
                        print("[Core] Agent is waiting for next event...")

            except Exception as e:
                print(f"[Core Error] LLM Call Failed: {e}")
                await asyncio.sleep(5) # Prevent tight error loops

    def resume(self, user_message: str = None):
        """Wakes the agent up, optionally with new information."""
        if user_message:
            self.add_message("user", user_message)
        self.suspended.set()
