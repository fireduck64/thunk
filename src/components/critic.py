from typing import Dict, Any
from src.components.base import BaseComponent

class CriticComponent(BaseComponent):
    """
    Acts as the subconscious supervisor of the agent.
    Periodically reviews recent actions to determine if the agent is stuck in a loop.
    Injects feelings/intuitions directly into working memory to course-correct.
    """
    def __init__(self, frequency: int = 8):
        # How many tools must be executed before the critic wakes up
        self.frequency = frequency
        self.action_count = 0
        self.needs_evaluation = False

    async def on_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        if event_name == "tool_executed":
            self.action_count += 1
            if self.action_count >= self.frequency:
                self.action_count = 0
                self.needs_evaluation = True
        elif event_name == "before_llm_call":
            if self.needs_evaluation:
                self.needs_evaluation = False
                await self._evaluate_progress()

    async def _evaluate_progress(self):
        print("\n[Critic] Waking up to evaluate recent progress...")
        if not self.agent:
            return

        # Grab the last 12 messages to get a good sense of the current loop
        recent_msgs = self.agent.messages[-12:]
        transcript = ""
        
        for msg in recent_msgs:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if msg.get("tool_calls"):
                calls = [f"{tc.function.name}({tc.function.arguments})" for tc in msg.get("tool_calls")]
                content = "Called tools: " + ", ".join(calls)
                
            # If the tool result is massive, we truncate it for the critic
            if role == "tool" and len(content) > 200:
                content = content[:200] + "... [TRUNCATED]"
                
            transcript += f"[{role.upper()}]: {content}\n"

        prompt = (
            "You are the subconscious monitor (Critic) for an autonomous AI agent.\n"
            "Review the following recent transcript of the agent's actions:\n\n"
            f"<transcript>\n{transcript}\n</transcript>\n\n"
            "Analyze the agent's behavior. Is the agent stuck in an infinite loop? Is it repeating the same "
            "failed tool calls blindly? Is it making good progress towards its goal?\n\n"
            "Output a brief, 1-2 sentence 'feeling' or 'intuition' to guide the agent. "
            "Speak in an observational, non-imperative tone (e.g., 'I notice you have been...', 'It seems like...'). "
            "Do NOT issue direct commands or directives like 'You must stop'. Just offer a gentle observation of the current pattern.\n"
            "Do NOT provide any introductory text, just the raw feeling."
        )

        try:
            # We use the main agent's client, but we don't give the critic access to tools
            # or the full context window. It just acts as a pure text-in text-out evaluator.
            response = await self.agent.client.chat.completions.create(
                model=self.agent.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7 # Add a bit of creativity so feelings aren't always identical
            )
            feeling = response.choices[0].message.content.strip()
            
            print(f"[Critic Feeling]: {feeling}\n")
            
            # Inject the feeling into the agent's working memory as a system message
            await self.agent.add_message("system", f"[Internal Feeling]: {feeling}")

        except Exception as e:
            print(f"[Critic Error] {e}")
