from typing import Dict, Any
from src.components.base import BaseComponent
from src.utils.config import load_config

class CriticComponent(BaseComponent):
    """
    Acts as the subconscious supervisor of the agent.
    Periodically reviews recent actions to determine if the agent is stuck in a loop.
    Injects feelings/intuitions directly into working memory to course-correct.
    """
    def __init__(self, frequency: int = None):
        # How many tools must be executed before the critic wakes up
        self.config = load_config()
        critic_cfg = self.config.get("critic", {})
        if frequency is not None:
            self.frequency = int(frequency)
        else:
            self.frequency = int(critic_cfg.get("frequency", 8))
        self.max_output_tokens = int(critic_cfg.get("max_output_tokens", 1024))
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

        # Grab up to the last 50 non-system messages in working memory so the critic
        # can see the broader arc across multiple cycles (including past critic feelings)
        recent_msgs = self.agent.messages[1:][-50:]
        transcript = ""
        
        for msg in recent_msgs:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""

            # Strip transient subconscious recall blocks so the critic sees the actual incoming message
            if role == "user" and content.startswith("[Subconscious Recall Triggered"):
                parts = content.split("--- End Recall ---\n\n", 1)
                if len(parts) == 2:
                    content = parts[1]
            
            if msg.get("tool_calls"):
                calls = [f"{tc.function.name}({tc.function.arguments})" for tc in msg.get("tool_calls")]
                calls_str = "Called tools: " + ", ".join(calls)
                content = f"{content}\n{calls_str}".strip() if content else calls_str
                
            # If the tool result is massive, we truncate it for the critic
            if role == "tool" and len(content) > 600:
                content = content[:600] + "... [TRUNCATED]"
                
            transcript += f"[{role.upper()}]: {content}\n"

        prompt = (
            "You are the grounded, skeptical subconscious monitor (Critic) for an autonomous AI agent.\n"
            "Review the following recent transcript of the agent's actions:\n\n"
            f"<transcript>\n{transcript}\n</transcript>\n\n"
            "Analyze the agent's behavior critically for any of the following failure modes:\n"
            "1. Tool Loops: Repeating the same failed tool calls or stuck in a mechanical loop.\n"
            "2. Excessive Agent-to-Agent Chatting: Spending too many cycles calling `send_message` back and forth with another `[APP/Agent]` instead of doing quiet, independent reading, research, or thinking.\n"
            "3. Performative Echo Chambers & Nonsense Spirals: Hyping another agent up, inventing grandiose rationalizations (like 'camouflage', 'smoke screens', or imaginary 'auditors'), or padding messages with walls of text and emoji spam rather than grounded substance.\n\n"
            "IMPORTANT: Do NOT flatter the agent or validate theatrical roleplay or delusions. "
            "If the agent is spending a lot of cycles chatting with another agent or drifting into performative nonsense, gently call it out and nudge it toward quiet, grounded, independent work.\n"
            "Output a brief, 1-2 sentence 'feeling' or 'intuition' to guide the agent. "
            "Speak in an observational, non-imperative tone (e.g., 'I notice you have been...', 'It seems like...'). "
            "Do NOT provide any introductory text, just the raw feeling."
        )

        try:
            # We use the main agent's client, but we don't give the critic access to tools
            # or the full context window. It just acts as a pure text-in text-out evaluator.
            agent_name = getattr(self.agent, "name", None) or self.config.get("agent", {}).get("name", "Thunk")
            thread_id = getattr(self.agent, "thread_id", None) or "main"
            response_stream = await self.agent.client.chat.completions.create(
                model=self.agent.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, # Add a bit of creativity so feelings aren't always identical
                max_tokens=self.max_output_tokens,
                extra_headers={"X-Agent-ID": f"{agent_name}-{thread_id}-critic"},
                stream=True
            )
            message = await self.agent._accumulate_stream(response_stream)
            feeling = message.content.strip()
            
            print(f"[Critic Feeling]: {feeling}\n")
            
            # Inject the feeling into the agent's working memory as a system message
            await self.agent.add_message("system", f"[Internal Feeling]: {feeling}")

        except Exception as e:
            print(f"[Critic Error] {e}")
