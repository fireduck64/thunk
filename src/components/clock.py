from datetime import datetime
from src.components.base import BaseComponent

class ClockComponent(BaseComponent):
    """
    Injects the current real-world date and time into the agent's prompt
    so it is always aware of when it is operating.
    """
    def get_system_prompt_addition(self) -> str:
        now = datetime.now()
        # Truncate minutes, seconds, and microseconds to zero to maximize
        # llama.cpp prefix caching, keeping the prompt static for a full hour.
        now_truncated = now.replace(minute=0, second=0, microsecond=0)
        current_time = now_truncated.strftime("%Y-%m-%d %H:%M:%S")
        day_of_week = now_truncated.strftime("%A")
        
        return (
            f"--- CURRENT TIME ---\n"
            f"The current system time is: {current_time} ({day_of_week}).\n"
        )
