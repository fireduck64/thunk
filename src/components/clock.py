from datetime import datetime
from src.components.base import BaseComponent

class ClockComponent(BaseComponent):
    """
    Injects the current real-world date and time into the agent's prompt
    so it is always aware of when it is operating.
    """
    def get_system_prompt_addition(self) -> str:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        day_of_week = now.strftime("%A")
        
        return (
            f"--- CURRENT TIME ---\n"
            f"The current system time is: {current_time} ({day_of_week}).\n"
        )
