import sqlite3
from typing import List, Callable
from src.components.base import BaseComponent

class TaskQueueComponent(BaseComponent):
    """
    Provides a priority-based task queue for the agent to manage its own 
    long-term goals and multi-step operations.
    """
    def __init__(self, db_path: str = "tasks.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute('''CREATE TABLE IF NOT EXISTS task_queue (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                priority INTEGER,
                                description TEXT
                              )''')

        finally:
            conn.close()
    def get_system_prompt_addition(self) -> str:
        return (
            "--- TASK MANAGEMENT ---\n"
            "You have a priority task queue to manage your long-term and multi-step goals.\n"
            "Use `add_task` to remember future work. Higher priority numbers are executed first (e.g., 10 is higher than 1).\n"
            "Use `pop_highest_priority_task` when you are ready for a new objective, or to get your next step.\n"
            "Use `list_tasks` to see what is currently in your backlog."
        )

    def get_tools(self) -> List[Callable]:
        return [self.add_task, self.pop_highest_priority_task, self.list_tasks]

    def add_task(self, description: str, priority: int) -> str:
        """Adds a task to your queue. Higher priority numbers are executed first."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("INSERT INTO task_queue (priority, description) VALUES (?, ?)", (priority, description))

        finally:
            conn.close()
        # Tell the agent core to update its prompt since the queue changed
        if hasattr(self, 'agent') and self.agent:
            self.agent.rebuild_system_prompt()
            
        return f"Task added successfully with priority {priority}."

    def pop_highest_priority_task(self) -> str:
        """Removes and returns the highest priority task from the queue so you can work on it."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute("SELECT id, priority, description FROM task_queue ORDER BY priority DESC, id ASC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    conn.execute("DELETE FROM task_queue WHERE id = ?", (row[0],))

                    if hasattr(self, 'agent') and self.agent:
                        self.agent.rebuild_system_prompt()

                    return f"Popped Task (Priority {row[1]}): {row[2]}"
                return "The task queue is empty."

        finally:
            conn.close()
    def list_tasks(self) -> str:
        """Returns the top 10 highest priority tasks currently in the queue."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute("SELECT priority, description FROM task_queue ORDER BY priority DESC, id ASC LIMIT 10")
                rows = cursor.fetchall()
                if not rows:
                    return "The task queue is empty."
                return "\n".join([f"- [Priority {r[0]}] {r[1]}" for r in rows])
        finally:
            conn.close()
