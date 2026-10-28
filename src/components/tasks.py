import os
import sqlite3
from typing import List, Callable
from src.components.base import BaseComponent
from src.utils.config import load_config

class TaskQueueComponent(BaseComponent):
    """
    Provides a priority-based task queue for the agent to manage its own 
    long-term goals and multi-step operations.
    """
    def __init__(self, db_path: str = None):
        if db_path is None:
            config = load_config()
            tasks_cfg = config.get("tasks", {})
            if "db_path" not in tasks_cfg or not tasks_cfg["db_path"]:
                raise ValueError("TaskQueueComponent requires 'db_path' to be explicitly defined in config under [tasks].db_path")
            self.db_path = tasks_cfg["db_path"]
        else:
            self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self):
        conn = self._connect_db()
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
            "Use `list_tasks` to see what is currently in your backlog (with each task's ID).\n"
            "Use `delete_task` with a `task_id` from `list_tasks` to remove a specific task."
        )

    def get_tools(self) -> List[Callable]:
        return [self.add_task, self.pop_highest_priority_task, self.list_tasks, self.delete_task]

    def add_task(self, description: str, priority: int) -> str:
        """Adds a task to your queue. Higher priority numbers are executed first."""
        conn = self._connect_db()
        try:
            with conn:
                cursor = conn.execute(
                    "INSERT INTO task_queue (priority, description) VALUES (?, ?)",
                    (priority, description),
                )
                task_id = cursor.lastrowid
        finally:
            conn.close()
        return f"Task added successfully (ID: {task_id}, Priority {priority})."

    def pop_highest_priority_task(self) -> str:
        """Removes and returns the highest priority task from the queue so you can work on it."""
        conn = self._connect_db()
        try:
            with conn:
                cursor = conn.execute("SELECT id, priority, description FROM task_queue ORDER BY priority DESC, id ASC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    conn.execute("DELETE FROM task_queue WHERE id = ?", (row[0],))
                    return f"Popped Task (ID: {row[0]}, Priority {row[1]}): {row[2]}"
                return "The task queue is empty."

        finally:
            conn.close()

    def delete_task(self, task_id: int) -> str:
        """Deletes a specific task from the queue by its ID number."""
        conn = self._connect_db()
        try:
            with conn:
                cursor = conn.execute(
                    "SELECT id, priority, description FROM task_queue WHERE id = ?",
                    (task_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return f"Error: Task with ID {task_id} not found."
                conn.execute("DELETE FROM task_queue WHERE id = ?", (task_id,))
                return f"Deleted Task (ID: {row[0]}, Priority {row[1]}): {row[2]}"
        finally:
            conn.close()

    def list_tasks(self) -> str:
        """Returns the top 10 highest priority tasks currently in the queue, including each task's ID."""
        conn = self._connect_db()
        try:
            with conn:
                cursor = conn.execute("SELECT id, priority, description FROM task_queue ORDER BY priority DESC, id ASC LIMIT 10")
                rows = cursor.fetchall()
                if not rows:
                    return "The task queue is empty."
                return "\n".join([f"- [ID: {r[0]} | Priority {r[1]}] {r[2]}" for r in rows])
        finally:
            conn.close()
