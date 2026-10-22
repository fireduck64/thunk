import os
import pytest
from src.utils.config import load_config
from src.components.critic import CriticComponent
from src.components.tasks import TaskQueueComponent


@pytest.mark.asyncio
async def test_critic_trigger_frequency():
    """Verifies CriticComponent tracks tool_executed events and flags needs_evaluation at the configured frequency."""
    critic = CriticComponent(frequency=3)
    assert not critic.needs_evaluation

    await critic.on_event("tool_executed", {"name": "read_next_chunk"})
    assert critic.action_count == 1
    assert not critic.needs_evaluation

    await critic.on_event("tool_executed", {"name": "read_next_chunk"})
    assert critic.action_count == 2
    assert not critic.needs_evaluation

    await critic.on_event("tool_executed", {"name": "read_next_chunk"})
    assert critic.action_count == 0
    assert critic.needs_evaluation


def test_task_queue_ids_and_delete():
    """Verifies TaskQueueComponent includes task IDs in list_tasks and supports delete_task(task_id)."""
    db_path = load_config()["tasks"]["db_path"]
    if os.path.exists(db_path):
        os.remove(db_path)

    tq = TaskQueueComponent(db_path=db_path)
    try:
        res1 = tq.add_task("Audit the clacks towers", priority=5)
        res2 = tq.add_task("Read Going Postal chapter 2", priority=10)
        assert "ID: 1" in res1
        assert "ID: 2" in res2

        listed = tq.list_tasks()
        assert "[ID: 2 | Priority 10] Read Going Postal chapter 2" in listed
        assert "[ID: 1 | Priority 5] Audit the clacks towers" in listed

        # Delete task 1 by ID
        del_res = tq.delete_task(1)
        assert "Deleted Task (ID: 1, Priority 5): Audit the clacks towers" in del_res

        # Deleting a non-existent task returns an error
        assert "not found" in tq.delete_task(999)

        # Only task 2 remains
        listed_after = tq.list_tasks()
        assert "ID: 1" not in listed_after
        assert "[ID: 2 | Priority 10] Read Going Postal chapter 2" in listed_after
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


