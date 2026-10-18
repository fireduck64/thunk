import pytest
from src.components.critic import CriticComponent


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

