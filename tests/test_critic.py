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


def test_discord_expand_user_and_role_mentions():
    """Verifies DiscordAdapter._expand_mentions expands user mentions (<@ID>, <@!ID>) and role mentions (<@&ID>)."""
    from src.adapters.discord_adapter import DiscordAdapter

    class DummyUser:
        def __init__(self, uid: int, display_name: str):
            self.id = uid
            self.display_name = display_name

    class DummyRoleTags:
        def __init__(self, bot_id: int = None):
            self.bot_id = bot_id

    class DummyRole:
        def __init__(self, rid: int, name: str, bot_id: int = None):
            self.id = rid
            self.name = name
            self.tags = DummyRoleTags(bot_id=bot_id)

    class DummyGuild:
        def __init__(self, roles_by_id: dict):
            self._roles = roles_by_id

        def get_role(self, rid: int):
            return self._roles.get(rid)

        def get_member(self, uid: int):
            return None

    class DummyAgent:
        name = "Thunk"

    adapter = DiscordAdapter(DummyAgent())
    self_user = DummyUser(1551318628543234190, "Thunk")
    wisp_user = DummyUser(1560498996789645402, "Wisp")
    adapter.client._connection.user = self_user

    thunk_role = DummyRole(1551396165621387286, "Thunk", bot_id=1551318628543234190)
    wisp_role = DummyRole(1560504734715088969, "Wisp", bot_id=1560498996789645402)
    stuffle_role = DummyRole(1560665089407979572, "Stuffleupagus")
    guild = DummyGuild({stuffle_role.id: stuffle_role})

    raw = (
        "Ok <@&1560504734715088969> and <@&1551396165621387286>, "
        "plus <@!1560498996789645402> and <@&1560665089407979572> April wants to know if you could see her artwork."
    )
    expanded = adapter._expand_mentions(
        raw,
        mentions=[wisp_user],
        role_mentions=[wisp_role, thunk_role],
        guild=guild,
    )

    assert "@Wisp (<@&1560504734715088969>)" in expanded
    assert "@Thunk (you, <@&1551396165621387286>)" in expanded
    assert "@Wisp (<@1560498996789645402>)" in expanded
    assert "@Stuffleupagus (<@&1560665089407979572>)" in expanded


@pytest.mark.asyncio
async def test_x_agent_id_headers(monkeypatch):
    """Verifies X-Agent-ID: <agent_name>-<component> headers are attached on all LLM and embedding clients/requests."""
    import requests
    from src.core.agent import AgentCore
    from src.components.subconscious import SubconsciousRecallComponent
    from src.components.spreading_activation import SpreadingActivationComponent
    from src.components.vector_memory import VectorMemoryComponent
    from src.components.knowledge import KnowledgeBaseComponent

    agent = AgentCore()
    assert agent.client.default_headers.get("X-Agent-ID") == f"{agent.name}-core"

    sub = SubconsciousRecallComponent()
    assert sub.client.default_headers.get("X-Agent-ID") == f"{agent.name}-subconscious"

    sa = SpreadingActivationComponent(persist=False)
    assert sa.client.default_headers.get("X-Agent-ID") == f"{agent.name}-spreading_activation"

    # Critic extra_headers check
    captured_critic_kwargs = {}

    class DummyMsg:
        content = "I notice steady progress."

    class DummyCompletions:
        async def create(self, **kwargs):
            captured_critic_kwargs.update(kwargs)
            return "stream"

    critic = CriticComponent(frequency=1)
    agent.register_component(critic)
    agent.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    monkeypatch.setattr(agent.client.chat, "completions", DummyCompletions())
    monkeypatch.setattr(agent, "_accumulate_stream", lambda s: DummyMsg())

    async def mock_acc(s):
        return DummyMsg()

    agent._accumulate_stream = mock_acc
    await critic._evaluate_progress()
    assert captured_critic_kwargs.get("extra_headers") == {"X-Agent-ID": f"{agent.name}-critic"}

    # Embeddings requests.post headers check
    captured_headers = []

    class DummyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"embedding": [0.1] * 768}

    def fake_post(url, json=None, headers=None, **kwargs):
        captured_headers.append(dict(headers or {}))
        return DummyResp()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(VectorMemoryComponent, "_ensure_collection", lambda self: None)

    vm = VectorMemoryComponent(collection_name="test_agent_memory")
    vm._get_embedding("hello")
    assert captured_headers[-1].get("X-Agent-ID") == f"{agent.name}-vector_memory"

    kb = KnowledgeBaseComponent()
    kb._get_embedding("world")
    assert captured_headers[-1].get("X-Agent-ID") == f"{agent.name}-knowledge"




