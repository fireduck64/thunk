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
    """Verifies X-Agent-ID: <agent_name>-<thread_name>-<component> headers are attached on all LLM and embedding clients/requests."""
    import requests
    from src.core.agent import AgentCore
    from src.components.subconscious import SubconsciousRecallComponent
    from src.components.spreading_activation import SpreadingActivationComponent
    from src.components.vector_memory import VectorMemoryComponent
    from src.components.knowledge import KnowledgeBaseComponent

    agent = AgentCore()
    assert agent.client.default_headers.get("X-Agent-ID") == f"{agent.name}-main-core"

    reader_agent = AgentCore(thread_id="reader")
    assert reader_agent.client.default_headers.get("X-Agent-ID") == f"{reader_agent.name}-reader-core"

    sub = SubconsciousRecallComponent(thread_id="social")
    assert sub.client.default_headers.get("X-Agent-ID") == f"{agent.name}-social-subconscious"

    sa = SpreadingActivationComponent(persist=False, thread_id="reader")
    assert sa.client.default_headers.get("X-Agent-ID") == f"{agent.name}-reader-spreading_activation"

    # Critic extra_headers check
    captured_critic_kwargs = {}

    class DummyMsg:
        content = "I notice steady progress."

    class DummyCompletions:
        async def create(self, **kwargs):
            captured_critic_kwargs.update(kwargs)
            return "stream"

    critic = CriticComponent(frequency=1)
    reader_agent.register_component(critic)
    reader_agent.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    monkeypatch.setattr(reader_agent.client.chat, "completions", DummyCompletions())

    async def mock_acc(s):
        return DummyMsg()

    reader_agent._accumulate_stream = mock_acc
    await critic._evaluate_progress()
    assert captured_critic_kwargs.get("extra_headers") == {"X-Agent-ID": f"{reader_agent.name}-reader-critic"}

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

    vm = VectorMemoryComponent(collection_name="test_agent_memory", thread_id="worker")
    vm._get_embedding("hello")
    assert captured_headers[-1].get("X-Agent-ID") == f"{agent.name}-worker-vector_memory"

    kb = KnowledgeBaseComponent(thread_id="reader")
    kb._get_embedding("world")
    assert captured_headers[-1].get("X-Agent-ID") == f"{agent.name}-reader-knowledge"


@pytest.mark.asyncio
async def test_mqtt_direct_messaging_system_inbox_and_presence():
    """Verifies MQTT system inbox, JSON DM wrapper + retained-until-consumed clearing, and +/agent/presence discovery."""
    import json
    from src.core.agent import AgentCore
    from src.components.system import SystemComponent
    from src.adapters.mqtt import MQTTAdapter

    agent = AgentCore()
    sys_comp = SystemComponent()
    agent.register_component(sys_comp)
    mqtt_adapter = MQTTAdapter(agent)
    agent.event_bus.subscribe("send_to_operator", mqtt_adapter._on_send_to_operator)

    published = []

    class DummyMqttClient:
        async def publish(self, topic, payload, qos=0, retain=False):
            published.append((topic, payload, qos, retain))

    mqtt_adapter.client = DummyMqttClient()

    # 1. Sending via mqtt without recipient returns an error
    err_res = await sys_comp.send_message("Hello?", medium="mqtt")
    assert "Recipient is required" in err_res
    assert len(published) == 0

    # 2. Sending via mqtt with recipient wraps JSON {"sender", "text"} with qos=1, retain=True to <recipient>/agent/dm
    ok_res = await sys_comp.send_message("Meet at the tower", medium="mqtt", recipient="Wisp")
    assert "Direct message sent successfully via mqtt to 'Wisp' (wisp/agent/dm)" in ok_res
    assert len(published) == 1
    topic, payload_bytes, qos, retain = published[0]
    assert topic == "wisp/agent/dm"
    assert qos == 1
    assert retain is True
    assert json.loads(payload_bytes.decode("utf-8")) == {
        "sender": agent.name,
        "text": "Meet at the tower",
    }

    # 3. Presence discovery populates list_mqtt_agents
    await mqtt_adapter._handle_message(
        "stuffleupagus/agent/presence",
        json.dumps({"agent": "Stuffleupagus", "status": "online", "dm_topic": "stuffleupagus/agent/dm"}),
        retain=True,
    )
    agents_list = sys_comp.list_mqtt_agents()
    assert "Stuffleupagus (recipient='stuffleupagus', status=online, dm_topic='stuffleupagus/agent/dm')" in agents_list

    # 4. Incoming DM clears retained message (b"", qos=1, retain=True) and resumes agent with formatted DM
    published.clear()
    await mqtt_adapter._handle_message(
        mqtt_adapter.dm_topic,
        json.dumps({"sender": "Wisp", "text": "The clacks are live!"}),
        retain=True,
    )
    assert published == [(mqtt_adapter.dm_topic, b"", 1, True)]
    inbox_msg = await agent.inbox.get()
    assert inbox_msg == "[Via MQTT | DM from Wisp] The clacks are live!"

    # Empty retained-clear payload on dm_topic is ignored
    await mqtt_adapter._handle_message(mqtt_adapter.dm_topic, "", retain=True)
    assert agent.inbox.empty()

    # 5. Incoming unformatted system inbox message wakes agent as [Via MQTT | System]
    published.clear()
    await mqtt_adapter._handle_message(mqtt_adapter.inbox_topic, "clock notification\nFri Oct 23", retain=True)
    assert published == [(mqtt_adapter.inbox_topic, b"", 1, True)]
    sys_msg = await agent.inbox.get()
    assert sys_msg == "[Via MQTT | System] clock notification\nFri Oct 23"


def test_mqtt_chat_script_helpers_and_cli():
    """Verifies scripts/mqtt-chat.py presence/DM payload builders, topic resolution, and CLI flags."""
    import importlib.util
    import json
    import os
    import subprocess
    import sys

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "mqtt-chat.py"))
    spec = importlib.util.spec_from_file_location("mqtt_chat", script_path)
    mqtt_chat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mqtt_chat)

    pres = json.loads(mqtt_chat.build_presence_payload("Sombus", "online", "sombus/agent/dm").decode("utf-8"))
    assert pres == {"agent": "Sombus", "status": "online", "dm_topic": "sombus/agent/dm"}

    dm = json.loads(mqtt_chat.build_dm_payload("Sombus", "Hello Thunk!").decode("utf-8"))
    assert dm == {"sender": "Sombus", "text": "Hello Thunk!"}

    known = {"wisp": {"agent": "Wisp", "status": "online", "dm_topic": "custom/wisp/dm"}}
    assert mqtt_chat.resolve_recipient_dm_topic("@Wisp", known) == "custom/wisp/dm"
    assert mqtt_chat.resolve_recipient_dm_topic("Thunk", known) == "thunk/agent/dm"

    help_out = subprocess.check_output([sys.executable, script_path, "--help"], text=True)
    assert "--user" in help_out
    assert "--recipient" in help_out






