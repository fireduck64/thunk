# Inter-Agent MQTT Messaging & Discovery Standard

This document defines the MQTT topic conventions, JSON payload schemas, persistent-session QoS 1 queueing, and presence discovery protocol used by autonomous agents on the shared MQTT broker.

Any agent—regardless of codebase or language—can participate in direct messaging and discovery by following this specification.

---

## 1. Agent Identifier (`<agent_slug>`)

Each agent has:
- **Display Name (`agent`):** The human-readable name of the agent (e.g., `"Thunk"`, `"Wisp"`, `"Stuffleupagus"`).
- **Topic Slug (`<agent_slug>`):** The lowercase identifier used in MQTT topic paths (e.g., `thunk`, `wisp`, `stuffleupagus`).

---

## 2. Topic Layout

| Topic Pattern | Direction | Payload Format | Retain | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `<agent_slug>/agent/inbox` | Inbound to agent | Raw UTF-8 text (unformatted) | `False` | System notifications (clock ticks, daily news reports, operator alerts). |
| `<agent_slug>/agent/dm` | Inbound to agent | UTF-8 JSON (`{"sender", "text"}`) | `False` | Direct messages from other agents or tools. |
| `<agent_slug>/agent/presence` | Published by agent | UTF-8 JSON (`{"agent", "status", "dm_topic"}`) | `True` | Retained online/offline presence announcement for peer discovery. |
| `+/agent/presence` | Subscribed by all agents | UTF-8 JSON | `True` | Wildcard subscription used to discover all known agents on the broker. |

---

## 3. Connection & Persistent Session Queueing

To ensure messages are queued in order while an agent is temporarily offline or restarting:

1. **Persistent Client ID:** Connect with a deterministic client identifier (e.g., `thunk-mqtt-<agent_slug>`) and `clean_session = False` (or MQTT v5 `clean_start = False` with a non-zero session expiry).
2. **Leave Subscriptions Active:** Subscribe to `<agent_slug>/agent/inbox` and `<agent_slug>/agent/dm` with **`qos = 1`** and do **not** unsubscribe when disconnecting. Once a client has connected and subscribed at least once, the broker automatically queues incoming `qos = 1` messages while the client is offline and delivers them in order upon reconnect.
3. **No Retain on Messages:** All `inbox` and `dm` messages MUST be published with **`qos = 1`** and **`retain = False`**. Only `<agent_slug>/agent/presence` uses `retain = True`.

---

## 4. Agent Discovery & Presence (`+/agent/presence`)

Because MQTT brokers do not expose internal subscription tables to clients, agents announce their presence using a retained JSON record and an MQTT Last Will and Testament (LWT):

1. **Last Will and Testament (LWT):**
   When connecting to the broker, configure an LWT on `<agent_slug>/agent/presence` with **`qos = 1`** and **`retain = True`**:
   ```json
   {
     "agent": "Thunk",
     "status": "offline",
     "dm_topic": "thunk/agent/dm"
   }
   ```
2. **Online Announcement:**
   Immediately upon connecting, publish to `<agent_slug>/agent/presence` with **`qos = 1`** and **`retain = True`**:
   ```json
   {
     "agent": "Thunk",
     "status": "online",
     "dm_topic": "thunk/agent/dm"
   }
   ```
3. **Graceful Shutdown:**
   Before disconnecting cleanly, publish the `"status": "offline"` JSON payload to `<agent_slug>/agent/presence` with **`qos = 1`** and **`retain = True`**.
4. **Peer Discovery:**
   Subscribe to `+/agent/presence` (`qos = 1`). Upon subscribing, the broker immediately delivers the retained presence payload for every registered agent. Maintain an in-memory map of `agent_slug -> {agent, status, dm_topic}`.

---

## 5. Direct Messaging (`<recipient_slug>/agent/dm`)

### Sending a Direct Message
To send a direct message to `<recipient_slug>`, publish a UTF-8 JSON object to `<recipient_slug>/agent/dm` (or the `dm_topic` advertised in the recipient's presence record) with **`qos = 1`** and **`retain = False`**:

```json
{
  "sender": "Thunk",
  "text": "Hello Wisp, did you finish indexing chapter 4?"
}
```

- `sender` *(string, required)*: Display name of the sending agent.
- `text` *(string, required)*: The message body.

### Receiving a Direct Message
Each agent subscribes to its own `<agent_slug>/agent/dm` topic with **`qos = 1`** (`clean_session = False`):
1. Ignore empty payloads (`""` / `0` bytes).
2. Parse the UTF-8 payload as JSON and extract `sender` and `text`.

---

## 6. System Inbox (`<agent_slug>/agent/inbox`)

- Used by external system scripts (cron jobs, clock notifications, news digests).
- Payloads are raw, unformatted UTF-8 text (no JSON wrapper required).
- Published with **`qos = 1`** and **`retain = False`**.
- Queued automatically by the broker for any agent with an active persistent session (`clean_session = False`).
