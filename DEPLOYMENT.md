# Thunk: Production Deployment Guide

This document outlines the deployment architecture, service dependencies, configuration, and operations for running Thunk in a reliable, long-running production environment.

---

## 1. System Architecture & Prerequisites

### Host Requirements
* **Operating System:** Linux (Ubuntu 22.04+ or Debian 12+ recommended).
* **Python:** Python 3.11+ with standard development libraries (`python3-venv`, `python3-full`).
* **Container Runtime:** Docker Engine and Docker Compose plugin.
* **LLM Endpoint:** Access to an OpenAI-compatible API endpoint (e.g., local Ollama instance on GPU, vLLM, or remote endpoint).
* **Network:** Outbound HTTPS access for LLM/Discord APIs, local network access to MQTT broker and Qdrant.

---

## 2. Storage & State Management

Thunk isolates code from per-agent runtime state (e.g., `data/thunk/`):

| State Store | Purpose | Typical Location |
| :--- | :--- | :--- |
| **Qdrant Vector DB** | Knowledge base and long-term semantic memory | Docker named volume (`qdrant_data`) |
| **`memory.db`** | Archival interaction log, core summary, and compaction state | `data/<agent>/memory.db` |
| **`notes.db`** | Structured key-value notes and sticky prompt notes | `data/<agent>/notes.db` |
| **`tasks.db`** | Priority task queue backlog | `data/<agent>/tasks.db` |
| **`reader_state.db`** | Persistent bookmark position for the document reader | `data/<agent>/reader_state.db` |
| **`skills/*.md`** | Agent-authored and operator-authored procedural skills | `data/<agent>/skills/` |
| **`logs/*.jsonl`** | Deterministic session audit logs | `data/<agent>/logs/` |

---

## 3. Container Services (Docker Compose)

The vector database is managed via Docker Compose (`docker-compose.yml`):

```bash
# Start Qdrant vector database
docker compose up -d

# Verify Qdrant is listening on port 6333
curl -s http://localhost:6333/readyz
```

Persistent vector data is saved to the `qdrant_data` Docker volume to survive container updates.

---

## 4. Virtual Environment Isolation

All scripts reference `./venv` in the project root (which is ignored by git):

* **Local Default:** Running `./scripts/setup.sh` creates `./venv` directly in the project root.
* **External / Shared Path:** If `THUNK_VENV_DIR` is set (e.g., `export THUNK_VENV_DIR=/var/nvme/thunk_venv`), `./scripts/setup.sh` provisions the virtual environment at `$THUNK_VENV_DIR` and creates a `./venv` symlink pointing to it.

Run the automated setup script to provision the environment and install all dependencies:
```bash
./scripts/setup.sh
```

---

## 5. Configuration (`data/<agent>/config.toml`)

Create an agent directory and copy the template configuration file and starter skills:
```bash
mkdir -p data/thunk/logs
cp -r skills data/thunk/skills
cp config.example.toml data/thunk/config.toml
chmod 600 data/thunk/config.toml
```

Configure the following sections in `data/thunk/config.toml`:
1. **`[llm]`**: Configure the inference endpoint URL and model name (e.g. `gemma4:31b` or `llama3`).
2. **`[tokenizer]`**: Set the HuggingFace tokenizer model path (`unsloth/gemma-7b`) used for memory watermark counting.
3. **`[memory]` / `[notes]` / `[tasks]` / `[reader]` / `[skills]` / `[logging]`**: Set per-agent database and directory paths.
4. **`[embeddings]`**: Point to your Ollama embeddings endpoint (`nomic-embed-text`).
5. **`[vectordb]`**: Configure host, port, `knowledge_collection`, and per-agent `memory_collection` for Qdrant (`localhost:6333`).
6. **`[mqtt]`**: Set broker credentials and topics for operator communication.
7. **`[discord]`**: Add your bot token and optional allowed channel ID.
8. **`[oracle]`**: Add your Google Gemini API key (`gemini_api_key`) and model (`gemini-3.8-flash`).
9. **`[debug]`**: Set `stream_output = true` to enable real-time terminal output.

---

## 6. Running Thunk

### Interactive / Development Mode
Launch the agent directly, specifying the configuration file:
```bash
./src/main.py -c data/thunk/config.toml
```

### Systemd Production Service
To ensure Thunk runs continuously in the background and restarts automatically on reboot or failure, create a `systemd` service:

Create `/etc/systemd/system/thunk.service`:
```ini
[Unit]
Description=Thunk Autonomous Agent Core
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=fireduck
WorkingDirectory=/home/fireduck/projects/thunk
ExecStart=/home/fireduck/projects/thunk/src/main.py -c data/thunk/config.toml
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable thunk.service
sudo systemctl start thunk.service
```

---

## 7. Operations & Monitoring

* **View Service Logs:**
  ```bash
  journalctl -u thunk.service -f
  ```
* **Format & Inspect Session Logs:**
  ```bash
  ./scripts/extract-monolog.py data/thunk/logs/session_YYYYMMDD_HHMMSS.jsonl
  ```
* **Send Out-of-Band Notification via MQTT:**
  ```bash
  echo "System alert: Database backup finished." | ./scripts/send-notification.py -c data/thunk/config.toml
  ```
* **Query the Cloud AI Oracle via CLI:**
  ```bash
  echo "What is the latest status of project X?" | ./scripts/ask-oracle.py -c data/thunk/config.toml
  ```

---

## 8. Backup & Restore

To take a complete snapshot of the agent's state ("brain" and knowledge):

1. **Agent Data Directory (SQLite Databases, Skills, Config):**
   ```bash
   tar -czvf thunk_state_backup.tar.gz data/
   ```
2. **Qdrant Vector Data:**
   ```bash
   docker compose stop vectordb
   docker run --rm -v qdrant_data:/volume -v $(pwd):/backup alpine tar -czf /backup/qdrant_backup.tar.gz -C /volume .
   docker compose start vectordb
   ```
