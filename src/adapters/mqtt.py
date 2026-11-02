import asyncio
import base64
import json
import mimetypes
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import aiomqtt
from src.core.agent import AgentCore
from src.utils.config import load_config
from src.utils.image import downsize_image_bytes, encode_jpeg_data_url, save_image_atomic

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
DOCUMENT_EXTENSIONS = {".pdf", ".epub"}
MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "application/pdf": ".pdf",
    "application/epub+zip": ".epub",
    "application/json": ".json",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
}


class MQTTAdapter:
    """
    Connects the AgentCore to an MQTT broker.
    - Listens on a system inbox topic (`<agent>/agent/inbox`) for unformatted system events or JSON multimedia messages.
    - Listens on a direct message topic (`<agent>/agent/dm`) for JSON agent-to-agent messages
      (`{"sender", "text", "filename", "file_mime_type", "file_bytes"}`).
    - Publishes and subscribes to `+/agent/presence` with retained LWT so agents can discover peers.
    """

    def __init__(self, agent: AgentCore):
        self.agent = agent
        self.config = load_config()
        self.agent_name = getattr(agent, "name", None) or self.config.get("agent", {}).get("name", "Thunk")
        self.agent_slug = self.agent_name.strip().lower()

        mqtt_cfg = self.config.get("mqtt", {})
        self.host = mqtt_cfg.get("host", "localhost")
        self.port = mqtt_cfg.get("port", 1883)
        self.inbox_topic = mqtt_cfg.get("inbox_topic", f"{self.agent_slug}/agent/inbox")
        self.dm_topic = mqtt_cfg.get("dm_topic", f"{self.agent_slug}/agent/dm")
        self.dm_topic_template = mqtt_cfg.get("dm_topic_template", "{recipient}/agent/dm")
        self.presence_topic = mqtt_cfg.get("presence_topic", f"{self.agent_slug}/agent/presence")
        self.presence_wildcard = mqtt_cfg.get("presence_wildcard", "+/agent/presence")
        self.outbox_topic = mqtt_cfg.get("outbox_topic", f"{self.agent_slug}/agent/outbox")
        self.client_id = mqtt_cfg.get("client_id", f"thunk-mqtt-{self.agent_slug}")

        discord_cfg = self.config.get("discord", {})
        libs_cfg = self.config.get("file_reader", {}).get("libraries", {})
        self.images_dir = (
            mqtt_cfg.get("images_dir")
            or discord_cfg.get("images_dir")
            or libs_cfg.get("discord_file")
            or libs_cfg.get("discord_images")
        )
        if self.images_dir:
            os.makedirs(self.images_dir, exist_ok=True)

        self.client = None
        self.known_agents: dict[str, dict] = {}
        self.agent.mqtt_adapter = self

    def resolve_dm_topic(self, recipient: str) -> str:
        """Resolves a recipient agent name to its MQTT direct-message topic."""
        slug = recipient.strip().lstrip("@").lower()
        if slug in self.known_agents and self.known_agents[slug].get("dm_topic"):
            return self.known_agents[slug]["dm_topic"]
        return self.dm_topic_template.format(recipient=slug)

    def _resolve_library_attachment(self, attachment_library: str, attachment_path: str) -> str:
        """Resolves and validates an attachment path inside a configured library (with path-traversal protection)."""
        raw_libs = dict(self.config.get("file_reader", {}).get("libraries") or {})
        ws_dir = self.config.get("workspace", {}).get("workspace_dir")
        if ws_dir and "workspace" not in raw_libs:
            raw_libs["workspace"] = ws_dir

        aliases = {
            "ebooks": "ebook",
            "ebook": "ebooks",
            "pdfs": "pdf",
            "pdf": "pdfs",
            "discord_images": "discord_file",
            "discord_files": "discord_file",
            "discord_file": "discord_images",
            "shared": "workspace",
            "shared_workspace": "workspace",
        }

        lib_key = (attachment_library or "").strip()
        lower_lib_key = lib_key.lower()
        resolved_lib = None
        if lib_key in raw_libs:
            resolved_lib = lib_key
        else:
            for k in raw_libs:
                if k.lower() == lower_lib_key:
                    resolved_lib = k
                    break
            if resolved_lib is None and lower_lib_key in aliases and aliases[lower_lib_key] in raw_libs:
                resolved_lib = aliases[lower_lib_key]

        if not resolved_lib or not raw_libs.get(resolved_lib):
            raise ValueError(
                f"Library '{attachment_library}' is not configured. Available libraries: {list(raw_libs.keys())}"
            )

        base_dir = os.path.abspath(raw_libs[resolved_lib])
        clean_rel = (attachment_path or "").strip()
        if "[path: " in clean_rel and clean_rel.endswith("]"):
            clean_rel = clean_rel.split("[path: ", 1)[1][:-1].strip()

        container_ws_prefix = (
            self.config.get("workspace", {}).get("container_workdir", "/home/agent/workspace").rstrip("/") + "/"
        )
        if resolved_lib.lower() == "workspace" and clean_rel.startswith(container_ws_prefix):
            clean_rel = clean_rel[len(container_ws_prefix) :]

        if lower_lib_key in ("shared", "shared_workspace") and not (
            clean_rel == "shared" or clean_rel.startswith("shared/")
        ):
            clean_rel = f"shared/{clean_rel.lstrip('/')}"

        safe_rel = clean_rel.lstrip(os.sep)
        if not safe_rel:
            raise ValueError("attachment_path must not be empty.")

        if resolved_lib.lower() == "workspace" and (safe_rel == "shared" or safe_rel.startswith("shared/")):
            ws_cfg = self.config.get("workspace", {})
            raw_shared = ws_cfg.get("shared_workspace_dir")
            if not raw_shared:
                if os.path.basename(os.path.dirname(base_dir)) == "test":
                    raw_shared = os.path.join(os.path.dirname(base_dir), "shared_workspace")
                else:
                    raw_shared = "data/shared/workspace"
            base_dir = os.path.abspath(raw_shared)
            safe_rel = safe_rel[len("shared") :].lstrip(os.sep)
            if not safe_rel:
                raise ValueError("attachment_path must reference a file inside shared/, not the directory itself.")

        abs_path = os.path.abspath(os.path.join(base_dir, safe_rel))
        if os.path.commonpath([base_dir, abs_path]) != base_dir:
            raise ValueError(f"Access denied: '{attachment_path}' is outside of library '{resolved_lib}'.")
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Attachment '{attachment_path}' does not exist in library '{resolved_lib}'.")
        if not os.path.isfile(abs_path):
            raise ValueError(f"Attachment '{attachment_path}' in library '{resolved_lib}' is not a regular file.")
        return abs_path

    @staticmethod
    def _summarize_payload_for_log(payload_raw: str) -> str:
        """Sanitizes JSON payloads containing `file_bytes` so base64 blobs do not flood console logs."""
        try:
            data = json.loads(payload_raw)
            if isinstance(data, dict) and "file_bytes" in data and data["file_bytes"]:
                summary = dict(data)
                b64_len = len(str(data["file_bytes"]))
                summary["file_bytes"] = f"<base64 {b64_len} chars>"
                return json.dumps(summary)
        except Exception:
            pass
        return payload_raw

    def _process_incoming_attachment(self, data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Decodes base64 `file_bytes` from an incoming MQTT JSON payload, downsizes images into
        ephemeral vision data URLs, and saves files atomically to `self.images_dir`.
        Returns (attached_images, attachment_notes).
        """
        raw_b64 = data.get("file_bytes")
        if not raw_b64 or not isinstance(raw_b64, str):
            return [], []

        b64_str = raw_b64.strip()
        inferred_mime = ""
        if b64_str.startswith("data:") and "," in b64_str:
            header, b64_str = b64_str.split(",", 1)
            if ";" in header and ":" in header:
                inferred_mime = header.split(":", 1)[1].split(";", 1)[0].strip().lower()

        mime_type = (str(data.get("file_mime_type") or inferred_mime or "").strip().lower())
        raw_filename = str(data.get("filename") or "").strip()
        ext = os.path.splitext(raw_filename.lower())[1] if raw_filename else ""
        if not ext and mime_type:
            ext = MIME_TO_EXT.get(mime_type) or (mimetypes.guess_extension(mime_type) or "")
        if not mime_type and ext:
            guessed, _ = mimetypes.guess_type(f"file{ext}")
            mime_type = (guessed or "application/octet-stream").lower()
        if not mime_type:
            mime_type = "application/octet-stream"

        stem = os.path.splitext(os.path.basename(raw_filename))[0] if raw_filename else "attachment"
        safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "attachment"
        display_name = raw_filename or f"{safe_stem}{ext or ''}"
        ts_ms = int(time.time() * 1000)

        try:
            raw_bytes = base64.b64decode(b64_str, validate=False)
        except Exception as e:
            return [], [f"[Attached File: '{display_name}' (type: {mime_type} — failed to decode base64 file_bytes: {e})]"]

        attached_images: List[str] = []
        attachment_notes: List[str] = []

        if mime_type.startswith("image/") or ext in IMAGE_EXTENSIONS:
            try:
                jpeg_bytes, width, height = downsize_image_bytes(raw_bytes, max_edge=1024)
                data_url = encode_jpeg_data_url(jpeg_bytes)
                attached_images.append(data_url)

                saved_name = f"mqtt_{ts_ms}_{safe_stem}.jpg"
                if self.images_dir:
                    dest_path = os.path.join(self.images_dir, saved_name)
                    save_image_atomic(jpeg_bytes, dest_path)
                    attachment_notes.append(
                        f"[Attached Image: '{display_name}' ({width}x{height}px, type: {mime_type}, saved in library 'discord_file' as '{saved_name}')]"
                    )
                else:
                    attachment_notes.append(
                        f"[Attached Image: '{display_name}' ({width}x{height}px, type: {mime_type})]"
                    )
            except Exception as e:
                attachment_notes.append(
                    f"[Attached File: '{display_name}' (type: {mime_type}, size: {len(raw_bytes)} bytes — failed to load image: {e})]"
                )
            return attached_images, attachment_notes

        safe_ext = re.sub(r"[^.A-Za-z0-9]+", "", ext) or MIME_TO_EXT.get(mime_type, ".bin")
        saved_name = f"mqtt_{ts_ms}_{safe_stem}{safe_ext}"
        if self.images_dir:
            try:
                dest_path = os.path.join(self.images_dir, saved_name)
                save_image_atomic(raw_bytes, dest_path)
                kind_label = (
                    "Attached Document"
                    if (mime_type in ("application/pdf", "application/epub+zip") or safe_ext in DOCUMENT_EXTENSIONS)
                    else "Attached File"
                )
                attachment_notes.append(
                    f"[{kind_label}: '{display_name}' (type: {mime_type}, size: {len(raw_bytes)} bytes, saved in library 'discord_file' as '{saved_name}' — use read_file or view_image on library 'discord_file' to inspect)]"
                )
            except Exception as e:
                attachment_notes.append(
                    f"[Attached File: '{display_name}' (type: {mime_type}, size: {len(raw_bytes)} bytes — failed to save attachment: {e})]"
                )
        else:
            attachment_notes.append(
                f"[Attached File: '{display_name}' (type: {mime_type}, size: {len(raw_bytes)} bytes)]"
            )
        return attached_images, attachment_notes

    async def _on_send_to_operator(self, event_name: str, payload: dict):
        """Intercept explicit send_message events and publish via MQTT if requested."""
        medium = payload.get("medium", "all").lower()
        if medium not in ["mqtt", "all"]:
            return

        msg_content = payload.get("message", "")
        recipient = (payload.get("recipient") or "").strip()
        att_lib = (payload.get("attachment_library") or "").strip()
        att_path = (payload.get("attachment_path") or "").strip()

        if medium == "mqtt" and not recipient:
            payload["error"] = (
                "Recipient is required when sending via 'mqtt' (e.g., recipient='wisp'). "
                "Use list_mqtt_agents to see available agents."
            )
            return

        if not msg_content and not (att_lib and att_path):
            return

        if not self.client:
            if medium == "mqtt":
                payload["error"] = "MQTT client is not currently connected."
            return

        if recipient:
            target_topic = self.resolve_dm_topic(recipient)
            envelope_dict: Dict[str, Any] = {"sender": self.agent_name, "text": msg_content}

            if (att_lib and not att_path) or (att_path and not att_lib):
                if medium == "mqtt":
                    payload["error"] = "Both 'attachment_library' and 'attachment_path' must be provided when attaching a file."
                    return
            elif att_lib and att_path:
                try:
                    resolved_file = self._resolve_library_attachment(att_lib, att_path)
                    with open(resolved_file, "rb") as f:
                        file_bytes = f.read()
                    guessed_mime, _ = mimetypes.guess_type(resolved_file)
                    envelope_dict["filename"] = os.path.basename(resolved_file)
                    envelope_dict["file_mime_type"] = guessed_mime or "application/octet-stream"
                    envelope_dict["file_bytes"] = base64.b64encode(file_bytes).decode("ascii")
                    payload["attached_file"] = os.path.basename(resolved_file)
                except Exception as e:
                    if medium == "mqtt":
                        payload["error"] = str(e)
                        return

            dm_envelope = json.dumps(envelope_dict)
            print(f"[MQTT] Sending DM to '{recipient}' on topic '{target_topic}'")
            try:
                await self.client.publish(
                    target_topic,
                    dm_envelope.encode("utf-8"),
                    qos=1,
                    retain=False,
                )
                payload["mqtt_recipient"] = recipient
                payload["mqtt_topic"] = target_topic
            except Exception as e:
                print(f"[MQTT Error] Failed to publish DM: {e}")
                payload["error"] = str(e)
        elif medium == "all" and self.outbox_topic and msg_content:
            print(f"[MQTT] Publishing broadcast to {self.outbox_topic}")
            try:
                await self.client.publish(self.outbox_topic, msg_content.encode("utf-8"), qos=1)
            except Exception as e:
                print(f"[MQTT Error] Failed to publish to outbox: {e}")

    async def _handle_message(self, topic_str: str, payload_raw: str, retain: bool = False):
        """Processes an incoming MQTT message from presence, DM, or system inbox topics."""
        # 1. Presence discovery topic (+/agent/presence)
        if topic_str.endswith("/agent/presence"):
            if not payload_raw.strip():
                return
            topic_slug = topic_str.split("/", 1)[0].strip().lower()
            try:
                data = json.loads(payload_raw)
                agent_display = data.get("agent") or topic_slug
                status = data.get("status", "online")
                dm_topic = data.get("dm_topic") or self.dm_topic_template.format(recipient=topic_slug)
            except Exception:
                agent_display = topic_slug
                status = payload_raw.strip() or "online"
                dm_topic = self.dm_topic_template.format(recipient=topic_slug)

            key = agent_display.strip().lower()
            self.known_agents[key] = {
                "agent": agent_display,
                "status": status,
                "dm_topic": dm_topic,
            }
            return

        # 2. Direct Message topic (<agent>/agent/dm)
        if topic_str == self.dm_topic:
            if not payload_raw.strip():
                return

            print(f"\n[MQTT Incoming DM] {topic_str}: {self._summarize_payload_for_log(payload_raw)}")
            if retain and self.client:
                try:
                    await self.client.publish(self.dm_topic, b"", qos=1, retain=True)
                except Exception:
                    pass

            attached_images: List[str] = []
            attachment_notes: List[str] = []
            try:
                data = json.loads(payload_raw)
                if isinstance(data, dict):
                    sender = data.get("sender") or "Unknown"
                    text = data.get("text", "") if ("text" in data or "file_bytes" in data) else payload_raw
                    attached_images, attachment_notes = self._process_incoming_attachment(data)
                else:
                    sender = "Unknown"
                    text = payload_raw
            except Exception:
                sender = "Unknown"
                text = payload_raw

            if attachment_notes:
                notes_str = "\n".join(attachment_notes)
                text = f"{text}\n{notes_str}".strip() if text else notes_str

            sender_key = str(sender).strip().lower()
            if sender_key and sender_key != "unknown" and sender_key not in self.known_agents:
                self.known_agents[sender_key] = {
                    "agent": str(sender).strip(),
                    "status": "online",
                    "dm_topic": self.dm_topic_template.format(recipient=sender_key),
                }

            formatted_payload = f"[Via MQTT | DM from {sender}] {text}"
            await self.agent.resume(formatted_payload, images=attached_images or None)
            return

        # 3. System Inbox topic (<agent>/agent/inbox)
        if topic_str == self.inbox_topic:
            if not payload_raw.strip():
                return

            print(f"\n[MQTT Incoming System] {topic_str}: {self._summarize_payload_for_log(payload_raw)}")
            if retain and self.client:
                try:
                    await self.client.publish(self.inbox_topic, b"", qos=1, retain=True)
                except Exception:
                    pass

            attached_images = []
            attachment_notes = []
            body_text = payload_raw
            if payload_raw.lstrip().startswith("{"):
                try:
                    data = json.loads(payload_raw)
                    if isinstance(data, dict) and ("text" in data or "file_bytes" in data):
                        body_text = str(data.get("text") or "")
                        sender = data.get("sender")
                        if sender:
                            body_text = f"{sender}: {body_text}" if body_text else str(sender)
                        attached_images, attachment_notes = self._process_incoming_attachment(data)
                except Exception:
                    pass

            if attachment_notes:
                notes_str = "\n".join(attachment_notes)
                body_text = f"{body_text}\n{notes_str}".strip() if body_text else notes_str

            formatted_payload = f"[Via MQTT | System] {body_text}"
            await self.agent.resume(formatted_payload, images=attached_images or None)

    async def start(self):
        """Connect to MQTT and start the listener loop with auto-reconnect."""
        self.agent.event_bus.subscribe("send_to_operator", self._on_send_to_operator)

        offline_will_payload = json.dumps(
            {
                "agent": self.agent_name,
                "status": "offline",
                "dm_topic": self.dm_topic,
            }
        ).encode("utf-8")
        online_presence_payload = json.dumps(
            {
                "agent": self.agent_name,
                "status": "online",
                "dm_topic": self.dm_topic,
            }
        ).encode("utf-8")

        while True:
            print(f"[MQTT] Connecting to broker at {self.host}:{self.port} as '{self.client_id}'...")
            try:
                will = aiomqtt.Will(
                    topic=self.presence_topic,
                    payload=offline_will_payload,
                    qos=1,
                    retain=True,
                )
                async with aiomqtt.Client(
                    self.host,
                    port=self.port,
                    identifier=self.client_id,
                    clean_session=False,
                    will=will,
                ) as client:
                    self.client = client
                    print(
                        f"[MQTT] Connected. Subscribing to system inbox '{self.inbox_topic}', "
                        f"DM topic '{self.dm_topic}', and presence '{self.presence_wildcard}'."
                    )
                    await client.subscribe(self.inbox_topic, qos=1)
                    await client.subscribe(self.dm_topic, qos=1)
                    await client.subscribe(self.presence_wildcard, qos=1)
                    await client.publish(
                        self.presence_topic,
                        online_presence_payload,
                        qos=1,
                        retain=True,
                    )

                    async for message in client.messages:
                        payload_raw = message.payload.decode("utf-8", errors="replace")
                        await self._handle_message(
                            str(message.topic),
                            payload_raw,
                            retain=bool(getattr(message, "retain", False)),
                        )

            except aiomqtt.MqttError as error:
                print(f"[MQTT Error] Connection failed or dropped: {error}. Reconnecting in 5 seconds...")
                self.client = None
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                print("[MQTT] Shutting down adapter.")
                if self.client:
                    try:
                        await self.client.publish(
                            self.presence_topic,
                            offline_will_payload,
                            qos=1,
                            retain=True,
                        )
                    except Exception:
                        pass
                break
            except Exception as e:
                print(f"[MQTT Error] Unexpected error: {e}. Reconnecting in 5 seconds...")
                self.client = None
                await asyncio.sleep(5)
