import asyncio
import os
import re
import discord
from src.core.agent import AgentCore
from src.utils.config import load_config
from src.utils.image import downsize_image_bytes, encode_jpeg_data_url, save_image_atomic

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
TEXT_EXTENSIONS = (
    ".txt", ".md", ".markdown", ".json", ".jsonl", ".py", ".sh",
    ".toml", ".yaml", ".yml", ".csv", ".log", ".xml", ".html",
    ".css", ".js", ".ts", ".sql", ".ini", ".cfg"
)
MAX_TEXT_ATTACHMENT_BYTES = 2 * 1024 * 1024

class DiscordAdapter:
    """
    Connects the AgentCore to a Discord bot.
    Listens for messages in a specific channel to wake the agent.
    Publishes the agent's explicit `send_to_operator` events back to Discord.
    """
    def __init__(self, agent: AgentCore):
        self.agent = agent
        self.config = load_config()
        
        discord_cfg = self.config.get("discord", {})
        self.token = discord_cfg.get("bot_token")
        self.allowed_channel_id = discord_cfg.get("allowed_channel_id")
        self.images_dir = (
            discord_cfg.get("images_dir")
            or self.config.get("file_reader", {}).get("libraries", {}).get("discord_images")
        )
        if self.images_dir:
            os.makedirs(self.images_dir, exist_ok=True)
        
        # Need message content intent to read what users say
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)
        
        # We need to know where to send messages back
        # If allowed_channel_id is set, we use that. 
        # Otherwise, we reply to the last channel we saw a message in.
        self.last_channel_id = self.allowed_channel_id

        self._setup_events()

    def _expand_mentions(self, text: str, mentions: list) -> str:
        """Expands raw <@ID> and <@!ID> tags into @DisplayName (<@ID>)."""
        if not text:
            return ""
        for user in mentions:
            if user == self.client.user:
                replacement = f"@{user.display_name} (you, <@{user.id}>)"
            else:
                replacement = f"@{user.display_name} (<@{user.id}>)"
            text = text.replace(f"<@{user.id}>", replacement).replace(f"<@!{user.id}>", replacement)
        return text

    def _setup_events(self):
        @self.client.event
        async def on_ready():
            print(f"[Discord] Logged in as {self.client.user}")
            if self.allowed_channel_id:
                print(f"[Discord] Restricted to channel ID: {self.allowed_channel_id}")
            else:
                print("[Discord] Listening on all channels the bot can see.")

        @self.client.event
        async def on_message(message):
            # Ignore messages from ourselves
            if message.author == self.client.user:
                return
                
            # If restricted to a specific channel, ignore others
            if self.allowed_channel_id and message.channel.id != self.allowed_channel_id:
                return
                
            # Update the last channel so we know where to reply
            self.last_channel_id = message.channel.id
            
            # Expand <@ID> and <@!ID> mentions so the agent sees both the display name and raw Discord tag
            content = self._expand_mentions(message.content, message.mentions)

            # Check if this message is a reply to another message
            reply_context = ""
            if message.reference and message.reference.message_id:
                ref_id = message.reference.message_id
                ref_msg = message.reference.resolved if isinstance(message.reference.resolved, discord.Message) else None
                if ref_msg is None:
                    try:
                        ref_msg = await message.channel.fetch_message(ref_id)
                    except Exception:
                        ref_msg = None
                if ref_msg is not None:
                    ref_author = "you" if ref_msg.author == self.client.user else ref_msg.author.display_name
                    ref_text = self._expand_mentions(ref_msg.content or "", ref_msg.mentions)
                    ref_snippet = ref_text[:120].replace("\n", " ")
                    reply_context = f' (replying to message_id={ref_id} by {ref_author}: "{ref_snippet}")'
                else:
                    reply_context = f" (replying to message_id={ref_id})"

            # Process any attachments (images, text files, or unsupported binary files) on the message
            attached_images = []
            attachment_notes = []
            for idx, attachment in enumerate(getattr(message, "attachments", []) or []):
                ctype = (attachment.content_type or "").lower()
                ctype_label = attachment.content_type or "unknown"
                size_bytes = getattr(attachment, "size", 0) or 0
                ext = os.path.splitext(attachment.filename.lower())[1]
                stem = os.path.splitext(os.path.basename(attachment.filename))[0]
                safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "attachment"

                if ctype.startswith("image/") or ext in IMAGE_EXTENSIONS:
                    try:
                        raw_bytes = await attachment.read()
                        jpeg_bytes, width, height = downsize_image_bytes(raw_bytes, max_edge=1024)
                        data_url = encode_jpeg_data_url(jpeg_bytes)
                        attached_images.append(data_url)

                        saved_name = f"{message.id}_{idx}_{safe_stem}.jpg"
                        if self.images_dir:
                            dest_path = os.path.join(self.images_dir, saved_name)
                            save_image_atomic(jpeg_bytes, dest_path)
                            attachment_notes.append(
                                f"[Attached Image: '{attachment.filename}' ({width}x{height}px, saved in library 'discord_images' as '{saved_name}')]"
                            )
                        else:
                            attachment_notes.append(
                                f"[Attached Image: '{attachment.filename}' ({width}x{height}px)]"
                            )
                    except Exception as e:
                        print(f"[Discord Error] Failed to process image attachment '{attachment.filename}': {e}")
                        attachment_notes.append(
                            f"[Attached File: '{attachment.filename}' (type: {ctype_label}, size: {size_bytes} bytes — failed to load image: {e})]"
                        )
                elif (
                    (ctype.startswith("text/") or "json" in ctype or "xml" in ctype or ext in TEXT_EXTENSIONS)
                    and size_bytes <= MAX_TEXT_ATTACHMENT_BYTES
                    and self.images_dir
                ):
                    try:
                        raw_bytes = await attachment.read()
                        safe_ext = re.sub(r"[^.A-Za-z0-9]+", "", ext) or ".txt"
                        saved_name = f"{message.id}_{idx}_{safe_stem}{safe_ext}"
                        dest_path = os.path.join(self.images_dir, saved_name)
                        save_image_atomic(raw_bytes, dest_path)
                        attachment_notes.append(
                            f"[Attached File: '{attachment.filename}' (type: {ctype_label}, size: {size_bytes} bytes, saved in library 'discord_images' as '{saved_name}' — use read_file to inspect)]"
                        )
                    except Exception as e:
                        print(f"[Discord Error] Failed to save text attachment '{attachment.filename}': {e}")
                        attachment_notes.append(
                            f"[Attached File: '{attachment.filename}' (type: {ctype_label}, size: {size_bytes} bytes — attachment not loaded)]"
                        )
                else:
                    attachment_notes.append(
                        f"[Attached File: '{attachment.filename}' (type: {ctype_label}, size: {size_bytes} bytes — attachment not loaded)]"
                    )

            if attachment_notes:
                notes_str = "\n".join(attachment_notes)
                content = f"{content}\n{notes_str}".strip() if content else notes_str

            # Format the payload so the agent knows the message_id, who is speaking, their mention tag, and their role
            role_tag = "[APP/Agent]" if message.author.bot else "[Operator]"
            payload = f"[Via Discord | message_id={message.id}] {message.author.display_name} (<@{message.author.id}>) {role_tag}{reply_context} says: {content}"
            print(f"[Discord Incoming] {payload}")
            
            # Wake the agent up!
            await self.agent.resume(payload, images=attached_images)

        @self.client.event
        async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
            # Ignore reactions added by ourselves
            if self.client.user and payload.user_id == self.client.user.id:
                return

            # If restricted to a specific channel, ignore others
            if self.allowed_channel_id and payload.channel_id != self.allowed_channel_id:
                return

            self.last_channel_id = payload.channel_id

            # Resolve who added the reaction
            reactor_name = str(payload.user_id)
            is_bot = False
            if payload.member:
                reactor_name = payload.member.display_name
                is_bot = payload.member.bot
            else:
                user = self.client.get_user(payload.user_id)
                if user is None:
                    try:
                        user = await self.client.fetch_user(payload.user_id)
                    except Exception:
                        user = None
                if user is not None:
                    reactor_name = user.display_name
                    is_bot = user.bot

            role_tag = "[APP/Agent]" if is_bot else "[Operator]"

            # Resolve the message that was reacted to
            target_desc = f"message_id={payload.message_id}"
            channel = self.client.get_channel(payload.channel_id)
            if channel:
                try:
                    target_msg = await channel.fetch_message(payload.message_id)
                    target_author = "your message" if target_msg.author == self.client.user else f"{target_msg.author.display_name}'s message"
                    target_text = self._expand_mentions(target_msg.content or "", target_msg.mentions)
                    snippet = target_text[:120].replace("\n", " ")
                    target_desc = f'{target_author} (message_id={payload.message_id}: "{snippet}")'
                except Exception:
                    pass

            event_str = f"[Via Discord | Reaction] {reactor_name} (<@{payload.user_id}>) {role_tag} reacted with {payload.emoji} to {target_desc}"
            print(f"[Discord Incoming] {event_str}")

            await self.agent.resume(event_str)

    async def _on_send_to_operator(self, event_name: str, payload: dict):
        """Intercept explicit operator messages and publish via Discord if requested."""
        # Only process if intended for Discord or broadcast
        medium = payload.get("medium", "all").lower()
        if medium not in ["discord", "all"]:
            return
            
        msg_content = payload.get("message", "")
        reply_to_raw = payload.get("reply_to_message_id", "")
        if msg_content and self.last_channel_id:
            try:
                channel = self.client.get_channel(self.last_channel_id)
                if channel:
                    reference = None
                    if reply_to_raw:
                        clean_id = str(reply_to_raw).replace("message_id=", "").strip()
                        if clean_id.isdigit():
                            reference = discord.MessageReference(
                                message_id=int(clean_id),
                                channel_id=channel.id,
                                fail_if_not_exists=False
                            )
                        else:
                            payload["error"] = f"Invalid reply_to_message_id '{reply_to_raw}'."
                            return

                    print(f"[Discord] Sending message to channel {self.last_channel_id}" + (f" (reply to {reply_to_raw})" if reference else ""))
                    # Discord has a 2000 character limit per message.
                    # We should chunk it if it's too long.
                    chunks = [msg_content[i:i+1900] for i in range(0, len(msg_content), 1900)]
                    first_sent = None
                    for idx, chunk in enumerate(chunks):
                        sent_msg = await channel.send(chunk, reference=reference if idx == 0 else None)
                        if first_sent is None:
                            first_sent = sent_msg
                    if first_sent is not None:
                        payload["sent_message_id"] = str(first_sent.id)
                else:
                    err = f"Could not find channel {self.last_channel_id}"
                    print(f"[Discord Error] {err}")
                    payload["error"] = err
            except Exception as e:
                print(f"[Discord Error] Failed to send message: {e}")
                payload["error"] = str(e)

    async def _on_react_to_message(self, event_name: str, payload: dict):
        """Add an emoji reaction to a specific Discord message."""
        msg_id_raw = payload.get("message_id", "")
        emoji = payload.get("emoji", "")
        if not msg_id_raw or not emoji:
            payload["error"] = "Both message_id and emoji are required."
            return

        if not self.last_channel_id:
            payload["error"] = "No active Discord channel configured."
            return

        clean_id = str(msg_id_raw).replace("message_id=", "").strip()
        if not clean_id.isdigit():
            payload["error"] = f"Invalid message_id '{msg_id_raw}'."
            return

        try:
            channel = self.client.get_channel(self.last_channel_id)
            if not channel:
                payload["error"] = f"Could not find channel {self.last_channel_id}"
                return

            partial_msg = channel.get_partial_message(int(clean_id))
            await partial_msg.add_reaction(emoji)
            print(f"[Discord] Reacted with {emoji} to message {clean_id}")
            payload["status"] = f"Reacted with {emoji} to Discord message {clean_id}."
        except Exception as e:
            print(f"[Discord Error] Failed to add reaction: {e}")
            payload["error"] = str(e)

    async def start(self):
        """Connect to Discord."""
        if not self.token:
            print("[Discord] No bot_token found in config.toml. Discord adapter disabled.")
            return

        # Subscribe to the agent's explicit output and reaction requests
        self.agent.event_bus.subscribe("send_to_operator", self._on_send_to_operator)
        self.agent.event_bus.subscribe("react_to_message", self._on_react_to_message)
        
        print("[Discord] Connecting to Discord...")
        try:
            # We must use start() instead of run() because run() is blocking
            await self.client.start(self.token)
        except Exception as e:
            print(f"[Discord Error] {e}")
