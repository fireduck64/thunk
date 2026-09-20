import asyncio
import discord
from src.core.agent import AgentCore
from src.utils.config import load_config

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
        
        # Need message content intent to read what users say
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)
        
        # We need to know where to send messages back
        # If allowed_channel_id is set, we use that. 
        # Otherwise, we reply to the last channel we saw a message in.
        self.last_channel_id = self.allowed_channel_id

        self._setup_events()

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
                
            # Update the last seen channel so we know where to reply
            self.last_channel_id = message.channel.id
            
            # Format the payload so the agent knows who is speaking
            payload = f"{message.author.display_name} says: {message.content}"
            print(f"[Discord Incoming] {payload}")
            
            # Wake the agent up!
            await self.agent.resume(payload)

    async def _on_send_to_operator(self, event_name: str, payload: dict):
        """Intercept explicit operator messages and publish via Discord."""
        msg_content = payload.get("message", "")
        if msg_content and self.last_channel_id:
            try:
                channel = self.client.get_channel(self.last_channel_id)
                if channel:
                    print(f"[Discord] Sending message to channel {self.last_channel_id}")
                    # Discord has a 2000 character limit per message.
                    # We should chunk it if it's too long.
                    chunks = [msg_content[i:i+1900] for i in range(0, len(msg_content), 1900)]
                    for chunk in chunks:
                        await channel.send(chunk)
                else:
                    print(f"[Discord Error] Could not find channel {self.last_channel_id}")
            except Exception as e:
                print(f"[Discord Error] Failed to send message: {e}")

    async def start(self):
        """Connect to Discord."""
        if not self.token:
            print("[Discord] No bot_token found in config.toml. Discord adapter disabled.")
            return

        # Subscribe to the agent's explicit output requests
        self.agent.event_bus.subscribe("send_to_operator", self._on_send_to_operator)
        
        print("[Discord] Connecting to Discord...")
        try:
            # We must use start() instead of run() because run() is blocking
            await self.client.start(self.token)
        except Exception as e:
            print(f"[Discord Error] {e}")
