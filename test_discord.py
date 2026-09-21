import asyncio
import discord
from src.utils.config import load_config

async def main():
    config = load_config()
    token = config.get("discord", {}).get("bot_token")
    
    if not token:
        print("No discord token in config.")
        return
        
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    
    @client.event
    async def on_ready():
        print(f"Logged in successfully as {client.user}!")
        print("Connected to the following servers:")
        for guild in client.guilds:
            print(f"- {guild.name} (id: {guild.id})")
        await client.close()

    print("Attempting to connect...")
    try:
        await client.start(token)
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
