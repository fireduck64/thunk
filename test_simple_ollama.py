import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(base_url="https://ollama-api.1209k.com/v1", api_key="CONSTRUCT_ADDITIONAL_PYLONS")
    response = await client.chat.completions.create(
        model="gemma4:12b",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
