import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(base_url="https://ollama-api.1209k.com/v1", api_key="CONSTRUCT_ADDITIONAL_PYLONS")
    tools = [{
        "type": "function",
        "function": {
            "name": "put_note",
            "description": "Save a note",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"}
                },
                "required": ["key", "value"]
            }
        }
    }]
    response = await client.chat.completions.create(
        model="gemma4:12b",
        messages=[{"role": "user", "content": "Please save a note with key 'test' and value '123'"}],
        tools=tools
    )
    msg = response.choices[0].message
    print("Tool calls:", msg.tool_calls)

asyncio.run(main())
