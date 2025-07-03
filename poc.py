import asyncio
from openai import AsyncOpenAI

from realtime import SESSION_CONFIG

OPENAI_API_KEY = ""

async def main():
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
    )

    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
        await connection.session.update(session=SESSION_CONFIG)

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello!"}],
            }
        )
        await connection.response.create()

        async for event in connection:
            if event.type == 'response.text.delta':
                print(event.delta, flush=True, end="")

            elif event.type == 'response.text.done':
                print()

            elif event.type == "response.done":
                break

asyncio.run(main())
