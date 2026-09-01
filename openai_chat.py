"""Simple streaming chat client using OpenRouter's OpenAI-compatible API."""

import os

from openai import OpenAI


MODEL = "deepseek-v4-flash"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SYSTEM_PROMPT = "You are a helpful assistant. Keep your answers clear and concise."
EXIT_COMMAND = "exit"
INPUT_PROMPT = "You: "
ASSISTANT_PREFIX = "Assistant: "
GOODBYE_MESSAGE = "Goodbye!"
SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
STREAMING_ENABLED = True


def main() -> None:
    client = OpenAI(
        api_key=os.environ[OPENROUTER_API_KEY_ENV],
        base_url=OPENROUTER_BASE_URL,
    )
    messages = [{"role": SYSTEM_ROLE, "content": SYSTEM_PROMPT}]

    print(f"Type {EXIT_COMMAND!r} to exit.")

    while True:
        user_input = input(INPUT_PROMPT).strip()
        if user_input.lower() == EXIT_COMMAND:
            print(GOODBYE_MESSAGE)
            return
        if not user_input:
            continue

        messages.append({"role": USER_ROLE, "content": user_input})
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=STREAMING_ENABLED,
        )

        print(ASSISTANT_PREFIX, end="", flush=True)
        assistant_response = []
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                assistant_response.append(content)
                print(content, end="", flush=True)

        print()
        messages.append(
            {"role": ASSISTANT_ROLE, "content": "".join(assistant_response)}
        )


if __name__ == "__main__":
    main()
