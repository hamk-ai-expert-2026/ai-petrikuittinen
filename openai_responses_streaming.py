"""Streaming example of the OpenAI Responses API."""

from openai import OpenAI


client = OpenAI()
prompt = input("Enter a prompt: ")

stream = client.responses.create(
    model="gpt-5.6-luna",
    reasoning={"effort": "none"},
    max_output_tokens=1024,
    input=prompt,
    stream=True,
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)

print()
