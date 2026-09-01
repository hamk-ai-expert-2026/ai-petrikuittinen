"""Minimal example of the OpenAI Responses API.

The Responses API is OpenAI's unified interface for generating model output
from text or multimodal input, with support for features such as reasoning and
tool use in a single request.
"""

from openai import OpenAI


client = OpenAI()

response = client.responses.create(
    model="gpt-5.6-luna",
    reasoning={"effort": "low"},
    max_output_tokens=512,
    input="Explain what is OpenAI Responses API in simple terms, max 2 sentences.",
)

print(response.output_text)
