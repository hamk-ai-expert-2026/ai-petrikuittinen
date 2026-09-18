"""Describe petri.jpg with a Gemini vision model."""

import base64
from pathlib import Path

from google import genai


MODEL = "gemini-3.8-flash"
IMAGE_PATH = Path(__file__).with_name("petri.jpg")


def main() -> None:
    image_data = base64.b64encode(IMAGE_PATH.read_bytes()).decode("utf-8")
    client = genai.Client()
    interaction = client.interactions.create(
        model=MODEL,
        input=[
            {
                "type": "text",
                "text": (
                    "Describe this image in brutally honest terms. "
                    "Be specific about what is actually visible, and "
                    "do not invent details."
                ),
            },
            {
                "type": "image",
                "data": image_data,
                "mime_type": "image/jpeg",
            },
        ],
    )

    print(interaction.output_text)


if __name__ == "__main__":
    main()