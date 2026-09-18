"""Describe petri.jpg with an OpenAI vision model."""

import base64
from pathlib import Path

from openai import OpenAI


MODEL = "gpt-5.6-luna"
IMAGE_PATH = Path(__file__).with_name("petri.jpg")


def main() -> None:
    image_data = base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
    response = OpenAI().responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Describe this image in brutally honest terms. "
                            "Be specific about what is actually visible, and "
                            "do not invent details."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_data}",
                    },
                ],
            }
        ],
    )

    print(response.output_text)


if __name__ == "__main__":
    main()