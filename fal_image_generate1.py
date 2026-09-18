"""Generate a children's book cover with Fal's GPT image model."""

import json
import os
from pathlib import Path
from urllib.request import urlopen

import fal_client


MODEL = "openai/gpt-image-2.5/sunburst/text-to-image"
PROMPT = "Children's book cover: Granda Ate My Hamster"
OUTPUT_PATH = Path(__file__).with_name("generated_book_cover.png")


def main() -> None:
    os.environ["FAL_KEY"] = os.environ["FAL_API_KEY"]
    result = fal_client.subscribe(
        MODEL,
        arguments={
            "prompt": PROMPT,
            "image_size": {"width": 2048, "height": 2736},
            "aspect_ratio": "3:4",
        },
    )

    print(json.dumps(result, indent=2))

    image_url = result["images"][0]["url"]
    with urlopen(image_url) as response:
        OUTPUT_PATH.write_bytes(response.read())
    print(f"Saved image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()