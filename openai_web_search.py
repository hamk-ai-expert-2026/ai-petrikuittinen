from openai import OpenAI


MODEL = "gpt-5.6-luna"
MAX_OUTPUT_TOKENS = 1024


def main() -> None:
    client = OpenAI()
    response = client.responses.create(
        model=MODEL,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        tools=[{"type": "web_search_preview"}],
        input=(
            "Use web search to find the official OpenAI website. "
            "Return its page title and the exact URL, followed by one sentence "
            "describing what OpenAI does."
        ),
    )

    if response.output_text.strip():
        print(response.output_text)
    else:
        print("The API returned no final text. Response item types:")
        for item in response.output:
            print(f"- {item.type}")


if __name__ == "__main__":
    main()