from openai import OpenAI


MODEL = "gpt-5.6-luna"
MAX_OUTPUT_TOKENS = 1024


def main() -> None:
    query = input("What would you like to search for? ").strip()
    if not query:
        print("No search query provided.")
        return

    client = OpenAI()
    response = client.responses.create(
        model=MODEL,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        tools=[{"type": "web_search_preview"}],
        input=(
            f"Use web search to answer this user query: {query}\n\n"
            "Display the most relevant results clearly. Include the source URL "
            "for every result, and do not omit URLs."
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