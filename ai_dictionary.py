import json
import sys

from openai import OpenAI


MODEL = "gpt-5.6-luna"
MAX_OUTPUT_TOKENS = 1024

DICTIONARY_SCHEMA = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "definition": {"type": "string"},
                    "synonym": {"type": "array", "items": {"type": "string"}},
                    "antonym": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["word", "definition", "synonym", "antonym"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entries"],
    "additionalProperties": False,
}


def validate_dictionary_output(output_text: str) -> dict:
    try:
        result = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"The model returned invalid JSON: {error}") from error

    if not isinstance(result, dict) or set(result) != {"entries"}:
        raise ValueError("The JSON must be an object containing only an entries field.")
    if not isinstance(result["entries"], list) or not result["entries"]:
        raise ValueError("The entries field must be a non-empty array.")

    required_fields = {"word", "definition", "synonym", "antonym"}
    for entry in result["entries"]:
        if not isinstance(entry, dict) or set(entry) != required_fields:
            raise ValueError(
                "Each entry must contain only word, definition, synonym, and antonym."
            )
        if not isinstance(entry["word"], str) or not entry["word"].strip():
            raise ValueError("Each word must be a non-empty string.")
        if not isinstance(entry["definition"], str) or not entry["definition"].strip():
            raise ValueError("Each definition must be a non-empty string.")
        if not isinstance(entry["synonym"], list) or not all(
            isinstance(value, str) for value in entry["synonym"]
        ):
            raise ValueError("Synonyms must be an array of strings.")
        if not isinstance(entry["antonym"], list) or not all(
            isinstance(value, str) for value in entry["antonym"]
        ):
            raise ValueError("Antonyms must be an array of strings.")

    return result


def lookup_words(client: OpenAI, words: str) -> dict:
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "medium"},
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input=(
            "Return dictionary information for every requested term. "
            "Use an empty array when a term has no common synonym or antonym. "
            "Requested terms: "
            f"{words}"
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "dictionary_entries",
                "strict": True,
                "schema": DICTIONARY_SCHEMA,
            }
        },
    )
    return validate_dictionary_output(response.output_text)


def main() -> None:
    client = OpenAI()

    while True:
        words = input("Words (press Enter to exit): ").strip()
        if not words:
            return

        try:
            result = lookup_words(client, words)
        except Exception as error:
            print(f"Dictionary lookup failed: {error}", file=sys.stderr)
            continue

        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()