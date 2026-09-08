import json
from pathlib import Path

from openai import OpenAI


MODEL = "gpt-5.6-luna"
MAX_OUTPUT_TOKENS = 512
MAX_TOOL_ROUNDS = 5


def read(path: str) -> str:
    """Return a file's text contents, or an error message."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return f"Error reading {path!r}: {error}"


TOOLS = [
    {
        "type": "function",
        "name": "read",
        "description": "Read a UTF-8 text file and return its contents or an error message.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the text file to read.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


def main() -> None:
    client = OpenAI()
    response = client.responses.create(
        model=MODEL,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        tools=TOOLS,
        input="Use the read tool to tell me what is in local.txt.",
    )

    for _ in range(MAX_TOOL_ROUNDS):
        tool_outputs = []
        for item in response.output:
            if item.type != "function_call":
                continue

            arguments = json.loads(item.arguments)
            result = read(arguments["path"])
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result,
                }
            )

        if not tool_outputs:
            print(response.output_text)
            return

        response = client.responses.create(
            model=MODEL,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            tools=TOOLS,
            input=response.output + tool_outputs,
        )

    raise RuntimeError("The model exceeded the maximum number of tool calls.")


if __name__ == "__main__":
    main()