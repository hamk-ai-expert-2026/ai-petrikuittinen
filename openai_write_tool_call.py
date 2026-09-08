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


def write(path: str, contents: str) -> str:
    """Write text to a UTF-8 file and return a status message."""
    try:
        Path(path).write_text(contents, encoding="utf-8")
        return f"Successfully wrote {path!r}."
    except (OSError, UnicodeError) as error:
        return f"Error writing {path!r}: {error}"


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
    },
    {
        "type": "function",
        "name": "write",
        "description": "Write text to a UTF-8 file and return a status message.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the file to write.",
                },
                "contents": {
                    "type": "string",
                    "description": "The text to write to the file.",
                },
            },
            "required": ["path", "contents"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def main() -> None:
    client = OpenAI()
    response = client.responses.create(
        model=MODEL,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        tools=TOOLS,
        input="Use the write tool to write the Chinese sentence '你好，世界！' as UTF-8 to chinese.txt.",
    )

    for _ in range(MAX_TOOL_ROUNDS):
        tool_outputs = []
        for item in response.output:
            if item.type != "function_call":
                continue

            arguments = json.loads(item.arguments)
            if item.name == "read":
                result = read(arguments["path"])
            elif item.name == "write":
                result = write(arguments["path"], arguments["contents"])
            else:
                result = f"Error: unsupported tool {item.name!r}."
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