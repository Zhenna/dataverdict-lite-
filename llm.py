"""Thin OpenAI adapter (Responses API). This is the only file that knows the provider's formats.

To switch provider, reimplement `make_client` and `call_llm` so they return the same `LLMTurn`.
"""

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    error: str | None = None  # set when the model produced unparseable arguments


@dataclass
class LLMTurn:
    output_items: list[Any]  # raw provider items, appended back into the next request
    text: str
    tool_calls: list[ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0


def make_client() -> Any:
    from openai import OpenAI  # imported lazily so tests can run without the SDK

    return OpenAI()  # reads OPENAI_API_KEY from the environment


def _to_provider_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # strict=False because the verdict schema uses optional fields and nested models,
    # which the strict-mode subset does not allow.
    return [
        {
            "type": "function",
            "name": s["name"],
            "description": s["description"],
            "parameters": s["parameters"],
            "strict": False,
        }
        for s in specs
    ]


def call_llm(
    client: Any,
    model: str,
    instructions: str,
    input_items: list[Any],
    tool_specs: list[dict[str, Any]],
) -> LLMTurn:
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_items,
        tools=_to_provider_tools(tool_specs),
    )

    tool_calls: list[ToolCall] = []
    for item in response.output:
        if item.type != "function_call":
            continue
        try:
            arguments = json.loads(item.arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
            error = None
        except ValueError as exc:
            arguments, error = {}, f"Arguments were not valid JSON: {exc}"
        tool_calls.append(ToolCall(id=item.call_id, name=item.name, arguments=arguments, error=error))

    usage = getattr(response, "usage", None)
    return LLMTurn(
        output_items=list(response.output),
        text=response.output_text or "",
        tool_calls=tool_calls,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )
