"""Structured-output prompt helpers for SDK agents."""

from __future__ import annotations

from pydantic import BaseModel


def structured_output_instruction(model: type[BaseModel]) -> str:
    schema = model.model_json_schema()
    required = schema.get("required", [])
    properties = sorted((schema.get("properties") or {}).keys())
    return (
        "\n\n# Structured Output Contract\n"
        "Return exactly one top-level JSON object parseable by Python json.loads. "
        "Do not wrap the final JSON in prose. If you also write the JSON to disk, "
        "the file payload must have the same shape.\n"
        f"Schema model: {model.__name__}\n"
        f"Required fields: {required}\n"
        f"Allowed top-level fields: {properties}\n"
    )


def append_structured_output_instruction(prompt: str, model: type[BaseModel] | None) -> str:
    if model is None:
        return prompt
    return prompt + structured_output_instruction(model)


__all__ = [
    "append_structured_output_instruction",
    "structured_output_instruction",
]
