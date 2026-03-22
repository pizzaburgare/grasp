"""Helpers for extracting token usage and cost metadata from LLM responses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _format_cost(cost_usd: float | None) -> str:
    return f"${cost_usd:.6f}" if cost_usd is not None else "n/a"


class UsageTracker:
    """Accumulates per-step LLM usage, prints each step inline, and can print a summary."""

    def __init__(self) -> None:
        self._steps: list[tuple[str, LLMUsage | None, bool]] = []

    def record(self, label: str, usage: LLMUsage | None = None, *, skipped: bool = False) -> None:
        self._steps.append((label, usage, skipped))
        if skipped:
            print(f"{label}: skipped (cache)")
        elif usage is None:
            print(f"{label}: usage unavailable")
        else:
            print(
                f"{label}: prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, "
                f"total={usage.total_tokens}, "
                f"cost={_format_cost(usage.cost_usd)}"
            )

    def summarize(self) -> dict[str, Any]:
        total_prompt = total_completion = total_tokens = 0
        known_cost = 0.0
        has_unknown = False

        print()
        print("OpenRouter usage summary")
        print("-" * 60)

        for label, usage, skipped in self._steps:
            if skipped:
                print(f"{label}: skipped (cache)")
                continue
            if usage is None:
                print(f"{label}: usage unavailable")
                has_unknown = True
                continue
            total_prompt += usage.prompt_tokens
            total_completion += usage.completion_tokens
            total_tokens += usage.total_tokens
            if usage.cost_usd is None:
                has_unknown = True
            else:
                known_cost += usage.cost_usd
            print(
                f"{label}: prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, "
                f"total={usage.total_tokens}, "
                f"cost={_format_cost(usage.cost_usd)}"
            )

        print("-" * 60)
        print(f"TOTAL: prompt={total_prompt}, completion={total_completion}, total={total_tokens}")
        suffix = " + unknown" if has_unknown else ""
        print(f"TOTAL COST: {known_cost:.6f} USD{suffix}")

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "known_cost_usd": known_cost,
            "has_unknown_cost": has_unknown,
        }


@dataclass(slots=True)
class LLMUsage:
    """Normalized usage and pricing metadata for one LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _find_numeric(payload: Any, keys: set[str]) -> float | None:
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key).lower()
            if key in keys:
                number = _as_float(value)
                if number is not None:
                    return number
        for value in payload.values():
            found = _find_numeric(value, keys)
            if found is not None:
                return found
        return None

    if isinstance(payload, (list, tuple)):
        for item in payload:
            found = _find_numeric(item, keys)
            if found is not None:
                return found
    return None


def _find_first(candidates: list[Any], keys: set[str]) -> float | None:
    for candidate in candidates:
        found = _find_numeric(candidate, keys)
        if found is not None:
            return found
    return None


def extract_llm_usage(response: Any) -> LLMUsage:
    """Extract prompt/completion/total tokens and USD cost from a response object."""

    candidates: list[Any] = []

    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is not None:
        candidates.append(usage_metadata)

    response_metadata = getattr(response, "response_metadata", None)
    if response_metadata is not None:
        candidates.append(response_metadata)
        if isinstance(response_metadata, Mapping):
            for key in ("token_usage", "usage", "model_extra"):
                value = response_metadata.get(key)
                if value is not None:
                    candidates.append(value)

    additional_kwargs = getattr(response, "additional_kwargs", None)
    if additional_kwargs is not None:
        candidates.append(additional_kwargs)

    prompt_tokens_raw = _find_first(candidates, {"prompt_tokens", "input_tokens"})
    completion_tokens_raw = _find_first(
        candidates,
        {"completion_tokens", "output_tokens"},
    )
    total_tokens_raw = _find_first(candidates, {"total_tokens"})

    prompt_tokens = int(prompt_tokens_raw or 0)
    completion_tokens = int(completion_tokens_raw or 0)
    total_tokens = int(total_tokens_raw or 0)
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens

    cost = _find_first(
        candidates,
        {"total_cost", "cost", "usd_cost", "cost_usd", "price", "total_price"},
    )
    if cost is None:
        prompt_cost = _find_first(candidates, {"prompt_cost"})
        completion_cost = _find_first(candidates, {"completion_cost"})
        if prompt_cost is not None or completion_cost is not None:
            cost = (prompt_cost or 0.0) + (completion_cost or 0.0)

    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
    )
