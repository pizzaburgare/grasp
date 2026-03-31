#!/usr/bin/env python3
"""Manual inspection tool for the document selector step.

Usage:
    uv run manual_inspection/document_selector.py ./courses/kosys/ "Kendall's notation"
"""

import argparse
from pathlib import Path

from src.preprocessing import DocumentSelectorAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the document selector and print what the LLM returns."
    )
    parser.add_argument("directory", type=Path, help="Directory to search.")
    parser.add_argument("topic", help="Lesson topic/query.")
    args = parser.parse_args()

    agent = DocumentSelectorAgent(base_dir=args.directory)
    selected, usage = agent.select(args.topic)

    print(f"Topic: {args.topic}")
    print(f"Directory: {args.directory.resolve()}")
    print("\nSelected files:")
    if not selected:
        print("  (none)")
    else:
        for path in selected:
            print(f"  - {path}")

    cost_str = f"${usage.cost_usd:.6f}" if usage.cost_usd is not None else "n/a"
    print(
        f"\nTokens: {usage.total_tokens} "
        f"(prompt={usage.prompt_tokens}, completion={usage.completion_tokens}) "
        f"cost={cost_str}"
    )


if __name__ == "__main__":
    main()
