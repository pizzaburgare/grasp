"""
Document Selector Agent

Given a topic, explores base_dir using LLM tool-calling and returns the
subset of files most relevant to the topic.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from src.llm_metrics import LLMUsage, extract_llm_usage
from src.settings import DOCUMENT_SELECTOR_MODEL

load_dotenv()
logger = logging.getLogger(__name__)

_SUMMARY_MAX_CHARS = 2_000
_SUMMARY_INPUT_MAX_CHARS = 8_000
_MAX_ITERATIONS = 10

_SYSTEM_PROMPT = """\
You are a document selection assistant for lesson creation.
Your task is to select source materials relevant to a specific lesson topic.

Explore the provided directory with list_files and get_summary.
The corpus contains lectures and exams.

Selection goals:
- Prioritize relevant lecture materials that explain the topic clearly.
- Include a handful of relevant exam files that assess the same topic (prefer about 3-6 exams when available).
- Favor diversity across exam sources/years if available, and avoid near-duplicate files.

Output rules:
- Select only files, not directories.
- Prefer source documents over derived artefacts.
- Exclude files that are not clearly relevant to the lesson topic.
"""


def _extract_markdown_frontmatter(text: str) -> dict[str, Any] | None:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", text, re.DOTALL)
    if not match:
        return None
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(metadata, dict):
        return None
    return metadata


class _Selection(BaseModel):
    files: list[str] = Field(description="File paths relative to base_dir that are relevant to the topic.")


class DocumentSelectorAgent:
    def __init__(self, base_dir: Path, model: str | None = None) -> None:
        self.base_dir = base_dir.resolve()
        self._llm = ChatOpenAI(
            model=model or DOCUMENT_SELECTOR_MODEL,
            api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Document Selector",
            },
        )

    def select(self, topic: str) -> tuple[list[Path], LLMUsage]:
        """Return (paths, usage) for files inside base_dir relevant to *topic*."""
        tools = self._build_tools()
        tool_map: dict[str, Any] = {t.name: t for t in tools}
        llm_with_tools = self._llm.bind_tools(tools)

        messages: list[Any] = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Topic: {topic}\nDirectory: {self.base_dir}"),
        ]

        usage = LLMUsage()

        for _ in range(_MAX_ITERATIONS):
            response: AIMessage = llm_with_tools.invoke(messages)
            _accumulate(usage, extract_llm_usage(response))
            messages.append(response)
            if not response.tool_calls:
                break
            for call in response.tool_calls:
                fn = tool_map.get(call["name"])
                result = fn.invoke(call["args"]) if fn else f"Unknown tool: {call['name']}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        structured = self._llm.with_structured_output(_Selection)
        final: Any = structured.invoke([*messages, HumanMessage(content=f"Now output the selected files for topic: {topic!r}.")])

        return self._validate(final.files if final else []), usage

    def _validate(self, candidates: list[str]) -> list[Path]:
        """Resolve relative paths, reject traversals and missing files."""
        result: list[Path] = []
        for rel in candidates:
            path = (self.base_dir / rel).resolve()
            try:
                path.relative_to(self.base_dir)
            except ValueError:
                logger.warning("Rejected out-of-bounds path: %s", rel)
                continue
            if path.is_file():
                result.append(path)
        return result

    def _build_tools(self) -> list:  # noqa: C901
        base = self.base_dir

        @tool
        def list_files(subdirectory: str = "") -> str:
            """List files and folders inside subdirectory (relative to base_dir). Default: base_dir."""
            target = (base / subdirectory).resolve() if subdirectory else base
            if not target.is_relative_to(base):
                return "Access denied."
            if not target.exists():
                return "Path does not exist."
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
            lines = [f"{e.relative_to(base)}{'/' if e.is_dir() else f'  ({e.stat().st_size:,} bytes)'}" for e in entries]
            return "\n".join(lines) or "Empty directory."

        @tool
        def get_summary(file_path: str) -> str:
            """Return a text preview of file_path (relative to base_dir)."""
            path = (base / file_path).resolve()
            if not path.is_relative_to(base):
                return "Access denied."
            if not path.is_file():
                return "Not a file."
            suffix = path.suffix.lower()

            if suffix not in {".md", ".markdown", ".txt"}:
                return "Discarded: unsupported file type."
            document_text = ""

            if suffix in {".md", ".markdown"}:
                try:
                    markdown_text = path.read_text(errors="replace")
                except OSError:
                    return f"{suffix} file, {path.stat().st_size:,} bytes"
                metadata = _extract_markdown_frontmatter(markdown_text)
                if metadata:
                    metadata_text = yaml.safe_dump(
                        metadata,
                        sort_keys=False,
                        allow_unicode=True,
                    ).strip()
                    return metadata_text[:_SUMMARY_MAX_CHARS]
                document_text = markdown_text.strip()
            else:
                try:
                    document_text = path.read_text(errors="replace").strip()
                except OSError:
                    return f"{suffix or 'binary'} file, {path.stat().st_size:,} bytes"

            if not document_text:
                return "No extractable text content."

            summary_response = self._llm.invoke(
                [
                    SystemMessage(content="You are a concise document summarizer."),
                    HumanMessage(
                        content=(
                            "Summarize this document concisely. Return ONLY the summary text. "
                            "Do not include introductory phrases like 'Here is a summary'. "
                            "Do not use double quotes, formatting, or line breaks in your response.\n\n"
                            f"Document:\n{document_text[:_SUMMARY_INPUT_MAX_CHARS]}"
                        )
                    ),
                ]
            )
            llm_output = str(summary_response.content)
            safe_summary = llm_output.strip().replace('"', "'")
            safe_summary = safe_summary.replace("\n", " ")
            return safe_summary[:_SUMMARY_MAX_CHARS]

        return [list_files, get_summary]


def _accumulate(total: LLMUsage, usage: LLMUsage) -> None:
    total.prompt_tokens += usage.prompt_tokens
    total.completion_tokens += usage.completion_tokens
    total.total_tokens += usage.total_tokens
    if usage.cost_usd is not None:
        total.cost_usd = (total.cost_usd or 0.0) + usage.cost_usd


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Select relevant files for a topic.")
    parser.add_argument("directory", type=Path, help="Directory to search.")
    parser.add_argument("query", help="Topic or query to select files for.")
    args = parser.parse_args()

    agent = DocumentSelectorAgent(base_dir=args.directory)
    selected, cost = agent.select(args.query)
    for path in selected:
        print(path)
    cost_str = f"${cost.cost_usd:.6f}" if cost.cost_usd is not None else "n/a"
    print(f"\nTokens: {cost.total_tokens} (prompt={cost.prompt_tokens}, completion={cost.completion_tokens})  cost={cost_str}")
