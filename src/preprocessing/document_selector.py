"""
Document Selector Agent

Given a topic, explores base_dir using LLM tool-calling and returns the
subset of files most relevant to the topic.
"""

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.core.llm_metrics import LLMUsage, extract_llm_usage, make_openrouter_llm
from src.core.settings import DOCUMENT_SELECTOR_MODEL

load_dotenv()
logger = logging.getLogger(__name__)

_SUMMARY_MAX_CHARS = 2_000
_MAX_ITERATIONS = 5
_MAX_PREVIEW_FILES = 20

_SYSTEM_PROMPT = """\
You are a document selection assistant for lesson creation.
Your task is to select source materials relevant to a specific lesson topic.

Explore the provided directory with list_files and get_summary.
The corpus contains lectures and exams.

Tool usage guidance:
- Use list_files to inspect folders.
- Use get_summary with a list of file paths.
- Batch requests when possible: pass multiple files in a single get_summary call
    instead of calling get_summary once per file.

Selection goals:
- Prioritize relevant lecture materials that explain the topic clearly.
- Include a handful of relevant exam files that assess the same topic
    (prefer about 3-6 exams when available).
- Favor diversity across exam sources/years if available, and avoid near-duplicate files.

Output rules:
- Select only files, not directories.
- Prefer source documents over derived artefacts.
- Exclude files that are not clearly relevant to the lesson topic.
- Return multiple files when they are relevant; do not limit yourself to one.
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
    files: list[str] = Field(
        description="File paths relative to base_dir that are relevant to the topic."
    )


class DocumentSelectorAgent:
    """Agent that selects relevant documents from a directory for a given topic."""

    def __init__(self, base_dir: Path, model: str | None = None) -> None:
        self.base_dir = base_dir.resolve()
        self._llm = make_openrouter_llm(model or DOCUMENT_SELECTOR_MODEL, title="Document Selector")

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
        final: Any = structured.invoke(
            [
                *messages,
                HumanMessage(content=f"Now output the selected files for topic: {topic!r}."),
            ]
        )

        return self._validate(final.files if final else []), usage

    def _validate(self, candidates: list[str]) -> list[Path]:
        """Resolve relative paths, reject traversals and missing files."""
        result: list[Path] = []
        for rel in candidates:
            abs_path = (self.base_dir / rel).resolve()
            try:
                abs_path.relative_to(self.base_dir)
            except ValueError:
                logger.warning("Rejected out-of-bounds path: %s", rel)
                continue
            if abs_path.is_file():
                result.append(abs_path)
        return result

    def _build_tools(self) -> list:  # noqa: C901
        base = self.base_dir

        def _preview_one_file(file_path: str) -> str:
            abs_path = (base / file_path).resolve()
            if not abs_path.is_relative_to(base):
                return f"[{file_path}]\nAccess denied."
            if not abs_path.is_file():
                return f"[{file_path}]\nNot a file."

            suffix = abs_path.suffix.lower()
            if suffix not in {".md", ".markdown", ".txt"}:
                return f"[{file_path}]\nDiscarded: unsupported file type."

            raw_text = abs_path.read_text(errors="replace")

            if suffix in {".md", ".markdown"}:
                metadata = _extract_markdown_frontmatter(raw_text)
                if metadata:
                    metadata_text = yaml.safe_dump(
                        metadata,
                        sort_keys=False,
                        allow_unicode=True,
                    ).strip()
                    return f"[{file_path}]\n{metadata_text[:_SUMMARY_MAX_CHARS]}"

            document_text = raw_text.strip()
            if not document_text:
                return f"[{file_path}]\nNo extractable text content."

            return f"[{file_path}]\n{document_text[:_SUMMARY_MAX_CHARS]}"

        @tool
        def list_files(subdirectory: str = "") -> str:
            """List files/folders inside subdirectory relative to base_dir.

            Default: base_dir.
            """
            target = (base / subdirectory).resolve() if subdirectory else base
            if not target.is_relative_to(base):
                return "Access denied."
            if not target.exists():
                return "Path does not exist."
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
            lines = [
                f"{e.relative_to(base)}{'/' if e.is_dir() else f'  ({e.stat().st_size:,} bytes)'}"
                for e in entries
            ]
            return "\n".join(lines) or "Empty directory."

        @tool
        def get_summary(file_paths: list[str]) -> str:
            """Return text previews for file paths (relative to base_dir).

            Accepts a list of paths and returns one preview block per file.
            """
            if not file_paths:
                return "No file paths provided."

            limited_paths = file_paths[:_MAX_PREVIEW_FILES]
            previews = [_preview_one_file(file_path) for file_path in limited_paths]

            if len(file_paths) > _MAX_PREVIEW_FILES:
                previews.append(
                    f"Truncated request: processed first {_MAX_PREVIEW_FILES} file paths "
                    f"out of {len(file_paths)}."
                )

            return "\n\n".join(previews)

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
    print(
        f"\nTokens: {cost.total_tokens} "
        f"(prompt={cost.prompt_tokens}, completion={cost.completion_tokens}) "
        f" cost={cost_str}"
    )
