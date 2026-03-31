"""PDF to Markdown conversion using local extraction or LLM-based transcription."""

import base64
import re
from datetime import date
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from markitdown import MarkItDown

from src.core.llm_metrics import LLMUsage, combine_llm_usage, extract_llm_usage, make_openrouter_llm
from src.core.paths import PDF_TRANSCRIBER_PROMPT

MIN_FENCE_LINES = 2


def strip_outer_markdown_fence(text: str) -> str:
    """Removes a single outer ```md/```markdown fence wrapper if present."""
    stripped = text.strip()
    lines = stripped.splitlines()
    fence_match = len(lines) >= MIN_FENCE_LINES and re.fullmatch(
        r"```(?:md|markdown)?\\s*", lines[0], re.IGNORECASE
    )
    if fence_match and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _clean_markdown_content(content: str) -> str:
    """Clean up markdown content by removing artifacts and normalizing whitespace."""
    content = re.sub(r"\(cid:\d+\)", "", content)  # Remove CID tags
    content = re.sub(r"\||-", "", content)  # Remove pipes and dashes
    content = re.sub(r"[^\S\r\n]+", " ", content)  # Collapse horizontal whitespace
    return re.sub(r"\n{3,}", "\n\n", content).strip()  # Collapse vertical whitespace


def local_pdf_conversion(input_path: str | Path, output_path: str | Path) -> None:
    """Convert PDF to Markdown locally, cleaning up artifacts and whitespace."""
    md = MarkItDown()
    result = md.convert(input_path)
    content = _clean_markdown_content(strip_outer_markdown_fence(result.markdown))

    Path(output_path).write_text(content, encoding="utf-8")
    print(f"Successfully cleaned and saved to: {output_path}")


def _transcribe_pdf(llm: ChatOpenAI, input_path: str) -> tuple[str, LLMUsage | None]:
    """Transcribe PDF to markdown using LLM."""
    encoded = base64.standard_b64encode(Path(input_path).read_bytes()).decode("utf-8")
    file_data = f"data:application/pdf;base64,{encoded}"
    response = llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant that transcribes documents."),
            HumanMessage(
                content=[
                    {
                        "type": "file",
                        "file": {"filename": Path(input_path).name, "file_data": file_data},
                    },
                    {"type": "text", "text": PDF_TRANSCRIBER_PROMPT.read_text(encoding="utf-8")},
                ]
            ),
        ]
    )
    return strip_outer_markdown_fence(str(response.content)), extract_llm_usage(response)


def _summarize_document(llm: ChatOpenAI, markdown: str) -> tuple[str, LLMUsage | None]:
    """Generate a summary of the document."""
    response = llm.invoke(
        [
            SystemMessage(content="You are a concise document summarizer."),
            HumanMessage(
                content=(
                    "Summarize this document concisely. Return ONLY the summary text. "
                    "Do not include introductory phrases like 'Here is a summary'. "
                    "Do not use double quotes, formatting, or line breaks in your response.\n\n"
                    f"Document:\n{markdown}"
                )
            ),
        ]
    )
    summary = str(response.content).strip().replace('"', "'").replace("\n", " ")
    return summary, extract_llm_usage(response)


def _build_frontmatter(summary: str) -> str:
    """Build YAML frontmatter with summary and date."""
    metadata = yaml.safe_dump(
        {"summary": summary, "date": date.today().isoformat()},
        sort_keys=False,
        allow_unicode=True,
    ).strip()
    return f"---\n{metadata}\n---\n\n"


def pdf_to_md_llm(
    input_path: str,
    output_path: str | None = None,
    model: str = "google/gemini-3-flash-preview",
) -> LLMUsage | None:
    """Transcribe PDF via LLM and optionally write Markdown with summary frontmatter."""
    llm = make_openrouter_llm(model, "PDF to Markdown")

    markdown, transcription_usage = _transcribe_pdf(llm, input_path)
    summary, summary_usage = _summarize_document(llm, markdown)

    if output_path:
        final_document = _build_frontmatter(summary) + markdown
        Path(output_path).write_text(final_document, encoding="utf-8")
        print(f"LLM transcription saved to: {output_path}")

    return combine_llm_usage([transcription_usage, summary_usage])


def convert_pdf_to_md(input_file: str, output_file: str, local: bool = False) -> LLMUsage | None:
    """Convert PDF to Markdown using local extraction or LLM-based transcription."""
    if local:
        local_pdf_conversion(input_file, output_file)
        return None
    return pdf_to_md_llm(input_file, output_file)


if __name__ == "__main__":
    INPUT_FILE = "test.pdf"
    OUTPUT_FILE = "test_output.md"
    convert_pdf_to_md(INPUT_FILE, OUTPUT_FILE, local=True)
