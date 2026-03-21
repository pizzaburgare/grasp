import base64
import os
import re
from datetime import date

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from markitdown import MarkItDown
from pydantic import SecretStr

from src.llm_metrics import extract_llm_usage
from src.paths import PDF_TRANSCRIBER_PROMPT


def strip_outer_markdown_fence(text: str) -> str:
    """Removes a single outer ```md/```markdown fence wrapper if present."""
    stripped = text.strip()
    lines = stripped.splitlines()
    if len(lines) >= 2 and re.fullmatch(r"```(?:md|markdown)?\\s*", lines[0], re.IGNORECASE) and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()

    return stripped


def local_pdf_conversion(input_path, output_path):
    """
    Converts a PDF to Markdown and strips out CID tags,
    excessive formatting characters, and redundant whitespace.
    """
    try:
        # Initialize converter and process file
        md = MarkItDown()
        result = md.convert(input_path)
        content = strip_outer_markdown_fence(result.markdown)

        # 1. Remove CID tags: (cid:123)
        content = re.sub(r"\(cid:\d+\)", "", content)

        # 2. Remove specific formatting characters like pipes and dashes
        # Added [ ] to the class if you want to add more chars later
        content = re.sub(r"\||-", "", content)

        # 3. Clean up horizontal whitespace (multiple spaces -> single space)
        # We use [^\S\r\n] to target spaces/tabs but NOT newlines
        content = re.sub(r"[^\S\r\n]+", " ", content)

        # 4. Clean up vertical whitespace (3+ newlines -> 2 newlines)
        # This preserves paragraph breaks while removing excessive gaps
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Successfully cleaned and saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")


def pdf_to_md_llm(
    input_path: str,
    output_path: str | None = None,
    model: str = "google/gemini-2.0-flash-001",
) -> float:
    """
    Sends a PDF to an LLM, then generates a concise summary.
    Optionally writes Markdown with summary frontmatter to output_path.
    """
    llm = ChatOpenAI(
        model=model,
        api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "PDF to Markdown",
        },
    )

    with open(input_path, "rb") as f:
        encoded = base64.standard_b64encode(f.read()).decode("utf-8")

    transcriber_prompt = PDF_TRANSCRIBER_PROMPT.read_text(encoding="utf-8")

    messages = [
        SystemMessage(content="You are a helpful assistant that transcribes documents."),
        HumanMessage(
            content=[
                {
                    "type": "file",
                    "file": {
                        "filename": os.path.basename(input_path),
                        "file_data": f"data:application/pdf;base64,{encoded}",
                    },
                },
                {
                    "type": "text",
                    "text": transcriber_prompt,
                },
            ]
        ),
    ]

    response = llm.invoke(messages)
    markdown = strip_outer_markdown_fence(str(response.content))

    transcription_cost = extract_llm_usage(response).cost_usd

    summary_response = llm.invoke(
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

    llm_output = str(summary_response.content)
    safe_summary = llm_output.strip().replace('"', "'")
    safe_summary = safe_summary.replace("\n", " ")

    yaml_metadata = yaml.safe_dump(
        {"summary": safe_summary, "date": date.today().isoformat()},
        sort_keys=False,
        allow_unicode=True,
    ).strip()
    frontmatter = f"---\n{yaml_metadata}\n---\n\n"
    final_document = frontmatter + markdown

    summary_cost = extract_llm_usage(summary_response).cost_usd

    cost = 0.0
    if transcription_cost is not None:
        cost += transcription_cost
    if summary_cost is not None:
        cost += summary_cost

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_document)
        print(f"LLM transcription saved to: {output_path}")

    return cost


def convert_pdf_to_md(input_file: str, output_file: str, local: bool = False):
    if local:
        local_pdf_conversion(input_file, output_file)
        return 0.0
    else:
        return pdf_to_md_llm(input_file, output_file)


if __name__ == "__main__":
    INPUT_FILE = "test.pdf"
    OUTPUT_FILE = "test_output.md"
    convert_pdf_to_md(INPUT_FILE, OUTPUT_FILE, local=True)
