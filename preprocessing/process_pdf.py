import base64
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from markitdown import MarkItDown
from pydantic import SecretStr

from src.llm_metrics import extract_llm_usage


def local_pdf_conversion(input_path, output_path):
    """
    Converts a PDF to Markdown and strips out CID tags,
    excessive formatting characters, and redundant whitespace.
    """
    try:
        # Initialize converter and process file
        md = MarkItDown()
        result = md.convert(input_path)
        content = result.markdown

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
    Sends a PDF to an LLM and returns the transcription as Markdown.
    Optionally writes the result to output_path.
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

    messages = [
        SystemMessage(
            content="You are a helpful assistant that transcribes documents."
        ),
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
                    "text": "You are an expert academic transcriber. Your task is to accurately transcribe all pages of the provided PDF into clean, well-structured Markdown. Follow these strict guidelines:\n\n1. Structure & Formatting: Preserve the original logical hierarchy. Use standard Markdown for headings (##, ###), bullet points, bolding, and italics. Ignore irrelevant page headers, footers, or page numbers.\n2. Math & Science (LaTeX): Use LaTeX for all mathematical equations, scientific formulas, and complex symbols. Strictly use `$` for inline equations (e.g., $E=mc^2$) and `$$` for block/display equations. Do not leave spaces between the delimiters and the math.\n3. Exams & Exercises: Carefully preserve all question numbers, sub-questions (a, b, c), multiple-choice options, and point/mark allocations.\n4. Tables & Visuals: Format tabular data into standard Markdown tables. If you encounter a graph, diagram, or image, do not attempt to draw it; instead, insert a descriptive placeholder like `[Image: Brief description of the graph/diagram]`.\n5. Handwriting/Illegibility: If transcribing handwritten notes and a word or phrase is completely unreadable, insert `[illegible]` rather than guessing.\n6. Output: Provide ONLY the final Markdown transcription. Do not include any conversational filler, introductions, or conclusions.",
                },
            ]
        ),
    ]

    response = llm.invoke(messages)
    markdown = str(response.content)

    cost = extract_llm_usage(response).cost_usd

    if cost is None:
        cost = 0.0

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
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
