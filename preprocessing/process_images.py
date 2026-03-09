import base64
import io
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import SecretStr

from src.llm_metrics import extract_llm_usage

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}

# Max dimension (width or height) before downscaling
_MAX_IMAGE_DIM = 1920


def _prepare_image(image_path: str) -> str:
    """Open an image with PIL, downscale if needed, and return a base64 JPEG data URL."""
    img: Image.Image = Image.open(image_path)

    # Convert to RGB so JPEG encoding works for any mode (RGBA, P, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Downscale proportionally if either dimension exceeds the limit
    w, h = img.size
    if w > _MAX_IMAGE_DIM or h > _MAX_IMAGE_DIM:
        img.thumbnail((_MAX_IMAGE_DIM, _MAX_IMAGE_DIM), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def image_to_md_llm(
    input_path: str,
    output_path: str | None = None,
    model: str = "google/gemini-2.0-flash-001",
) -> float:
    """
    Sends an image to an LLM and returns a Markdown transcription.
    Optionally writes the result to output_path.
    """
    llm = ChatOpenAI(
        model=model,
        api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Image to Markdown",
        },
    )

    image_url = _prepare_image(input_path)

    messages = [
        SystemMessage(
            content="You are a helpful assistant that transcribes documents."
        ),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                    "type": "text",
                    "text": """
                    You are an expert academic transcriber.
                    Your task is to accurately transcribe and describe the content of the provided image into clean, well-structured Markdown.
                    Follow these strict guidelines:

                    1. Contextual Summary: Begin your response by describing the overall scene, context, or visual narrative of the image. For math courses, describe the type of problem being solved, the mathematical concepts shown on the board/screen, and the general action taking place. Format this exactly as: `**Summary**: <Scene description>`
                    2. Structure & Formatting: Preserve the original logical hierarchy. Use standard Markdown for headings (##, ###), bullet points, bolding, and italics. Ignore irrelevant page headers, footers, or page numbers.
                    3. Math & Science (LaTeX): Use LaTeX for all mathematical equations, scientific formulas, and complex symbols. Strictly use `$` for inline equations (e.g., $E=mc^2$) and `$$` for block/display equations. Do not leave spaces between the delimiters and the math.
                    4. Exams & Exercises: Carefully preserve all question numbers, sub-questions (a, b, c), multiple-choice options, and point/mark allocations.
                    5. Tables & Visuals: Format tabular data into standard Markdown tables. If you encounter a graph, diagram, or image within the image, do not attempt to draw it; instead, insert a descriptive placeholder like `[Image: Brief description of the graph/diagram]`.
                    6. Handwriting/Illegibility: If transcribing handwritten notes and a word or phrase is completely unreadable, insert `[illegible]` rather than guessing.
                    7. Output: Provide ONLY the final Markdown transcription, starting with the Summary. Do not include any conversational filler, introductions, or conclusions.
                    """,
                },
            ]
        ),
    ]

    response = llm.invoke(messages)
    markdown = str(response.content)

    cost = extract_llm_usage(response).cost_usd or 0.0

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"LLM transcription saved to: {output_path}")

    return cost


def convert_image_to_md(input_file: str, output_file: str) -> float:
    return image_to_md_llm(input_file, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe an image to Markdown using an LLM."
    )
    parser.add_argument("input", help="Path to the input image file")
    parser.add_argument("output", help="Path to the output Markdown file")
    args = parser.parse_args()

    convert_image_to_md(args.input, args.output)
