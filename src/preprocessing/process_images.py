import base64
import io

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from src.llm_metrics import LLMUsage, extract_llm_usage, make_openrouter_llm
from src.paths import IMAGE_TRANSCRIBER_PROMPT

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
) -> LLMUsage | None:
    """
    Sends an image to an LLM and returns a Markdown transcription.
    Optionally writes the result to output_path.
    """
    llm = make_openrouter_llm(model=model, title="image-to-md")

    image_url = _prepare_image(input_path)
    transcriber_prompt = IMAGE_TRANSCRIBER_PROMPT.read_text(encoding="utf-8")

    messages = [
        SystemMessage(content="You are a helpful assistant that transcribes documents."),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                    "type": "text",
                    "text": transcriber_prompt,
                },
            ]
        ),
    ]

    response = llm.invoke(messages)
    markdown = str(response.content)

    usage = extract_llm_usage(response)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"LLM transcription saved to: {output_path}")

    return usage


def convert_image_to_md(input_file: str, output_file: str) -> LLMUsage | None:
    return image_to_md_llm(input_file, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe an image to Markdown using an LLM.")
    parser.add_argument("input", help="Path to the input image file")
    parser.add_argument("output", help="Path to the output Markdown file")
    args = parser.parse_args()

    convert_image_to_md(args.input, args.output)
