"""
Input File Processor
Scans an input directory and converts all files into LLM-ready content:
- PDFs: each page rendered as a JPEG image
- Videos: sampled at 1 fps into base64 JPEG frames
- Images: base64-encoded directly
- Text/Markdown: read as-is
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium
from moviepy import VideoFileClip
from PIL import Image

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

MAX_VIDEO_FRAMES = 120
MAX_PDF_PAGES = 30


def _encode_image_file(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def _frame_to_data_uri(frame: np.ndarray, quality: int = 70) -> str:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{data}"


def extract_pdf_pages(
    path: Path, scale: int = 1, max_pages: int = MAX_PDF_PAGES
) -> list[str]:
    """Render each page of a PDF as a JPEG data-URI."""
    doc = pdfium.PdfDocument(str(path))
    n_pages = min(len(doc), max_pages)
    if len(doc) > max_pages:
        logger.warning(
            "PDF %s has %d pages; capping to %d", path.name, len(doc), max_pages
        )
    uris: list[str] = []
    for i, page in enumerate(doc):
        if i >= n_pages:
            break
        bitmap = page.render(scale=scale, rotation=0)
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
        data = base64.b64encode(buf.getvalue()).decode()
        uris.append(f"data:image/jpeg;base64,{data}")
    doc.close()
    return uris


def extract_video_frames(
    path: Path,
    fps: int = 1,
    max_frames: int = MAX_VIDEO_FRAMES,
) -> list[str]:
    """Sample a video at *fps* frames per second and return base64 data-URIs."""
    clip = VideoFileClip(str(path))
    duration = clip.duration
    interval = 1.0 / fps
    timestamps = [t * interval for t in range(int(duration * fps))]

    if len(timestamps) > max_frames:
        logger.warning(
            "Video %s has %d frames at %d fps; capping to %d",
            path.name,
            len(timestamps),
            fps,
            max_frames,
        )
        step = len(timestamps) / max_frames
        timestamps = [timestamps[int(i * step)] for i in range(max_frames)]

    uris: list[str] = []
    for t in timestamps:
        frame = clip.get_frame(t)
        if frame is not None:
            uris.append(_frame_to_data_uri(frame))
    clip.close()
    return uris


def process_input_dir(
    input_dir: str | Path, max_video_frames: int = MAX_VIDEO_FRAMES
) -> list[dict[str, Any]]:
    """
    Walk *input_dir* and return a list of LangChain-compatible content parts
    (``{"type": "text", ...}`` or ``{"type": "image_url", ...}``).
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    parts: list[dict] = []
    files = sorted(
        f for f in input_path.rglob("*") if f.is_file() and not f.name.startswith(".")
    )

    if not files:
        logger.warning("Input directory %s is empty", input_dir)
        return parts

    for file in files:
        ext = file.suffix.lower()
        rel = file.relative_to(input_path)

        if ext in TEXT_EXTENSIONS:
            text = file.read_text(errors="replace")
            parts.append({"type": "text", "text": f"--- File: {rel} ---\n{text}"})

        elif ext in PDF_EXTENSIONS:
            page_uris = extract_pdf_pages(file)
            parts.append(
                {"type": "text", "text": f"[PDF: {rel} — {len(page_uris)} page(s)]"}
            )
            for uri in page_uris:
                parts.append({"type": "image_url", "image_url": {"url": uri}})

        elif ext in IMAGE_EXTENSIONS:
            parts.append({"type": "text", "text": f"[Image: {rel}]"})
            parts.append(
                {"type": "image_url", "image_url": {"url": _encode_image_file(file)}}
            )

        elif ext in VIDEO_EXTENSIONS:
            frame_uris = extract_video_frames(file, fps=1, max_frames=max_video_frames)
            parts.append(
                {
                    "type": "text",
                    "text": f"[Video: {rel}] — {len(frame_uris)} frames sampled at 1 fps.",
                }
            )
            for uri in frame_uris:
                parts.append({"type": "image_url", "image_url": {"url": uri}})

        else:
            logger.debug("Skipping unsupported file: %s", rel)

    return parts
