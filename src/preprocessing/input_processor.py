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

import numpy as np
from moviepy import VideoFileClip
from PIL import Image

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

MAX_VIDEO_FRAMES = 240


def _frame_to_data_uri(frame: np.ndarray, quality: int = 70) -> str:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{data}"


def extract_video_frames(
    path: Path,
    fps: int = 1,
    max_frames: int = MAX_VIDEO_FRAMES,
) -> list[str]:
    """Sample a video at *fps* frames per second and return base64 data-URIs."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")

    clip = VideoFileClip(str(path))
    try:
        duration = float(clip.duration or 0.0)
        if duration <= 0:
            logger.warning("Video %s has non-positive duration; skipping", path.name)
            return []

        interval = 1.0 / fps
        n_samples = max(1, int(duration * fps))
        max_t = max(0.0, duration - 1e-3)
        timestamps = [min(t * interval, max_t) for t in range(n_samples)]

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
        return uris
    finally:
        clip.close()
