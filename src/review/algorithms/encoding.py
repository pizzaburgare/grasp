"""Encoding helpers for turning frames into review payload parts."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from src.review.algorithms.constants import REVIEW_FRAME_QUALITY
from src.utils import format_timestamp


def encode_selected_frames(
    selected: list[tuple[float, np.ndarray]],
    jpeg_quality: int = REVIEW_FRAME_QUALITY,
) -> list[tuple[str, dict[str, Any]]]:
    """Encode selected frames into image_url payloads for LLM review."""
    parts: list[tuple[str, dict[str, Any]]] = []
    for chosen_t, frame in selected:
        img = Image.fromarray(frame)  # type: ignore[arg-type]
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        data = base64.b64encode(buf.getvalue()).decode()
        label = f"Frame at {format_timestamp(chosen_t)}"
        parts.append(
            (
                label,
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                },
            )
        )
    return parts
