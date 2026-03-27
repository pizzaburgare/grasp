"""Top-level frame extraction entrypoints by algorithm name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.review.algorithms.brightness_peaks import select_brightness_peak_frames
from src.review.algorithms.encoding import encode_selected_frames
from src.review.algorithms.settled_ssim import select_settled_ssim_frames

Selector = Any


def extract_video_frames_parts(
    video_path: Path,
    algorithm: str = "settled-ssim",
) -> list[tuple[str, dict[str, Any]]]:
    """Extract and encode review frames using the selected algorithm."""
    selectors: dict[str, Selector] = {
        "settled-ssim": select_settled_ssim_frames,
        "brightness-peaks": select_brightness_peak_frames,
    }
    selector = selectors.get(algorithm)
    if selector is None:
        raise ValueError(f"Unknown frame extraction algorithm: {algorithm}")

    selected = selector(video_path)
    return encode_selected_frames(selected)
