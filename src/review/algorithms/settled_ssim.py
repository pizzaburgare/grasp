"""Settled-frame scanning and SSIM-based selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from moviepy import VideoFileClip

from src.review.algorithms.constants import (
    HIGH_SSIM_THRESHOLD,
    MIN_NON_BLANK_FRAME_STD,
    REVIEW_TARGET_FRAMES,
    SCENE_SCAN_INTERVAL,
    SCENE_SETTLE_THRESHOLD,
)
from src.review.algorithms.similarity import frame_ssim


def scan_settled_frames(video_path: Path) -> list[tuple[float, np.ndarray]]:
    """Return settled, non-blank frames with at least 1 second spacing."""
    clip = VideoFileClip(str(video_path))
    try:
        duration = float(clip.duration or 0.0)
        if duration <= 0:
            return []

        candidates: list[tuple[float, np.ndarray]] = []
        last_kept_t = -float("inf")

        t = 0.0
        curr_frame: np.ndarray = clip.get_frame(0.0)  # type: ignore[assignment]
        while t < duration:
            next_t = min(t + SCENE_SCAN_INTERVAL, duration)
            next_frame: np.ndarray = clip.get_frame(next_t)  # type: ignore[assignment]

            mae = float(np.mean(np.abs(curr_frame.astype(np.int16) - next_frame.astype(np.int16))))
            not_blank = float(np.std(curr_frame)) >= MIN_NON_BLANK_FRAME_STD
            if mae <= SCENE_SETTLE_THRESHOLD and t - last_kept_t >= 1.0 and not_blank:
                candidates.append((t, curr_frame.copy()))
                last_kept_t = t

            curr_frame = next_frame
            t = next_t

        return candidates
    finally:
        clip.close()


def select_settled_ssim_frames(video_path: Path) -> list[tuple[float, np.ndarray]]:
    """Run settled scan, then SSIM dedup, then evenly subsample to target size."""
    candidates = scan_settled_frames(video_path)
    if not candidates:
        return []

    deduped: list[tuple[float, np.ndarray]] = [candidates[0]]
    for ts, frame in candidates[1:]:
        if frame_ssim(deduped[-1][1], frame) < HIGH_SSIM_THRESHOLD:
            deduped.append((ts, frame))

    if len(deduped) <= REVIEW_TARGET_FRAMES:
        return deduped

    step = len(deduped) / REVIEW_TARGET_FRAMES
    indices = list(
        dict.fromkeys(min(round(i * step), len(deduped) - 1) for i in range(REVIEW_TARGET_FRAMES))
    )
    return [deduped[i] for i in indices]
