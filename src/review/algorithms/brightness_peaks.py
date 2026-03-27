"""Brightness-peaks frame selection for manual inspection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from moviepy import VideoFileClip

from src.review.algorithms.constants import (
    BRIGHTNESS_DEDUP_SSIM_THRESHOLD,
    BRIGHTNESS_SCAN_INTERVAL,
    LIGHTNESS_LUMA_BLACK_THRESHOLD,
    NON_BLACK_MAX_RATIO,
    REVIEW_TARGET_FRAMES,
)
from src.review.algorithms.similarity import frame_ssim


def lightness_score(frame: np.ndarray) -> float:
    """Return lightness score in [0, 1] as inverse black-pixel ratio."""
    gray = (
        0.299 * frame[:, :, 0].astype(np.float64)
        + 0.587 * frame[:, :, 1].astype(np.float64)
        + 0.114 * frame[:, :, 2].astype(np.float64)
    )
    black_ratio = float(np.mean(gray <= LIGHTNESS_LUMA_BLACK_THRESHOLD))
    return 1.0 - black_ratio


def sample_every_second(video_path: Path) -> list[tuple[float, np.ndarray]]:
    """Sample one frame every second, plus one final frame at clip end."""
    clip = VideoFileClip(str(video_path))
    try:
        duration = float(clip.duration or 0.0)
        if duration <= 0:
            return []

        sampled: list[tuple[float, np.ndarray]] = []
        t = 0.0
        while t < duration:
            sampled.append((t, clip.get_frame(t)))  # type: ignore[arg-type]
            t += BRIGHTNESS_SCAN_INTERVAL

        if not sampled or sampled[-1][0] < duration:
            sampled.append((duration, clip.get_frame(duration)))  # type: ignore[arg-type]

        return sampled
    finally:
        clip.close()


def select_brightness_peak_frames(video_path: Path) -> list[tuple[float, np.ndarray]]:
    """Pick local lightness maxima and drop highly similar darker duplicates."""
    sampled = sample_every_second(video_path)
    if not sampled:
        return []

    lightness = [lightness_score(frame) for _, frame in sampled]
    black_ratios = [1.0 - score for score in lightness]

    peaks: list[tuple[float, np.ndarray, float]] = []
    for i in range(1, len(sampled) - 1):
        if (
            lightness[i] > lightness[i - 1]
            and lightness[i] > lightness[i + 1]
            and black_ratios[i] < NON_BLACK_MAX_RATIO
        ):
            ts, frame = sampled[i]
            peaks.append((ts, frame, lightness[i]))

    if not peaks:
        for i, (ts, frame) in enumerate(sampled):
            if black_ratios[i] < NON_BLACK_MAX_RATIO:
                peaks.append((ts, frame, lightness[i]))
    if not peaks:
        return []

    peaks.sort(key=lambda item: item[0])
    ssim_deduped: list[tuple[float, np.ndarray, float]] = [peaks[0]]
    for candidate in peaks[1:]:
        prev = ssim_deduped[-1]
        ssim = frame_ssim(prev[1], candidate[1])
        if ssim >= BRIGHTNESS_DEDUP_SSIM_THRESHOLD:
            if candidate[2] > prev[2]:
                ssim_deduped[-1] = candidate
        else:
            ssim_deduped.append(candidate)

    ssim_deduped.sort(key=lambda item: item[2], reverse=True)
    ssim_deduped = ssim_deduped[:REVIEW_TARGET_FRAMES]
    ssim_deduped.sort(key=lambda item: item[0])
    return [(ts, frame) for ts, frame, _ in ssim_deduped]
