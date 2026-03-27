"""Video compositor: progress-sidebar overlay and multi-scene concatenation.

Responsibilities:
- Render a left-side progress panel (showing all section names with the current
  one highlighted) as a PIL image.
- Overlay that panel onto a rendered content-scene video using MoviePy.
- Concatenate intro + content scenes + outro into the final video file.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

import numpy as np
from moviepy import VideoFileClip
from moviepy import concatenate_videoclips as _moviepy_concat
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Visual constants
# ------------------------------------------------------------------

_BG_COLOR = (13, 17, 35)          # deep dark blue
_PANEL_ALPHA = 220                 # 0-255 transparency of the sidebar
_BORDER_COLOR = (99, 179, 237)     # accent blue for current section border
_TEXT_ACTIVE = (255, 255, 255)     # white — current section label
_TEXT_DONE = (120, 140, 160)       # dim blue-grey — completed sections
_TEXT_FUTURE = (70, 90, 110)       # darker — upcoming sections
_HIGHLIGHT_BG = (25, 45, 80)       # subtle blue tint behind current section
_SIDEBAR_FRACTION = 0.22           # fraction of video width used by sidebar


# ------------------------------------------------------------------
# Font helpers
# ------------------------------------------------------------------

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common system font paths; fall back to PIL bitmap font."""
    candidates: list[str] = []
    if bold:
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


# ------------------------------------------------------------------
# Sidebar image generation
# ------------------------------------------------------------------

def _create_sidebar_image(
    sections: list[str],
    current_idx: int,
    width: int,
    height: int,
) -> Image.Image:
    """Return a PIL RGBA image of the progress sidebar.

    Parameters
    ----------
    sections:
        Ordered list of section display names.
    current_idx:
        0-based index of the currently active section.
    width:
        Pixel width of the sidebar image.
    height:
        Pixel height of the sidebar image (should match video height).
    """
    img = Image.new("RGBA", (width, height), (*_BG_COLOR, _PANEL_ALPHA))
    draw = ImageDraw.Draw(img)

    # Header
    header_font = _load_font(max(10, width // 12), bold=True)
    header_text = "PROGRESS"
    header_y = max(10, height // 30)
    draw.text(
        (width // 2, header_y), header_text, font=header_font, fill=(180, 200, 220), anchor="mt"
    )

    # Divider line below header
    divider_y = header_y + max(22, height // 25)
    draw.line([(10, divider_y), (width - 10, divider_y)], fill=(40, 60, 90), width=1)

    # Section items — distribute vertically in remaining space
    n = len(sections)
    top_margin = divider_y + max(12, height // 40)
    bottom_margin = max(12, height // 40)
    available_height = height - top_margin - bottom_margin
    row_height = max(30, available_height // max(n, 1))

    label_font_size = max(9, min(width // 14, row_height // 3))
    label_font = _load_font(label_font_size)
    num_font = _load_font(max(8, label_font_size - 2), bold=True)

    for i, section_name in enumerate(sections):
        row_top = top_margin + i * row_height
        row_mid = row_top + row_height // 2

        if i == current_idx:
            # Highlighted row background
            draw.rectangle(
                [(4, row_top + 2), (width - 4, row_top + row_height - 2)],
                fill=(*_HIGHLIGHT_BG, 255),
            )
            # Left accent border
            draw.rectangle(
                [(4, row_top + 2), (7, row_top + row_height - 2)],
                fill=(*_BORDER_COLOR, 255),
            )
            text_color = _TEXT_ACTIVE
            num_color = _BORDER_COLOR
        elif i < current_idx:
            # Completed sections — slightly dimmed, show a small check mark
            text_color = _TEXT_DONE
            num_color = _TEXT_DONE
        else:
            # Upcoming sections
            text_color = _TEXT_FUTURE
            num_color = _TEXT_FUTURE

        # Section number
        num_str = str(i + 1)
        num_x = 18
        draw.text((num_x, row_mid), num_str, font=num_font, fill=num_color, anchor="lm")

        # Section name — wrap long names at spaces
        text_x = num_x + max(18, width // 10)
        max_chars = max(8, (width - text_x - 6) // max(label_font_size // 2, 1))
        wrapped = _wrap_text(section_name, max_chars)
        # If multi-line, offset the first line up slightly
        if len(wrapped) > 1:
            line_h = label_font_size + 2
            total_text_h = len(wrapped) * line_h
            text_y = row_mid - total_text_h // 2
            for line in wrapped:
                draw.text((text_x, text_y), line, font=label_font, fill=text_color, anchor="lm")
                text_y += line_h
        else:
            draw.text(
                (text_x, row_mid),
                wrapped[0] if wrapped else "",
                font=label_font,
                fill=text_color,
                anchor="lm",
            )

    # Right edge separator line
    draw.line([(width - 1, 0), (width - 1, height)], fill=(30, 50, 80, 200), width=2)

    return img


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Wrap *text* to lines of at most *max_chars* characters, breaking at spaces."""
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [text]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def add_progress_sidebar(
    video_path: Path,
    sections: list[str],
    current_section_idx: int,
    output_path: Path,
) -> Path:
    """Overlay a left-side progress sidebar on *video_path* and write *output_path*.

    The sidebar occupies ~22 % of the video width on the left edge.  Content
    scene generators are instructed to keep their visuals in the right 75 % of
    the frame so there is no meaningful overlap.

    Parameters
    ----------
    video_path:
        Path to the rendered content-scene MP4.
    sections:
        Ordered list of all section display names for this video.
    current_section_idx:
        0-based index of the section shown in *video_path*.
    output_path:
        Destination MP4 path.
    """
    moviepy_logger = logging.getLogger("moviepy")

    with contextlib.ExitStack() as stack:
        stack.callback(moviepy_logger.setLevel, moviepy_logger.level)
        moviepy_logger.setLevel(logging.ERROR)

        clip = VideoFileClip(str(video_path))
        stack.callback(clip.close)

        vid_w, vid_h = clip.size
        sidebar_w = max(120, int(vid_w * _SIDEBAR_FRACTION))

        # Build sidebar as numpy array (H x W x 4 = RGBA)
        sidebar_img = _create_sidebar_image(sections, current_section_idx, sidebar_w, vid_h)
        sidebar_arr = np.array(sidebar_img)  # RGBA uint8

        # Split alpha and composite manually: we need RGB for MoviePy ImageClip
        alpha = sidebar_arr[:, :, 3:4] / 255.0  # (H, W, 1) float
        rgb = sidebar_arr[:, :, :3].astype(float)

        # We will produce a per-frame composited RGB array via a make_frame lambda.
        # For a static overlay this is straightforward: blend sidebar over video at x=0.
        sidebar_rgb = rgb  # shape (H, sidebar_w, 3)
        sidebar_alpha = alpha  # shape (H, sidebar_w, 1)

        # image_transform passes the already-retrieved frame array to the
        # function, NOT a time value — so the argument is np.ndarray, not float.
        def _blend_frame(frame: np.ndarray) -> np.ndarray:
            result = frame.astype(float)
            result[:, :sidebar_w, :] = (
                sidebar_rgb * sidebar_alpha + result[:, :sidebar_w, :] * (1 - sidebar_alpha)
            )
            return result.astype(np.uint8)

        composited = clip.image_transform(_blend_frame)
        stack.callback(composited.close)

        # Preserve original audio
        if clip.audio is not None:
            composited = composited.with_audio(clip.audio)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        composited.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            logger=None,
            threads=4,
        )

    return output_path


def concatenate_videos(video_paths: list[Path], output_path: Path) -> Path:
    """Concatenate *video_paths* in order and write the result to *output_path*.

    Parameters
    ----------
    video_paths:
        Ordered list of MP4 paths (intro, section_0, section_1, ..., outro).
    output_path:
        Destination MP4 path for the assembled video.
    """
    if not video_paths:
        raise ValueError("concatenate_videos: video_paths must not be empty")

    moviepy_logger = logging.getLogger("moviepy")

    with contextlib.ExitStack() as stack:
        stack.callback(moviepy_logger.setLevel, moviepy_logger.level)
        moviepy_logger.setLevel(logging.ERROR)

        clips = [VideoFileClip(str(p)) for p in video_paths]
        for c in clips:
            stack.callback(c.close)

        final = _moviepy_concat(clips, method="compose")
        stack.callback(final.close)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            logger=None,
            threads=4,
        )

    return output_path
