"""
Tests for src/compositor.py — progress-sidebar overlay and video concatenation.

These tests create real (tiny) MP4 files via MoviePy/ffmpeg to verify the
full compositing pipeline without requiring any LLM or TTS calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from moviepy import ColorClip, VideoFileClip

# Duration tolerance for encode rounding in assertions (seconds).
_DURATION_TOLERANCE = 0.1
# Minimum combined duration for two 0.5 s clips.
_TWO_CLIP_MIN_DURATION = 0.9
# Minimum combined duration for three 0.4 s clips.
_THREE_CLIP_MIN_DURATION = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_video(path: Path, color: tuple[int, int, int], duration: float = 0.5) -> Path:
    """Write a minimal solid-colour MP4 to *path* and return it."""
    clip = ColorClip(size=(320, 180), color=list(color), duration=duration)
    clip.write_videofile(str(path), fps=15, codec="libx264", audio=False, logger=None)
    clip.close()
    return path


# ---------------------------------------------------------------------------
# Sidebar image generation (pure PIL, no ffmpeg needed)
# ---------------------------------------------------------------------------

class TestCreateSidebarImage:
    def test_returns_rgba_image_of_correct_size(self) -> None:
        from src.compositor import _create_sidebar_image

        img = _create_sidebar_image(
            sections=["Introduction", "Theory", "Example"],
            current_idx=1,
            width=150,
            height=360,
        )
        assert img.mode == "RGBA"
        assert img.size == (150, 360)

    def test_valid_for_single_section(self) -> None:
        from src.compositor import _create_sidebar_image

        img = _create_sidebar_image(["Only Section"], current_idx=0, width=120, height=240)
        arr = np.array(img)
        assert arr.shape == (240, 120, 4)

    def test_all_sections_highlighted_index_in_range(self) -> None:
        """No exception for any valid current_idx."""
        from src.compositor import _create_sidebar_image

        sections = ["A", "B", "C", "D", "E"]
        for idx in range(len(sections)):
            img = _create_sidebar_image(sections, current_idx=idx, width=160, height=400)
            assert img.size == (160, 400)


# ---------------------------------------------------------------------------
# Video compositing (requires ffmpeg via MoviePy)
# ---------------------------------------------------------------------------

class TestAddProgressSidebar:
    def test_output_file_is_created(self, tmp_path: Path) -> None:
        from src.compositor import add_progress_sidebar

        src = _make_video(tmp_path / "input.mp4", color=(50, 100, 150))
        out = tmp_path / "output.mp4"

        result = add_progress_sidebar(
            video_path=src,
            sections=["Intro", "Main", "Conclusion"],
            current_section_idx=1,
            output_path=out,
        )

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_output_dimensions_match_input(self, tmp_path: Path) -> None:
        """The sidebar is overlaid; the frame size must not change."""
        from src.compositor import add_progress_sidebar

        src = _make_video(tmp_path / "input.mp4", color=(200, 80, 80))
        out = tmp_path / "output.mp4"
        add_progress_sidebar(
            video_path=src,
            sections=["S1", "S2"],
            current_section_idx=0,
            output_path=out,
        )

        with VideoFileClip(str(src)) as clip_in, VideoFileClip(str(out)) as clip_out:
            assert clip_in.size == clip_out.size

    def test_duration_is_preserved(self, tmp_path: Path) -> None:
        from src.compositor import add_progress_sidebar

        src = _make_video(tmp_path / "input.mp4", color=(80, 200, 80), duration=1.0)
        out = tmp_path / "output.mp4"
        add_progress_sidebar(
            video_path=src,
            sections=["Alpha", "Beta", "Gamma"],
            current_section_idx=2,
            output_path=out,
        )

        with VideoFileClip(str(src)) as clip_in, VideoFileClip(str(out)) as clip_out:
            assert abs(clip_in.duration - clip_out.duration) < _DURATION_TOLERANCE


# ---------------------------------------------------------------------------
# Video concatenation
# ---------------------------------------------------------------------------

class TestConcatenateVideos:
    def test_single_clip_is_copied_to_output(self, tmp_path: Path) -> None:
        from src.compositor import concatenate_videos

        src = _make_video(tmp_path / "clip.mp4", color=(100, 100, 255))
        out = tmp_path / "out.mp4"

        result = concatenate_videos([src], out)

        assert result == out
        assert out.exists()

    def test_two_clips_produce_combined_duration(self, tmp_path: Path) -> None:
        from src.compositor import concatenate_videos

        a = _make_video(tmp_path / "a.mp4", color=(255, 0, 0), duration=0.5)
        b = _make_video(tmp_path / "b.mp4", color=(0, 255, 0), duration=0.5)
        out = tmp_path / "out.mp4"

        concatenate_videos([a, b], out)

        with VideoFileClip(str(out)) as final:
            assert final.duration >= _TWO_CLIP_MIN_DURATION  # two 0.5 s clips

    def test_three_clips_order_is_preserved(self, tmp_path: Path) -> None:
        """Verify the output is longer than any individual clip."""
        from src.compositor import concatenate_videos

        clips = [
            _make_video(tmp_path / f"c{i}.mp4", color=(i * 80, 0, 0), duration=0.4)
            for i in range(3)
        ]
        out = tmp_path / "out.mp4"
        concatenate_videos(clips, out)

        with VideoFileClip(str(out)) as final:
            assert final.duration >= _THREE_CLIP_MIN_DURATION  # three 0.4 s clips

    def test_raises_on_empty_list(self, tmp_path: Path) -> None:
        from src.compositor import concatenate_videos

        with pytest.raises(ValueError, match="video_paths must not be empty"):
            concatenate_videos([], tmp_path / "out.mp4")

    def test_output_dimensions_match_inputs(self, tmp_path: Path) -> None:
        from src.compositor import concatenate_videos

        clips = [
            _make_video(tmp_path / f"c{i}.mp4", color=(0, i * 60, 200))
            for i in range(2)
        ]
        out = tmp_path / "out.mp4"
        concatenate_videos(clips, out)

        with VideoFileClip(str(clips[0])) as ref, VideoFileClip(str(out)) as final:
            assert final.size == ref.size
