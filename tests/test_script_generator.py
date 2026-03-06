"""Tests for ManimScriptGenerator helpers (no LLM calls)."""

from __future__ import annotations

import numpy as np
import pytest

from src.script_generator import ManimScriptGenerator


# ---------------------------------------------------------------------------
# _frame_similarity
# ---------------------------------------------------------------------------


class TestFrameSimilarity:
    """Verify the similarity metric catches real content changes
    even on Manim-style (mostly-dark) frames."""

    @staticmethod
    def _make_dark_frame(h: int = 120, w: int = 160) -> np.ndarray:
        """Return a black frame (uint8)."""
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_identical_frames_return_one(self):
        frame = self._make_dark_frame()
        assert ManimScriptGenerator._frame_similarity(frame, frame) == pytest.approx(
            1.0
        )

    def test_completely_different_frames(self):
        black = self._make_dark_frame()
        white = np.full_like(black, 255)
        sim = ManimScriptGenerator._frame_similarity(black, white)
        assert sim == pytest.approx(0.0, abs=0.01)

    def test_small_content_on_dark_background_detected(self):
        """A small bright region on a dark frame must be detected as different.

        This is the scenario that was broken when using cosine similarity:
        two mostly-black frames with different small content areas scored
        > 0.95 cosine similarity because the zero-valued background
        dominated the dot product.
        """
        h, w = 120, 160
        frame_a = self._make_dark_frame(h, w)
        frame_b = self._make_dark_frame(h, w)

        # Draw a bright rectangle in different positions
        frame_a[10:30, 10:50] = 255  # top-left block
        frame_b[80:100, 100:140] = 255  # bottom-right block

        sim = ManimScriptGenerator._frame_similarity(frame_a, frame_b)
        # Must be below the dedup threshold so both frames are kept
        assert sim < 0.95, (
            f"Similarity {sim:.4f} >= 0.95; small content changes on a dark "
            "background are being hidden by the metric"
        )

    def test_same_content_same_position_is_similar(self):
        """Frames with identical content should still score high."""
        frame_a = self._make_dark_frame()
        frame_b = self._make_dark_frame()
        frame_a[10:30, 10:50] = 200
        frame_b[10:30, 10:50] = 200
        sim = ManimScriptGenerator._frame_similarity(frame_a, frame_b)
        assert sim == pytest.approx(1.0)

    def test_slight_brightness_change_is_high_similarity(self):
        """A tiny global brightness bump should score close to 1."""
        frame_a = np.full((120, 160, 3), 100, dtype=np.uint8)
        frame_b = np.full((120, 160, 3), 105, dtype=np.uint8)
        sim = ManimScriptGenerator._frame_similarity(frame_a, frame_b)
        assert sim > 0.99


# ---------------------------------------------------------------------------
# _extract_video_frames  (uses a synthetic clip via numpy)
# ---------------------------------------------------------------------------


class TestExtractVideoFrames:
    """Integration-style tests for frame extraction without a real video file.

    We monkeypatch moviepy's VideoFileClip to return a synthetic clip so we
    don't need an actual video on disk.
    """

    @staticmethod
    def _make_synthetic_frames(n: int, h: int = 120, w: int = 160) -> list[np.ndarray]:
        """Return *n* distinguishable dark frames with unique bright patches."""
        frames = []
        for i in range(n):
            f = np.zeros((h, w, 3), dtype=np.uint8)
            # Place a bright column that shifts right with each frame
            col = (i * 20) % w
            f[:, col : col + 5] = 255
            frames.append(f)
        return frames

    def test_extracts_multiple_distinct_frames(self, monkeypatch):
        """With clearly distinct frames, extraction should keep more than 1."""
        frames = self._make_synthetic_frames(10)
        duration = 10.0

        class FakeClip:
            def __init__(self, path):
                self.duration = duration

            def get_frame(self, t):
                idx = min(int(t), len(frames) - 1)
                return frames[idx]

            def close(self):
                pass

        monkeypatch.setattr("src.script_generator.VideoFileClip", FakeClip)
        from pathlib import Path

        parts = ManimScriptGenerator._extract_video_frames(Path("fake.mp4"))
        assert len(parts) > 1, (
            f"Only {len(parts)} frame(s) extracted from {len(frames)} distinct frames"
        )

    def test_identical_frames_deduplicated_to_one(self, monkeypatch):
        """If every sampled frame is identical, only 1 should be kept."""
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        class FakeClip:
            def __init__(self, path):
                self.duration = 10.0

            def get_frame(self, t):
                return frame.copy()

            def close(self):
                pass

        monkeypatch.setattr("src.script_generator.VideoFileClip", FakeClip)
        from pathlib import Path

        parts = ManimScriptGenerator._extract_video_frames(Path("fake.mp4"))
        assert len(parts) == 1

    def test_zero_duration_returns_empty(self, monkeypatch):
        class FakeClip:
            def __init__(self, path):
                self.duration = 0.0

            def close(self):
                pass

        monkeypatch.setattr("src.script_generator.VideoFileClip", FakeClip)
        from pathlib import Path

        parts = ManimScriptGenerator._extract_video_frames(Path("fake.mp4"))
        assert parts == []


# ---------------------------------------------------------------------------
# VideoReview model
# ---------------------------------------------------------------------------


class TestVideoReview:
    def test_no_issues(self):
        from src.script_generator import VideoReview

        r = VideoReview(
            text_clipped=False,
            overlapping_content=False,
            broken_animations=False,
            content_overflow=False,
            latex_rendering=False,
        )
        assert not r.has_issues
        assert r.failed_criteria() == []

    def test_all_issues(self):
        from src.script_generator import VideoReview

        r = VideoReview(
            text_clipped=True,
            overlapping_content=True,
            broken_animations=True,
            content_overflow=True,
            latex_rendering=True,
        )
        assert r.has_issues
        assert len(r.failed_criteria()) == 5

    def test_single_issue(self):
        from src.script_generator import VideoReview

        r = VideoReview(
            text_clipped=True,
            overlapping_content=False,
            broken_animations=False,
            content_overflow=False,
            latex_rendering=False,
        )
        assert r.has_issues
        assert len(r.failed_criteria()) == 1
        assert "clipped" in r.failed_criteria()[0].lower()
