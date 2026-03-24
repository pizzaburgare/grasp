#!/usr/bin/env python3
"""
Manual inspection tool for the video review step.

Uses the same settled-ssim extraction path as production review. The
brightness-peaks mode is implemented here for manual inspection only,
while keeping JPEG encoding compatible with the review payload format.

Navigate with ← / → arrow keys or the Prev / Next buttons.

Usage:
    uv run manual_inspection/review_step.py path/to/video.mp4
    uv run manual_inspection/review_step.py --review path/to/video.mp4
    uv run manual_inspection/review_step.py --all-stills path/to/video.mp4
"""

import argparse
import base64
import contextlib
import io
import sys
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from matplotlib.widgets import Button  # type: ignore
from moviepy import VideoFileClip
from PIL import Image

from src.script_generator import ManimScriptGenerator, VideoReview

BG = "#0d0d0d"
SSIM_GREEN_THRESHOLD = 90
SSIM_YELLOW_THRESHOLD = 70
LIGHTNESS_LUMA_BLACK_THRESHOLD = 16
BRIGHTNESS_SCAN_INTERVAL = 1.0
NON_BLACK_MAX_RATIO = 0.98
BRIGHTNESS_DEDUP_SSIM_THRESHOLD = 0.50
REVIEW_FRAME_QUALITY = 70


# ---------------------------------------------------------------------------
# SSIM - local helper for the inspector UI only (not used by the agent)
# ---------------------------------------------------------------------------


def _global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Image-level SSIM on BT.601 grayscale. Returns value in [-1, 1]."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    def to_gray(x: np.ndarray) -> np.ndarray:
        f = x.astype(np.float64)
        return 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]

    ga, gb = to_gray(a), to_gray(b)
    mu_a, mu_b = ga.mean(), gb.mean()
    var_a, var_b = ga.var(), gb.var()
    cov = float(np.mean((ga - mu_a) * (gb - mu_b)))
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(np.clip(num / den, -1.0, 1.0))


def _lightness_score(frame: np.ndarray) -> float:
    """Return lightness score in [0, 1] as inverse black-pixel ratio."""
    gray = (
        0.299 * frame[:, :, 0].astype(np.float64)
        + 0.587 * frame[:, :, 1].astype(np.float64)
        + 0.114 * frame[:, :, 2].astype(np.float64)
    )
    black_ratio = float(np.mean(gray <= LIGHTNESS_LUMA_BLACK_THRESHOLD))
    return 1.0 - black_ratio


def _sample_every_second(video_path: Path) -> list[tuple[float, np.ndarray]]:
    """Sample one frame every second, plus one at clip end."""
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


def _select_brightness_peak_frames(video_path: Path) -> list[tuple[float, np.ndarray]]:
    """Pick local lightness maxima and drop very similar darker duplicates."""
    sampled = _sample_every_second(video_path)
    if not sampled:
        return []

    lightness = [_lightness_score(frame) for _, frame in sampled]
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
        ssim = ManimScriptGenerator._frame_similarity(prev[1], candidate[1])
        if ssim >= BRIGHTNESS_DEDUP_SSIM_THRESHOLD:
            if candidate[2] > prev[2]:
                ssim_deduped[-1] = candidate
        else:
            ssim_deduped.append(candidate)

    ssim_deduped.sort(key=lambda item: item[2], reverse=True)
    ssim_deduped = ssim_deduped[:50]
    ssim_deduped.sort(key=lambda item: item[0])
    return [(ts, frame) for ts, frame, _ in ssim_deduped]


def _encode_selected_frames(
    selected: list[tuple[float, np.ndarray]],
) -> list[tuple[str, dict[str, Any]]]:
    """Encode selected frames into the same image_url payload shape as production."""
    parts: list[tuple[str, dict[str, Any]]] = []
    for chosen_t, frame in selected:
        img = Image.fromarray(frame)  # type: ignore[arg-type]
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=REVIEW_FRAME_QUALITY)
        data = base64.b64encode(buf.getvalue()).decode()
        label = f"Frame at {ManimScriptGenerator._format_timestamp(chosen_t)}"
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


def _extract_frames_for_manual_review(
    video_path: Path,
    algorithm: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Extract frames for inspector modes while preserving production payload shape."""
    if algorithm == "settled-ssim":
        return ManimScriptGenerator._extract_video_frames(video_path, algorithm=algorithm)
    if algorithm == "brightness-peaks":
        selected = _select_brightness_peak_frames(video_path)
        return _encode_selected_frames(selected)
    raise ValueError(f"Unknown frame extraction algorithm: {algorithm}")


# ---------------------------------------------------------------------------
# LLM review - reuses the production review pipeline per-frame
# ---------------------------------------------------------------------------


def run_review(
    raw_parts: list[tuple[str, dict]],
    topic: str = "unknown",
) -> list[VideoReview | None]:
    """Run the real per-frame review LLM on each extracted frame.

    Returns a list parallel to *raw_parts*: a ``VideoReview`` for every frame.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    gen = ManimScriptGenerator()

    structured_llm = gen.review_llm.with_structured_output(VideoReview, include_raw=True)
    review_text = f"Topic: {topic}\n\n{gen.review_prompt_template}"
    sys_msg = SystemMessage(content=gen.system_prompt)

    reviews: list[VideoReview | None] = []
    for i, (label, img_part) in enumerate(raw_parts, 1):
        user_content: list = [
            {"type": "text", "text": review_text},
            {"type": "text", "text": label},
            img_part,
        ]
        try:
            result = structured_llm.invoke([sys_msg, HumanMessage(content=user_content)])
            review: VideoReview = result["parsed"]  # type: ignore[index]
            reviews.append(review)
            status = (
                f"ISSUES ({', '.join(review.failed_criteria())})" if review.has_issues else "OK"
            )
        except Exception as exc:  # noqa: BLE001
            reviews.append(None)
            status = f"ERROR ({exc})"
        print(f"  [{i}/{len(raw_parts)}] {label}: {status}")

    return reviews


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------


class FrameViewer:
    def __init__(
        self,
        frames: list[tuple[str, np.ndarray]],
        reviews: list[VideoReview | None] | None = None,
        lightness_scores: list[float] | None = None,
    ) -> None:
        self.frames = frames
        self.reviews = reviews
        self.lightness_scores = lightness_scores
        self.n = len(frames)
        self.idx = 0

        # Pre-compute SSIM for each frame vs. the previous one
        self.sims: list[float | None] = [None]
        print("Computing similarity scores ...", end="", flush=True)
        for i in range(1, self.n):
            self.sims.append(_global_ssim(frames[i - 1][1], frames[i][1]))
        print(" done.")

        self._build_ui()
        self._render()
        plt.show()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        has_reviews = self.reviews is not None
        self.fig = plt.figure(figsize=(20, 9), facecolor=BG)
        with contextlib.suppress(Exception):
            self.fig.canvas.manager.set_window_title("Frame Inspector")  # type: ignore[union-attr]

        # Rows: header | images (+ optional review panel) | controls
        outer = gridspec.GridSpec(
            3,
            1,
            figure=self.fig,
            height_ratios=[0.055, 0.845, 0.10],
            hspace=0.05,
            left=0.01,
            right=0.99,
            top=0.99,
            bottom=0.02,
        )

        # Header
        self.ax_hdr = self.fig.add_subplot(outer[0])
        self.ax_hdr.set_facecolor(BG)
        self.ax_hdr.axis("off")
        self.hdr_text = self.ax_hdr.text(
            0.5,
            0.5,
            "",
            transform=self.ax_hdr.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#cdd6f4",
            fontfamily="monospace",
        )

        # Middle row: images + optional review panel
        if has_reviews:
            mid_gs = gridspec.GridSpecFromSubplotSpec(
                1,
                3,
                subplot_spec=outer[1],
                wspace=0.03,
                width_ratios=[0.35, 0.35, 0.30],
            )
        else:
            mid_gs = gridspec.GridSpecFromSubplotSpec(
                1,
                2,
                subplot_spec=outer[1],
                wspace=0.03,
            )

        self.ax_prev = self.fig.add_subplot(mid_gs[0])
        self.ax_curr = self.fig.add_subplot(mid_gs[1])
        for ax in (self.ax_prev, self.ax_curr):
            ax.set_facecolor(BG)
            ax.axis("off")

        # Review panel (only when --review)
        self.ax_review = None
        self.review_text_obj = None
        if has_reviews:
            self.ax_review = self.fig.add_subplot(mid_gs[2])
            self.ax_review.set_facecolor("#1e1e2e")
            self.ax_review.axis("off")
            self.review_text_obj = self.ax_review.text(
                0.05,
                0.95,
                "",
                transform=self.ax_review.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="#cdd6f4",
                fontfamily="monospace",
                wrap=True,
            )

        # Controls row: [prev btn] [similarity bar] [next btn]
        ctrl_gs = gridspec.GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=outer[2],
            wspace=0.05,
            width_ratios=[0.12, 0.76, 0.12],
        )
        ax_pb = self.fig.add_subplot(ctrl_gs[0])
        self.ax_sim = self.fig.add_subplot(ctrl_gs[1])
        ax_nb = self.fig.add_subplot(ctrl_gs[2])

        self.ax_sim.set_facecolor(BG)
        self.ax_sim.axis("off")
        self.sim_text = self.ax_sim.text(
            0.5,
            0.5,
            "",
            transform=self.ax_sim.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            fontfamily="monospace",
        )

        btn_kw: dict = dict(color="#1e1e2e", hovercolor="#313244")
        self.btn_prev = Button(ax_pb, "◀  Prev", **btn_kw)
        self.btn_next = Button(ax_nb, "Next  ▶", **btn_kw)
        for btn in (self.btn_prev, self.btn_next):
            btn.label.set_color("#cdd6f4")
            btn.label.set_fontsize(11)

        self.btn_prev.on_clicked(lambda _: self.go(-1))
        self.btn_next.on_clicked(lambda _: self.go(+1))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def go(self, delta: int) -> None:
        new = (self.idx + delta) % self.n
        if new != self.idx:
            self.idx = new
            self._render()

    def _on_key(self, event: Any) -> None:
        if event.key in ("right", "l"):
            self.go(+1)
        elif event.key in ("left", "h"):
            self.go(-1)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:  # noqa: PLR0912
        label_curr, frame_curr = self.frames[self.idx]

        # Current frame
        self.ax_curr.cla()
        self.ax_curr.imshow(frame_curr, interpolation="lanczos")
        self.ax_curr.set_title(
            f"► {label_curr}   [{self.idx + 1} / {self.n}]",
            color="#cdd6f4",
            fontsize=10,
            fontfamily="monospace",
            pad=6,
        )
        if self.lightness_scores is not None:
            curr_lightness = self.lightness_scores[self.idx] * 100
            self.ax_curr.text(
                0.5,
                -0.06,
                f"Lightness: {curr_lightness:5.1f}%",
                transform=self.ax_curr.transAxes,
                ha="center",
                va="top",
                color="#a6e3a1",
                fontsize=10,
                fontfamily="monospace",
            )
        self.ax_curr.axis("off")

        # Previous frame (or placeholder)
        self.ax_prev.cla()
        self.ax_prev.set_facecolor(BG)
        self.ax_prev.axis("off")
        if self.idx > 0:
            label_prev, frame_prev = self.frames[self.idx - 1]
            self.ax_prev.imshow(frame_prev, interpolation="lanczos")
            self.ax_prev.set_title(
                f"  {label_prev}   [{self.idx} / {self.n}]",
                color="#585b70",
                fontsize=10,
                fontfamily="monospace",
                pad=6,
            )
            if self.lightness_scores is not None:
                prev_lightness = self.lightness_scores[self.idx - 1] * 100
                self.ax_prev.text(
                    0.5,
                    -0.06,
                    f"Lightness: {prev_lightness:5.1f}%",
                    transform=self.ax_prev.transAxes,
                    ha="center",
                    va="top",
                    color="#585b70",
                    fontsize=10,
                    fontfamily="monospace",
                )
        else:
            self.ax_prev.text(
                0.5,
                0.5,
                "(no previous frame)",
                transform=self.ax_prev.transAxes,
                ha="center",
                va="center",
                color="#45475a",
                fontsize=13,
            )

        # Similarity score
        sim = self.sims[self.idx]
        if sim is None:
            self.sim_text.set_text("SSIM vs prev:  -")
            self.sim_text.set_color("#585b70")
        else:
            pct = sim * 100
            filled = round(pct / 5)  # 20-block bar
            bar = "█" * filled + "░" * (20 - filled)
            if pct >= SSIM_GREEN_THRESHOLD:
                color = "#a6e3a1"  # green  - nearly identical
            elif pct >= SSIM_YELLOW_THRESHOLD:
                color = "#f9e2af"  # yellow - noticeable change
            else:
                color = "#f38ba8"  # red    - very different
            self.sim_text.set_text(f"SSIM vs prev:  {pct:5.1f}%   {bar}")
            self.sim_text.set_color(color)

        # Header
        self.hdr_text.set_text(
            f"Frame Inspector  •  {self.idx + 1} of {self.n}  •  ← / → to navigate"
        )

        # Review panel
        if self.review_text_obj is not None and self.reviews is not None:
            review = self.reviews[self.idx]
            if review is None:
                self.review_text_obj.set_text("Review: ERROR")
                self.review_text_obj.set_color("#f38ba8")
            elif review.has_issues:
                lines = ["REVIEW: ISSUES FOUND\n"]
                for criterion in review.failed_criteria():
                    lines.append(f"  • {criterion}")
                self.review_text_obj.set_text("\n".join(lines))
                self.review_text_obj.set_color("#f38ba8")
            else:
                self.review_text_obj.set_text("REVIEW: OK\n\n  No issues detected.")
                self.review_text_obj.set_color("#a6e3a1")

        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect video frames (and optionally run the review LLM)."
    )
    parser.add_argument("video", type=Path, help="Path to the .mp4 video file.")
    parser.add_argument(
        "--review",
        action="store_true",
        help="Run the per-frame review LLM and display results.",
    )
    parser.add_argument(
        "--all-stills",
        action="store_true",
        help="Show every settled non-blank frame (>=1s gap). No SSIM dedup or target limit.",
    )
    parser.add_argument(
        "--topic",
        default="unknown",
        help="Topic string passed to the review prompt (default: 'unknown').",
    )
    parser.add_argument(
        "--algorithm",
        choices=["settled-ssim", "brightness-peaks"],
        default="settled-ssim",
        help=(
            "Frame extraction algorithm: 'settled-ssim' (default production flow) "
            "or 'brightness-peaks' (1-second samples, keep local lightness maxima)."
        ),
    )
    args = parser.parse_args()

    video_path: Path = args.video
    if not video_path.exists():
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    if args.all_stills:
        print(f"Extracting ALL still frames from: {video_path}")
        raw = ManimScriptGenerator._scan_settled_frames(video_path)
        if not raw:
            print("No still frames found.")
            sys.exit(1)
        still_frames = [
            (f"Frame at {ManimScriptGenerator._format_timestamp(t)}", f) for t, f in raw
        ]
        print(f"{len(still_frames)} still frames found (no SSIM dedup, no target limit).")
        FrameViewer(still_frames)
        return

    print(f"Extracting frames from: {video_path}")
    print(f"Using extraction algorithm: {args.algorithm}")
    raw_parts = _extract_frames_for_manual_review(video_path, algorithm=args.algorithm)
    if not raw_parts:
        print("No frames could be extracted.")
        sys.exit(1)

    # Decode JPEG bytes to numpy for display
    decoded_frames: list[tuple[str, np.ndarray]] = []
    lightness_scores: list[float] = []
    for label, img_part in raw_parts:
        data_url: str = img_part["image_url"]["url"]
        b64 = data_url.split(",", 1)[1]
        frame = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))
        decoded_frames.append((label, frame))
        lightness_scores.append(_lightness_score(frame))

    print(f"{len(decoded_frames)} unique frames extracted.")

    reviews: list[VideoReview | None] | None = None
    if args.review:
        print(f"Running review LLM on each frame (topic={args.topic!r}) ...")
        reviews = run_review(raw_parts, topic=args.topic)
        flagged = sum(1 for r in reviews if r is not None and r.has_issues)
        print(f"Review complete: {flagged}/{len(reviews)} frames flagged.")

    FrameViewer(decoded_frames, reviews=reviews, lightness_scores=lightness_scores)


if __name__ == "__main__":
    main()
