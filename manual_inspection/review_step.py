#!/usr/bin/env python3
"""
Manual inspection tool for the video review step.

Uses ManimScriptGenerator._extract_video_frames — the exact same function
called by the review agent — so what you see here is pixel-for-pixel what
the LLM receives (JPEG-compressed at the production quality setting).

Navigate with ← / → arrow keys or the Prev / Next buttons.

Usage:
    uv run manual_inspection/review_step.py path/to/video.mp4
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

from src.script_generator import ManimScriptGenerator

BG = "#0d0d0d"


# ---------------------------------------------------------------------------
# Frame extraction — delegates entirely to the production code
# ---------------------------------------------------------------------------


def extract_frames(video_path: Path) -> list[tuple[str, np.ndarray]]:
    """Call the real agent extractor and decode the JPEG bytes back to numpy.

    The numpy arrays represent exactly what the LLM would receive, including
    any JPEG compression artefacts from _REVIEW_FRAME_QUALITY.
    """
    parts = ManimScriptGenerator._extract_video_frames(video_path)
    frames: list[tuple[str, np.ndarray]] = []
    for label, img_part in parts:
        data_url: str = img_part["image_url"]["url"]
        # data_url is  "data:image/jpeg;base64,<b64>"
        b64 = data_url.split(",", 1)[1]
        frame = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))
        frames.append((label, frame))
    return frames


# ---------------------------------------------------------------------------
# SSIM — local helper for the inspector UI only (not used by the agent)
# ---------------------------------------------------------------------------


def _global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Image-level SSIM on BT.601 grayscale. Returns value in [-1, 1]."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    def to_gray(x: np.ndarray) -> np.ndarray:
        f = x.astype(np.float64)
        return 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]

    ga, gb = to_gray(a), to_gray(b)
    mu_a, mu_b = ga.mean(), gb.mean()
    var_a, var_b = ga.var(), gb.var()
    cov = float(np.mean((ga - mu_a) * (gb - mu_b)))
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)
    return float(np.clip(num / den, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------


class FrameViewer:
    def __init__(self, frames: list[tuple[str, np.ndarray]]) -> None:
        self.frames = frames
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
        self.fig = plt.figure(figsize=(20, 9), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title("Frame Inspector")  # type: ignore[union-attr]
        except Exception:
            pass

        # Three rows: header | images | controls
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

        # Side-by-side images
        img_gs = gridspec.GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=outer[1],
            wspace=0.03,
        )
        self.ax_prev = self.fig.add_subplot(img_gs[0])
        self.ax_curr = self.fig.add_subplot(img_gs[1])
        for ax in (self.ax_prev, self.ax_curr):
            ax.set_facecolor(BG)
            ax.axis("off")

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

    def _on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key in ("right", "l"):
            self.go(+1)
        elif event.key in ("left", "h"):
            self.go(-1)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
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
            self.sim_text.set_text("SSIM vs prev:  —")
            self.sim_text.set_color("#585b70")
        else:
            pct = sim * 100
            filled = round(pct / 5)  # 20-block bar
            bar = "█" * filled + "░" * (20 - filled)
            if pct >= 90:
                color = "#a6e3a1"  # green  – nearly identical
            elif pct >= 70:
                color = "#f9e2af"  # yellow – noticeable change
            else:
                color = "#f38ba8"  # red    – very different
            self.sim_text.set_text(f"SSIM vs prev:  {pct:5.1f}%   {bar}")
            self.sim_text.set_color(color)

        # Header
        self.hdr_text.set_text(
            f"Frame Inspector  •  {self.idx + 1} of {self.n}  •  ← / → to navigate"
        )

        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run manual_inspection/review_step.py <video.mp4>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    print(f"Extracting frames from: {video_path}")
    frames = extract_frames(video_path)
    if not frames:
        print("No frames could be extracted.")
        sys.exit(1)

    print(f"{len(frames)} unique frames extracted.")
    FrameViewer(frames)


if __name__ == "__main__":
    main()
