import inspect
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from moviepy import VideoFileClip  # type: ignore

from src.review.algorithms.brightness_peaks import lightness_score
from src.review.algorithms.similarity import frame_ssim

# Using the type definition from your code
type Frame = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]

# Variables to change
known_scene_switches = [0.0, 10.0, 20.0]  # Where scene switches are known to occur (in seconds)
video_path = "./output/TMLect13KF9-18.mp4"  # REPLACE WITH YOUR VIDEO PATH
sample_fps = 1


# Algorithm definitions
def lightness_diff(frame1: Frame, frame2: Frame) -> float:
    return 1 - abs(lightness_score(frame1) - lightness_score(frame2))


def absolute_diff(frame1: Frame, frame2: Frame) -> float:
    return float(1 - np.mean(np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))) / 255)


# Map for algorithms: name and function reference
algorithms: list[tuple[str, Callable]] = [
    ("lightness", lightness_score),
    ("ssim_prev_curr", frame_ssim),
    ("lightness_diff", lightness_diff),
    ("absolute_diff", absolute_diff),
]

clip = VideoFileClip(video_path)


# Data structures to hold results
timestamps: list[float] = []
metric_series: dict[str, list[float]] = {name: [] for name, _ in algorithms}
prev_frame: Frame | None = None

print(f"Processing video: {clip.duration} seconds at {clip.fps} FPS...")

for t, frame in clip.iter_frames(with_times=True, fps=sample_fps):
    # Two-frame algorithms need a previous frame; skip the first sampled frame.
    if prev_frame is None:
        prev_frame = frame
        continue

    timestamps.append(t)
    for name, algorithm in algorithms:
        param_count = len(inspect.signature(algorithm).parameters)

        if param_count == 1:
            value = algorithm(frame)
        elif param_count == 2:  # noqa: PLR2004
            value = algorithm(prev_frame, frame)
        else:
            raise ValueError(
                f"Algorithm '{name}' must accept 1 or 2 positional args, got {param_count}."
            )
        metric_series[name].append(float(value))

    prev_frame = frame

clip.close()
print("Processing complete!")


plt.figure(figsize=(14, 6))

for name, values in metric_series.items():
    plt.plot(timestamps, values, linewidth=1.5, label=name)

for i, t_switch in enumerate(known_scene_switches):
    plt.axvline(
        x=t_switch,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label="Known scene switch" if i == 0 else None,
    )

plt.title("Frame Metrics Over Time", fontsize=14)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Metric Value", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()

plt.show()
