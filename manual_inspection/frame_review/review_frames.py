#!/usr/bin/env python3
"""
Script to review saved frames in the flagged and not-flagged directories
using the same review logic as the production pipeline.
"""

import argparse
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add the project root to sys.path so we can import from src and manual_inspection
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def review_frames(show_viewer: bool = True) -> None:
    from manual_inspection.review_step import FrameViewer, run_review
    from src.review.algorithms import lightness_score

    base_dir = Path(__file__).parent

    raw_parts = []
    expected_issues = []
    file_names = []
    categories = ["flagged", "not-flagged"]

    for category in categories:
        cat_dir = base_dir / category
        if not cat_dir.exists():
            continue

        images = (
            list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpeg"))
        )
        for img_path in images:
            with open(img_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")

            mime_type = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
            img_part = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
            }

            raw_parts.append((f"{category}/{img_path.name}", img_part))
            expected_issues.append(category == "flagged")
            file_names.append(f"{category}/{img_path.name}")

    if not raw_parts:
        print("No images found in flagged/not-flagged directories.")
        return

    print(f"Found {len(raw_parts)} frames to review.")
    print("Running review LLM on each frame...\n")

    # run_review is imported from manual_inspection.review_step
    reviews = run_review(raw_parts)

    correct = 0
    total = len(reviews)

    print("\n--- Review Results ---")
    for i, review in enumerate(reviews):
        name = file_names[i]
        expected = expected_issues[i]

        if review is None:
            print(f"[ERROR] {name}: Review failed")
            continue

        actual = review.has_issues

        if expected == actual:
            correct += 1
            print(f"[MATCH] {name}: Expected flagged={expected}, Actual flagged={actual}")
        else:
            issue_str = ", ".join(review.failed_criteria()) if actual else "None"
            print(
                "[MISMATCH] "
                f"{name}: Expected flagged={expected}, "
                f"Actual flagged={actual} (Issues: {issue_str})"
            )

    print(f"\nSummary: {correct}/{total} matched their folder classification.")
    if not show_viewer:
        print("Skipping interactive viewer (--no-viewer).")
        return

    # Decode images for interactive viewer, matching review_step behavior.
    decoded_frames = []
    lightness_scores = []
    for label, img_part in raw_parts:
        data_url = img_part["image_url"]["url"]
        b64 = data_url.split(",", 1)[1]
        frame = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))
        decoded_frames.append((label, frame))
        lightness_scores.append(lightness_score(frame))

    print("Opening interactive frame viewer...")
    FrameViewer(decoded_frames, reviews=reviews, lightness_scores=lightness_scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review saved frames from flagged/not-flagged folders using production review logic."
        )
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run review and print summary only; do not open the interactive viewer.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    review_frames(show_viewer=not args.no_viewer)
