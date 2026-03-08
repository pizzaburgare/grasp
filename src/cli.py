"""
CLI entry point for the AI Course Lesson Generator.

Usage (after uv sync):
  uv run lesson "LU Decomposition"
  uv run lesson "Fourier Transform" --input-dir ./slides
  uv run lesson "QR Decomposition" --final
"""

import argparse

from src.settings import (
    LESSON_PLANNER_MODEL,
    MANIM_GENERATOR_MODEL,
    VIDEO_REVIEW_MODEL,
)
from src.workflow import CourseWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an AI course lesson video with narration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "topic",
        type=str,
        help='Topic to teach, e.g. "LU Decomposition"',
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory with reference materials (PDFs, videos, images, text)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        metavar="DIR",
        help="Where to place the final video (default: ./output)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help=(
            "Override all pipeline stages with a single OpenRouter model. "
            "When omitted, each stage uses its env-configured model "
            f"(planner: {LESSON_PLANNER_MODEL}, manim: {MANIM_GENERATOR_MODEL}, "
            f"review: {VIDEO_REVIEW_MODEL})."
        ),
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Render at high quality (-qh) instead of low quality (-ql)",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip the LLM video-review loop and go straight to the final render",
    )

    args = parser.parse_args()

    workflow = CourseWorkflow(model=args.model)
    workflow.run_full_pipeline(
        topic=args.topic,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        final_quality=args.final,
        skip_review=args.skip_review,
    )
