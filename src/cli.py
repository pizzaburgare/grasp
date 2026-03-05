"""
CLI entry point for the AI Course Lesson Generator.

Usage (after uv sync):
  uv run lesson "LU Decomposition"
  uv run lesson "Fourier Transform" --input-dir ./slides
  uv run lesson "QR Decomposition" --final
"""

import argparse

from src.settings import DEFAULT_LLM_MODEL
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
        default=DEFAULT_LLM_MODEL,
        metavar="MODEL",
        help=f"OpenRouter model for planning and script generation (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Render at high quality (-qh) instead of low quality (-ql)",
    )

    args = parser.parse_args()

    workflow = CourseWorkflow(model=args.model)
    workflow.run_full_pipeline(
        topic=args.topic,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        final_quality=args.final,
    )
