"""
CLI entry point for the AI Course Lesson Generator.

Usage (after uv sync):
  uv run lesson "LU Decomposition"
  uv run lesson "Fourier Transform" --input-dir ./slides
  uv run lesson "QR Decomposition" --final
  uv run lesson --lesson-plan ./output/course_plan.yml --id 1.1
"""

import argparse
import sys
from pathlib import Path

import yaml

from src.core.settings import (
    LESSON_PLANNER_MODEL,
    MANIM_GENERATOR_MODEL,
    VIDEO_REVIEW_MODEL,
)
from src.planning.models import CoursePlan
from src.workflow import CourseWorkflow


def _load_subtopic_from_plan(plan_path: Path, subtopic_id: str) -> tuple[str, str]:
    """Load a subtopic from a course plan YAML file.

    Returns (name, description).
    """
    if not plan_path.exists():
        print(f"Error: Course plan file not found: {plan_path}", file=sys.stderr)
        sys.exit(1)

    plan_data = yaml.safe_load(plan_path.read_text())
    plan = CoursePlan.model_validate(plan_data)

    subtopic = plan.get_subtopic_by_id(subtopic_id)
    if subtopic is None:
        available_ids = [s.id for s in plan.get_all_subtopics()]
        print(f"Error: Subtopic '{subtopic_id}' not found in plan.", file=sys.stderr)
        print(f"Available IDs: {', '.join(available_ids)}", file=sys.stderr)
        sys.exit(1)

    return subtopic_id + "_" + subtopic.name, subtopic.description


def main() -> None:
    """Parse CLI arguments and run the course video generation workflow."""
    parser = argparse.ArgumentParser(
        description="Generate an AI course lesson video with narration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "topic",
        type=str,
        nargs="?",
        default=None,
        help='Topic to teach, e.g. "LU Decomposition"',
    )
    parser.add_argument(
        "--lesson-plan",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to course plan YAML file (use with --id)",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        dest="subtopic_id",
        metavar="ID",
        help="Subtopic ID from the course plan (e.g., '1.1')",
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
    parser.add_argument(
        "--script-hash",
        type=str,
        default=None,
        metavar="HASH",
        help=(
            "Reuse a specific cached script hash. Useful for testing rendering "
            "(e.g., TTS/quality changes) without regenerating the lesson plan "
            "and script."
        ),
    )

    args = parser.parse_args()

    # Determine topic and context from args
    topic: str
    lesson_context: str | None = None

    if args.lesson_plan and args.subtopic_id:
        topic, lesson_context = _load_subtopic_from_plan(args.lesson_plan, args.subtopic_id)
    elif args.lesson_plan or args.subtopic_id:
        print("Error: --lesson-plan and --id must be used together.", file=sys.stderr)
        sys.exit(1)
    elif args.topic:
        topic = args.topic
    else:
        print("Error: Either provide a topic or use --lesson-plan with --id.", file=sys.stderr)
        sys.exit(1)

    workflow = CourseWorkflow(model=args.model)
    workflow.run_full_pipeline(
        topic=topic,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        final_quality=args.final,
        skip_review=args.skip_review,
        user_script_hash=args.script_hash,
        lesson_context=lesson_context,
    )
