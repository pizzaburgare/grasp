"""Run all lessons from a course plan sequentially in fresh subprocesses."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from src.planning.models import CoursePlan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all subtopics in a lesson plan sequentially. "
            "Each lesson runs in its own subprocess."
        )
    )
    parser.add_argument(
        "--lesson-plan",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to course plan YAML file",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory with reference materials (forwarded to lesson CLI)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        metavar="DIR",
        help="Output directory (forwarded to lesson CLI)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help="Model override for all pipeline stages (forwarded to lesson CLI)",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Render each lesson at high quality",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip review loop for each lesson",
    )
    parser.add_argument(
        "--script-hash",
        type=str,
        default=None,
        metavar="HASH",
        help="Reuse a specific cached script hash for each lesson",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining lessons if one lesson fails",
    )
    return parser.parse_args()


def _load_subtopic_ids(plan_path: Path) -> list[str]:
    if not plan_path.exists():
        raise FileNotFoundError(f"Course plan file not found: {plan_path}")

    plan_data = yaml.safe_load(plan_path.read_text())
    plan = CoursePlan.model_validate(plan_data)
    return [subtopic.id for subtopic in plan.get_all_subtopics()]


def _build_command(args: argparse.Namespace, subtopic_id: str) -> list[str]:
    command = [
        "uv",
        "run",
        "lesson",
        "--lesson-plan",
        str(args.lesson_plan),
        "--id",
        subtopic_id,
        "--output-dir",
        args.output_dir,
    ]

    if args.input_dir:
        command.extend(["--input-dir", args.input_dir])
    if args.model:
        command.extend(["--model", args.model])
    if args.final:
        command.append("--final")
    if args.skip_review:
        command.append("--skip-review")
    if args.script_hash:
        command.extend(["--script-hash", args.script_hash])

    return command


def main() -> None:
    args = _parse_args()

    try:
        subtopic_ids = _load_subtopic_ids(args.lesson_plan)
    except (FileNotFoundError, yaml.YAMLError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not subtopic_ids:
        print("No subtopics found in lesson plan.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(subtopic_ids)} lessons in {args.lesson_plan}")

    failures: list[tuple[str, int]] = []

    for index, subtopic_id in enumerate(subtopic_ids, start=1):
        command = _build_command(args, subtopic_id)
        printable_command = shlex.join(command)
        print(f"\n[{index}/{len(subtopic_ids)}] Running: {printable_command}")

        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            print(f"[{index}/{len(subtopic_ids)}] Success: {subtopic_id}")
            continue

        print(
            f"[{index}/{len(subtopic_ids)}] Failed: {subtopic_id} (exit code {result.returncode})",
            file=sys.stderr,
        )
        failures.append((subtopic_id, result.returncode))

        if not args.continue_on_error:
            print("Stopping after first failure. Use --continue-on-error to keep going.")
            sys.exit(result.returncode)

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for subtopic_id, exit_code in failures:
            print(f"- {subtopic_id}: exit code {exit_code}", file=sys.stderr)
        sys.exit(1)

    print("\nAll lessons completed successfully.")


if __name__ == "__main__":
    main()
