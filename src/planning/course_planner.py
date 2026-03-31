"""
Course Planner Agent

Analyzes course materials and extracts a structured plan of topics and subtopics
suitable for lesson generation.
"""

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm_metrics import LLMUsage, extract_llm_usage, make_openrouter_llm
from src.core.paths import COURSE_PLANNER_PROMPT
from src.core.settings import LESSON_PLANNER_MODEL
from src.planning.models import CoursePlan
from src.preprocessing.batch_process import batch_process

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 100_000


class CoursePlanner:
    """Agent that analyzes course materials and produces a structured course plan."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or LESSON_PLANNER_MODEL
        self._llm = make_openrouter_llm(self._model, title="Course Planner")
        self._prompt_template = COURSE_PLANNER_PROMPT.read_text()

    def plan(self, course_dir: Path | str) -> tuple[CoursePlan, LLMUsage]:
        """Analyze course materials and return a structured plan.

        Args:
            course_dir: Path to course directory (e.g., "./courses/kosys").
                        Expects a "raw" subdirectory with source materials.

        Returns:
            Tuple of (CoursePlan, LLMUsage).
        """
        course_path = Path(course_dir).resolve()
        raw_dir = course_path / "raw"
        processed_dir = course_path / "processed"

        if not raw_dir.is_dir():
            raise ValueError(f"Expected 'raw' subdirectory under {course_dir}")

        # Preprocess raw materials
        logger.info("Preprocessing course materials from %s", raw_dir)
        batch_process(raw_dir, processed_dir)

        # Collect processed content
        content_parts = self._collect_content(processed_dir)

        # Generate plan via LLM
        plan, usage = self._generate_plan(course_path.name, content_parts)

        return plan, usage

    def _collect_content(self, processed_dir: Path) -> list[dict[str, Any]]:
        """Collect text content from processed directory."""
        content_parts: list[dict[str, Any]] = []
        total_chars = 0

        # Prioritize lecture materials
        for md_file in sorted(processed_dir.rglob("*.md")):
            if total_chars >= _MAX_CONTENT_CHARS:
                logger.warning("Content truncated at %d chars", _MAX_CONTENT_CHARS)
                break

            rel_path = md_file.relative_to(processed_dir)
            text = md_file.read_text(errors="replace")

            # Truncate individual files if needed
            remaining = _MAX_CONTENT_CHARS - total_chars
            if len(text) > remaining:
                text = text[:remaining] + "\n[... truncated ...]"

            content_parts.append(
                {
                    "type": "text",
                    "text": f"--- File: {rel_path} ---\n{text}",
                }
            )
            total_chars += len(text)

        return content_parts

    def _generate_plan(
        self,
        course_name: str,
        content_parts: list[dict[str, Any]],
    ) -> tuple[CoursePlan, LLMUsage]:
        """Call LLM to generate the course plan."""
        system_message = SystemMessage(content=self._prompt_template)

        if content_parts:
            user_content: list[str | dict[str, Any]] = [
                {
                    "type": "text",
                    "text": f"Analyze the following course materials for: {course_name}\n",
                },
                *content_parts,
            ]
        else:
            user_content = [
                {
                    "type": "text",
                    "text": f"Create a course plan for: {course_name} (no materials provided)",
                }
            ]

        messages = [system_message, HumanMessage(content=user_content)]

        # Use structured output to get CoursePlan directly
        structured_llm = self._llm.with_structured_output(CoursePlan)
        response: Any = structured_llm.invoke(messages)

        # Extract usage from the underlying response if available
        usage = LLMUsage()
        usage = extract_llm_usage(response)

        if not isinstance(response, CoursePlan):
            raise TypeError(f"Unexpected response type: {type(response)}")

        return response, usage


if __name__ == "__main__":
    import argparse

    import yaml

    from src.core.paths import OUTPUT_DIR

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate a course plan from materials.")
    parser.add_argument("directory", type=Path, help="Course directory (with raw/ subdirectory)")
    args = parser.parse_args()

    planner = CoursePlanner()
    plan, cost = planner.plan(args.directory)

    # Write to output/course_plan.yml
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "course_plan.yml"
    output_path.write_text(yaml.dump(plan.model_dump(), sort_keys=False, allow_unicode=True))

    print(f"\nCourse plan written to: {output_path}")
    cost_str = f"${cost.cost_usd:.6f}" if cost.cost_usd is not None else "n/a"
    print(
        f"Tokens: {cost.total_tokens} (prompt={cost.prompt_tokens}, "
        f"completion={cost.completion_tokens}) cost={cost_str}"
    )
