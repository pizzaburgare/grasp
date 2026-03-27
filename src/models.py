"""Shared Pydantic data models for structured LLM outputs."""

from pydantic import BaseModel, Field


class VideoSection(BaseModel):
    """A single logical section of the video, shown in the progress sidebar."""

    name: str = Field(
        description=(
            "Short display name (2-5 words) shown in the progress sidebar, "
            "e.g. 'Derivatives Review', 'Worked Example', 'Real Application'"
        )
    )
    description: str = Field(
        description=(
            "What this section covers — used to guide the Manim script generator "
            "for this specific section"
        )
    )


class StructuredLessonPlan(BaseModel):
    """Structured output from the lesson planner containing both the lesson content
    and the video section metadata needed for the progress sidebar."""

    title: str = Field(
        description="One to two word, engaging video title, e.g. 'Example', 'Real Application'"
    )
    sections: list[VideoSection] = Field(
        description=(
            "3-6 ordered sections that structure the video content from introduction "
            "to application. These names appear verbatim in the progress sidebar."
        )
    )
    lesson_markdown: str = Field(
        description=(
            "Full detailed lesson plan in Markdown format, following the 4-part "
            "lesson structure (Walkthrough, Target Problem, Step-by-Step Solution, "
            "Recap). This is the primary content used to generate each video section."
        )
    )
