"""Structured review and fix models used in video QA and patching."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VideoReview(BaseModel):
    """Structured result from the video review agent."""

    text_clipped: bool = Field(
        description="Text or equations are clipped / cut off at the frame edges."
    )
    overlapping_content: bool = Field(
        description="Content overlaps or is rendered unreadably on top of other content."
    )
    broken_animations: bool = Field(
        description="Visual artifacts, glitches, or misplaced objects are visible."
    )
    content_overflow: bool = Field(
        description="Content extends outside the visible frame boundary."
    )
    latex_rendering: bool = Field(
        description=(
            "LaTeX is incorrectly rendered (broken symbols, blank boxes, malformed equations)."
        )
    )
    notes: str | None = Field(
        default=None,
        description=(
            "Optional 4-8 word description of the problem(s) found. "
            "Only set when at least one criterion is true."
        ),
    )

    @property
    def has_issues(self) -> bool:
        return any(
            [
                self.text_clipped,
                self.overlapping_content,
                self.broken_animations,
                self.content_overflow,
                self.latex_rendering,
            ]
        )

    def failed_criteria(self) -> list[str]:
        labels = {
            "text_clipped": "Text or equations clipped / cut off at the edges",
            "overlapping_content": "Overlapping or unreadable content",
            "broken_animations": "Broken or glitchy animations (artifacts, misplaced objects)",
            "content_overflow": "Content overflowing outside the visible frame",
            "latex_rendering": "Incorrect rendering of LaTeX",
        }
        return [desc for field, desc in labels.items() if getattr(self, field)]


class CodeEdit(BaseModel):
    """A single search/replace edit to apply to the script."""

    old_code: str = Field(
        description="Verbatim substring to find in the source (include 5+ lines of context)."
    )
    new_code: str = Field(description="Replacement text.")


class CodeFix(BaseModel):
    """Structured list of targeted edits from the fix agent."""

    edits: list[CodeEdit] = Field(description="Ordered list of search/replace edits to apply.")
