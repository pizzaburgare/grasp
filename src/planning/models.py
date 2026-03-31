"""Pydantic models for course planning."""

from pydantic import BaseModel, Field


class Subtopic(BaseModel):
    """A single lesson-sized subtopic within a main topic."""

    id: str = Field(description="Subtopic ID in format 'X.Y' (e.g., '1.2')")
    name: str = Field(description="Short, descriptive name of the subtopic")
    description: str = Field(description="Brief description of what this subtopic covers")


class Topic(BaseModel):
    """A main concept/topic containing multiple subtopics."""

    id: str = Field(description="Topic ID (e.g., '1', '2')")
    name: str = Field(description="Name of the main topic/concept")
    description: str = Field(description="Overview of what this topic covers")
    subtopics: list[Subtopic] = Field(
        default_factory=list,
        description="Ordered list of subtopics, each small enough for one lesson",
    )


class CoursePlan(BaseModel):
    """Complete course plan with topics and subtopics."""

    course_name: str = Field(description="Name of the course")
    description: str = Field(description="Brief description of the course")
    topics: list[Topic] = Field(
        default_factory=list,
        description="Ordered list of main topics/concepts",
    )

    def get_all_subtopics(self) -> list[Subtopic]:
        """Return a flat list of all subtopics in order."""
        return [subtopic for topic in self.topics for subtopic in topic.subtopics]

    def get_subtopic_by_id(self, subtopic_id: str) -> Subtopic | None:
        """Look up a subtopic by its ID."""
        for topic in self.topics:
            for subtopic in topic.subtopics:
                if subtopic.id == subtopic_id:
                    return subtopic
        return None
