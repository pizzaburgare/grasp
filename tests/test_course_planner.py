"""Tests for the course planning module."""

import pytest

from src.planning.models import CoursePlan, Subtopic, Topic

EXPECTED_SUBTOPIC_COUNT_TOPIC = 2
EXPECTED_SUBTOPIC_COUNT_PLAN = 3


class TestSubtopic:
    def test_create_subtopic(self) -> None:
        subtopic = Subtopic(
            id="1.1",
            name="Introduction to Linear Equations",
            description="Basic concepts of linear equations",
        )
        assert subtopic.id == "1.1"
        assert subtopic.name == "Introduction to Linear Equations"


class TestTopic:
    def test_create_topic(self) -> None:
        topic = Topic(
            id="1",
            name="Linear Algebra",
            description="Fundamentals of linear algebra",
        )
        assert topic.id == "1"
        assert topic.subtopics == []

    def test_topic_with_subtopics(self) -> None:
        topic = Topic(
            id="1",
            name="Linear Algebra",
            description="Fundamentals of linear algebra",
            subtopics=[
                Subtopic(id="1.1", name="Intro", description="Introduction"),
                Subtopic(id="1.2", name="Vectors", description="Vector basics"),
            ],
        )
        assert len(topic.subtopics) == EXPECTED_SUBTOPIC_COUNT_TOPIC


class TestCoursePlan:
    @pytest.fixture
    def sample_plan(self) -> CoursePlan:
        return CoursePlan(
            course_name="Test Course",
            description="A test course",
            topics=[
                Topic(
                    id="1",
                    name="Topic One",
                    description="First topic",
                    subtopics=[
                        Subtopic(id="1.1", name="Subtopic 1.1", description="First subtopic"),
                        Subtopic(id="1.2", name="Subtopic 1.2", description="Second subtopic"),
                    ],
                ),
                Topic(
                    id="2",
                    name="Topic Two",
                    description="Second topic",
                    subtopics=[
                        Subtopic(id="2.1", name="Subtopic 2.1", description="Third subtopic"),
                    ],
                ),
            ],
        )

    def test_get_all_subtopics(self, sample_plan: CoursePlan) -> None:
        subtopics = sample_plan.get_all_subtopics()
        assert len(subtopics) == EXPECTED_SUBTOPIC_COUNT_PLAN
        assert [s.id for s in subtopics] == ["1.1", "1.2", "2.1"]

    def test_get_subtopic_by_id(self, sample_plan: CoursePlan) -> None:
        subtopic = sample_plan.get_subtopic_by_id("1.2")
        assert subtopic is not None
        assert subtopic.name == "Subtopic 1.2"

    def test_get_subtopic_by_id_not_found(self, sample_plan: CoursePlan) -> None:
        subtopic = sample_plan.get_subtopic_by_id("99.99")
        assert subtopic is None

    def test_serialization_roundtrip(self, sample_plan: CoursePlan) -> None:
        json_str = sample_plan.model_dump_json()
        restored = CoursePlan.model_validate_json(json_str)
        assert restored.course_name == sample_plan.course_name
        assert len(restored.get_all_subtopics()) == EXPECTED_SUBTOPIC_COUNT_PLAN
