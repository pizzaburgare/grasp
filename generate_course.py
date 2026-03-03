"""
Complete Automated Workflow for AI Course Generation
Orchestrates: Lesson Planning -> Manim Script Generation -> Video/Audio -> Merge
"""

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from moviepy import AudioFileClip, VideoFileClip
from pydantic import SecretStr

from create_lesson.input_processor import process_input_dir
from create_lesson.script_generator import ManimScriptGenerator

load_dotenv()
logger = logging.getLogger(__name__)


class CourseWorkflow:
    """Orchestrates the complete course generation pipeline"""

    def __init__(
        self,
        course: str = "FMNF05",
        planning_model: str = "google/gemini-3.1-pro-preview",
        script_model: str = "google/gemini-3.1-pro-preview",
    ):
        """
        Initialize the workflow

        Args:
            course: Course code (e.g., "FMNF05")
            planning_model: Model for lesson planning (cheap model recommended)
            script_model: Model for Manim script generation
        """
        self.course = course
        self.planning_model = planning_model

        # Initialize lesson planner LLM
        self.planner_llm = ChatOpenAI(
            model=planning_model,
            api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Math Lesson Planner",
            },
        )

        # Initialize script generator
        self.script_generator = ManimScriptGenerator(model=script_model)

        # Load lesson planning prompt
        prompt_path = "create_lesson/prompt.md"
        with open(prompt_path, "r") as f:
            self.lesson_prompt_template = f.read()

    def generate_lesson_plan(
        self,
        topic: str,
        input_parts: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate a detailed lesson plan for the topic.

        Args:
            topic: The topic to teach (e.g., "LU Decomposition")
            input_parts: Optional multimodal content parts from input_processor

        Returns:
            Generated lesson content as string
        """
        print(f"📚 Generating lesson plan for: {topic}")
        print(f"🤖 Using planning model: {self.planning_model}")

        system_content = self.lesson_prompt_template.replace("<topic>", topic)

        # Build the user message with optional multimodal input
        user_content: list[dict] | str
        if input_parts:
            user_content = [
                {
                    "type": "text",
                    "text": f"Create a lesson plan for: {topic}\n\nHere are reference materials:",
                },
                *input_parts,
            ]
        else:
            user_content = f"Create a lesson plan for: {topic}"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]

        response = self.planner_llm.invoke(messages)
        lesson_content = (
            response.content if hasattr(response, "content") else str(response)
        )

        print("✅ Lesson plan generated")
        return lesson_content

    @staticmethod
    def _detect_scene_class(script_path: Path) -> str:
        """Parse the script to find the first Scene subclass name."""
        import ast

        tree = ast.parse(script_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    name = getattr(base, "id", getattr(base, "attr", ""))
                    if name == "Scene":
                        return node.name
        raise ValueError(f"No Scene subclass found in {script_path}")

    def render_and_merge(self, script_path: Path, out: Path) -> Path:
        """
        1. Run manim at low quality to render the video into out/
        2. Merge the TTS audio clips with the video
        Returns the path to the final merged video.
        """
        scene_class = self._detect_scene_class(script_path)
        print(f"🎬 Rendering scene: {scene_class}")

        env = os.environ.copy()
        env["AUDIO_OUTPUT_DIR"] = str(out)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "manim",
                "-ql",
                "--media_dir",
                str(out),
                str(script_path),
                scene_class,
            ],
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError("Manim render failed — check output above.")

        # Find the rendered MP4 (manim puts it under videos/<stem>/480p15/)
        # Exclude any previously merged _final.mp4 at the root of out/
        mp4_files = [p for p in out.rglob("*.mp4") if not p.name.endswith("_final.mp4")]
        if not mp4_files:
            raise FileNotFoundError(f"No MP4 found under {out} after render")
        video_path = mp4_files[0]
        print(f"✅ Video rendered: {video_path}")

        # Merge audio
        merged_audio = out / "merged_audio.wav"
        if not merged_audio.exists():
            print("⚠️  No merged_audio.wav found — exporting video without audio")
            return video_path

        print("🔊 Merging audio into video …")
        video_clip = VideoFileClip(str(video_path))
        audio_clip = AudioFileClip(str(merged_audio))
        final = video_clip.with_audio(audio_clip)
        final_path = out / f"{scene_class}_final.mp4"
        final.write_videofile(str(final_path), codec="libx264", audio_codec="aac")
        video_clip.close()
        audio_clip.close()
        print(f"✅ Final video: {final_path}")
        # Open the merged video (not the raw manim render)
        subprocess.run(["open", str(final_path)])
        return final_path

    @staticmethod
    def _topic_to_slug(topic: str) -> str:
        """Convert a topic name to a filesystem-safe slug"""
        slug = topic.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug).strip("-")
        return slug

    def run_full_pipeline(
        self,
        topic: str,
        input_dir: Optional[str] = None,
        output_dir: str = "output",
        skip_lesson_planning: bool = False,
        lesson_content: Optional[str] = None,
    ) -> dict:
        """
        Run the complete pipeline from topic to Manim script.

        Args:
            topic: The topic to teach
            input_dir: Directory with reference materials (PDFs, videos, images, text)
            output_dir: Where all outputs are placed
            skip_lesson_planning: Skip lesson planning if you provide lesson_content
            lesson_content: Pre-generated lesson content (if skip_lesson_planning=True)

        Returns:
            Dictionary with output_dir, script_path, lesson_plan, and topic
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        script_path = out / "lesson.py"
        lesson_plan_path = out / "lesson_plan.md"

        print("=" * 60)
        print("🚀 Starting AI Course Generation Pipeline")
        print(f"📖 Topic: {topic}")
        print(f"🎓 Course: {self.course}")
        if input_dir:
            print(f"📂 Input dir : {input_dir}")
        print(f"📁 Output dir: {out}")
        print("=" * 60)

        # Step 0: Process input materials (if provided)
        input_parts: list[dict] | None = None
        if input_dir:
            print("📎 Processing input files …")
            input_parts = process_input_dir(input_dir)
            text_count = sum(1 for p in input_parts if p["type"] == "text")
            image_count = sum(1 for p in input_parts if p["type"] == "image_url")
            print(f"   {text_count} text parts, {image_count} image parts")

        # Step 1: Generate or use provided lesson plan
        if not skip_lesson_planning:
            lesson = self.generate_lesson_plan(topic, input_parts=input_parts)
            with open(lesson_plan_path, "w") as f:
                f.write(lesson)
            print(f"💾 Lesson plan saved to: {lesson_plan_path}")
        else:
            if not lesson_content:
                raise ValueError(
                    "Must provide lesson_content if skip_lesson_planning=True"
                )
            lesson = lesson_content
            print("⏭️  Skipping lesson planning, using provided content")

        print()

        # Step 2: Generate Manim script (also receives input context)
        saved_path = self.script_generator.generate_and_save(
            lesson_content=lesson,
            topic=topic,
            output_path=str(script_path),
            input_parts=input_parts,
        )

        print()

        # Step 3: Render video and merge audio
        final_video = self.render_and_merge(script_path, out)

        print()
        print("=" * 60)
        print("✅ Pipeline Complete!")
        print("=" * 60)
        print(f"📄 Lesson plan  : {lesson_plan_path}")
        print(f"🐍 Manim script : {saved_path}")
        print(f"🎥 Final video  : {final_video}")
        print("=" * 60)

        return {
            "output_dir": str(out),
            "lesson_plan": lesson,
            "script_path": saved_path,
            "final_video": str(final_video),
            "topic": topic,
        }


def main():
    """Example usage of the workflow"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate AI course lesson with automated Manim script"
    )
    parser.add_argument(
        "topic", type=str, help="Topic to teach (e.g., 'LU Decomposition')"
    )
    parser.add_argument(
        "--course", type=str, default="FMNF05", help="Course code (default: FMNF05)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory with input materials (PDFs, videos, images, text)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: output/{topic-slug}/)",
    )
    parser.add_argument(
        "--planning-model",
        type=str,
        default="google/gemini-3.1-pro-preview",
        help="Model for lesson planning (must support vision if --input-dir is used)",
    )
    parser.add_argument(
        "--script-model",
        type=str,
        default="google/gemini-3.1-pro-preview",
        help="Model for Manim script generation",
    )

    args = parser.parse_args()

    # Resolve output dir
    output_dir = args.output_dir
    if not output_dir:
        slug = CourseWorkflow._topic_to_slug(args.topic)
        output_dir = str(Path("output") / slug)

    # Create and run workflow
    workflow = CourseWorkflow(
        course=args.course,
        planning_model=args.planning_model,
        script_model=args.script_model,
    )

    workflow.run_full_pipeline(
        topic=args.topic,
        input_dir=args.input_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
