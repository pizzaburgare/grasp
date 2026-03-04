"""
Course Generation Workflow
Orchestrates: Input Processing → Lesson Planning → Script Generation → Render → Merge
"""

import ast
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from moviepy import AudioFileClip, VideoFileClip
from pydantic import SecretStr

from src.input_processor import process_input_dir
from src.cache import (
    get_audio_cache_dir,
    get_cached_video,
    get_lesson_cache_dir,
    hash_context,
    save_video_to_cache,
)
from src.paths import (
    CACHE_AUDIO_DIR,
    CACHE_DIR,
    CACHE_MANIM_DIR,
    LESSON_PROMPT,
)
from src.script_generator import ManimScriptGenerator

load_dotenv()
logger = logging.getLogger(__name__)


class CourseWorkflow:
    def __init__(self, model: str = "google/gemini-3.1-pro-preview"):
        self.model = model

        self.planner_llm = ChatOpenAI(
            model=model,
            api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Math Lesson Planner",
            },
        )

        self.script_generator = ManimScriptGenerator(model=model)

        self.lesson_prompt_template = LESSON_PROMPT.read_text()

    # ------------------------------------------------------------------
    # Lesson planning
    # ------------------------------------------------------------------

    def generate_lesson_plan(
        self,
        topic: str,
        input_parts: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        print(f"Generating lesson plan for: {topic}")
        print(f"Using model: {self.model}")

        system_content = self.lesson_prompt_template.replace("<topic>", topic)

        if input_parts:
            user_content: list[str | dict[str, Any]] | str = [
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
        print("Lesson plan generated")
        return str(lesson_content)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_scene_class(script_path: Path) -> str:
        tree = ast.parse(script_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    name = getattr(base, "id", getattr(base, "attr", ""))
                    if name == "Scene":
                        return node.name
        raise ValueError(f"No Scene subclass found in {script_path}")

    def render_and_merge(
        self,
        script_path: Path,
        output_dir: Path,
        topic_slug: str,
        final_quality: bool = False,
        *,
        lesson_name: str | None = None,
        context_hash: str | None = None,
    ) -> Path:
        """Render the Manim script and merge TTS audio into the final video.

        When *lesson_name* and *context_hash* are supplied:
        - ``AUDIO_CACHE_DIR`` is set to ``<lesson>/audio/`` (persistent hash store)
        - ``AUDIO_OUTPUT_DIR`` is set to ``<lesson>/audio/work/`` (numbered working
          files + ``merged_audio.wav``; cleaned up on success)
        - The finished video is written into the video cache.
        """
        import shutil as _shutil

        scene_class = self._detect_scene_class(script_path)
        quality_flag = "-qh" if final_quality else "-ql"
        quality_label = "high" if final_quality else "low"
        print(f"Rendering scene: {scene_class} ({quality_label} quality)")

        if lesson_name:
            audio_cache_dir = get_audio_cache_dir(lesson_name)
            audio_work_dir = audio_cache_dir / "work"
        else:
            audio_cache_dir = CACHE_AUDIO_DIR
            audio_work_dir = CACHE_AUDIO_DIR
        cache_manim = CACHE_MANIM_DIR
        audio_cache_dir.mkdir(parents=True, exist_ok=True)
        audio_work_dir.mkdir(parents=True, exist_ok=True)
        cache_manim.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["AUDIO_OUTPUT_DIR"] = str(audio_work_dir)
        if lesson_name:
            env["AUDIO_CACHE_DIR"] = str(audio_cache_dir)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "manim",
                quality_flag,
                "--media_dir",
                str(cache_manim),
                str(script_path),
                scene_class,
            ],
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError("Manim render failed — check output above.")

        # Find the rendered mp4 (exclude partial movie files)
        mp4_files = [
            p
            for p in cache_manim.rglob("*.mp4")
            if "partial_movie_files" not in p.parts
        ]
        if not mp4_files:
            raise FileNotFoundError(f"No MP4 found under {cache_manim} after render")
        video_path = mp4_files[0]
        print(f"Video rendered: {video_path}")

        # Merge audio
        merged_audio = audio_work_dir / "merged_audio.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{topic_slug}.mp4"

        if not merged_audio.exists():
            print("No merged_audio.wav found — copying video without audio")
            _shutil.copy2(video_path, final_path)
        else:
            print("Merging audio into video ...")
            video_clip = VideoFileClip(str(video_path))
            audio_clip = AudioFileClip(str(merged_audio))
            final = video_clip.with_audio(audio_clip)
            final.write_videofile(str(final_path), codec="libx264", audio_codec="aac")
            video_clip.close()
            audio_clip.close()

        print(f"Final video: {final_path}")

        if lesson_name and context_hash:
            save_video_to_cache(lesson_name, context_hash, final_path)
            print(
                f"Video cached: {get_lesson_cache_dir(lesson_name) / 'video' / f'{context_hash}.mp4'}"
            )

        # Clean up working audio files — hash-named cache files are preserved
        if audio_work_dir != audio_cache_dir:
            _shutil.rmtree(audio_work_dir, ignore_errors=True)

        return final_path

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        topic: str,
        input_dir: Optional[str] = None,
        output_dir: str = "output",
        final_quality: bool = False,
    ) -> dict:
        import shutil

        slug = self._topic_to_slug(topic)
        out = Path(output_dir)
        context_hash = hash_context(topic, input_dir)

        tts_engine = os.environ.get("TTS_ENGINE", "kokoro").lower()

        print("=" * 60)
        print("Starting AI Course Generation Pipeline")
        print(f"Topic:        {topic}")
        print(f"LLM model:    {self.model}")
        print(f"TTS engine:   {tts_engine}")
        if input_dir:
            print(f"Input dir:    {input_dir}")
        print(f"Output dir:   {out}")
        print(f"Cache dir:    {CACHE_DIR}")
        print(f"Context hash: {context_hash}")
        print("=" * 60)

        # Fast path: nothing has changed — return the cached final video immediately
        cached_video = get_cached_video(slug, context_hash)
        if cached_video:
            print()
            print("=" * 60)
            print("Nothing has changed — all assets already cached.")
            print(f"  script : .cache/{slug}/script/{context_hash}.py")
            print(f"  video  : .cache/{slug}/video/{context_hash}.mp4")
            print("=" * 60)
            out.mkdir(parents=True, exist_ok=True)
            final_path = out / f"{slug}.mp4"
            shutil.copy2(cached_video, final_path)
            print(f"Final video: {final_path}")
            return {
                "output_dir": str(out),
                "lesson_plan": None,
                "script_path": None,
                "final_video": str(final_path),
                "topic": topic,
                "cache_hit": "video",
            }

        # Step 0: Process input materials
        input_parts: list[dict[str, Any]] | None = None
        if input_dir:
            print("Processing input files ...")
            input_parts = process_input_dir(input_dir)
            text_count = sum(1 for p in input_parts if p["type"] == "text")
            image_count = sum(1 for p in input_parts if p["type"] == "image_url")
            print(f"   {text_count} text parts, {image_count} image parts")

        # Step 1+2: Lesson plan & Manim script (both keyed by context hash)
        script_path = get_lesson_cache_dir(slug) / "script" / f"{context_hash}.py"
        lesson_plan_path = script_path.with_suffix(".md")

        if script_path.exists():
            print(f"Cache hit (script): {script_path}")
            lesson = lesson_plan_path.read_text() if lesson_plan_path.exists() else ""
        else:
            # Step 1: Generate lesson plan
            lesson = self.generate_lesson_plan(topic, input_parts=input_parts)
            lesson_plan_path.parent.mkdir(parents=True, exist_ok=True)
            lesson_plan_path.write_text(lesson)
            print(f"Lesson plan: {lesson_plan_path}")
            print()
            # Step 2: Generate Manim script
            self.script_generator.generate_and_save(
                lesson_content=lesson,
                topic=topic,
                output_path=script_path,
                input_parts=input_parts,
            )

        print()

        # Step 3: Render + merge
        final_video = self.render_and_merge(
            script_path,
            out,
            slug,
            final_quality,
            lesson_name=slug,
            context_hash=context_hash,
        )

        print()
        print("=" * 60)
        print("Pipeline complete.")
        print(f"Final video : {final_video}")
        print("=" * 60)

        return {
            "output_dir": str(out),
            "lesson_plan": lesson,
            "script_path": str(script_path),
            "final_video": str(final_video),
            "topic": topic,
        }

    @staticmethod
    def _topic_to_slug(topic: str) -> str:
        slug = topic.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug).strip("-")
        return slug
