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
import time
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
from src.llm_metrics import LLMUsage, extract_llm_usage
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
        self._last_lesson_usage: LLMUsage | None = None

    @staticmethod
    def _format_cost(cost_usd: float | None) -> str:
        return f"${cost_usd:.6f}" if cost_usd is not None else "n/a"

    def _print_openrouter_step(
        self,
        label: str,
        usage: LLMUsage | None,
        *,
        skipped: bool = False,
    ) -> None:
        if skipped:
            print(f"{label}: cache hit — OpenRouter call skipped.")
            return

        if usage is None:
            print(f"{label}: OpenRouter usage unavailable.")
            return

        print(
            f"{label}: prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}, "
            f"cost={self._format_cost(usage.cost_usd)}"
        )

    def _print_openrouter_summary(
        self,
        steps: list[tuple[str, LLMUsage | None, bool]],
    ) -> dict[str, Any]:
        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        known_cost_total = 0.0
        has_unknown_cost = False

        print()
        print("OpenRouter usage summary")
        print("-" * 60)

        for label, usage, skipped in steps:
            if skipped:
                print(f"{label}: skipped (cache)")
                continue

            if usage is None:
                print(f"{label}: usage unavailable")
                has_unknown_cost = True
                continue

            total_prompt += usage.prompt_tokens
            total_completion += usage.completion_tokens
            total_tokens += usage.total_tokens

            if usage.cost_usd is None:
                has_unknown_cost = True
            else:
                known_cost_total += usage.cost_usd

            print(
                f"{label}: prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, "
                f"total={usage.total_tokens}, "
                f"cost={self._format_cost(usage.cost_usd)}"
            )

        print("-" * 60)
        print(
            "TOTAL: "
            f"prompt={total_prompt}, completion={total_completion}, total={total_tokens}"
        )
        if has_unknown_cost:
            print(f"TOTAL COST: {known_cost_total:.6f} USD + unknown")
        else:
            print(f"TOTAL COST: {known_cost_total:.6f} USD")

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "known_cost_usd": known_cost_total,
            "has_unknown_cost": has_unknown_cost,
        }

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
        self._last_lesson_usage = extract_llm_usage(response)
        lesson_content = (
            response.content if hasattr(response, "content") else str(response)
        )
        print("Lesson plan generated")
        return str(lesson_content)

    # TODO: Use existing library for spinner
    @staticmethod
    def _run_command_with_spinner(
        command: list[str],
        env: dict[str, str],
        status_label: str,
    ) -> subprocess.CompletedProcess[str]:
        spinner = ("|", "/", "-", "\\")
        start = time.perf_counter()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        i = 0
        while process.poll() is None:
            elapsed = time.perf_counter() - start
            print(
                f"\r{status_label} {spinner[i % len(spinner)]} {elapsed:5.1f}s",
                end="",
                flush=True,
            )
            time.sleep(0.2)
            i += 1

        stdout, stderr = process.communicate()
        elapsed = time.perf_counter() - start
        print(f"\r{status_label} ✓ {elapsed:5.1f}s")

        return subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode if process.returncode is not None else 1,
            stdout=stdout,
            stderr=stderr,
        )

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
        env["AUDIO_MANAGER_VERBOSE"] = "0"
        if lesson_name:
            env["AUDIO_CACHE_DIR"] = str(audio_cache_dir)

        result = self._run_command_with_spinner(
            command=[
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
            status_label="Compiling video",
        )
        if result.returncode != 0:
            combined = (result.stderr or "").strip() or (result.stdout or "").strip()
            if combined:
                tail = "\n".join(combined.splitlines()[-30:])
                print("Manim error output (last 30 lines):")
                print(tail)
            raise RuntimeError("Manim render failed — check output above.")

        # Find the rendered mp4 (exclude partial movie files)
        # Sort by modification time and take the most recent to handle persistent cache
        mp4_files = [
            p
            for p in cache_manim.rglob("*.mp4")
            if "partial_movie_files" not in p.parts
        ]
        if not mp4_files:
            raise FileNotFoundError(f"No MP4 found under {cache_manim} after render")
        video_path = max(mp4_files, key=lambda p: p.stat().st_mtime)
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

            # Suppress MoviePy verbose output during merge
            moviepy_logger = logging.getLogger("moviepy")
            old_level = moviepy_logger.level
            moviepy_logger.setLevel(logging.ERROR)
            try:
                final.write_videofile(
                    str(final_path),
                    codec="libx264",
                    audio_codec="aac",
                    logger=None,
                )
            finally:
                moviepy_logger.setLevel(old_level)

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
        self._last_lesson_usage = None
        if hasattr(self.script_generator, "last_generation_usage"):
            self.script_generator.last_generation_usage = None
        usage_steps: list[tuple[str, LLMUsage | None, bool]] = []

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
            usage_steps.extend(
                [
                    ("Step 1 — Lesson planning", LLMUsage(), True),
                    ("Step 2 — Script generation", LLMUsage(), True),
                ]
            )
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
            usage_summary = self._print_openrouter_summary(usage_steps)
            return {
                "output_dir": str(out),
                "lesson_plan": None,
                "script_path": None,
                "final_video": str(final_path),
                "topic": topic,
                "cache_hit": "video",
                "openrouter_usage": usage_summary,
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
            usage_steps.extend(
                [
                    ("Step 1 — Lesson planning", LLMUsage(), True),
                    ("Step 2 — Script generation", LLMUsage(), True),
                ]
            )
            self._print_openrouter_step(
                "Step 1 — Lesson planning",
                usage=None,
                skipped=True,
            )
            self._print_openrouter_step(
                "Step 2 — Script generation",
                usage=None,
                skipped=True,
            )
        else:
            # Step 1: Generate lesson plan
            lesson = self.generate_lesson_plan(topic, input_parts=input_parts)
            usage_steps.append(
                ("Step 1 — Lesson planning", self._last_lesson_usage, False)
            )
            self._print_openrouter_step(
                "Step 1 — Lesson planning",
                usage=self._last_lesson_usage,
            )
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
            script_usage = getattr(self.script_generator, "last_generation_usage", None)
            if not isinstance(script_usage, LLMUsage):
                script_usage = None
            usage_steps.append(("Step 2 — Script generation", script_usage, False))
            self._print_openrouter_step(
                "Step 2 — Script generation",
                usage=script_usage,
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

        usage_summary = self._print_openrouter_summary(usage_steps)

        return {
            "output_dir": str(out),
            "lesson_plan": lesson,
            "script_path": str(script_path),
            "final_video": str(final_video),
            "topic": topic,
            "openrouter_usage": usage_summary,
        }

    @staticmethod
    def _topic_to_slug(topic: str) -> str:
        slug = topic.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug).strip("-")
        return slug
