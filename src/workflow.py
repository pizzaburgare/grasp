"""
Course Generation Workflow
Orchestrates: Input Processing → Lesson Planning → Script Generation → Render → Merge
"""

import ast
import hashlib
import logging
import os
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
    lesson_name_to_key,
    save_video_to_cache,
)
from src.llm_metrics import LLMUsage, extract_llm_usage
from src.paths import (
    CACHE_AUDIO_DIR,
    CACHE_DIR,
    CACHE_MANIM_DIR,
    INPUT_DIR,
    LESSON_PROMPT,
    MANIM_PROMPT,
)
from src.settings import DEFAULT_LLM_MODEL, DEFAULT_TTS_ENGINE, MAX_SCRIPT_ITERATIONS
from src.script_generator import ManimScriptGenerator

load_dotenv()
logger = logging.getLogger(__name__)


class CourseWorkflow:
    def __init__(self, model: str = DEFAULT_LLM_MODEL):
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
        self.manim_prompt_template = MANIM_PROMPT.read_text()
        self._last_lesson_usage: LLMUsage | None = None

    @staticmethod
    def _prompt_digest(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def _tts_config_fingerprint(tts_engine: str) -> dict[str, str]:
        """Return env vars that materially affect the selected TTS output."""
        keys_by_engine: dict[str, tuple[str, ...]] = {
            "qwen": (
                "QWEN_TTS_MODEL",
                "QWEN_TTS_SPEAKER",
                "QWEN_TTS_LANGUAGE",
                "QWEN_TTS_REF_AUDIO",
                "QWEN_TTS_REF_TEXT",
            ),
            "piper": ("PIPER_MODEL",),
            "kokoro": ("KOKORO_VOICE", "KOKORO_LANG_CODE", "KOKORO_SPEED"),
        }
        keys = keys_by_engine.get(tts_engine, ())
        return {
            "engine": tts_engine,
            **{key: os.environ.get(key, "") for key in keys},
        }

    def _build_cache_context(
        self, tts_engine: str, final_quality: bool
    ) -> dict[str, Any]:
        """Build deterministic metadata that should invalidate cached script/video."""
        return {
            "model": self.model,
            "quality": "high" if final_quality else "low",
            "lesson_prompt": self._prompt_digest(self.lesson_prompt_template),
            "manim_prompt": self._prompt_digest(self.manim_prompt_template),
            "tts": self._tts_config_fingerprint(tts_engine),
        }

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
        input_files: Optional[list[str]] = None,
    ) -> str:
        print(f"Generating lesson plan for: {topic}")
        if input_files:
            print(f"  Using {len(input_files)} input file(s):")
            for f in input_files:
                print(f"    - {f}")
        else:
            print("  No input files — using topic name only")
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

        def _base_name(base: ast.expr) -> str:
            if isinstance(base, ast.Name):
                return base.id
            if isinstance(base, ast.Attribute):
                return base.attr
            return ""

        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        scene_like: list[str] = []
        changed = True

        while changed:
            changed = False
            for cls in classes:
                if cls.name in scene_like:
                    continue
                base_names = [_base_name(base) for base in cls.bases]
                if any(
                    name
                    and (
                        name == "Scene" or name.endswith("Scene") or name in scene_like
                    )
                    for name in base_names
                ):
                    scene_like.append(cls.name)
                    changed = True

        if scene_like:
            return scene_like[-1]

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
            final_clip = None

            # Suppress MoviePy verbose output during merge
            moviepy_logger = logging.getLogger("moviepy")
            old_level = moviepy_logger.level
            moviepy_logger.setLevel(logging.ERROR)
            try:
                video_clip = VideoFileClip(str(video_path))
                try:
                    audio_clip = AudioFileClip(str(merged_audio))
                    try:
                        final_clip = video_clip.with_audio(audio_clip)
                        final_clip.write_videofile(
                            str(final_path),
                            codec="libx264",
                            audio_codec="aac",
                            logger=None,
                        )
                    finally:
                        if final_clip is not None and hasattr(final_clip, "close"):
                            final_clip.close()
                        audio_clip.close()
                finally:
                    video_clip.close()
            finally:
                moviepy_logger.setLevel(old_level)

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
        tts_engine = os.environ.get("TTS_ENGINE", DEFAULT_TTS_ENGINE).lower()

        # Auto-detect default input directory when none is specified
        if input_dir is None and INPUT_DIR.is_dir():
            input_dir = str(INPUT_DIR)

        context_hash = hash_context(
            topic,
            input_dir,
            extra_context=self._build_cache_context(tts_engine, final_quality),
        )
        self._last_lesson_usage = None
        if hasattr(self.script_generator, "last_generation_usage"):
            self.script_generator.last_generation_usage = None
        usage_steps: list[tuple[str, LLMUsage | None, bool]] = []

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
        input_files: list[str] | None = None
        if input_dir:
            input_path = Path(input_dir)
            input_files = [
                f.relative_to(input_path).as_posix()
                for f in sorted(input_path.rglob("*"))
                if f.is_file() and not f.name.startswith(".")
            ]
            print("Processing input files ...")
            input_parts = process_input_dir(input_dir)
            image_count = sum(1 for p in input_parts if p["type"] == "image_url")
            print(f"   {len(input_files)} file(s) → {image_count} image parts")

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
            lesson = self.generate_lesson_plan(
                topic, input_parts=input_parts, input_files=input_files
            )
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

        # Step 3: Render + merge (with iterative review / error-fix loop)
        max_iters = MAX_SCRIPT_ITERATIONS
        final_video: Path | None = None

        for iteration in range(max_iters):
            iter_label = f"[iter {iteration + 1}/{max_iters}]"
            print(f"Step 3 {iter_label}: Render + merge")

            try:
                final_video = self.render_and_merge(
                    script_path,
                    out,
                    slug,
                    final_quality,
                    lesson_name=slug,
                    context_hash=context_hash,
                )
            except (RuntimeError, FileNotFoundError) as exc:
                # Compilation / render failure
                if iteration >= max_iters - 1:
                    raise  # exhausted retries

                error_text = str(exc)
                print(f"  Render failed {iter_label}: {error_text[:200]}")
                current_script = script_path.read_text()
                fixed = self.script_generator.fix_compilation_error(
                    script=current_script,
                    error_output=error_text,
                    topic=topic,
                    lesson_content=lesson,
                )
                script_path.write_text(fixed)
                print(f"  Overwritten script: {script_path}")

                fix_usage = getattr(self.script_generator, "last_fix_usage", None)
                if not isinstance(fix_usage, LLMUsage):
                    fix_usage = None
                usage_steps.append(
                    (f"Step 3 {iter_label} — error fix", fix_usage, False)
                )
                self._print_openrouter_step(
                    f"Step 3 {iter_label} — error fix", usage=fix_usage
                )
                continue  # retry render

            # Render succeeded — ask the LLM to review the video
            if iteration >= max_iters - 1:
                break  # last iteration, skip review

            current_script = script_path.read_text()
            new_script, changed = self.script_generator.review_video(
                script=current_script,
                video_path=final_video,
                topic=topic,
                lesson_content=lesson,
            )

            review_usage = getattr(self.script_generator, "last_review_usage", None)
            if not isinstance(review_usage, LLMUsage):
                review_usage = None
            usage_steps.append(
                (f"Step 3 {iter_label} — video review", review_usage, False)
            )
            self._print_openrouter_step(
                f"Step 3 {iter_label} — video review", usage=review_usage
            )

            if not changed:
                break  # video approved, early exit

            script_path.write_text(new_script)
            print(f"  Overwritten script: {script_path}")

        if final_video is None:
            raise RuntimeError("All render iterations failed.")

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
        return lesson_name_to_key(topic)
