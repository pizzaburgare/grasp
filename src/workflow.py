"""
Course Generation Workflow
Orchestrates: Input Processing → Lesson Planning → Script Generation → Render → Merge
"""

import ast
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from moviepy import AudioFileClip, VideoFileClip

from src.cache import (
    get_audio_cache_dir,
    get_cached_video,
    get_lesson_cache_dir,
    hash_context,
    lesson_name_to_key,
    save_video_to_cache,
)
from src.llm_metrics import LLMUsage, UsageTracker, extract_llm_usage, make_openrouter_llm
from src.paths import (
    CACHE_AUDIO_DIR,
    CACHE_DIR,
    CACHE_MANIM_DIR,
    INPUT_DIR,
    LESSON_PROMPT,
    MANIM_PROMPT,
)
from src.preprocessing.batch_process import batch_process
from src.script_generator import ManimScriptGenerator
from src.settings import (
    DEFAULT_TTS_ENGINE,
    LESSON_PLANNER_MODEL,
    MANIM_GENERATOR_MODEL,
    MAX_SCRIPT_ITERATIONS,
    VIDEO_REVIEW_MODEL,
    tts_config_fingerprint,
)

load_dotenv()
logger = logging.getLogger(__name__)


def _rel(path: Path | str) -> Path:
    """Return path relative to cwd, or absolute if not possible."""
    try:
        return Path(path).relative_to(Path.cwd())
    except ValueError:
        return Path(path)


def _print_pipeline_header(
    topic: str,
    wf: "CourseWorkflow",
    tts_engine: str,
    input_dir: str | None,
    out: Path,
    script_hash: str,
    video_hash: str,
) -> None:
    print("=" * 60)
    print("Starting AI Course Generation Pipeline")
    print(f"Topic:          {topic}")
    print(f"Planner model:  {wf.model}")
    print(f"Script model:   {wf.manim_model}")
    print(f"Review model:   {wf.review_model}")
    print(f"Fix model:      {wf.script_generator.fix_model}")
    print(f"TTS engine:     {tts_engine}")
    if input_dir:
        print(f"Input dir:      {_rel(input_dir)}")
    print(f"Output dir:     {_rel(out)}")
    print(f"Cache dir:      {_rel(CACHE_DIR)}")
    print(f"Script hash:    {script_hash}")
    print(f"Video hash:     {video_hash}")
    print("=" * 60)


class CourseWorkflow:
    def __init__(self, model: str | None = None) -> None:
        # If a global model override is supplied (e.g. via --model CLI flag)
        # it takes precedence over every stage's env-configured model.
        planner_model = model or LESSON_PLANNER_MODEL
        manim_model = model or MANIM_GENERATOR_MODEL
        review_model = model or VIDEO_REVIEW_MODEL

        # Expose the planner model as self.model for cache-key / logging use.
        self.model = planner_model
        self.manim_model = manim_model
        self.review_model = review_model

        self.planner_llm = make_openrouter_llm(planner_model, title="Math Lesson Planner")

        self.script_generator = ManimScriptGenerator(
            generation_model=manim_model,
            review_model=review_model,
        )

        self.lesson_prompt_template = LESSON_PROMPT.read_text()

    def _build_script_context(self) -> dict[str, Any]:
        """Metadata that determines whether to regenerate the lesson plan + script.

        Intentionally excludes model names, render quality and TTS config so that
        switching models (or TTS engines / quality) reuses the same cached script.
        """
        manim_hash = hashlib.sha256(MANIM_PROMPT.read_bytes()).hexdigest()[:16]
        lesson_hash = hashlib.sha256(self.lesson_prompt_template.encode()).hexdigest()[:16]
        return {"lesson_prompt": lesson_hash, "manim_prompt": manim_hash}

    def _build_cache_context(self, tts_engine: str, final_quality: bool) -> dict[str, Any]:
        """Full metadata including render settings - determines video cache validity."""
        return {
            **self._build_script_context(),
            "quality": "high" if final_quality else "low",
            "tts": tts_config_fingerprint(tts_engine),
        }

    # ------------------------------------------------------------------
    # Lesson planning
    # ------------------------------------------------------------------

    def generate_lesson_plan(
        self,
        topic: str,
        input_parts: list[dict[str, Any]] | None = None,
        input_files: list[str] | None = None,
    ) -> tuple[str, LLMUsage]:
        print(f"Generating lesson plan for: {topic}")
        if input_files:
            print(f"  Using {len(input_files)} input file(s):")
            for f in input_files:
                print(f"    - {f}")
        else:
            print("  No input files - using topic name only")
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
        usage = extract_llm_usage(response)
        lesson_content = str(response.content if hasattr(response, "content") else response)
        print("Lesson plan generated")
        return lesson_content, usage

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

        # Drain stdout/stderr on background threads to prevent the OS pipe
        # buffer (~64 KB) from filling up and deadlocking the subprocess.
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def _drain(pipe: IO[str], buf: list[str]) -> None:
            for chunk in iter(lambda: pipe.read(4096), ""):
                buf.append(chunk)

        t_out = threading.Thread(target=_drain, args=(process.stdout, stdout_chunks), daemon=True)
        t_err = threading.Thread(target=_drain, args=(process.stderr, stderr_chunks), daemon=True)
        t_out.start()
        t_err.start()

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

        t_out.join()
        t_err.join()
        elapsed = time.perf_counter() - start
        print(f"\r{status_label} ✓ {elapsed:5.1f}s")

        return subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode if process.returncode is not None else 1,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
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
                if any(name and (name == "Scene" or name.endswith("Scene") or name in scene_like) for name in base_names):
                    scene_like.append(cls.name)
                    changed = True

        if scene_like:
            return scene_like[-1]

        raise ValueError(f"No Scene subclass found in {script_path}")

    def render_and_merge(  # noqa: PLR0912
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
            raise RuntimeError("Manim render failed - check output above.")

        # Find the rendered mp4 (exclude partial movie files)
        # Sort by modification time and take the most recent to handle persistent cache
        mp4_files = [p for p in cache_manim.rglob("*.mp4") if "partial_movie_files" not in p.parts]
        if not mp4_files:
            raise FileNotFoundError(f"No MP4 found under {cache_manim} after render")
        video_path = max(mp4_files, key=lambda p: p.stat().st_mtime)
        print(f"Video rendered: {video_path}")

        # Merge audio
        merged_audio = audio_work_dir / "merged_audio.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{topic_slug}.mp4"

        if not merged_audio.exists():
            print("No merged_audio.wav found - copying video without audio")
            shutil.copy2(video_path, final_path)
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
            print(f"Video cached: {get_lesson_cache_dir(lesson_name) / 'video' / f'{context_hash}.mp4'}")

        # Clean up working audio files - hash-named cache files are preserved
        if audio_work_dir != audio_cache_dir:
            shutil.rmtree(audio_work_dir, ignore_errors=True)

        return final_path

    # ------------------------------------------------------------------
    # Render + review loop
    # ------------------------------------------------------------------

    def _render_loop(
        self,
        topic: str,
        lesson: str,
        slug: str,
        script_path: Path,
        out: Path,
        video_hash: str,
        final_quality: bool,
        skip_review: bool,
        tracker: UsageTracker,
    ) -> Path:
        max_iters = MAX_SCRIPT_ITERATIONS
        final_video: Path | None = None

        for iteration in range(max_iters):
            iter_label = f"[iter {iteration + 1}/{max_iters}]"
            print(f"Step 3 {iter_label}: Render + merge (low quality)")

            try:
                final_video = self.render_and_merge(
                    script_path,
                    out,
                    slug,
                    False,  # always low quality during iteration
                    lesson_name=slug,
                    context_hash=None,  # don't cache intermediate renders
                )
            except (RuntimeError, FileNotFoundError) as exc:
                if iteration >= max_iters - 1:
                    raise

                error_text = str(exc)
                print(f"  Render failed {iter_label}: {error_text[:200]}")
                current_script = script_path.read_text()
                fixed, fix_usage = self.script_generator.fix_compilation_error(
                    script=current_script,
                    error_output=error_text,
                    topic=topic,
                    lesson_content=lesson,
                )
                script_path.write_text(fixed)
                print(f"  Overwritten script: {script_path}")
                tracker.record(f"Step 3 {iter_label} - error fix", fix_usage)
                continue

            if skip_review or iteration >= max_iters - 1:
                break

            current_script = script_path.read_text()
            new_script, changed, review_usage = self.script_generator.review_video(
                script=current_script,
                video_path=final_video,
                topic=topic,
                lesson_content=lesson,
            )
            tracker.record(f"Step 3 {iter_label} - video review", review_usage)

            if not changed:
                break

            script_path.write_text(new_script)
            print(f"  Overwritten script: {script_path}")

        if final_video is None:
            raise RuntimeError("All render iterations failed.")

        if final_quality:
            print("Step 3 [final]: Render + merge (high quality)")
            final_video = self.render_and_merge(
                script_path,
                out,
                slug,
                final_quality,
                lesson_name=slug,
                context_hash=video_hash,
            )
        else:
            save_video_to_cache(slug, video_hash, final_video)

        return final_video

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        topic: str,
        input_dir: str | None = None,
        output_dir: str = "output",
        final_quality: bool = False,
        skip_review: bool = False,
    ) -> dict[str, Any]:
        slug = lesson_name_to_key(topic)
        out = Path(output_dir)
        tts_engine = os.environ.get("TTS_ENGINE", DEFAULT_TTS_ENGINE).lower()

        if input_dir is None and INPUT_DIR.is_dir():
            input_dir = str(INPUT_DIR)

        script_hash = hash_context(topic, input_dir, extra_context=self._build_script_context())
        video_hash = hash_context(topic, input_dir, extra_context=self._build_cache_context(tts_engine, final_quality))

        _print_pipeline_header(topic, self, tts_engine, input_dir, out, script_hash, video_hash)
        tracker = UsageTracker()

        # Fast path: cached video
        if cached_video := get_cached_video(slug, video_hash):
            tracker.record("Step 1 - Lesson planning", skipped=True)
            tracker.record("Step 2 - Script generation", skipped=True)
            print()
            print("=" * 60)
            print("Nothing has changed - all assets already cached.")
            print(f"  script : .cache/{slug}/script/{script_hash}.py")
            print(f"  video  : .cache/{slug}/video/{video_hash}.mp4")
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
                "openrouter_usage": tracker.summarize(),
            }

        # Process input materials
        input_parts: list[dict[str, Any]] | None = None
        input_files: list[str] | None = None
        if input_dir:
            input_path = Path(input_dir)

            # If input points at a course directory, preprocess raw assets first.
            raw_dir = input_path / "raw"
            processed_dir = input_path / "processed"

            assert raw_dir.is_dir(), f"Expected 'raw' subdirectory under {input_dir} for course inputs"

            print("Preprocessing raw input files ...")
            total_cost, input_parts = batch_process(raw_dir, processed_dir)

            input_files = [
                f.relative_to(processed_dir).as_posix()
                for f in sorted(processed_dir.rglob("*"))
                if f.is_file() and not f.name.startswith(".")
            ]

            print(f"{len(input_files)} file(s), total LLM cost for preprocessing: ${total_cost:.4f}")

        # Steps 1+2: lesson plan + Manim script
        script_path = get_lesson_cache_dir(slug) / "script" / f"{script_hash}.py"
        lesson_plan_path = script_path.with_suffix(".md")

        if script_path.exists():
            print(f"Cache hit (script): {script_path}")
            lesson = lesson_plan_path.read_text() if lesson_plan_path.exists() else ""
            tracker.record("Step 1 - Lesson planning", skipped=True)
            tracker.record("Step 2 - Script generation", skipped=True)
        else:
            lesson, lesson_usage = self.generate_lesson_plan(topic, input_parts=input_parts, input_files=input_files)
            tracker.record("Step 1 - Lesson planning", lesson_usage)
            lesson_plan_path.parent.mkdir(parents=True, exist_ok=True)
            lesson_plan_path.write_text(lesson)
            print(f"Lesson plan: {lesson_plan_path}\n")
            script_usage = self.script_generator.generate_and_save(
                lesson_content=lesson,
                topic=topic,
                output_path=script_path,
                input_parts=input_parts,
            )
            tracker.record("Step 2 - Script generation", script_usage)

        print()

        # Step 3: Render + review loop
        final_video = self._render_loop(topic, lesson, slug, script_path, out, video_hash, final_quality, skip_review, tracker)

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
            "openrouter_usage": tracker.summarize(),
        }
