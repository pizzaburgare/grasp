"""
Course Generation Workflow
Orchestrates: Input Processing → Lesson Planning → Script Generation → Render → Merge
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from src.cache import (
    get_cached_video,
    get_lesson_cache_dir,
    hash_context,
    lesson_name_to_key,
    save_video_to_cache,
)
from src.document_selector import DocumentSelectorAgent
from src.llm_metrics import LLMUsage, UsageTracker, extract_llm_usage, make_openrouter_llm
from src.paths import (
    CACHE_DIR,
    INPUT_DIR,
    LESSON_PROMPT,
    MANIM_PROMPT,
)
from src.preprocessing.batch_process import batch_process
from src.rendering import render_and_merge
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
    ) -> tuple[str, LLMUsage]:
        print(f"Generating lesson plan for: {topic}")

        if not input_parts:
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
                final_video = render_and_merge(
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
            final_video = render_and_merge(
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
        video_hash = hash_context(
            topic, input_dir, extra_context=self._build_cache_context(tts_engine, final_quality)
        )

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

        # Steps 1+2: lesson plan + Manim script
        script_path = get_lesson_cache_dir(slug) / "script" / f"{script_hash}.py"
        lesson_plan_path = script_path.with_suffix(".md")

        if script_path.exists():
            print(f"Cache hit (script): {script_path}")
            lesson = lesson_plan_path.read_text() if lesson_plan_path.exists() else ""
            tracker.record("Step 1 - Lesson planning", skipped=True)
            tracker.record("Step 2 - Script generation", skipped=True)
        else:
            # Process input materials
            input_parts: list[dict[str, Any]] | None = None

            if input_dir:
                input_path = Path(input_dir)

                # If input points at a course directory, preprocess raw assets first.
                raw_dir = (input_path / "raw").resolve()
                processed_dir = (input_path / "processed").resolve()

                assert raw_dir.is_dir(), (
                    f"Expected 'raw' subdirectory under {input_dir} for course inputs"
                )

                print("Preprocessing raw input files ...")
                total_cost = batch_process(raw_dir, processed_dir)
                selector_agent = DocumentSelectorAgent(processed_dir)

                selected, selection_cost = selector_agent.select(topic)
                cost_str = (
                    f"${selection_cost.cost_usd:.6f}"
                    if selection_cost.cost_usd is not None
                    else "n/a"
                )

                print(f"Selected files cost={cost_str}")
                for path in selected:
                    print(f"  - {path.resolve().relative_to(processed_dir).as_posix()}")
                input_parts = []
                for path in selected:
                    rel_path = path.resolve().relative_to(processed_dir).as_posix()
                    input_parts.append(
                        {
                            "type": "text",
                            "text": f"--- File: {rel_path} ---\n{path.read_text(errors='replace')}",
                        }
                    )

                print(
                    f"{len(input_parts)} file(s),"
                    f"total LLM cost for preprocessing: ${total_cost:.4f}"
                )

            lesson, lesson_usage = self.generate_lesson_plan(topic, input_parts=input_parts)
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
        final_video = self._render_loop(
            topic, lesson, slug, script_path, out, video_hash, final_quality, skip_review, tracker
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
            "openrouter_usage": tracker.summarize(),
        }
