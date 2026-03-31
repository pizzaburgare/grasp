"""
Course Generation Workflow
Orchestrates: Input Processing → Lesson Planning → Script Generation → Render → Merge
"""

import base64
import binascii
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.cache import (
    get_cached_video,
    get_lesson_cache_dir,
    hash_context,
    lesson_name_to_key,
    save_video_to_cache,
)
from src.core.llm_metrics import LLMUsage, UsageTracker, extract_llm_usage, make_openrouter_llm
from src.core.paths import (
    CACHE_DIR,
    INPUT_DIR,
    LESSON_PROMPT,
    MANIM_PROMPT,
)
from src.core.settings import (
    DEFAULT_TTS_ENGINE,
    LESSON_PLANNER_MODEL,
    MANIM_GENERATOR_MODEL,
    MAX_SCRIPT_ITERATIONS,
    VIDEO_REVIEW_MODEL,
    tts_config_fingerprint,
)
from src.preprocessing import DocumentSelectorAgent
from src.preprocessing.batch_process import batch_process
from src.rendering import render_and_merge
from src.scripting import ManimScriptGenerator

load_dotenv()
logger = logging.getLogger(__name__)


class PipelineContext(TypedDict):
    """Context for a single pipeline run."""

    topic: str
    prompt_topic: str
    slug: str
    input_dir: str | None
    out: Path
    script_hash: str
    video_hash: str
    tts_engine: str
    final_quality: bool
    skip_review: bool


def _try_decode_topic(topic: str) -> tuple[str, bool]:
    """Auto-detect base64 topics and return (decoded_or_original, was_decoded)."""
    # Fast-fail: Valid base64 strings (with padding) are always a multiple of 4 in length.
    # We also check for truthiness to avoid processing empty strings.
    if not topic or len(topic) % 4 != 0:
        return topic, False

    try:
        # validate=True strictly enforces the base64 alphabet, throwing binascii.Error if it fails
        decoded = base64.b64decode(topic, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return topic, False
    return decoded, True


def _rel(path: Path | str) -> Path:
    """Return path relative to cwd, or absolute if not possible."""
    try:
        return Path(path).relative_to(Path.cwd())
    except ValueError:
        return Path(path)


def _print_pipeline_header(ctx: PipelineContext, wf: "CourseWorkflow") -> None:
    """Print pipeline configuration header."""
    print("=" * 60)
    print("Starting AI Course Generation Pipeline")
    print(f"Topic:          {ctx['topic']}")
    if ctx["topic"] != ctx["prompt_topic"]:
        print(f"Prompt topic:   {ctx['prompt_topic']}")
    print(f"Planner model:  {wf.model}")
    print(f"Script model:   {wf.manim_model}")
    print(f"Review model:   {wf.review_model}")
    print(f"Fix model:      {wf.script_generator.fix_model}")
    print(f"TTS engine:     {ctx['tts_engine']}")
    if ctx["input_dir"]:
        print(f"Input dir:      {_rel(ctx['input_dir'])}")
    print(f"Output dir:     {_rel(ctx['out'])}")
    print(f"Cache dir:      {_rel(CACHE_DIR)}")
    print(f"Script hash:    {ctx['script_hash']}")
    print(f"Video hash:     {ctx['video_hash']}")
    print("=" * 60)


class CourseWorkflow:
    """Orchestrates the full course generation pipeline."""

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
        """Generate a lesson plan for the given topic using the planner LLM."""
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

    def _run_single_iteration(self, *, slug: str, script_path: Path, out: Path) -> Path:
        """Run a single low-quality render iteration."""
        return render_and_merge(script_path, out, slug, False, lesson_name=slug, context_hash=None)

    def _handle_render_error(
        self, *, exc: Exception, script_path: Path, topic: str, iter_label: str
    ) -> LLMUsage:
        """Fix a compilation error and return the usage."""
        error_text = str(exc)
        print(f"  Render failed {iter_label}: {error_text[:200]}")
        current_script = script_path.read_text()
        fixed, fix_usage = self.script_generator.fix_compilation_error(
            script=current_script, error_output=error_text, topic=topic
        )
        script_path.write_text(fixed)
        print(f"  Overwritten script: {script_path}")
        return fix_usage

    def _review_and_update(
        self, *, script_path: Path, video_path: Path, topic: str
    ) -> tuple[bool, LLMUsage]:
        """Review video and update script if needed. Returns (changed, usage)."""
        current_script = script_path.read_text()
        new_script, changed, review_usage = self.script_generator.review_video(
            script=current_script, video_path=video_path, topic=topic
        )
        if changed:
            script_path.write_text(new_script)
            print(f"  Overwritten script: {script_path}")
        return changed, review_usage

    def _render_loop(self, ctx: PipelineContext, script_path: Path, tracker: UsageTracker) -> Path:
        """Iterate render/review until video passes or max iterations reached."""
        slug, topic, out = ctx["slug"], ctx["prompt_topic"], ctx["out"]
        max_iters = MAX_SCRIPT_ITERATIONS
        final_video: Path | None = None
        self.script_generator.reset_review_cache()

        for iteration in range(max_iters):
            iter_label = f"[iter {iteration + 1}/{max_iters}]"
            print(f"Step 3 {iter_label}: Render + merge (low quality)")

            try:
                final_video = self._run_single_iteration(
                    slug=slug, script_path=script_path, out=out
                )
            except (RuntimeError, FileNotFoundError) as exc:
                if iteration >= max_iters - 1:
                    raise
                fix_usage = self._handle_render_error(
                    exc=exc, script_path=script_path, topic=topic, iter_label=iter_label
                )
                tracker.record(f"Step 3 {iter_label} - error fix", fix_usage)
                continue

            if ctx["skip_review"] or iteration >= max_iters - 1:
                break

            changed, review_usage = self._review_and_update(
                script_path=script_path, video_path=final_video, topic=topic
            )
            tracker.record(f"Step 3 {iter_label} - video review", review_usage)
            if not changed:
                break

        if final_video is None:
            raise RuntimeError("All render iterations failed.")

        if ctx["final_quality"]:
            print("Step 3 [final]: Render + merge (high quality)")
            final_video = render_and_merge(
                script_path, out, slug, True, lesson_name=slug, context_hash=ctx["video_hash"]
            )
        else:
            save_video_to_cache(slug, ctx["video_hash"], final_video)

        return final_video

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def _handle_cached_video(self, ctx: PipelineContext) -> dict[str, Any] | None:
        """Check for cached video and return result dict if found, None otherwise."""
        slug, video_hash, script_hash = ctx["slug"], ctx["video_hash"], ctx["script_hash"]
        out, topic = ctx["out"], ctx["topic"]

        cached_video = get_cached_video(slug, video_hash)
        if not cached_video:
            return None

        tracker = UsageTracker()
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

    def _process_input_materials(
        self, input_dir: str, prompt_topic: str, tracker: UsageTracker
    ) -> list[dict[str, Any]]:
        """Process input directory and return LLM-ready content parts."""
        input_path = Path(input_dir)
        raw_dir = (input_path / "raw").resolve()
        processed_dir = (input_path / "processed").resolve()

        assert raw_dir.is_dir(), f"Expected 'raw' subdirectory under {input_dir} for course inputs"

        print("Preprocessing raw input files ...")
        preprocessing_usage = batch_process(raw_dir, processed_dir)
        selector_agent = DocumentSelectorAgent(processed_dir)

        selected, selection_usage = selector_agent.select(prompt_topic)
        tracker.record("Step 0 - Selecting documents", selection_usage)

        cost_str = (
            f"${selection_usage.cost_usd:.6f}" if selection_usage.cost_usd is not None else "n/a"
        )
        print(f"Selected files cost={cost_str}")
        for path in selected:
            print(f"  - {path.resolve().relative_to(processed_dir).as_posix()}")

        input_parts = [
            {
                "type": "text",
                "text": f"--- File: {path.resolve().relative_to(processed_dir).as_posix()} ---\n"
                f"{path.read_text(errors='replace')}",
            }
            for path in selected
        ]

        tracker.record("Step -1 - Preparing input for lesson plan", preprocessing_usage)
        preprocessing_cost_str = (
            f"${preprocessing_usage.cost_usd:.4f}"
            if preprocessing_usage.cost_usd is not None
            else "n/a"
        )
        print(
            f"{len(input_parts)} file(s), "
            f"total LLM cost for preprocessing: {preprocessing_cost_str}"
        )
        return input_parts

    def _generate_script(self, ctx: PipelineContext, tracker: UsageTracker) -> tuple[Path, str]:
        """Generate or load cached lesson plan and script. Returns (script_path, lesson)."""
        slug, script_hash = ctx["slug"], ctx["script_hash"]
        prompt_topic, input_dir = ctx["prompt_topic"], ctx["input_dir"]

        script_path = get_lesson_cache_dir(slug) / "script" / f"{script_hash}.py"
        lesson_plan_path = script_path.with_suffix(".md")

        if script_path.exists():
            print(f"Cache hit (script): {script_path}")
            lesson = lesson_plan_path.read_text() if lesson_plan_path.exists() else ""
            tracker.record("Step 1 - Lesson planning", skipped=True)
            tracker.record("Step 2 - Script generation", skipped=True)
            return script_path, lesson

        input_parts: list[dict[str, Any]] | None = None
        if input_dir:
            input_parts = self._process_input_materials(input_dir, prompt_topic, tracker)
        else:
            tracker.record("Step 0 - Selecting documents", skipped=True)

        lesson, lesson_usage = self.generate_lesson_plan(prompt_topic, input_parts=input_parts)
        tracker.record("Step 1 - Lesson planning", lesson_usage)
        lesson_plan_path.parent.mkdir(parents=True, exist_ok=True)
        lesson_plan_path.write_text(lesson)
        print(f"Lesson plan: {lesson_plan_path}\n")

        script_usage = self.script_generator.generate_and_save(
            lesson_content=lesson,
            topic=prompt_topic,
            output_path=script_path,
            input_parts=input_parts,
        )
        tracker.record("Step 2 - Script generation", script_usage)
        return script_path, lesson

    def run_full_pipeline(
        self,
        topic: str,
        input_dir: str | None = None,
        output_dir: str = "output",
        *,
        final_quality: bool = False,
        skip_review: bool = False,
        user_script_hash: str | None = None,
    ) -> dict[str, Any]:
        """Run the complete course generation pipeline from topic to final video."""
        prompt_topic, _ = _try_decode_topic(topic)
        tts_engine = os.environ.get("TTS_ENGINE", DEFAULT_TTS_ENGINE).lower()

        if input_dir is None and INPUT_DIR.is_dir():
            input_dir = str(INPUT_DIR)

        ctx: PipelineContext = {
            "topic": topic,
            "prompt_topic": prompt_topic,
            "slug": lesson_name_to_key(topic),
            "input_dir": input_dir,
            "out": Path(output_dir),
            "script_hash": user_script_hash
            or hash_context(topic, input_dir, extra_context=self._build_script_context()),
            "video_hash": hash_context(
                topic, input_dir, extra_context=self._build_cache_context(tts_engine, final_quality)
            ),
            "tts_engine": tts_engine,
            "final_quality": final_quality,
            "skip_review": skip_review,
        }

        _print_pipeline_header(ctx, self)

        # Fast path: cached video
        if cached := self._handle_cached_video(ctx):
            return cached

        tracker = UsageTracker()
        script_path, lesson = self._generate_script(ctx, tracker)
        print()

        final_video = self._render_loop(ctx, script_path, tracker)

        print()
        print("=" * 60)
        print("Pipeline complete.")
        print(f"Final video : {final_video}")
        print("=" * 60)

        return {
            "output_dir": str(ctx["out"]),
            "lesson_plan": lesson,
            "script_path": str(script_path),
            "final_video": str(final_video),
            "topic": topic,
            "openrouter_usage": tracker.summarize(),
        }
