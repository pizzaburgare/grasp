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
    get_cached_scene_video,
    get_cached_video,
    get_lesson_cache_dir,
    hash_context,
    lesson_name_to_key,
    save_scene_video_to_cache,
    save_video_to_cache,
)
from src.compositor import add_progress_sidebar, concatenate_videos
from src.document_selector import DocumentSelectorAgent
from src.llm_metrics import LLMUsage, UsageTracker, extract_llm_usage, make_openrouter_llm
from src.models import StructuredLessonPlan
from src.paths import (
    CACHE_DIR,
    INPUT_DIR,
    LESSON_PROMPT,
    MANIM_PROMPT,
    SECTION_SCRIPT_PROMPT,
)
from src.preprocessing.batch_process import batch_process
from src.rendering import render_and_merge
from src.script_generator import ManimScriptGenerator
from src.settings import (
    DEFAULT_TTS_ENGINE,
    FEEDBACK_URL,
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


def _scene_render_hash(script_path: Path, tts_engine: str, final_quality: bool) -> str:
    """Return a 16-char hash unique to the script content + render settings.

    Changes whenever the script file is modified (e.g. after a fix-loop patch)
    or when TTS engine / quality flags change — ensuring the cached video is
    always in sync with what would be produced by a fresh render.
    """
    h = hashlib.sha256()
    h.update(script_path.read_bytes())
    h.update(tts_engine.encode())
    h.update(b"hq" if final_quality else b"lq")
    return h.hexdigest()[:16]


def _sidebar_scene_hash(raw_scene_hash: str, section_names: list[str], idx: int) -> str:
    """Return a 16-char hash for a composited (sidebar-overlaid) section video.

    Depends on the underlying section video hash, the full list of section
    names (shown in the sidebar), and the highlighted index.
    """
    h = hashlib.sha256()
    h.update(raw_scene_hash.encode())
    h.update("|".join(section_names).encode())
    h.update(str(idx).encode())
    return h.hexdigest()[:16]


def _try_generate_qr_code(url: str, output_path: Path) -> bool:
    """Generate a QR-code PNG at *output_path* for *url*.

    Returns ``True`` on success, ``False`` when the ``qrcode`` package is not
    installed (the outro scene will fall back to plain text in that case).
    """
    try:
        import qrcode  # type: ignore[import-untyped]
        from qrcode.image.pil import PilImage  # type: ignore[import-untyped]

        qr: qrcode.QRCode = qrcode.QRCode(  # type: ignore[type-arg]
            version=1,
            error_correction=qrcode.ERROR_CORRECT_L,  # type: ignore[attr-defined]
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img: PilImage = qr.make_image(fill_color="white", back_color="black")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path))
    except ImportError:
        logger.warning("qrcode package not installed - outro will show URL text only.")
        return False
    else:
        return True


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
        """Metadata that determines whether to regenerate the lesson plan + scripts.

        Intentionally excludes model names, render quality and TTS config so that
        switching models (or TTS engines / quality) reuses the same cached scripts.
        """
        manim_hash = hashlib.sha256(MANIM_PROMPT.read_bytes()).hexdigest()[:16]
        section_hash = hashlib.sha256(SECTION_SCRIPT_PROMPT.read_bytes()).hexdigest()[:16]
        lesson_hash = hashlib.sha256(self.lesson_prompt_template.encode()).hexdigest()[:16]
        return {
            "lesson_prompt": lesson_hash,
            "manim_prompt": manim_hash,
            "section_prompt": section_hash,
        }

    def _build_cache_context(self, tts_engine: str, final_quality: bool) -> dict[str, Any]:
        """Full metadata including render settings - determines video cache validity."""
        return {
            **self._build_script_context(),
            "quality": "high" if final_quality else "low",
            "tts": tts_config_fingerprint(tts_engine),
        }

    # ------------------------------------------------------------------
    # Lesson planning (structured output)
    # ------------------------------------------------------------------

    def generate_lesson_plan(
        self,
        topic: str,
        input_parts: list[dict[str, Any]] | None = None,
    ) -> tuple[StructuredLessonPlan, LLMUsage]:
        """Return a structured lesson plan for *topic*.

        Uses LangChain structured output to produce a :class:`StructuredLessonPlan`
        containing the video title, a list of sections (for the progress sidebar),
        and the full lesson Markdown.
        """
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

        structured_llm = self.planner_llm.with_structured_output(
            StructuredLessonPlan, include_raw=True
        )
        result = structured_llm.invoke(messages)
        plan: StructuredLessonPlan = result["parsed"]  # type: ignore[index]
        usage = extract_llm_usage(result["raw"])  # type: ignore[index]

        print(f"Lesson plan generated — {len(plan.sections)} section(s):")
        for i, s in enumerate(plan.sections):
            print(f"  {i + 1}. {s.name}")
        return plan, usage

    # ------------------------------------------------------------------
    # Script generation for all scenes
    # ------------------------------------------------------------------

    def _generate_all_scripts(
        self,
        plan: StructuredLessonPlan,
        topic: str,
        script_dir: Path,
        script_hash: str,
        input_parts: list[dict[str, Any]] | None,
        tracker: UsageTracker,
        qr_image_path: Path | None,
    ) -> dict[str, Path]:
        """Generate intro, per-section, and outro Manim scripts.

        Returns a dict with keys:
            ``"intro"``        → intro script path
            ``"sections"``     → list of section script paths (as JSON-encoded key)
            ``"outro"``        → outro script path
        and section keys ``"section_0"``, ``"section_1"``, etc.
        """
        scripts: dict[str, Path] = {}

        # Intro
        intro_path = script_dir / f"{script_hash}_intro.py"
        self.script_generator.generate_intro_script(plan.title, intro_path)
        scripts["intro"] = intro_path

        # Content sections — one LLM call each
        print(f"Generating {len(plan.sections)} section script(s) ...")
        for idx, section in enumerate(plan.sections):
            sec_path = script_dir / f"{script_hash}_section_{idx}.py"
            usage = self.script_generator.generate_section_script(
                section=section,
                all_sections=plan.sections,
                lesson_markdown=plan.lesson_markdown,
                topic=topic,
                section_idx=idx,
                output_path=sec_path,
                input_parts=input_parts,
            )
            tracker.record(f"Step 2 - Section {idx} script generation", usage)
            scripts[f"section_{idx}"] = sec_path

        # Outro
        outro_path = script_dir / f"{script_hash}_outro.py"
        self.script_generator.generate_outro_script(
            feedback_url=FEEDBACK_URL,
            output_path=outro_path,
            qr_image_path=qr_image_path,
        )
        scripts["outro"] = outro_path

        return scripts

    # ------------------------------------------------------------------
    # Render + review loop (for a single scene)
    # ------------------------------------------------------------------

    def _render_loop(
        self,
        topic: str,
        slug: str,
        script_path: Path,
        scene_tag: str,
        out: Path,
        final_quality: bool,
        skip_review: bool,
        tracker: UsageTracker,
    ) -> Path:
        """Render one scene with iterative error-fix and visual-review cycles.

        Returns the path of the rendered (low-quality) video.  High-quality
        re-render is intentionally deferred to after all scenes are composed.
        """
        max_iters = MAX_SCRIPT_ITERATIONS
        final_video: Path | None = None

        self.script_generator.reset_review_cache()

        for iteration in range(max_iters):
            iter_label = f"[iter {iteration + 1}/{max_iters}]"
            print(f"  Render {iter_label}: {scene_tag}")

            try:
                final_video = render_and_merge(
                    script_path,
                    out,
                    slug,
                    False,  # always low quality during iteration
                    lesson_name=slug,
                    context_hash=None,  # don't cache intermediate renders
                    scene_tag=scene_tag,
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
                )
                script_path.write_text(fixed)
                tracker.record(f"{scene_tag} {iter_label} - error fix", fix_usage)
                continue

            if skip_review or iteration >= max_iters - 1:
                break

            current_script = script_path.read_text()
            new_script, changed, review_usage = self.script_generator.review_video(
                script=current_script,
                video_path=final_video,
                topic=topic,
            )
            tracker.record(f"{scene_tag} {iter_label} - video review", review_usage)

            if not changed:
                break

            script_path.write_text(new_script)
            print(f"  Overwritten script: {script_path}")

        if final_video is None:
            raise RuntimeError(f"All render iterations failed for scene: {scene_tag}")

        return final_video

    def _prepare_input_materials(
        self,
        topic: str,
        input_dir: str,
        tracker: UsageTracker,
    ) -> list[dict[str, Any]]:
        """Preprocess raw course files and select the most relevant documents."""
        input_path = Path(input_dir)
        raw_dir = (input_path / "raw").resolve()
        processed_dir = (input_path / "processed").resolve()

        assert raw_dir.is_dir(), (
            f"Expected 'raw' subdirectory under {input_dir} for course inputs"
        )

        print("Preprocessing raw input files ...")
        preprocessing_usage = batch_process(raw_dir, processed_dir)
        selector_agent = DocumentSelectorAgent(processed_dir)

        selected, selection_usage = selector_agent.select(topic)
        tracker.record("Step 0 - Selecting documents", selection_usage)

        cost_str = (
            f"${selection_usage.cost_usd:.6f}"
            if selection_usage.cost_usd is not None
            else "n/a"
        )
        print(f"Selected files cost={cost_str}")
        for path in selected:
            print(f"  - {path.resolve().relative_to(processed_dir).as_posix()}")

        input_parts: list[dict[str, Any]] = []
        for path in selected:
            rel_path = path.resolve().relative_to(processed_dir).as_posix()
            input_parts.append(
                {
                    "type": "text",
                    "text": f"--- File: {rel_path} ---\n{path.read_text(errors='replace')}",
                }
            )
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

    def _ensure_plan_and_scripts(  # noqa: PLR0912
        self,
        topic: str,
        slug: str,
        script_hash: str,
        input_dir: str | None,
        tracker: UsageTracker,
    ) -> StructuredLessonPlan:
        """Load or regenerate the lesson plan and all per-scene Manim scripts.

        Unlike the previous all-or-nothing approach, this method checks each
        ``.py`` file individually and only regenerates what is missing.  This
        means a single deleted or corrupted script no longer forces all expensive
        LLM section-generation calls to re-run.

        Returns the :class:`StructuredLessonPlan` (from cache or freshly
        generated) after ensuring every scene script file exists on disk.
        """
        script_dir = get_lesson_cache_dir(slug) / "script"
        lesson_plan_path = script_dir / f"{script_hash}.json"

        # ------------------------------------------------------------------
        # Partial or full cache hit: plan already exists on disk
        # ------------------------------------------------------------------
        if lesson_plan_path.exists():
            plan = StructuredLessonPlan.model_validate_json(lesson_plan_path.read_text())
            print(f"Cache hit (lesson plan): {lesson_plan_path}")
            tracker.record("Step 1 - Lesson planning", skipped=True)

            intro_path = script_dir / f"{script_hash}_intro.py"
            outro_path = script_dir / f"{script_hash}_outro.py"
            missing_section_idxs = [
                i
                for i in range(len(plan.sections))
                if not (script_dir / f"{script_hash}_section_{i}.py").exists()
            ]
            need_intro = not intro_path.exists()
            need_outro = not outro_path.exists()

            if not need_intro and not need_outro and not missing_section_idxs:
                print("Cache hit (all scripts) - skipping generation.")
                tracker.record("Step 0 - Selecting documents", skipped=True)
                tracker.record("Step 2 - Script generation", skipped=True)
                return plan

            # Partial hit — log what is missing
            missing_parts: list[str] = []
            if need_intro:
                missing_parts.append("intro")
            if missing_section_idxs:
                missing_parts.append(
                    f"{len(missing_section_idxs)} section(s) "
                    f"({', '.join(str(i) for i in missing_section_idxs)})"
                )
            if need_outro:
                missing_parts.append("outro")
            print(f"Partial script cache — regenerating: {', '.join(missing_parts)}")

            # Only call the expensive document-selector when LLM section scripts
            # are actually missing.  Template scripts (intro/outro) need no input.
            input_parts: list[dict[str, Any]] | None = None
            if missing_section_idxs and input_dir:
                input_parts = self._prepare_input_materials(topic, input_dir, tracker)
            else:
                tracker.record("Step 0 - Selecting documents", skipped=True)

            qr_image_path = script_dir / f"{script_hash}_qr.png"
            if not qr_image_path.exists():
                _try_generate_qr_code(FEEDBACK_URL, qr_image_path)

            script_dir.mkdir(parents=True, exist_ok=True)

            if need_intro:
                self.script_generator.generate_intro_script(plan.title, intro_path)

            for idx in missing_section_idxs:
                sec_path = script_dir / f"{script_hash}_section_{idx}.py"
                usage = self.script_generator.generate_section_script(
                    section=plan.sections[idx],
                    all_sections=plan.sections,
                    lesson_markdown=plan.lesson_markdown,
                    topic=topic,
                    section_idx=idx,
                    output_path=sec_path,
                    input_parts=input_parts,
                )
                tracker.record(f"Step 2 - Section {idx} script generation", usage)

            if need_outro:
                self.script_generator.generate_outro_script(
                    feedback_url=FEEDBACK_URL,
                    output_path=outro_path,
                    qr_image_path=qr_image_path if qr_image_path.exists() else None,
                )

            return plan

        # ------------------------------------------------------------------
        # Full cache miss: generate lesson plan + all scripts from scratch
        # ------------------------------------------------------------------
        input_parts_full: list[dict[str, Any]] | None = None
        if input_dir:
            input_parts_full = self._prepare_input_materials(topic, input_dir, tracker)
        else:
            tracker.record("Step 0 - Selecting documents", skipped=True)

        plan, lesson_usage = self.generate_lesson_plan(topic, input_parts=input_parts_full)
        tracker.record("Step 1 - Lesson planning", lesson_usage)

        script_dir.mkdir(parents=True, exist_ok=True)
        lesson_plan_path.write_text(plan.model_dump_json(indent=2))
        (script_dir / f"{script_hash}.md").write_text(plan.lesson_markdown)
        print(f"Lesson plan saved: {lesson_plan_path}")

        qr_image_path = script_dir / f"{script_hash}_qr.png"
        _try_generate_qr_code(FEEDBACK_URL, qr_image_path)

        print(f"\nGenerating Manim scripts ({self.manim_model}) ...")
        self._generate_all_scripts(
            plan=plan,
            topic=topic,
            script_dir=script_dir,
            script_hash=script_hash,
            input_parts=input_parts_full,
            tracker=tracker,
            qr_image_path=qr_image_path if qr_image_path.exists() else None,
        )
        return plan

    def _render_simple(
        self,
        slug: str,
        script_path: Path,
        scene_tag: str,
        out: Path,
        final_quality: bool,
    ) -> Path:
        """Render a template scene (intro/outro) with no review loop."""
        return render_and_merge(
            script_path,
            out,
            slug,
            final_quality,
            lesson_name=slug,
            context_hash=None,
            scene_tag=scene_tag,
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(  # noqa: C901, PLR0912
        self,
        topic: str,
        input_dir: str | None = None,
        output_dir: str = "output",
        final_quality: bool = False,
        skip_review: bool = False,
        user_script_hash: str | None = None,
    ) -> dict[str, Any]:
        slug = lesson_name_to_key(topic)
        out = Path(output_dir)
        scenes_dir = out / "scenes"  # intermediate per-scene clips
        tts_engine = os.environ.get("TTS_ENGINE", DEFAULT_TTS_ENGINE).lower()

        if input_dir is None and INPUT_DIR.is_dir():
            input_dir = str(INPUT_DIR)

        script_hash = user_script_hash or hash_context(
            topic,
            input_dir,
            extra_context=self._build_script_context(),
        )
        video_hash = hash_context(
            topic, input_dir, extra_context=self._build_cache_context(tts_engine, final_quality)
        )

        _print_pipeline_header(topic, self, tts_engine, input_dir, out, script_hash, video_hash)
        tracker = UsageTracker()

        # ----------------------------------------------------------------
        # Fast path: cached final video
        # ----------------------------------------------------------------
        if cached_video := get_cached_video(slug, video_hash):
            tracker.record("Step 1 - Lesson planning", skipped=True)
            tracker.record("Step 2 - Script generation", skipped=True)
            print()
            print("=" * 60)
            print("Nothing has changed - all assets already cached.")
            print(f"  video  : .cache/{slug}/video/{video_hash}.mp4")
            print("=" * 60)
            out.mkdir(parents=True, exist_ok=True)
            final_path = out / f"{slug}.mp4"
            shutil.copy2(cached_video, final_path)
            print(f"Final video: {final_path}")
            return {
                "output_dir": str(out),
                "lesson_plan": None,
                "scripts": {},
                "final_video": str(final_path),
                "topic": topic,
                "cache_hit": "video",
                "openrouter_usage": tracker.summarize(),
            }

        # ----------------------------------------------------------------
        # Steps 0-2: load/generate lesson plan + all scene scripts
        # ----------------------------------------------------------------
        plan = self._ensure_plan_and_scripts(
            topic=topic,
            slug=slug,
            script_hash=script_hash,
            input_dir=input_dir,
            tracker=tracker,
        )
        script_dir = get_lesson_cache_dir(slug) / "script"

        print()

<<<<<<< Updated upstream
        # Step 3: Render + review loop
        final_video = self._render_loop(
            topic, slug, script_path, out, video_hash, final_quality, skip_review, tracker
        )
=======
        # ----------------------------------------------------------------
        # Step 3: render each scene (with per-scene video caching)
        #
        # Each scene's cache key is a hash of its script content + TTS engine
        # + quality flag.  Because the fix-loop may rewrite the script in-place,
        # we re-hash AFTER the loop to capture any applied patches.  On the next
        # run the script file already contains the patched content, so the hash
        # matches what was saved and we get an instant cache hit.
        # ----------------------------------------------------------------
        print("Step 3: Rendering scenes ...")
        scenes_dir.mkdir(parents=True, exist_ok=True)

        # 3a: Intro (template, no review)
        intro_script = script_dir / f"{script_hash}_intro.py"
        print("\n[Intro]")
        intro_scene_hash = _scene_render_hash(intro_script, tts_engine, False)
        if cached_intro := get_cached_scene_video(slug, "intro", intro_scene_hash):
            print("  Cache hit (intro video) - skipping render.")
            intro_video = scenes_dir / f"{slug}_intro.mp4"
            shutil.copy2(cached_intro, intro_video)
        else:
            intro_video = self._render_simple(slug, intro_script, "intro", scenes_dir, False)
            save_scene_video_to_cache(slug, "intro", intro_scene_hash, intro_video)

        # 3b: Content sections (LLM-generated, with review loop)
        section_videos: list[Path] = []
        section_scene_hashes: list[str] = []

        for idx in range(len(plan.sections)):
            sec_script = script_dir / f"{script_hash}_section_{idx}.py"
            tag = f"section_{idx}"
            print(f"\n[Section {idx + 1}/{len(plan.sections)}: {plan.sections[idx].name}]")

            pre_hash = _scene_render_hash(sec_script, tts_engine, False)
            if cached_sec := get_cached_scene_video(slug, tag, pre_hash):
                print(f"  Cache hit (section {idx} video) - skipping render.")
                sec_video = scenes_dir / f"{slug}_{tag}.mp4"
                shutil.copy2(cached_sec, sec_video)
                section_scene_hashes.append(pre_hash)
            else:
                sec_video = self._render_loop(
                    topic=topic,
                    slug=slug,
                    script_path=sec_script,
                    scene_tag=tag,
                    out=scenes_dir,
                    final_quality=False,
                    skip_review=skip_review,
                    tracker=tracker,
                )
                # Re-hash after the loop: the script may have been patched in-place.
                final_hash = _scene_render_hash(sec_script, tts_engine, False)
                save_scene_video_to_cache(slug, tag, final_hash, sec_video)
                section_scene_hashes.append(final_hash)

            section_videos.append(sec_video)

        # 3c: Outro (template, no review)
        outro_script = script_dir / f"{script_hash}_outro.py"
        print("\n[Outro]")
        outro_scene_hash = _scene_render_hash(outro_script, tts_engine, False)
        if cached_outro := get_cached_scene_video(slug, "outro", outro_scene_hash):
            print("  Cache hit (outro video) - skipping render.")
            outro_video = scenes_dir / f"{slug}_outro.mp4"
            shutil.copy2(cached_outro, outro_video)
        else:
            outro_video = self._render_simple(slug, outro_script, "outro", scenes_dir, False)
            save_scene_video_to_cache(slug, "outro", outro_scene_hash, outro_video)

        # ----------------------------------------------------------------
        # Step 4: add progress sidebar to each content section
        # ----------------------------------------------------------------
        print("\nStep 4: Compositing progress sidebars ...")
        section_names = [s.name for s in plan.sections]
        composited_section_videos: list[Path] = []

        for idx, (sec_video, section) in enumerate(
            zip(section_videos, plan.sections, strict=True)
        ):
            sidebar_tag = f"section_{idx}_sidebar"
            s_hash = _sidebar_scene_hash(section_scene_hashes[idx], section_names, idx)
            sidebar_out = scenes_dir / f"{slug}_{sidebar_tag}.mp4"

            if cached_sidebar := get_cached_scene_video(slug, sidebar_tag, s_hash):
                print(f"  Cache hit (section {idx} sidebar) - skipping compositing.")
                shutil.copy2(cached_sidebar, sidebar_out)
            else:
                print(f"  Adding sidebar to section {idx + 1}: {section.name!r}")
                add_progress_sidebar(
                    video_path=sec_video,
                    sections=section_names,
                    current_section_idx=idx,
                    output_path=sidebar_out,
                )
                save_scene_video_to_cache(slug, sidebar_tag, s_hash, sidebar_out)

            composited_section_videos.append(sidebar_out)

        # ----------------------------------------------------------------
        # Step 5: concatenate all scenes into final video
        # ----------------------------------------------------------------
        print("\nStep 5: Concatenating scenes ...")
        out.mkdir(parents=True, exist_ok=True)
        final_path = out / f"{slug}.mp4"

        all_clips = [intro_video, *composited_section_videos, outro_video]
        concatenate_videos(all_clips, final_path)

        # Cache the final assembled video
        save_video_to_cache(slug, video_hash, final_path)
        print(f"Video cached: .cache/{slug}/video/{video_hash}.mp4")

        # High-quality re-render if requested.
        # We only re-render the content sections (the most important part); intro
        # and outro are already short template scenes.
        if final_quality:
            print("\nStep 5b: High-quality re-render of content sections ...")
            hq_section_videos: list[Path] = []
            for idx in range(len(plan.sections)):
                sec_script = script_dir / f"{script_hash}_section_{idx}.py"
                tag = f"section_{idx}_hq"
                pre_hq_hash = _scene_render_hash(sec_script, tts_engine, True)
                if cached_hq := get_cached_scene_video(slug, tag, pre_hq_hash):
                    print(f"  Cache hit (section {idx} HQ video) - skipping render.")
                    hq_video = scenes_dir / f"{slug}_{tag}.mp4"
                    shutil.copy2(cached_hq, hq_video)
                else:
                    hq_video = self._render_simple(slug, sec_script, tag, scenes_dir, True)
                    post_hq_hash = _scene_render_hash(sec_script, tts_engine, True)
                    save_scene_video_to_cache(slug, tag, post_hq_hash, hq_video)
                hq_section_videos.append(hq_video)

            hq_composited: list[Path] = []
            for idx, hq_vid in enumerate(hq_section_videos):
                hq_sidebar_tag = f"section_{idx}_sidebar_hq"
                hq_raw_hash = _scene_render_hash(
                    script_dir / f"{script_hash}_section_{idx}.py", tts_engine, True
                )
                hq_s_hash = _sidebar_scene_hash(hq_raw_hash, section_names, idx)
                hq_sidebar_out = scenes_dir / f"{slug}_{hq_sidebar_tag}.mp4"

                if cached_hq_sidebar := get_cached_scene_video(
                    slug, hq_sidebar_tag, hq_s_hash
                ):
                    print(f"  Cache hit (section {idx} HQ sidebar) - skipping compositing.")
                    shutil.copy2(cached_hq_sidebar, hq_sidebar_out)
                else:
                    add_progress_sidebar(
                        video_path=hq_vid,
                        sections=section_names,
                        current_section_idx=idx,
                        output_path=hq_sidebar_out,
                    )
                    save_scene_video_to_cache(slug, hq_sidebar_tag, hq_s_hash, hq_sidebar_out)
                hq_composited.append(hq_sidebar_out)

            all_hq = [intro_video, *hq_composited, outro_video]
            concatenate_videos(all_hq, final_path)
            save_video_to_cache(slug, video_hash, final_path)
>>>>>>> Stashed changes

        print()
        print("=" * 60)
        print("Pipeline complete.")
        print(f"Final video : {final_path}")
        print("=" * 60)

        script_paths = {
            "intro": str(intro_script),
            "sections": [
                str(script_dir / f"{script_hash}_section_{i}.py")
                for i in range(len(plan.sections))
            ],
            "outro": str(outro_script),
        }

        return {
            "output_dir": str(out),
            "lesson_plan": plan.lesson_markdown,
            "scripts": script_paths,
            "final_video": str(final_path),
            "topic": topic,
            "openrouter_usage": tracker.summarize(),
        }
