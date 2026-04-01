"""
Script Generator for Manim Educational Videos
Uses OpenRouter to generate Manim code from lesson plans
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm_metrics import (
    LLMUsage,
    accumulate_llm_usage,
    extract_llm_usage,
    make_openrouter_llm,
)
from src.core.paths import MANIM_PROMPT, VIDEO_FIX_PROMPT, VIDEO_REVIEW_PROMPT
from src.core.settings import MANIM_GENERATOR_MODEL, VIDEO_FIX_MODEL, VIDEO_REVIEW_MODEL
from src.review.algorithms import (
    encode_selected_frames,
    frame_ssim,
    select_brightness_peak_frames,
)
from src.review.models import CodeFix, VideoReview
from src.scripting.search_replace import flexible_search_and_replace

load_dotenv()

REVIEW_OK_CACHE_SSIM_THRESHOLD = 0.98


class ManimScriptGenerator:
    """Generates Manim Python scripts using LLM"""

    def __init__(
        self,
        generation_model: str = MANIM_GENERATOR_MODEL,
        review_model: str = VIDEO_REVIEW_MODEL,
        fix_model: str = VIDEO_FIX_MODEL,
    ) -> None:
        self.model = generation_model
        self.review_model = review_model
        self.fix_model = fix_model

        self.llm = make_openrouter_llm(
            generation_model, title="Manim Script Generator", temperature=0.7
        )
        self.review_llm = make_openrouter_llm(
            review_model, title="Manim Video Reviewer", temperature=0.0
        )
        self.fix_llm = make_openrouter_llm(fix_model, title="Manim Video Fixer")

        with open(MANIM_PROMPT, encoding="utf-8") as f:
            self.system_prompt = f.read()

        with open(VIDEO_REVIEW_PROMPT, encoding="utf-8") as f:
            self.review_prompt_template = f.read()

        with open(VIDEO_FIX_PROMPT, encoding="utf-8") as f:
            self.fix_prompt = f.read()

        # Caches approved frames during one render/review loop so similar
        # frames in later iterations can skip redundant LLM checks.
        self._approved_review_frames: list[np.ndarray] = []

    def reset_review_cache(self) -> None:
        """Clear the per-loop approved-frame cache used by review_video."""
        self._approved_review_frames.clear()

    def _matches_approved_frame(self, frame: np.ndarray) -> bool:
        """Return True if frame is highly similar to any previously approved frame."""
        return any(
            frame_ssim(approved, frame) >= REVIEW_OK_CACHE_SSIM_THRESHOLD
            for approved in self._approved_review_frames
        )

    def generate_script(
        self,
        lesson_content: str,
        topic: str,
        input_parts: list[dict[str, Any]] | None = None,
    ) -> tuple[str, LLMUsage]:
        """Generate a Manim script from lesson content using the LLM."""
        text = f"""Topic: {topic}

Lesson Content:
\"\"\"
{lesson_content}
\"\"\"

Generate a complete Manim script that:
1. Teaches this topic step-by-step with clear visuals
2. Uses AudioManager for narration at each step
3. Includes mathematical equations and geometric animations where appropriate
4. Follows the exact pattern from the system prompt
5. Is production-ready and can be directly executed by manim
"""

        if input_parts:
            user_content: list[str | dict[str, Any]] = [
                {"type": "text", "text": text},
                {"type": "text", "text": "Reference materials from input directory:"},
                *input_parts,
            ]
        else:
            user_content = [{"type": "text", "text": text}]

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]

        response = self.llm.invoke(messages)
        usage = extract_llm_usage(response)
        return self._clean_code_output(str(response.content)), usage

    def _clean_code_output(self, code: str) -> str:
        code = code.strip()
        # Extract the content of the first fenced code block if any.
        # This handles leading preamble text, trailing commentary, and any
        # amount of whitespace around the fences.
        m = re.search(r"```(?:python)?\s*\n(.*?)\n?```", code, re.DOTALL)
        if m:
            code = m.group(1).strip()
        return self._sanitize_latex(code)

    @staticmethod
    def _sanitize_latex(code: str) -> str:
        """Escape bare LaTeX special chars inside Text() and Title() string args."""

        # Match Text("...") or Title("...") - single or double quoted, non-greedy
        def _escape_arg(m: re.Match) -> str:
            prefix = m.group(1)  # Text( or Title(
            quote = m.group(2)  # " or '
            content = m.group(3)

            return f"{prefix}{quote}{content}{quote}"

        pattern = re.compile(r'((?:Text|Title)\()(["\'])(.*?)\2', re.DOTALL)
        return pattern.sub(_escape_arg, code)

    def generate_and_save(
        self,
        lesson_content: str,
        topic: str,
        output_path: str | Path,
        input_parts: list[dict[str, Any]] | None = None,
    ) -> LLMUsage:
        """Generate a Manim script and save it to the specified path."""
        print(f"Generating Manim script ({self.model}) ...")
        script, usage = self.generate_script(lesson_content, topic, input_parts)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(script)
        print(f"Script saved: {out}")
        return usage

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def review_video(
        self,
        script: str,
        video_path: Path,
        topic: str,
    ) -> tuple[str, bool, LLMUsage]:
        """Review the rendered video for visual issues using structured output.

        Each frame is reviewed individually.  Only frames with issues are
        forwarded to the fix agent, together with their specific failure
        criteria.

        Returns ``(script, changed)``.  *changed* is ``False`` when the video
        passed review; otherwise *script* contains the corrected code.
        """
        print("Reviewing rendered video for visual issues ...")
        selected_frames = select_brightness_peak_frames(video_path)
        if not selected_frames:
            print("  Could not extract frames - skipping review.")
            return script, False, LLMUsage()
        frames = encode_selected_frames(selected_frames)

        print(f"  Reviewing {len(frames)} unique frames one-by-one ...")

        structured_llm = self.review_llm.with_structured_output(VideoReview, include_raw=True)
        review_text = f"Topic: {topic}\n\n{self.review_prompt_template}"
        sys_msg = SystemMessage(content=self.system_prompt)

        # Collect frames that have issues
        flagged: list[tuple[str, dict[str, Any], list[str], str | None]] = []
        total_usage = LLMUsage()
        skipped_from_cache = 0
        reviewed_count = 0

        for i, ((_, frame_arr), (label, img_part)) in enumerate(
            zip(selected_frames, frames, strict=True), 1
        ):
            if self._matches_approved_frame(frame_arr):
                skipped_from_cache += 1
                print(f"  [{i}/{len(frames)}] {label}: SKIP (SSIM >= 90% to approved frame)")
                continue

            reviewed_count += 1
            user_content: list[str | dict[str, Any]] = [
                {"type": "text", "text": review_text},
                {"type": "text", "text": label},
                img_part,
            ]
            result = structured_llm.invoke([sys_msg, HumanMessage(content=user_content)])
            review: VideoReview = result["parsed"]  # type: ignore[index]
            usage = extract_llm_usage(result["raw"])  # type: ignore[index]
            accumulate_llm_usage(total_usage, usage)

            if review.has_issues:
                failed = review.failed_criteria()
                flagged.append((label, img_part, failed, review.notes))
                status = f"ISSUES ({', '.join(failed)})"
            else:
                self._approved_review_frames.append(frame_arr.copy())
                status = "OK"
            print(f"  [{i}/{len(frames)}] {label}: {status}")

        if skipped_from_cache:
            print(
                f"  Skipped {skipped_from_cache} frame(s) due to SSIM cache "
                f"(threshold {int(REVIEW_OK_CACHE_SSIM_THRESHOLD * 100)}%)."
            )

        if not flagged:
            if reviewed_count == 0:
                print("  Video review: APPROVED - all frames matched previously approved content.")
            else:
                print("  Video review: APPROVED - no issues found in any reviewed frame.")
            return script, False, total_usage

        # Aggregate all unique failed criteria across flagged frames
        all_failed: dict[str, None] = {}
        for _, _, criteria, _ in flagged:
            for c in criteria:
                all_failed[c] = None
        all_failed_list = list(all_failed)

        print(
            f"  Video review: {len(flagged)}/{len(frames)} frames flagged, "
            f"{len(all_failed_list)} unique criteria failed:"
        )
        for criterion in all_failed_list:
            print(f"    \u2022 {criterion}")

        flagged_frames = [(label, img_part, notes) for label, img_part, _, notes in flagged]
        fixed_script = self._fix_visual_issues(script, all_failed_list, flagged_frames, topic)
        return fixed_script, True, total_usage

    def _fix_visual_issues(
        self,
        script: str,
        failed_criteria: list[str],
        frames: list[tuple[str, dict[str, Any], str | None]],
        topic: str,
    ) -> str:
        """Send the failed criteria + frames to the fix agent, then apply edits."""
        print("  Sending to fix agent ...")

        criteria_block = "\n".join(f"- {c}" for c in failed_criteria)
        fix_text = (
            f"Topic: {topic}\n\n"
            f"The following visual criteria FAILED:\n{criteria_block}\n\n"
            f"--- SOURCE CODE ---\n```python\n{script}\n```"
        )

        user_content: list[str | dict[str, Any]] = [
            {"type": "text", "text": fix_text},
        ]
        for label, img_part, notes in frames:
            annotation = f"{label}" + (f" - {notes}" if notes else "")
            user_content.append({"type": "text", "text": annotation})
            user_content.append(img_part)

        structured_fix_llm = self.fix_llm.with_structured_output(CodeFix)
        messages = [
            SystemMessage(content=self.fix_prompt),
            HumanMessage(content=user_content),
        ]

        fix: CodeFix = structured_fix_llm.invoke(messages)  # type: ignore[assignment]

        if not fix.edits:
            print("  Fix agent returned no edits - returning original script.")
            return script

        # Ensure the original ends with a newline for dmp_lines_apply
        result = script if script.endswith("\n") else script + "\n"
        applied = 0
        for edit in fix.edits:
            search = edit.old_code if edit.old_code.endswith("\n") else edit.old_code + "\n"
            replace = edit.new_code if edit.new_code.endswith("\n") else edit.new_code + "\n"
            patched = flexible_search_and_replace([search, replace, result])
            if patched is not None:
                result = patched
                applied += 1
            else:
                print(f"  Warning: edit could not be applied - skipping: {edit.old_code[:60]!r}")

        print(f"  Applied {applied}/{len(fix.edits)} edits from fix agent.")
        return self._clean_code_output(result)

    def fix_compilation_error(
        self,
        script: str,
        error_output: str,
        topic: str,
    ) -> tuple[str, LLMUsage]:
        """Ask the LLM to fix a script that failed to render.

        Returns the corrected script and usage.
        """
        print("Sending compilation error to LLM for a fix ...")

        user_content: list[str | dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Topic: {topic}\n\n"
                    "The following Manim script failed to render. "
                    "Below is the script followed by the error output.\n\n"
                    "Fix the script so it compiles and renders correctly. "
                    "Output ONLY the complete corrected Python code, no markdown "
                    "fences, no explanations.\n\n"
                    f"--- SOURCE CODE ---\n```python\n{script}\n```\n\n"
                    f"--- ERROR OUTPUT (last 60 lines) ---\n```\n{error_output}\n```"
                ),
            },
        ]

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]

        response = self.llm.invoke(messages)
        usage = extract_llm_usage(response)
        return self._clean_code_output(str(response.content)), usage
