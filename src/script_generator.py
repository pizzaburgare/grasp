"""
Script Generator for Manim Educational Videos
Uses OpenRouter to generate Manim code from lesson plans
"""

import base64
import io
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from moviepy import VideoFileClip
from PIL import Image
from pydantic import BaseModel, Field, SecretStr

from src.llm_metrics import LLMUsage, extract_llm_usage
from src.paths import MANIM_PROMPT, VIDEO_FIX_PROMPT, VIDEO_REVIEW_PROMPT
from src.search_replace import flexible_search_and_replace
from src.settings import MANIM_GENERATOR_MODEL, VIDEO_FIX_MODEL, VIDEO_REVIEW_MODEL

load_dotenv()

# Target number of video frames sent for review
_REVIEW_TARGET_FRAMES = 50
_REVIEW_FRAME_QUALITY = 70
# Dense scan settings for scene-change detection
_SCENE_SCAN_INTERVAL = 0.5  # seconds between scan samples
_SCENE_SETTLE_THRESHOLD = 0.01  # max mean absolute pixel difference to count as "settled"


# ---------------------------------------------------------------------------
# Image similarity
# ---------------------------------------------------------------------------


def _frame_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute global SSIM on BT.601 luma.  Returns a value in [-1, 1]."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ga = np.dot(a[..., :3].astype(np.float64), [0.299, 0.587, 0.114])
    gb = np.dot(b[..., :3].astype(np.float64), [0.299, 0.587, 0.114])
    mu_a, mu_b = ga.mean(), gb.mean()
    var_a, var_b = ga.var(), gb.var()
    cov = float(np.mean((ga - mu_a) * (gb - mu_b)))
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)
    return float(np.clip(num / den, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class VideoReview(BaseModel):
    """Structured result from the video review agent."""

    text_clipped: bool = Field(description="Text or equations are clipped / cut off at the frame edges.")
    overlapping_content: bool = Field(description="Content overlaps or is rendered unreadably on top of other content.")
    broken_animations: bool = Field(description="Visual artifacts, glitches, or misplaced objects are visible.")
    content_overflow: bool = Field(description="Content extends outside the visible frame boundary.")
    latex_rendering: bool = Field(description="LaTeX is incorrectly rendered (broken symbols, blank boxes, malformed equations).")
    notes: str | None = Field(
        default=None,
        description="Optional 4-8 word description of the problem(s) found. Only set when at least one criterion is true.",
    )

    @property
    def has_issues(self) -> bool:
        return any(
            [
                self.text_clipped,
                self.overlapping_content,
                self.broken_animations,
                self.content_overflow,
                self.latex_rendering,
            ]
        )

    def failed_criteria(self) -> list[str]:
        labels = {
            "text_clipped": "Text or equations clipped / cut off at the edges",
            "overlapping_content": "Overlapping or unreadable content",
            "broken_animations": "Broken or glitchy animations (artifacts, misplaced objects)",
            "content_overflow": "Content overflowing outside the visible frame",
            "latex_rendering": "Incorrect rendering of LaTeX",
        }
        return [desc for field, desc in labels.items() if getattr(self, field)]


class CodeEdit(BaseModel):
    """A single search/replace edit to apply to the script."""

    old_code: str = Field(description="Verbatim substring to find in the source (include 5+ lines of context).")
    new_code: str = Field(description="Replacement text.")


class CodeFix(BaseModel):
    """Structured list of targeted edits from the fix agent."""

    edits: list[CodeEdit] = Field(description="Ordered list of search/replace edits to apply.")


class ManimScriptGenerator:
    """Generates Manim Python scripts using LLM"""

    def __init__(
        self,
        generation_model: str = MANIM_GENERATOR_MODEL,
        review_model: str = VIDEO_REVIEW_MODEL,
        fix_model: str = VIDEO_FIX_MODEL,
    ):
        self.model = generation_model
        self.review_model = review_model
        self.fix_model = fix_model

        self.llm = ChatOpenAI(
            model=generation_model,
            api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Manim Script Generator",
            },
            temperature=0.7,
        )

        self.review_llm = ChatOpenAI(
            model=review_model,
            api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Manim Video Reviewer",
            },
        )

        self.fix_llm = ChatOpenAI(
            model=fix_model,
            api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Manim Video Fixer",
            },
        )

        with open(MANIM_PROMPT) as f:
            self.system_prompt = f.read()

        with open(VIDEO_REVIEW_PROMPT) as f:
            self.review_prompt_template = f.read()

        with open(VIDEO_FIX_PROMPT) as f:
            self.fix_prompt = f.read()

        self.last_generation_usage: LLMUsage | None = None

    def generate_script(
        self,
        lesson_content: str,
        topic: str,
        input_parts: list[dict[str, Any]] | None = None,
    ) -> str:
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
        self.last_generation_usage = extract_llm_usage(response)
        return self._clean_code_output(str(response.content))

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
    ) -> Path:
        print(f"Generating Manim script ({self.model}) ...")
        script = self.generate_script(lesson_content, topic, input_parts)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(script)
        print(f"Script saved: {out}")
        return out

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Backward-compatible wrapper for frame similarity comparisons."""
        return _frame_ssim(a, b)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format *seconds* as ``H:MM:SS`` or ``M:SS`` (no milliseconds)."""
        total = int(seconds)
        h, remainder = divmod(total, 3600)
        m, s = divmod(remainder, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    @staticmethod
    def _scan_settled_frames(
        video_path: Path,
    ) -> list[tuple[float, np.ndarray]]:
        """Scan a video and return all settled, non-blank frames with ≥ 1 s gap.

        A frame at time *T* is "settled" when its mean absolute pixel
        difference from the frame at *T + _SCENE_SCAN_INTERVAL* is below
        ``_SCENE_SETTLE_THRESHOLD``.  This tolerates tiny codec rounding
        artifacts that prevent perfect pixel equality.

        Near-uniform frames (std < 2) are skipped, and a minimum 1 s gap
        between kept frames is enforced.

        Returns a list of ``(timestamp, frame_array)`` tuples.  The caller
        owns the numpy arrays (they are copies, not views into the clip).
        """
        clip = VideoFileClip(str(video_path))
        try:
            duration = float(clip.duration or 0.0)
            if duration <= 0:
                return []

            candidates: list[tuple[float, np.ndarray]] = []
            last_kept_t = -float("inf")

            t = 0.0
            curr_frame: np.ndarray = clip.get_frame(0.0)  # type: ignore[assignment]
            while t < duration:
                next_t = min(t + _SCENE_SCAN_INTERVAL, duration)
                next_frame: np.ndarray = clip.get_frame(next_t)  # type: ignore[assignment]

                mae = float(np.mean(np.abs(curr_frame.astype(np.int16) - next_frame.astype(np.int16))))
                if mae <= _SCENE_SETTLE_THRESHOLD and t - last_kept_t >= 1.0 and float(np.std(curr_frame)) >= 2.0:  # not blank
                    candidates.append((t, curr_frame.copy()))
                    last_kept_t = t

                curr_frame = next_frame
                t = next_t

            return candidates
        finally:
            clip.close()

    @staticmethod
    def _extract_video_frames(
        video_path: Path,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Sample frames from a video, preferring moments after animations settle.

        Returns a list of ``(timestamp_label, image_url_part)`` tuples, targeting
        up to ``_REVIEW_TARGET_FRAMES`` frames.

        Algorithm:
        1. ``_scan_settled_frames`` collects all settled, non-blank frames
           with ≥ 1 s gap (single sequential scan, MAE-based settle check).
        2. SSIM-dedup the full candidate set (threshold 0.99): walk sequentially,
           skip any frame with SSIM ≥ 0.99 vs. the previous kept frame.
        3. Evenly subsample to at most ``_REVIEW_TARGET_FRAMES``.
        4. JPEG-encode the final frames.
        """
        candidates = ManimScriptGenerator._scan_settled_frames(video_path)
        if not candidates:
            return []

        # ----------------------------------------------------------
        # SSIM dedup (sequential, 0.99 threshold)
        # ----------------------------------------------------------
        deduped: list[tuple[float, np.ndarray]] = [candidates[0]]
        for ts, frame in candidates[1:]:
            if ManimScriptGenerator._frame_similarity(deduped[-1][1], frame) < 0.99:
                deduped.append((ts, frame))

        # ----------------------------------------------------------
        # Evenly subsample to target
        # ----------------------------------------------------------
        if len(deduped) > _REVIEW_TARGET_FRAMES:
            step = len(deduped) / _REVIEW_TARGET_FRAMES
            indices = list(dict.fromkeys(min(round(i * step), len(deduped) - 1) for i in range(_REVIEW_TARGET_FRAMES)))
            deduped = [deduped[i] for i in indices]

        # ----------------------------------------------------------
        # JPEG-encode the final frames
        # ----------------------------------------------------------
        parts: list[tuple[str, dict[str, Any]]] = []
        for chosen_t, frame in deduped:
            img = Image.fromarray(frame)  # type: ignore[arg-type]
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_REVIEW_FRAME_QUALITY)
            data = base64.b64encode(buf.getvalue()).decode()
            label = f"Frame at {ManimScriptGenerator._format_timestamp(chosen_t)}"
            parts.append(
                (
                    label,
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                    },
                )
            )
        return parts

    def review_video(
        self,
        script: str,
        video_path: Path,
        topic: str,
        lesson_content: str,
    ) -> tuple[str, bool]:
        """Review the rendered video for visual issues using structured output.

        Each frame is reviewed individually.  Only frames with issues are
        forwarded to the fix agent, together with their specific failure
        criteria.

        Returns ``(script, changed)``.  *changed* is ``False`` when the video
        passed review; otherwise *script* contains the corrected code.
        """
        print("Reviewing rendered video for visual issues ...")
        frames = self._extract_video_frames(video_path)
        if not frames:
            print("  Could not extract frames - skipping review.")
            return script, False

        print(f"  Reviewing {len(frames)} unique frames one-by-one ...")

        structured_llm = self.review_llm.with_structured_output(VideoReview, include_raw=True)
        review_text = f"Topic: {topic}\n\n{self.review_prompt_template}"
        sys_msg = SystemMessage(content=self.system_prompt)

        # Collect frames that have issues
        flagged: list[tuple[str, dict[str, Any], list[str], str | None]] = []
        total_usage = LLMUsage()

        for i, (label, img_part) in enumerate(frames, 1):
            user_content: list[str | dict[str, Any]] = [
                {"type": "text", "text": review_text},
                {"type": "text", "text": label},
                img_part,
            ]
            result = structured_llm.invoke([sys_msg, HumanMessage(content=user_content)])
            review: VideoReview = result["parsed"]  # type: ignore[index]
            usage = extract_llm_usage(result["raw"])  # type: ignore[index]
            total_usage = LLMUsage(
                prompt_tokens=total_usage.prompt_tokens + usage.prompt_tokens,
                completion_tokens=total_usage.completion_tokens + usage.completion_tokens,
                total_tokens=total_usage.total_tokens + usage.total_tokens,
                cost_usd=(total_usage.cost_usd or 0) + (usage.cost_usd or 0) or None,
            )

            if review.has_issues:
                failed = review.failed_criteria()
                flagged.append((label, img_part, failed, review.notes))
                status = f"ISSUES ({', '.join(failed)})"
            else:
                status = "OK"
            print(f"  [{i}/{len(frames)}] {label}: {status}")

        self.last_review_usage = total_usage

        if not flagged:
            print("  Video review: APPROVED - no issues found in any frame.")
            return script, False

        # Aggregate all unique failed criteria across flagged frames
        all_failed: dict[str, None] = {}
        for _, _, criteria, _ in flagged:
            for c in criteria:
                all_failed[c] = None
        all_failed_list = list(all_failed)

        print(f"  Video review: {len(flagged)}/{len(frames)} frames flagged, {len(all_failed_list)} unique criteria failed:")
        for criterion in all_failed_list:
            print(f"    \u2022 {criterion}")

        flagged_frames = [(label, img_part, notes) for label, img_part, _, notes in flagged]
        fixed_script = self._fix_visual_issues(script, all_failed_list, flagged_frames, topic)
        return fixed_script, True

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
            patched = flexible_search_and_replace((search, replace, result))
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
        lesson_content: str,
    ) -> str:
        """Ask the LLM to fix a script that failed to render.

        Returns the corrected script.
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
        self.last_fix_usage = extract_llm_usage(response)
        return self._clean_code_output(str(response.content))
