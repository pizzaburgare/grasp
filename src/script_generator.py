"""
Script Generator for Manim Educational Videos
Uses OpenRouter to generate Manim code from lesson plans
"""

import base64
import io
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from moviepy import VideoFileClip
from PIL import Image
from pydantic import SecretStr

from src.llm_metrics import LLMUsage, extract_llm_usage
from src.paths import MANIM_PROMPT, VIDEO_REVIEW_PROMPT
from src.settings import MANIM_GENERATOR_MODEL, VIDEO_REVIEW_MODEL

load_dotenv()

# Maximum number of video frames sent for review
_REVIEW_MAX_FRAMES = 50
_REVIEW_FRAME_QUALITY = 70


class ManimScriptGenerator:
    """Generates Manim Python scripts using LLM"""

    def __init__(
        self,
        generation_model: str = MANIM_GENERATOR_MODEL,
        review_model: str = VIDEO_REVIEW_MODEL,
    ):
        self.model = generation_model
        self.review_model = review_model

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

        with open(MANIM_PROMPT) as f:
            self.system_prompt = f.read()

        with open(VIDEO_REVIEW_PROMPT) as f:
            self.review_prompt_template = f.read()

        self.last_generation_usage: LLMUsage | None = None

    def generate_script(
        self,
        lesson_content: str,
        topic: str,
        input_parts: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        text = f"""Topic: {topic}

Lesson Content:
{lesson_content}

Generate a complete Manim script that:
1. Teaches this topic step-by-step with clear visuals
2. Uses AudioManager for narration at each step
3. Includes mathematical equations and geometric visualizations where appropriate
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
        if code.startswith("```python"):
            code = code[len("```python") :].strip()
        elif code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        return self._sanitize_latex(code)

    @staticmethod
    def _sanitize_latex(code: str) -> str:
        """Escape bare LaTeX special chars inside Text() and Title() string args."""
        import re

        # Match Text("...") or Title("...") — single or double quoted, non-greedy
        def _escape_arg(m: re.Match) -> str:
            prefix = m.group(1)  # Text( or Title(
            quote = m.group(2)  # " or '
            content = m.group(3)

            # Escape & % # that aren't already escaped and aren't inside math ($...$)
            # Only touch characters that are definitely outside math mode
            def _escape_outside_math(s: str) -> str:
                result = []
                in_math = False
                i = 0
                while i < len(s):
                    if s[i] == "$":
                        in_math = not in_math
                        result.append(s[i])
                    elif not in_math and s[i] == "&" and (i == 0 or s[i - 1] != "\\"):
                        result.append("\\&")
                    elif not in_math and s[i] == "%" and (i == 0 or s[i - 1] != "\\"):
                        result.append("\\%")
                    else:
                        result.append(s[i])
                    i += 1
                return "".join(result)

            escaped = _escape_outside_math(content)
            return f"{prefix}{quote}{escaped}{quote}"

        pattern = re.compile(r'((?:Text|Title)\()(["\'])(.*?)\2', re.DOTALL)
        return pattern.sub(_escape_arg, code)

    def generate_and_save(
        self,
        lesson_content: str,
        topic: str,
        output_path: str | Path,
        input_parts: Optional[list[dict[str, Any]]] = None,
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
    def _extract_video_frames(video_path: Path) -> list[dict[str, Any]]:
        """Sample frames from a video and return them as LLM image_url parts."""
        clip = VideoFileClip(str(video_path))
        try:
            duration = float(clip.duration or 0.0)
            if duration <= 0:
                return []
            # Pick a frame interval that yields at most _REVIEW_MAX_FRAMES
            interval = max(1.0, duration / _REVIEW_MAX_FRAMES)
            parts: list[dict[str, Any]] = []
            t = 0.0
            while t < duration:
                frame: np.ndarray = clip.get_frame(t)  # type: ignore[assignment]
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=_REVIEW_FRAME_QUALITY)
                data = base64.b64encode(buf.getvalue()).decode()
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{data}",
                        },
                    }
                )
                t += interval
            return parts
        finally:
            clip.close()

    def review_video(
        self,
        script: str,
        video_path: Path,
        topic: str,
        lesson_content: str,
    ) -> tuple[str, bool]:
        """Review the rendered video for visual issues.

        Returns ``(script, changed)``.  When *changed* is ``False`` the video
        passed review; otherwise *script* contains the corrected code.
        """
        print("Reviewing rendered video for visual issues ...")
        frames = self._extract_video_frames(video_path)
        if not frames:
            print("  Could not extract frames — skipping review.")
            return script, False

        review_text = (
            f"Topic: {topic}\n\n"
            + self.review_prompt_template
            + f"\n\n--- SOURCE CODE ---\n```python\n{script}\n```"
        )

        user_content: list[str | dict[str, Any]] = [
            {"type": "text", "text": review_text},
            *frames,
        ]

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]

        response = self.review_llm.invoke(messages)
        self.last_review_usage = extract_llm_usage(response)
        answer = str(response.content).strip()

        if answer.upper().startswith("APPROVED"):
            print("  Video review: APPROVED - no issues found.")
            return script, False

        print("  Video review: issues found - regenerating script.")
        new_script = self._clean_code_output(answer)
        return new_script, True

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
