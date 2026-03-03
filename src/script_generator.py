"""
Script Generator for Manim Educational Videos
Uses OpenRouter to generate Manim code from lesson plans
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()


class ManimScriptGenerator:
    """Generates Manim Python scripts using LLM"""

    def __init__(self, model: str = "google/gemini-3.1-pro-preview"):
        self.model = model

        self.llm = ChatOpenAI(
            model=model,
            api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Manim Script Generator",
            },
            temperature=0.7,
        )

        prompt_path = Path(__file__).parent / "manim_prompt.md"
        with open(prompt_path) as f:
            self.system_prompt = f.read()

    def generate_script(
        self,
        lesson_content: str,
        topic: str,
        input_parts: Optional[list[dict]] = None,
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
            user_content: list[dict] = [
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
        input_parts: Optional[list[dict]] = None,
    ) -> Path:
        print(f"Generating Manim script ({self.model}) ...")
        script = self.generate_script(lesson_content, topic, input_parts)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(script)
        print(f"Script saved: {out}")
        return out
