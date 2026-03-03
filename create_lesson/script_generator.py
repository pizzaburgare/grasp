"""
Script Generator for Manim Educational Videos
Uses OpenRouter to generate Manim code from lesson plans
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()


class ManimScriptGenerator:
    """Generates Manim Python scripts using LLM"""

    def __init__(
        self,
        model: str = "google/gemini-3.1-pro-preview",
        max_retries: int = 3,
    ):
        """
        Initialize the script generator

        Args:
            model: OpenRouter model to use (default: free Gemini model for cost-efficiency)
            max_retries: Maximum number of retry attempts for code generation
        """
        self.model = model
        self.max_retries = max_retries

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or ""),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Manim Script Generator",
            },
            temperature=0.7,  # Some creativity but not too much
        )

        # Load system prompt
        prompt_path = os.path.join(os.path.dirname(__file__), "manim_prompt.md")
        with open(prompt_path, "r") as f:
            self.system_prompt = f.read()

    def generate_script(
        self,
        lesson_content: str,
        topic: str,
        additional_context: Optional[str] = None,
        input_parts: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate a Manim script from lesson content

        Args:
            lesson_content: The lesson plan/content to visualize
            topic: The topic name (e.g., "LU Decomposition")
            additional_context: Optional additional instructions or examples
            input_parts: Optional multimodal content parts (images, text) from input_processor

        Returns:
            Generated Python code as a string
        """
        # Build the user message text
        text = f"""Topic: {topic}

Lesson Content:
{lesson_content}
"""

        if additional_context:
            text += f"\n\nAdditional Context:\n{additional_context}"

        text += """

Generate a complete Manim script that:
1. Teaches this topic step-by-step with clear visuals
2. Uses AudioManager for narration at each step
3. Includes mathematical equations and geometric visualizations where appropriate
4. Follows the exact pattern from the system prompt
5. Is production-ready and can be directly saved as main.py
"""

        # If we have multimodal input, send as structured content
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

        # Generate the code
        response = self.llm.invoke(messages)
        generated_code = response.content

        # Clean up the response (remove markdown code blocks if present)
        generated_code = self._clean_code_output(generated_code)

        return generated_code

    def _clean_code_output(self, code: str) -> str:
        """Remove markdown code blocks and extra formatting"""
        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code[len("```python") :].strip()
        elif code.startswith("```"):
            code = code[len("```") :].strip()

        if code.endswith("```"):
            code = code[:-3].strip()

        return code

    def generate_and_save(
        self,
        lesson_content: str,
        topic: str,
        output_path: str = "main.py",
        additional_context: Optional[str] = None,
        input_parts: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate script and save directly to file

        Args:
            lesson_content: The lesson plan/content
            topic: The topic name
            output_path: Path to save the generated script (default: main.py)
            additional_context: Optional additional instructions
            input_parts: Optional multimodal content parts from input_processor

        Returns:
            Path to the saved file
        """
        print(f"Generating Manim script for: {topic}")
        print(f"Using model: {self.model}")

        script = self.generate_script(
            lesson_content=lesson_content,
            topic=topic,
            additional_context=additional_context,
            input_parts=input_parts,
        )

        # Save to file
        with open(output_path, "w") as f:
            f.write(script)

        print(f"Script saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage
    generator = ManimScriptGenerator()

    example_lesson = """
    Quick Walkthrough:
    The QR decomposition breaks a matrix A into:
    - Q: an orthogonal matrix (columns are perpendicular unit vectors)
    - R: an upper triangular matrix
    
    The Target Problem:
    Given A = [[1, 2], [1, 0]], find Q and R using Gram-Schmidt.
    
    Step-by-Step Solution:
    1. Extract columns: a1 = [1, 1], a2 = [2, 0]
    2. Normalize a1: e1 = a1/||a1|| = [1/√2, 1/√2]
    3. Orthogonalize a2: u2 = a2 - (a2·e1)e1 = [1, -1]
    4. Normalize u2: e2 = u2/||u2|| = [1/√2, -1/√2]
    5. Build Q from e1, e2 and calculate R
    """

    generator.generate_and_save(
        lesson_content=example_lesson,
        topic="QR Decomposition with Gram-Schmidt",
        output_path="main.py",
    )
