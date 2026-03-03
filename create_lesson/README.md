# Lesson Creation Modules

This directory contains the AI-powered lesson planning and script generation system.

## Files

### `script_generator.py`
**ManimScriptGenerator class** - Generates Manim Python code from lesson content
- Uses OpenRouter LLM to convert lesson plans into working Manim animations
- Handles code cleanup (removes markdown formatting)
- Saves directly to `main.py` or custom path

**Usage:**
```python
from create_lesson.script_generator import ManimScriptGenerator

generator = ManimScriptGenerator(model="google/gemini-2.0-flash-exp:free")
generator.generate_and_save(
    lesson_content="Your lesson plan here...",
    topic="LU Decomposition",
    output_path="main.py"
)
```

### `lesson_planner.py`
**Legacy standalone lesson planner** - Generates structured lesson content
- Uses OpenRouter with the lesson planning prompt
- Outputs lesson text only (no Manim code)
- Can be used independently for lesson planning

### `prompt.md`
**Lesson structure template** - Defines the pedagogical structure
- Used by lesson_planner.py
- Enforces: Quick Walkthrough → Target Problem → Step-by-Step Solution → Recap
- Designed for exam-focused STEM teaching

### `manim_prompt.md`
**Manim code generation template** - Instructions for generating Manim scripts
- Used by script_generator.py
- Includes AudioManager integration requirements
- Specifies Manim Community Edition syntax patterns
- Enforces visual and narration best practices

## Integration

These modules are orchestrated by `../generate_course.py` which runs the full pipeline:
1. Generate lesson plan (using `prompt.md`)
2. Generate Manim script (using `manim_prompt.md` + script_generator)
3. Output ready-to-render `main.py`
