# CLAUDE.md

## What is Grasp?

**Grasp** creates animated engineering explainer videos from university course materials (lecture slides in PDF format, videos in MP4 format). Our core approach applies cognitive science research to produce videos that outperform traditional lectures in learning outcomes and retention. We use Manim to generate high-quality animations with LLM:s, and our custom TTS engine to add narration.

We have a kanban board on GitHub to track progress and tasks. Keep each task small and focused, and always link to the relevant issue when making commits.

**Video quality is our highest priority.**

Primary TTS engine: **Qwen**

## Architecture

Agentic pipeline where each step is a **pure function** of its inputs → outputs:

1. **Input Processing** — raw materials → LLM-ready content
2. **Lesson Planning** — content → structured lesson plan
3. **Script Generation** — plan → Manim Python script
4. **Render + Review** — script → video frames → quality feedback
5. **Fix Loop** — feedback + script → patched script (iterates with step 4)

Keep agents pure. No side effects, no shared state between steps. This makes it easier to test, debug, and maintain each component independently.

## Critical rules

1. **NEVER run the full lesson generation pipeline autonomously** (`uv run lesson ...`). This is expensive and must be human-initiated.
2. **Before completing any task**, always make sure all tests and linters pass:
```bash
uv run ruff check && uv run pyright && uv run pytest
```

## Commands

```bash
uv sync                # Install dependencies
uv run pytest          # Run tests
uv run ruff check      # Lint
uv run pyright         # Type check
```
