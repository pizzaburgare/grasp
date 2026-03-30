# Grasp

![Grasp logo](assets/logo.png)

**Turn any university course into high quality video lessons - 100% automatically**

Website: [grasp-team.se](https://grasp-team.se)
YouTube: [@grasp-team](https://www.youtube.com/@grasp-team)

## Problem Statement

University is broken for most students.

- **One teacher for 200 students.** Lectures move at one speed - too fast for some, too slow for others. There's no way to personalize at scale.
- **Disengaged teaching.** Many lecturers would rather do research. The result: recycled 10-year-old slides and labs, and feedback that goes nowhere.
- **Canvas is a mess.** No standards for course structure. Students are buried under hundreds of slide pages and 1,000+ page textbooks.

Overall, students waste enormous amounts of time - and it doesn't have to be this way.

## Solution

Grasp takes raw course materials and transforms them into concise, animated explainer videos optimized for learning and exam results.

1. **Upload** all course materials - slides, exams, labs, textbooks.
2. **AI analysis** - a data pipeline processes each file type and an AI agent analyzes the content to create a structured learning plan.
3. **Animated video** - a Python animation script is generated and rendered using Manim.
4. **AI review loop** - a second AI agent reviews the rendered video, requesting fixes until quality criteria pass.
5. **Voice synthesis** - narration is generated via TTS and merged with the animation.

The result: a 57-minute lecture becomes a 15-minute Grasp video. Same concepts, 75% shorter, optimized for retention.

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.13 |
| **LLM orchestration** | LangChain, LangGraph |
| **LLM provider** | OpenRouter (Gemini, Claude, etc.) |
| **Animation** | Manim Community Edition |
| **TTS engines** | Kokoro, Qwen TTS, Piper |
| **Video processing** | FFmpeg, MoviePy |
| **Math rendering** | LaTeX |
| **Package manager** | uv |

## How to Run

### Prerequisites

Install the following before setting up the project:

| Dependency | Why | Install (macOS) |
|------------|-----|-----------------|
| **Python 3.13+** | Runtime | `brew install python@3.13` |
| **uv** | Fast Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **ffmpeg** | Video encoding (used by Manim & MoviePy) | `brew install ffmpeg` |
| **LaTeX** | Equation rendering in Manim | `brew install --cask mactex-no-gui` |

<details>
<summary>Linux (Debian/Ubuntu)</summary>

```bash
sudo apt update && sudo apt install -y python3.13 ffmpeg texlive-full
curl -LsSf https://astral.sh/uv/install.sh | sh
```
</details>

## Installation

```bash
git clone https://github.com/pizzaburgare/AI_Courses_lundaihackathon.git && cd AI_Courses_lundaihackathon
uv sync
```

> **One-time extra setup for the Kokoro engine** (uv venvs don't ship `pip` by default):
> ```bash
> uv pip install pip
> uv pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
> ```

Create a `.env` file in the project root:

```env
# ── Required ──────────────────────────────────────────────
OPENROUTER_API_KEY=your_key_here

# ── LLM models (all optional, override per pipeline stage) ─
# Defaults to google/gemini-3.1-pro-preview for all stages.
# Use --model flag to override all at once, or set individually:
LESSON_PLANNER_MODEL=google/gemini-3.1-pro-preview
MANIM_GENERATOR_MODEL=google/gemini-3.1-pro-preview
VIDEO_REVIEW_MODEL=google/gemini-3.1-pro-preview
VIDEO_FIX_MODEL=google/gemini-3.1-pro-preview

# ── TTS engine: "kokoro" | "qwen" | "piper" ──────────────
TTS_ENGINE=kokoro

# Kokoro overrides (all optional)
KOKORO_VOICE=am_adam          # see kokoro docs for available voices
KOKORO_LANG_CODE=a            # a=American EN, b=British EN, j=Japanese …
KOKORO_SPEED=1.2

# Qwen overrides (all optional)
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
QWEN_TTS_SPEAKER=Ryan
QWEN_TTS_LANGUAGE=English
QWEN_TTS_REF_AUDIO=           # path to .wav for voice cloning (falls back to bundled src/tts/clone.wav)
QWEN_TTS_REF_TEXT=             # transcript of reference audio (improves clone quality)

# Piper overrides (all optional)
PIPER_MODEL=models/en_US-ryan-high.onnx

# ── Audio / safety limits ────────────────────────────────
AUDIO_OUTPUT_DIR=.cache/audio
AUDIO_MANAGER_VERBOSE=1       # 0 = quiet during render, 1 = log each TTS call
TTS_MAX_SECONDS_PER_WORD=1.0  # reject audio exceeding this rate
TTS_SYNTHESIS_TIMEOUT_SECONDS=300
```

## Quick Start

```bash
# basic lesson
uv run lesson "LU Decomposition"

# with reference materials (PDFs, slides, images, videos)
uv run lesson "Fourier Transform" --input-dir ./slides

# high-quality final render
uv run lesson "QR Decomposition" --final

# override LLM model for all stages
uv run lesson "QR Decomposition" --model anthropic/claude-sonnet-4-5

# custom output directory
uv run lesson "Eigenvalues" --output-dir ./renders

# reuse a cached script to test rendering without regenerating script/lesson plan
uv run lesson "Kendall's notation" --input-dir ./courses/kosys --script-hash 641aab71c6b2b647 --skip-review
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `topic` (positional) | *required* | Topic to generate a lesson for |
| `--input-dir` | auto-detects `./input` | Directory with reference materials |
| `--output-dir` | `./output` | Where to place the final video |
| `--model` | per-stage env vars | Override all pipeline stages with a single model |
| `--final` | off | Render at high quality (`-qh`) instead of low (`-ql`) |
| `--script-hash` | auto | Reuse a specific cached script hash to test rendering without regenerating the lesson plan/script |

### Supported Input Formats

Place reference materials in the input directory. All formats are converted to LLM-ready content automatically:

| Type | Extensions |
|------|-----------|
| Text | `.txt`, `.md`, `.markdown`, `.rst`, `.csv` |
| PDF | `.pdf` (each page rendered as image, max 60 pages) |
| Images | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp` |
| Videos | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` (sampled at 1 fps, max 240 frames) |

## Preprocessing

Before generating a lesson, raw course materials can be batch-preprocessed into a normalized format using `batch_process.py`. It recursively scans a `raw/` subdirectory and converts each file based on its type, writing results to a `processed/` subdirectory alongside it.

| Input type | Output |
|------------|--------|
| `.pdf` | Converted to `.md`; TOC-based PDFs are split into topic files |
| `.mp4` | Transcribed to `.txt` with timestamps and screenshots processed by a VLM |
| Image files (`.png`, `.jpg`, `.webp`, etc.) | Converted to `.md` via VLM |
| `.md`, `.txt` | Copied as-is |

```bash
uv run src/preprocessing/batch_process.py <directory>
```

`<directory>` should be a course folder that contains a `raw/` subdirectory, e.g.:

```bash
uv run src/preprocessing/batch_process.py courses/FMNF05
```

Use local (non-LLM) PDF conversion:

```bash
uv run src/preprocessing/batch_process.py courses/FMNF05 --local
```

Processed files are written to `<directory>/processed/`, mirroring the original folder structure. Existing non-empty outputs are skipped.

## Document Selection

After preprocessing, use the document selector to pick the most relevant files for a lesson topic from a large `processed/` corpus.

The selector uses an LLM agent that:
- explores folders/files,
- summarizes candidate documents,
- returns the most relevant source files (typically lectures + a handful of exams).

Run it on a processed course directory:

```bash
uv run src/document_selector.py <processed_dir> "<topic query>"
```

Example:

```bash
uv run src/document_selector.py courses/FMNF05/processed "QR decomposition using Gram-Schmidt orthogonalization"
```

Output format:
- one selected file path per line,
- then token/cost usage summary.

Notes:
- Supported candidate file types are `.md`, `.markdown`, and `.txt`.
- Paths printed by the selector are absolute paths on your machine.

## Text-to-Speech

Audio narration is generated locally. Three engines are available, selected via `TTS_ENGINE`:

| Engine | Model | Notes |
|--------|-------|-------|
| `kokoro` | hexgrad/Kokoro-82M | Fast, high quality, no GPU needed |
| `qwen` | Qwen/Qwen3-TTS-12Hz-1.7B-Base | GPU recommended; supports voice cloning via reference audio |
| `piper` | en_US-ryan-high | Fastest, fully offline, CPU-only |

```bash
# optional: FlashAttention 2 for lower VRAM usage on CUDA (Qwen only)
uv run pip install flash-attn --no-build-isolation
```

## Pipeline Overview

```mermaid
flowchart TD
    A[Topic] --> B["1. Lesson Plan (LESSON_PLANNER_MODEL)"]
    B --> C["2. Manim Script (MANIM_GENERATOR_MODEL)"]
    C --> LOOP

    subgraph LOOP["3. Iterate up to 8 times at low quality"]
        R[Render + TTS] --> ERR{Error?}
        ERR -- Yes --> FIX_COMPILE["Fix agent applies search/replace edits"] --> R
        ERR -- No --> REV["Review agent (5 visual criteria)"]
        REV -- All pass --> DONE[Approved]
        REV -- Any fail --> FIX_VISUAL["Fix agent receives failed criteria + frames"] --> R
    end

    DONE --> OUT
    LOOP --> OUT

    OUT{--final?}
    OUT -- Yes --> HQ[High-quality re-render]
    OUT -- No --> V[Final Video]
    HQ --> V
```

**Review criteria:** text clipping, overlapping content, broken animations, content overflow, LaTeX rendering.

## Caching

Results are cached in `.cache/` to avoid redundant work:

| Asset | Cache key | Reused across |
|-------|-----------|---------------|
| Lesson plan + script | topic + inputs + prompt versions | Quality levels & TTS engines |
| Audio clips | text hash + TTS engine config | Renders of the same script |
| Final video | script hash + quality + TTS config | Nothing (unique per combo) |

Delete `.cache/` to force a full regeneration.

## Configuration

The default LLM model for all stages is `google/gemini-3.1-pro-preview`. Override all stages at once with `--model`, or set individual stage models via env vars (`LESSON_PLANNER_MODEL`, `MANIM_GENERATOR_MODEL`, `VIDEO_REVIEW_MODEL`, `VIDEO_FIX_MODEL`).

## Tests

```bash
uv run pytest                                          # all tests
uv run pytest -m integration                           # end-to-end only
uv run pytest tests/test_audiomanager.py               # single file
```

## Linting

```bash
uv run ruff check && uv run pyright   # both together
uv run ruff check                     # ruff only
uv run pyright                        # pyright only
```

## Project Structure

```
src/
  cli.py              # Entry point & argument parsing
  workflow.py          # Orchestrates the full pipeline
  script_generator.py  # LLM-driven Manim script generation & fixing
  input_processor.py   # PDF/video/image/text → LLM content parts
  audiomanager.py      # TTS integration & audio synchronization
  cache.py             # Content-addressed caching system
  settings.py          # Env var loading & defaults
  paths.py             # Central directory configuration
  search_replace.py    # Fuzzy search/replace for script patching
  llm_metrics.py       # Token usage & cost tracking
  tts/                 # TTS engine implementations (kokoro, piper, qwen)
prompts/               # LLM system prompts for each pipeline stage
models/                # Local TTS model weights (Piper)
```


## Acknowledgments

* Audio for cloning from [*Hypatia* by Charles Kingsley](https://librivox.org/), provided by [LibriVox](https://librivox.org/) (Public Domain).


## License

This project is licensed under the [MIT License](LICENSE).
