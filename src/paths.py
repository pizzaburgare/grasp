"""Central definition of all project paths."""

from pathlib import Path

# Project root (one level above this file's directory)
ROOT = Path(__file__).parent.parent

# --- Cache (ephemeral build artifacts, gitignored) ---
CACHE_DIR = ROOT / ".cache"
CACHE_AUDIO_DIR = CACHE_DIR / "audio"
CACHE_MANIM_DIR = CACHE_DIR / "manim"

# --- Models (local weights) ---
MODELS_DIR = ROOT / "models"
PIPER_DEFAULT_MODEL = MODELS_DIR / "en_US-ryan-high.onnx"

# --- Prompts ---
PROMPTS_DIR = ROOT / "prompts"
LESSON_PROMPT = PROMPTS_DIR / "lesson_prompt.md"
MANIM_PROMPT = PROMPTS_DIR / "manim_prompt.md"
VIDEO_REVIEW_PROMPT = PROMPTS_DIR / "video_review_prompt.md"
VIDEO_FIX_PROMPT = PROMPTS_DIR / "video_fix_prompt.md"
IMAGE_TRANSCRIBER_PROMPT = PROMPTS_DIR / "image_transcriber_prompt.md"
PDF_TRANSCRIBER_PROMPT = PROMPTS_DIR / "pdf_transcriber_prompt.md"
EXAM_CLASSIFIER_PROMPT = PROMPTS_DIR / "exam_classifier_prompt.md"

# --- Default input ---
INPUT_DIR = ROOT / "input"

# --- Default output ---
OUTPUT_DIR = ROOT / "output"
