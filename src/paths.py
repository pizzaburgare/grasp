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

# --- Source prompts ---
SRC_DIR = ROOT / "src"
LESSON_PROMPT = SRC_DIR / "lesson_prompt.md"
MANIM_PROMPT = SRC_DIR / "manim_prompt.md"

# --- Default input ---
INPUT_DIR = ROOT / "input"

# --- Default output ---
OUTPUT_DIR = ROOT / "output"
