"""Project-wide runtime defaults and tunables."""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LLM_MODEL = "google/gemini-3.1-pro-preview"
DEFAULT_TTS_ENGINE = "kokoro"
AUDIO_DURATION_BUFFER_SECONDS = 1.0
MAX_SCRIPT_ITERATIONS = 8

# TTS safety limits.
# TTS_MAX_SECONDS_PER_WORD: reject synthesised audio longer than this many
#   seconds per word (catches runaway/corrupt model output, e.g. 10-min noise).
# TTS_SYNTHESIS_TIMEOUT_SECONDS: hard wall-clock limit per synthesis call.
TTS_MAX_SECONDS_PER_WORD: float = float(os.getenv("TTS_MAX_SECONDS_PER_WORD", "1.0"))
TTS_SYNTHESIS_TIMEOUT_SECONDS: int = int(
    os.getenv("TTS_SYNTHESIS_TIMEOUT_SECONDS", "300")
)

# Per-stage model configuration.
# Each can be overridden independently via the corresponding .env variable;
# all three fall back to DEFAULT_LLM_MODEL when the variable is unset.
LESSON_PLANNER_MODEL: str = os.getenv("LESSON_PLANNER_MODEL", DEFAULT_LLM_MODEL)
MANIM_GENERATOR_MODEL: str = os.getenv("MANIM_GENERATOR_MODEL", DEFAULT_LLM_MODEL)
VIDEO_REVIEW_MODEL: str = os.getenv("VIDEO_REVIEW_MODEL", DEFAULT_LLM_MODEL)
VIDEO_FIX_MODEL: str = os.getenv("VIDEO_FIX_MODEL", DEFAULT_LLM_MODEL)
DOCUMENT_SELECTOR_MODEL: str = os.getenv("DOCUMENT_SELECTOR_MODEL", DEFAULT_LLM_MODEL)
