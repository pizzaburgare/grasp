import os

from .base import TTSEngine
from .piper import PiperTTSEngine
from .qwen import QwenTTSEngine

__all__ = ["TTSEngine", "QwenTTSEngine", "PiperTTSEngine", "get_default_engine"]


def get_default_engine() -> TTSEngine:
    """Instantiate the TTS engine selected by the TTS_ENGINE env var (default: qwen)."""
    engine = os.environ.get("TTS_ENGINE", "qwen").lower()
    if engine == "qwen":
        return QwenTTSEngine.from_env()
    if engine == "piper":
        return PiperTTSEngine.from_env()
    raise ValueError(f"Unknown TTS_ENGINE '{engine}'. Choose 'qwen' or 'piper'.")
