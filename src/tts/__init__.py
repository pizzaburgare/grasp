import os

from src.settings import DEFAULT_TTS_ENGINE

from .base import TTSEngine
from .kokoro import KokoroTTSEngine
from .piper import PiperTTSEngine
from .qwen import QwenTTSEngine

# Registry is built automatically from ENGINE_NAME on each engine class.
# To add a new engine: create the class with ENGINE_NAME and add it here.
_REGISTRY: dict[str, type[TTSEngine]] = {
    cls.ENGINE_NAME: cls  # type: ignore[attr-defined]
    for cls in (PiperTTSEngine, QwenTTSEngine, KokoroTTSEngine)
}

_DEFAULT_ENGINE = DEFAULT_TTS_ENGINE

__all__ = [
    "KokoroTTSEngine",
    "PiperTTSEngine",
    "QwenTTSEngine",
    "TTSEngine",
    "available_engines",
    "get_default_engine",
]


def available_engines() -> list[str]:
    """Return the names of all registered TTS engines."""
    return list(_REGISTRY.keys())


def get_default_engine() -> TTSEngine:
    """Instantiate the TTS engine selected by the TTS_ENGINE env var."""
    name = os.environ.get("TTS_ENGINE", _DEFAULT_ENGINE).lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown TTS_ENGINE '{name}'. Choose one of: {', '.join(_REGISTRY)}.")
    return _REGISTRY[name].from_env()
