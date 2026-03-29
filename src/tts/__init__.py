import os
from typing import TYPE_CHECKING

from src.settings import DEFAULT_TTS_ENGINE

from .base import TTSEngine

if TYPE_CHECKING:
    from .kokoro import KokoroTTSEngine
    from .piper import PiperTTSEngine
    from .qwen import QwenTTSEngine

# Lazy registry: map engine name -> (module_name, class_name) to avoid
# importing heavyweight dependencies (torch, kokoro, etc.) at init time.
_LAZY_REGISTRY: dict[str, tuple[str, str]] = {
    "piper": (".piper", "PiperTTSEngine"),
    "qwen": (".qwen", "QwenTTSEngine"),
    "kokoro": (".kokoro", "KokoroTTSEngine"),
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
    return list(_LAZY_REGISTRY.keys())


def _resolve_engine_class(name: str) -> type[TTSEngine]:
    """Import and return the engine class for *name*."""
    if name not in _LAZY_REGISTRY:
        engines = ", ".join(_LAZY_REGISTRY)
        raise ValueError(f"Unknown TTS_ENGINE '{name}'. Choose one of: {engines}.")
    module_name, class_name = _LAZY_REGISTRY[name]
    import importlib

    module = importlib.import_module(module_name, package=__name__)
    return getattr(module, class_name)  # type: ignore[no-any-return]


def get_default_engine() -> TTSEngine:
    """Instantiate the TTS engine selected by the TTS_ENGINE env var."""
    name = os.environ.get("TTS_ENGINE", _DEFAULT_ENGINE).lower()
    cls = _resolve_engine_class(name)
    return cls.from_env()
