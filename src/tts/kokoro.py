"""TTS engine implementation using Kokoro text-to-speech."""

import os
from typing import Any

import numpy as np

from .base import TTSEngine

_DEFAULT_VOICE = "am_adam"
_DEFAULT_LANG_CODE = "a"  # 'a' = American English
_SAMPLE_RATE = 24_000
_DEFAULT_SPEED_ = 1.2


class KokoroTTSEngine(TTSEngine):
    """TTS engine using the Kokoro-82M model for speech synthesis."""

    ENGINE_NAME = "kokoro"

    def __init__(
        self,
        voice: str = _DEFAULT_VOICE,
        lang_code: str = _DEFAULT_LANG_CODE,
        speed: float = _DEFAULT_SPEED_,
    ) -> None:
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self._pipeline = None

    @classmethod
    def from_env(cls) -> "KokoroTTSEngine":
        return cls(
            voice=os.environ.get("KOKORO_VOICE", _DEFAULT_VOICE),
            lang_code=os.environ.get("KOKORO_LANG_CODE", _DEFAULT_LANG_CODE),
            speed=float(os.environ.get("KOKORO_SPEED", _DEFAULT_SPEED_)),
        )

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code=self.lang_code, repo_id="hexgrad/Kokoro-82M")
        return self._pipeline

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        pipeline = self._load_pipeline()
        chunks = []
        for _gs, _ps, audio in pipeline(text, voice=self.voice, speed=self.speed):
            if hasattr(audio, "detach"):  # torch.Tensor → numpy
                audio = audio.detach().cpu().numpy()  # type: ignore[union-attr]
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            return np.zeros(0, dtype=np.float32), _SAMPLE_RATE
        return np.concatenate(chunks), _SAMPLE_RATE
