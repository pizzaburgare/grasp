import io
import os
import wave

import numpy as np

from src.paths import PIPER_DEFAULT_MODEL

from .base import TTSEngine

_DEFAULT_MODEL = str(PIPER_DEFAULT_MODEL)


class PiperTTSEngine(TTSEngine):
    ENGINE_NAME = "piper"

    def __init__(self, model_path: str = _DEFAULT_MODEL):
        self.model_path = model_path
        self._voice = None

    @classmethod
    def from_env(cls) -> "PiperTTSEngine":
        return cls(model_path=os.environ.get("PIPER_MODEL", _DEFAULT_MODEL))

    def _load_voice(self):
        if self._voice is None:
            from piper.voice import PiperVoice  # noqa: PLC0415 — lazy import

            self._voice = PiperVoice.load(self.model_path)
        return self._voice

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        voice = self._load_voice()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize(text, wav_file)
        buf.seek(0)
        with wave.open(buf, "rb") as wav_file:
            sr = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
        audio_int16 = np.frombuffer(frames, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32767.0, sr
