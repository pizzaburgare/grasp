import logging
import os

import numpy as np
import torch

from .base import TTSEngine

# Suppress verbose logging from qwen_tts and its transformers dependency
logging.getLogger("qwen_tts").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
_DEFAULT_INSTRUCT = "Academic, clear, quick."


class QwenTTSEngine(TTSEngine):
    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        speaker: str = "Ryan",
        language: str = "English",
        instruct: str | None = _DEFAULT_INSTRUCT,
    ):
        self.model_id = model_id
        self.speaker = speaker
        self.language = language
        self.instruct = instruct
        self._model = None

    @classmethod
    def from_env(cls) -> "QwenTTSEngine":
        return cls(
            model_id=os.environ.get("QWEN_TTS_MODEL", _DEFAULT_MODEL),
            speaker=os.environ.get("QWEN_TTS_SPEAKER", "Ryan"),
            language=os.environ.get("QWEN_TTS_LANGUAGE", "English"),
            instruct=os.environ.get("QWEN_TTS_INSTRUCT", _DEFAULT_INSTRUCT) or None,
        )

    def _load_model(self):
        if self._model is None:
            from qwen_tts import Qwen3TTSModel  # noqa: PLC0415 — lazy import

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self._device_map(),
                dtype=self._dtype(),
            )
        return self._model

    @staticmethod
    def _device_map() -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _dtype() -> torch.dtype:
        if torch.cuda.is_available():
            return torch.bfloat16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.bfloat16
        return torch.float32

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        model = self._load_model()
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=self.language,
            speaker=self.speaker,
            instruct=self.instruct,
        )
        audio = wavs[0]
        if hasattr(audio, "detach"):  # torch.Tensor → numpy
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio[0]
        return audio, sr
