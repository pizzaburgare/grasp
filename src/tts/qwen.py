import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import TTSEngine

if TYPE_CHECKING:
    import torch

# Default reference audio bundled alongside this file (used for voice cloning
# when QWEN_TTS_REF_AUDIO is not set and the model type is 'base').
_BUNDLED_REF_AUDIO = str(Path(__file__).parent / "clone.wav")

# Suppress verbose logging from qwen_tts and its transformers dependency
logging.getLogger("qwen_tts").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_MIN_AUDIO_CHANNELS = 2


class QwenTTSEngine(TTSEngine):
    ENGINE_NAME = "qwen"

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        speaker: str = "Ryan",
        language: str = "English",
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.speaker = speaker
        self.language = language
        self.ref_audio = ref_audio  # required for base / voice-clone models
        self.ref_text = ref_text  # optional transcript of ref_audio (improves clone quality)
        self._model = None

    @classmethod
    def from_env(cls) -> QwenTTSEngine:
        return cls(
            model_id=os.environ.get("QWEN_TTS_MODEL", _DEFAULT_MODEL),
            speaker=os.environ.get("QWEN_TTS_SPEAKER", "Ryan"),
            language=os.environ.get("QWEN_TTS_LANGUAGE", "English"),
            ref_audio=os.environ.get("QWEN_TTS_REF_AUDIO") or _BUNDLED_REF_AUDIO,
            ref_text=os.environ.get("QWEN_TTS_REF_TEXT") or None,
        )

    def _load_model(self) -> Any:
        if self._model is None:
            from qwen_tts import Qwen3TTSModel

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self._device_map(),
                dtype=self._dtype(),
            )
        return self._model

    @staticmethod
    def _device_map() -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _dtype() -> torch.dtype:
        import torch

        if torch.cuda.is_available():
            return torch.bfloat16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.bfloat16
        return torch.float32

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        import torch

        model = self._load_model()
        model_type = model.model.tts_model_type

        with torch.inference_mode():
            if model_type == "custom_voice":
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=self.language,
                    speaker=self.speaker,
                    temperature=0.7,
                )
            elif model_type == "base":
                if not self.ref_audio:
                    raise ValueError(
                        "The Qwen base model requires a reference audio for voice cloning. "
                        "Set QWEN_TTS_REF_AUDIO to a .wav file path."
                    )
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=self.language,
                    ref_audio=self.ref_audio,
                    ref_text=self.ref_text or None,
                    x_vector_only_mode=not bool(self.ref_text),
                )
            elif model_type == "voice_design":
                wavs, sr = model.generate_voice_design(
                    text=text,
                    instruct=self.speaker,
                    language=self.language,
                )
            else:
                raise ValueError(f"Unsupported Qwen TTS model type: {model_type!r}")

        audio = wavs[0]
        if hasattr(audio, "detach"):  # torch.Tensor → numpy
            audio = audio.detach().cpu().numpy()  # type: ignore[union-attr]
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == _MIN_AUDIO_CHANNELS:
            audio = audio[0]
        return audio, sr

    @staticmethod
    def _to_numpy_mono(wav: np.ndarray | torch.Tensor) -> np.ndarray:
        """Convert a single waveform to float32 mono numpy array."""
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()  # type: ignore[union-attr]
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim == _MIN_AUDIO_CHANNELS:
            arr = arr[0]
        return arr

    def synthesize_batch(self, texts: list[str]) -> list[tuple[np.ndarray, int]]:
        """Synthesize multiple texts using native batch support."""
        import torch

        if not texts:
            return []

        model = self._load_model()
        model_type = model.model.tts_model_type
        n = len(texts)

        with torch.inference_mode():
            if model_type == "custom_voice":
                wavs, sr = model.generate_custom_voice(
                    text=texts,
                    language=[self.language] * n,
                    speaker=[self.speaker] * n,
                    temperature=0.7,
                )
            elif model_type == "base":
                if not self.ref_audio:
                    raise ValueError("The Qwen base model requires a reference audio.")
                wavs, sr = model.generate_voice_clone(
                    text=texts,
                    language=[self.language] * n,
                    ref_audio=[self.ref_audio] * n,
                    ref_text=[self.ref_text or None] * n,
                    x_vector_only_mode=[not bool(self.ref_text)] * n,
                )
            elif model_type == "voice_design":
                wavs, sr = model.generate_voice_design(
                    text=texts,
                    instruct=[self.speaker] * n,
                    language=[self.language] * n,
                )
            else:
                raise ValueError(f"Unsupported Qwen TTS model type: {model_type!r}")

        return [(self._to_numpy_mono(wav), sr) for wav in wavs]
