from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class TTSEngine(ABC):
    @classmethod
    @abstractmethod
    def from_env(cls) -> Self:
        """Instantiate this engine from environment variables."""
        ...

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Returns:
            (audio_float32_mono, sample_rate)
        """
        ...
