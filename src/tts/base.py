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

    def synthesize_batch(self, texts: list[str]) -> list[tuple[np.ndarray, int]]:
        """Synthesize multiple texts. Default falls back to sequential calls.

        Engines that support native batching should override this.
        """
        return [self.synthesize(t) for t in texts]
