from abc import ABC, abstractmethod

import numpy as np


class TTSEngine(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Returns:
            (audio_float32_mono, sample_rate)
        """
        ...
