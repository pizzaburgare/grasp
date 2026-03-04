import os
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from manim import Scene

from .tts import TTSEngine, get_default_engine
from src.paths import CACHE_AUDIO_DIR

load_dotenv()

CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


def _audio_dir() -> Path:
    d = Path(os.environ.get("AUDIO_OUTPUT_DIR", str(CACHE_AUDIO_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_wav(text_to_speak: str, i: int, engine: TTSEngine) -> float:
    audio, sr = engine.synthesize(text_to_speak)
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    out_path = _audio_dir() / f"audio_{i}.wav"
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(sr)
        wav_file.writeframes(audio_int16.tobytes())

    return (len(audio_int16) / sr) + 1  # +1s buffer


class AudioManager:
    def __init__(self, scene: Scene, engine: TTSEngine | None = None):
        self.i: int = 0
        self.scene = scene
        self.engine = engine or get_default_engine()
        self.times: list[float] = []
        self.audio_durations: list[float] = []

    def say(self, text: str) -> None:
        print(f"AudioManager: {text} at {self.scene.renderer.time:.2f} seconds")
        self.i += 1
        self.times.append(self.scene.renderer.time)
        duration = create_wav(text, self.i, self.engine)
        self.audio_durations.append(duration)
        print(f"AudioManager: Audio duration is {duration:.2f} seconds")

    def done_say(self) -> None:
        print("AudioManager: Done saying text")
        time_since_started = self.scene.renderer.time - self.times[-1]
        print(
            f"AudioManager: Time since started saying text: {time_since_started:.2f} seconds"
        )
        to_sleep = self.audio_durations[-1] - time_since_started
        if to_sleep > 0:
            print(f"AudioManager: Sleeping for {to_sleep:.2f} seconds")
            self.scene.wait(to_sleep)

    def merge_audio(self) -> None:
        if self.i == 0:
            print("AudioManager: No audio files to merge")
            return

        audio_dir = _audio_dir()

        with wave.open(str(audio_dir / "audio_1.wav"), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()

        last_time = self.times[-1]
        last_duration = self.audio_durations[-1]
        total_duration = last_time + last_duration
        total_frames = int(total_duration * sample_rate) + sample_rate  # add buffer

        audio_data = np.zeros(total_frames, dtype=np.int16)

        for audio_idx in range(1, self.i + 1):
            with wave.open(str(audio_dir / f"audio_{audio_idx}.wav"), "rb") as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_chunk = np.frombuffer(frames, dtype=np.int16)

                start_frame = int(self.times[audio_idx - 1] * sample_rate)
                end_frame = start_frame + len(audio_chunk)
                audio_data[start_frame:end_frame] = audio_chunk

        merged_path = audio_dir / "merged_audio.wav"
        with wave.open(str(merged_path), "wb") as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        print(f"AudioManager: Merged {self.i} audio files to {merged_path}")
