import os
import shutil
import wave
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from manim import Scene

from src.cache import hash_text
from src.paths import CACHE_AUDIO_DIR
from src.settings import (
    AUDIO_DURATION_BUFFER_SECONDS,
    TTS_MAX_SECONDS_PER_WORD,
    TTS_SYNTHESIS_TIMEOUT_SECONDS,
)

from .tts import TTSEngine, get_default_engine

load_dotenv()

CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


def _audio_logs_enabled() -> bool:
    value = os.environ.get("AUDIO_MANAGER_VERBOSE", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _audio_log(text: str) -> None:
    if _audio_logs_enabled():
        print(text)


def _audio_dir() -> Path:
    d = Path(os.environ.get("AUDIO_OUTPUT_DIR", str(CACHE_AUDIO_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _audio_cache_dir() -> Path | None:
    """Return the persistent audio cache directory, or None if not configured.

    Set the ``AUDIO_CACHE_DIR`` environment variable to enable per-text-hash
    caching so that identical narration snippets are only synthesised once.
    """
    if d := os.environ.get("AUDIO_CACHE_DIR"):
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return None


def _engine_cache_salt(engine: TTSEngine) -> str | None:
    """Return a stable salt that captures TTS settings affecting waveform output."""
    fields = (
        "ENGINE_NAME",
        "model_id",
        "speaker",
        "language",
        "ref_audio",
        "ref_text",
        "model_path",
        "voice",
        "lang_code",
        "speed",
    )
    pairs: list[str] = []
    engine_dict = getattr(engine, "__dict__", {})
    class_engine_name = getattr(type(engine), "ENGINE_NAME", None)

    for field in fields:
        if field == "ENGINE_NAME":
            value = class_engine_name
        elif field in engine_dict:
            value = engine_dict[field]
        else:
            continue

        if value is None:
            continue
        pairs.append(f"{field}={value}")

    if not pairs:
        return None
    return "|".join(pairs)


def create_wav(text_to_speak: str, i: int, engine: TTSEngine) -> float:
    """Synthesise *text_to_speak* and write it as ``audio_{i}.wav``.

    When ``AUDIO_CACHE_DIR`` is set the function first checks whether a WAV
    for this exact text already exists (keyed by SHA-256 hash).  On a cache
    hit the TTS engine is bypassed entirely.  On a cache miss the synthesised
    file is stored in the cache for future reuse.
    """
    out_path = _audio_dir() / f"audio_{i}.wav"
    cache_dir = _audio_cache_dir()
    cache_salt = _engine_cache_salt(engine)
    cache_key = hash_text(text_to_speak, salt=cache_salt)
    cached_wav: Path | None = None

    if cache_dir is not None:
        cached_wav = cache_dir / f"{cache_key}.wav"
        if cached_wav.exists():
            shutil.copy2(cached_wav, out_path)
            with wave.open(str(out_path), "rb") as wf:
                return (wf.getnframes() / wf.getframerate()) + AUDIO_DURATION_BUFFER_SECONDS

    word_count = max(len(text_to_speak.split()), 1)
    with ThreadPoolExecutor(max_workers=1) as _executor:
        _future = _executor.submit(engine.synthesize, text_to_speak)
        try:
            audio, sr = _future.result(timeout=TTS_SYNTHESIS_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            raise RuntimeError(f"TTS synthesis timed out after {TTS_SYNTHESIS_TIMEOUT_SECONDS}s " f"({word_count} words)")

    actual_duration = len(audio) / sr
    max_duration = word_count * TTS_MAX_SECONDS_PER_WORD
    if actual_duration > max_duration:
        raise RuntimeError(
            f"TTS output rejected: {actual_duration:.1f}s for {word_count} words "
            f"(limit {max_duration:.1f}s at {TTS_MAX_SECONDS_PER_WORD}s/word). "
            "Likely corrupt/runaway model output."
        )

    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(sr)
        wav_file.writeframes(audio_int16.tobytes())

    if cached_wav is not None:
        shutil.copy2(out_path, cached_wav)

    return (len(audio_int16) / sr) + AUDIO_DURATION_BUFFER_SECONDS


class AudioManager:
    def __init__(self, scene: Scene, engine: TTSEngine | None = None):
        self.i: int = 0
        self.scene = scene
        self.engine = engine or get_default_engine()
        self.times: list[float] = []
        self.audio_durations: list[float] = []

    def say(self, text: str) -> None:
        _audio_log(f"AudioManager: {text} at {self.scene.renderer.time:.2f} seconds")
        self.i += 1
        self.times.append(self.scene.renderer.time)
        duration = create_wav(text, self.i, self.engine)
        self.audio_durations.append(duration)
        _audio_log(f"AudioManager: Audio duration is {duration:.2f} seconds")

    def done_say(self) -> None:
        if not self.times or not self.audio_durations:
            _audio_log("AudioManager: done_say() called before say(); skipping")
            return

        _audio_log("AudioManager: Done saying text")
        time_since_started = self.scene.renderer.time - self.times[-1]
        _audio_log(f"AudioManager: Time since started saying text: {time_since_started:.2f} seconds")
        to_sleep = self.audio_durations[-1] - time_since_started
        if to_sleep > 0:
            _audio_log(f"AudioManager: Sleeping for {to_sleep:.2f} seconds")
            self.scene.wait(to_sleep)

    def merge_audio(self) -> None:
        if self.i == 0:
            _audio_log("AudioManager: No audio files to merge")
            return

        if not self.times or not self.audio_durations:
            _audio_log("AudioManager: Missing timing metadata; merge skipped")
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
                if start_frame >= len(audio_data):
                    continue
                end_frame = min(start_frame + len(audio_chunk), len(audio_data))
                audio_data[start_frame:end_frame] = audio_chunk[: end_frame - start_frame]

        merged_path = audio_dir / "merged_audio.wav"
        with wave.open(str(merged_path), "wb") as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        _audio_log(f"AudioManager: Merged {self.i} audio files to {merged_path}")
