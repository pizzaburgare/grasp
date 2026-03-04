"""Cache management for lesson generation assets.

Cache structure::

    .cache/
        {lesson-name-slug}/
            script/
                {context_hash}.py
            audio/
                {text_hash}.wav
            video/
                {context_hash}.mp4

*Context hash* is derived from the lesson topic + all bytes of every file in the
input directory. *Text hash* is derived from the exact narration text that TTS
will speak.
"""

import hashlib
import re
import shutil
from pathlib import Path
from typing import Optional

from src.paths import CACHE_DIR


# ---------------------------------------------------------------------------
# Lesson key / directory helpers
# ---------------------------------------------------------------------------


def lesson_name_to_key(lesson_name: str) -> str:
    """Normalise *lesson_name* to a safe, filesystem-friendly directory key."""
    key = lesson_name.lower()
    key = re.sub(r"[^\w\s-]", "", key)
    key = re.sub(r"[\s_]+", "-", key).strip("-")
    return key or "default"


def get_lesson_cache_dir(lesson_name: str) -> Path:
    """Return the per-lesson cache root (not yet created)."""
    return CACHE_DIR / lesson_name_to_key(lesson_name)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def hash_context(topic: str, input_dir: Optional[str] = None) -> str:
    """Return a 16-char SHA-256 hex digest of *topic* + all file bytes in *input_dir*.

    The hash changes whenever the topic text or any input file changes, making
    it a stable cache key for the generated script and final video.
    """
    h = hashlib.sha256()
    h.update(topic.encode())
    if input_dir:
        p = Path(input_dir)
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(f.name.encode())
                h.update(f.read_bytes())
    return h.hexdigest()[:16]


def hash_text(text: str) -> str:
    """Return a 16-char SHA-256 hex digest of *text* (used as audio cache key)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Script cache
# ---------------------------------------------------------------------------


def get_cached_script(lesson_name: str, context_hash: str) -> Optional[Path]:
    """Return the cached Manim script path if it exists, else ``None``."""
    p = get_lesson_cache_dir(lesson_name) / "script" / f"{context_hash}.py"
    return p if p.exists() else None


def save_script_to_cache(
    lesson_name: str, context_hash: str, script_path: Path
) -> Path:
    """Copy *script_path* into the script cache and return the destination path."""
    dest = get_lesson_cache_dir(lesson_name) / "script" / f"{context_hash}.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_path, dest)
    return dest


# ---------------------------------------------------------------------------
# Audio cache
# ---------------------------------------------------------------------------


def get_audio_cache_dir(lesson_name: str) -> Path:
    """Return the per-lesson audio cache directory (not yet created)."""
    return get_lesson_cache_dir(lesson_name) / "audio"


def get_cached_audio(lesson_name: str, text: str) -> Optional[Path]:
    """Return the cached WAV path for *text* if it exists, else ``None``."""
    p = get_audio_cache_dir(lesson_name) / f"{hash_text(text)}.wav"
    return p if p.exists() else None


def save_audio_to_cache(lesson_name: str, text: str, wav_path: Path) -> Path:
    """Copy *wav_path* into the audio cache keyed by the hash of *text*."""
    dest = get_audio_cache_dir(lesson_name) / f"{hash_text(text)}.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wav_path, dest)
    return dest


# ---------------------------------------------------------------------------
# Video cache
# ---------------------------------------------------------------------------


def get_cached_video(lesson_name: str, context_hash: str) -> Optional[Path]:
    """Return the cached rendered video path if it exists, else ``None``."""
    p = get_lesson_cache_dir(lesson_name) / "video" / f"{context_hash}.mp4"
    return p if p.exists() else None


def save_video_to_cache(lesson_name: str, context_hash: str, video_path: Path) -> Path:
    """Copy *video_path* into the video cache and return the destination path."""
    dest = get_lesson_cache_dir(lesson_name) / "video" / f"{context_hash}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, dest)
    return dest
