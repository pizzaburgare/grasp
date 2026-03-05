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

*Context hash* is derived from the lesson topic + input file paths/bytes +
optional generation metadata (model, quality, TTS config, ...). *Text hash* is
derived from the exact narration text that TTS will speak.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

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


def _json_default(value: Any) -> str:
    return str(value)


def hash_context(
    topic: str,
    input_dir: Optional[str] = None,
    *,
    extra_context: Mapping[str, Any] | None = None,
) -> str:
    """Return a 16-char SHA-256 hex digest of *topic* + all file bytes in *input_dir*.

    The hash changes whenever the topic text, any input file, or optional
    ``extra_context`` metadata changes, making it a stable cache key for the
    generated script and final video.
    """
    h = hashlib.sha256()
    h.update(topic.encode())
    if input_dir:
        p = Path(input_dir)
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(f.relative_to(p).as_posix().encode())
                h.update(f.read_bytes())

    if extra_context:
        context_payload = json.dumps(
            dict(extra_context),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_json_default,
        )
        h.update(context_payload.encode())

    return h.hexdigest()[:16]


def hash_text(text: str, *, salt: str | None = None) -> str:
    """Return a 16-char SHA-256 hex digest of *text* (optionally salted)."""
    h = hashlib.sha256()
    if salt:
        h.update(salt.encode())
    h.update(text.encode())
    return h.hexdigest()[:16]


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


def get_cached_audio(
    lesson_name: str, text: str, *, salt: str | None = None
) -> Optional[Path]:
    """Return the cached WAV path for *text* if it exists, else ``None``."""
    p = get_audio_cache_dir(lesson_name) / f"{hash_text(text, salt=salt)}.wav"
    return p if p.exists() else None


def save_audio_to_cache(
    lesson_name: str,
    text: str,
    wav_path: Path,
    *,
    salt: str | None = None,
) -> Path:
    """Copy *wav_path* into the audio cache keyed by the hash of *text*."""
    dest = get_audio_cache_dir(lesson_name) / f"{hash_text(text, salt=salt)}.wav"
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
