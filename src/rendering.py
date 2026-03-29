import ast
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from src.cache import get_audio_cache_dir, get_lesson_cache_dir, hash_text, save_video_to_cache
from src.command_runner import run_command
from src.paths import CACHE_AUDIO_DIR, CACHE_MANIM_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio pre-synthesis helpers
# ---------------------------------------------------------------------------


def _extract_say_texts(script_path: Path) -> list[str]:
    """Extract all string arguments from ``audio_manager.say(...)`` calls via AST.

    Supports plain string literals and implicit string concatenation.  Returns
    them in source order so the cache can be pre-warmed before Manim renders.
    """
    tree = ast.parse(script_path.read_text())
    texts: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match *.say(...)
        if isinstance(func, ast.Attribute) and func.attr == "say" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                texts.append(arg.value)
            elif isinstance(arg, ast.JoinedStr):
                pass  # f-strings can't be pre-synthesized
    return texts


def _presynthesise_audio(
    texts: list[str],
    audio_cache_dir: Path,
    env: dict[str, str],
) -> int:
    """Synthesise all *texts* into *audio_cache_dir* using the TTS engine.

    Only texts whose cache file is missing are synthesised.  Returns the number
    of texts that required synthesis (cache misses).
    """
    from src.audiomanager import _engine_cache_salt
    from src.tts import get_default_engine

    engine = get_default_engine()
    salt = _engine_cache_salt(engine)

    # Determine which texts are missing from cache
    missing: list[tuple[str, str, Path]] = []
    for text in texts:
        key = hash_text(text, salt=salt)
        cached_path = audio_cache_dir / f"{key}.wav"
        if not cached_path.exists():
            missing.append((text, key, cached_path))

    if not missing:
        print(f"Audio pre-synthesis: all {len(texts)} clips cached, skipping")
        return 0

    print(f"Audio pre-synthesis: {len(missing)}/{len(texts)} clips need synthesis")

    import wave

    import numpy as np

    channels = 1
    sample_width = 2  # 16-bit

    # Use batch synthesis when the engine supports it natively.
    # Process in chunks to avoid GPU OOM on large scripts.
    batch_size = int(os.environ.get("TTS_BATCH_SIZE", "4"))
    missing_texts = [text for text, _key, _path in missing]
    results: list[tuple[np.ndarray, int]] = []
    for chunk_start in range(0, len(missing_texts), batch_size):
        chunk = missing_texts[chunk_start : chunk_start + batch_size]
        results.extend(engine.synthesize_batch(chunk))

    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    for i, ((_text, _key, cached_path), (audio, sr)) in enumerate(
        zip(missing, results, strict=True), 1
    ):
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(cached_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())

        dur = len(audio_int16) / sr
        print(f"  [{i}/{len(missing)}] {dur:.1f}s audio")

    return len(missing)


def detect_scene_class(script_path: Path) -> str:
    tree = ast.parse(script_path.read_text())

    def _base_name(base: ast.expr) -> str:
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
        return ""

    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    scene_like: list[str] = []
    changed = True

    while changed:
        changed = False
        for cls in classes:
            if cls.name in scene_like:
                continue
            base_names = [_base_name(base) for base in cls.bases]
            if any(
                name and (name == "Scene" or name.endswith("Scene") or name in scene_like)
                for name in base_names
            ):
                scene_like.append(cls.name)
                changed = True

    if scene_like:
        return scene_like[-1]

    raise ValueError(f"No Scene subclass found in {script_path}")


def _find_rendered_video(cache_manim: Path) -> Path:
    """Return the most recently modified MP4 under *cache_manim*, excluding partial files."""
    mp4_files = [p for p in cache_manim.rglob("*.mp4") if "partial_movie_files" not in p.parts]
    if not mp4_files:
        raise FileNotFoundError(f"No MP4 found under {cache_manim} after render")
    return max(mp4_files, key=lambda p: p.stat().st_mtime)


def render_and_merge(
    script_path: Path,
    output_dir: Path,
    topic_slug: str,
    final_quality: bool = False,
    *,
    lesson_name: str | None = None,
    context_hash: str | None = None,
) -> Path:
    """Render the Manim script and merge TTS audio into the final video.

    When *lesson_name* and *context_hash* are supplied:
    - ``AUDIO_CACHE_DIR`` is set to ``<lesson>/audio/`` (persistent hash store)
    - ``AUDIO_OUTPUT_DIR`` is set to ``<lesson>/audio/work/`` (numbered working
      files + ``merged_audio.wav``; cleaned up on success)
    - The finished video is written into the video cache.
    """
    scene_class = detect_scene_class(script_path)
    quality_flag = "-qh" if final_quality else "-ql"
    quality_label = "high" if final_quality else "low"
    print(f"Rendering scene: {scene_class} ({quality_label} quality)")

    # Set up audio directories and environment.
    # The Python variables are used for dir creation, cleanup, and caching;
    # the env vars are forwarded to the Manim subprocess (TTS engine reads them).
    env = os.environ.copy()
    env["AUDIO_MANAGER_VERBOSE"] = "0"
    if lesson_name:
        audio_cache_dir = get_audio_cache_dir(lesson_name)
        audio_work_dir = audio_cache_dir / "work"
        env["AUDIO_CACHE_DIR"] = str(audio_cache_dir)
    else:
        audio_cache_dir = CACHE_AUDIO_DIR
        audio_work_dir = CACHE_AUDIO_DIR
    env["AUDIO_OUTPUT_DIR"] = str(audio_work_dir)

    cache_manim = CACHE_MANIM_DIR
    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    audio_work_dir.mkdir(parents=True, exist_ok=True)
    cache_manim.mkdir(parents=True, exist_ok=True)

    # Pre-synthesise all audio so the Manim subprocess only hits cache.
    say_texts = _extract_say_texts(script_path)
    if say_texts:
        t0 = time.perf_counter()
        _presynthesise_audio(say_texts, audio_cache_dir, env)
        print(f"Audio pre-synthesis total: {time.perf_counter() - t0:.1f}s")

    result = run_command(
        command=[
            sys.executable,
            "-m",
            "manim",
            quality_flag,
            "--media_dir",
            str(cache_manim),
            str(script_path),
            scene_class,
        ],
        env=env,
        status_label="Compiling video",
    )
    if result.returncode != 0:
        raise RuntimeError("Manim render failed - check output above.")

    video_path = _find_rendered_video(cache_manim)
    print(f"Video rendered: {video_path}")

    # Merge audio
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{topic_slug}.mp4"

    shutil.copy2(video_path, final_path)

    print(f"Final video: {final_path}")

    if lesson_name and context_hash:
        save_video_to_cache(lesson_name, context_hash, final_path)
        print(
            f"Video cached: {get_lesson_cache_dir(lesson_name) / 'video' / f'{context_hash}.mp4'}"
        )

    # Clean up working audio files - hash-named cache files are preserved
    if audio_work_dir != audio_cache_dir:
        shutil.rmtree(audio_work_dir, ignore_errors=True)

    return final_path
