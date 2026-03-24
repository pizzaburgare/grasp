import ast
import contextlib
import logging
import os
import shutil
import sys
from pathlib import Path

from moviepy import AudioFileClip, VideoFileClip

from src.cache import get_audio_cache_dir, get_lesson_cache_dir, save_video_to_cache
from src.command_runner import CommandRunner
from src.paths import CACHE_AUDIO_DIR, CACHE_MANIM_DIR

logger = logging.getLogger(__name__)


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

    result = CommandRunner.run(
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
    merged_audio = audio_work_dir / "merged_audio.wav"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{topic_slug}.mp4"

    if not merged_audio.exists():
        print("No merged_audio.wav found - copying video without audio")
        shutil.copy2(video_path, final_path)
    else:
        print("Merging audio into video ...")
        moviepy_logger = logging.getLogger("moviepy")
        with contextlib.ExitStack() as stack:
            stack.callback(moviepy_logger.setLevel, moviepy_logger.level)
            moviepy_logger.setLevel(logging.ERROR)
            video_clip = VideoFileClip(str(video_path))
            stack.callback(video_clip.close)
            audio_clip = AudioFileClip(str(merged_audio))
            stack.callback(audio_clip.close)
            final_clip = video_clip.with_audio(audio_clip)
            stack.callback(final_clip.close)
            final_clip.write_videofile(
                str(final_path), codec="libx264", audio_codec="aac", logger=None, threads=4
            )

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
