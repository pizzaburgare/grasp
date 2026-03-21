import ast
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO

from moviepy import AudioFileClip, VideoFileClip

from src.cache import get_audio_cache_dir, get_lesson_cache_dir, save_video_to_cache
from src.paths import CACHE_AUDIO_DIR, CACHE_MANIM_DIR

logger = logging.getLogger(__name__)


# TODO: Use existing library for spinner
def run_command_with_spinner(
    command: list[str],
    env: dict[str, str],
    status_label: str,
) -> subprocess.CompletedProcess[str]:
    spinner = ("|", "/", "-", "\\")
    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Drain stdout/stderr on background threads to prevent the OS pipe
    # buffer (~64 KB) from filling up and deadlocking the subprocess.
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _drain(pipe: IO[str], buf: list[str]) -> None:
        for chunk in iter(lambda: pipe.read(4096), ""):
            buf.append(chunk)

    t_out = threading.Thread(target=_drain, args=(process.stdout, stdout_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(process.stderr, stderr_chunks), daemon=True)
    t_out.start()
    t_err.start()

    i = 0
    while process.poll() is None:
        elapsed = time.perf_counter() - start
        print(
            f"\r{status_label} {spinner[i % len(spinner)]} {elapsed:5.1f}s",
            end="",
            flush=True,
        )
        time.sleep(0.2)
        i += 1

    t_out.join()
    t_err.join()
    elapsed = time.perf_counter() - start
    print(f"\r{status_label} ✓ {elapsed:5.1f}s")

    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode if process.returncode is not None else 1,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


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
            if any(name and (name == "Scene" or name.endswith("Scene") or name in scene_like) for name in base_names):
                scene_like.append(cls.name)
                changed = True

    if scene_like:
        return scene_like[-1]

    raise ValueError(f"No Scene subclass found in {script_path}")


def render_and_merge(  # noqa: PLR0912
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

    if lesson_name:
        audio_cache_dir = get_audio_cache_dir(lesson_name)
        audio_work_dir = audio_cache_dir / "work"
    else:
        audio_cache_dir = CACHE_AUDIO_DIR
        audio_work_dir = CACHE_AUDIO_DIR
    cache_manim = CACHE_MANIM_DIR
    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    audio_work_dir.mkdir(parents=True, exist_ok=True)
    cache_manim.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["AUDIO_OUTPUT_DIR"] = str(audio_work_dir)
    env["AUDIO_MANAGER_VERBOSE"] = "0"
    if lesson_name:
        env["AUDIO_CACHE_DIR"] = str(audio_cache_dir)

    result = run_command_with_spinner(
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
        combined = (result.stderr or "").strip() or (result.stdout or "").strip()
        if combined:
            tail = "\n".join(combined.splitlines()[-30:])
            print("Manim error output (last 30 lines):")
            print(tail)
        raise RuntimeError("Manim render failed - check output above.")

    # Find the rendered mp4 (exclude partial movie files)
    # Sort by modification time and take the most recent to handle persistent cache
    mp4_files = [p for p in cache_manim.rglob("*.mp4") if "partial_movie_files" not in p.parts]
    if not mp4_files:
        raise FileNotFoundError(f"No MP4 found under {cache_manim} after render")
    video_path = max(mp4_files, key=lambda p: p.stat().st_mtime)
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
        final_clip = None

        # Suppress MoviePy verbose output during merge
        moviepy_logger = logging.getLogger("moviepy")
        old_level = moviepy_logger.level
        moviepy_logger.setLevel(logging.ERROR)
        try:
            video_clip = VideoFileClip(str(video_path))
            try:
                audio_clip = AudioFileClip(str(merged_audio))
                try:
                    final_clip = video_clip.with_audio(audio_clip)
                    final_clip.write_videofile(
                        str(final_path),
                        codec="libx264",
                        audio_codec="aac",
                        logger=None,
                    )
                finally:
                    if final_clip is not None and hasattr(final_clip, "close"):
                        final_clip.close()
                    audio_clip.close()
            finally:
                video_clip.close()
        finally:
            moviepy_logger.setLevel(old_level)

    print(f"Final video: {final_path}")

    if lesson_name and context_hash:
        save_video_to_cache(lesson_name, context_hash, final_path)
        print(f"Video cached: {get_lesson_cache_dir(lesson_name) / 'video' / f'{context_hash}.mp4'}")

    # Clean up working audio files - hash-named cache files are preserved
    if audio_work_dir != audio_cache_dir:
        shutil.rmtree(audio_work_dir, ignore_errors=True)

    return final_path
