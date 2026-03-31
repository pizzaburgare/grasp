"""Manim rendering and audio management."""

from src.rendering.audio import AudioManager
from src.rendering.render import detect_scene_class, render_and_merge

__all__ = [
    "AudioManager",
    "detect_scene_class",
    "render_and_merge",
]
