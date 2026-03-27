import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

from pytest import MonkeyPatch


class TestRenderAndMergeMoviepySafety:
    def test_moviepy_current_version_has_no_verbose_fps_kwarg(self) -> None:
        from src import rendering

        signature = inspect.signature(rendering.VideoFileClip.write_videofile)
        assert "verbose_fps" not in signature.parameters

    def _setup_render_fixture(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> tuple[Path, Path]:
        manim_dir = tmp_path / "manim"
        audio_dir = tmp_path / "audio"
        output_dir = tmp_path / "output"

        monkeypatch.setattr("src.rendering.CACHE_MANIM_DIR", manim_dir)
        monkeypatch.setattr("src.rendering.CACHE_AUDIO_DIR", audio_dir)

        rendered_mp4 = manim_dir / "videos" / "hash" / "480p15" / "Scene.mp4"
        rendered_mp4.parent.mkdir(parents=True, exist_ok=True)
        rendered_mp4.write_bytes(b"fake-mp4")

        merged_audio = audio_dir / "merged_audio.wav"
        merged_audio.parent.mkdir(parents=True, exist_ok=True)
        merged_audio.write_bytes(b"fake-wav")

        script = tmp_path / "scene.py"
        script.write_text(
            "from manim import *\nclass SceneA(Scene):\n    def construct(self): pass\n"
        )

        return script, output_dir

    def test_write_videofile_does_not_pass_verbose_fps(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        from src.rendering import render_and_merge

        script, output_dir = self._setup_render_fixture(tmp_path, monkeypatch)

        video_clip = MagicMock()
        audio_clip = MagicMock()
        final_clip = MagicMock()
        video_clip.with_audio.return_value = final_clip

        with (
            patch("src.rendering.run_command", return_value=MagicMock(returncode=0)),
            patch("src.rendering.VideoFileClip", return_value=video_clip),
            patch("src.rendering.AudioFileClip", return_value=audio_clip),
        ):
            render_and_merge(script, output_dir, "topic")

        assert final_clip.write_videofile.call_count == 1
        kwargs = final_clip.write_videofile.call_args.kwargs
        assert "verbose_fps" not in kwargs
        assert kwargs["logger"] is None
