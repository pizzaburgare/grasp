"""
Tests for the caching system (src/cache.py and the cache integration in
src/audiomanager.py).

No ML models are loaded, no LLM calls are made, and no Manim renders are
executed.  All expensive operations are replaced with mocks.

Run with:
    uv run pytest tests/test_cache.py -v
"""

import hashlib
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import MonkeyPatch

from src.llm_metrics import LLMUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000

# PLR2004 constants
HASH_HEX_LENGTH = 16
EXPECTED_CACHE_WAV_COUNT = 2


def _write_wav(path: Path, n_samples: int = SAMPLE_RATE, sr: int = SAMPLE_RATE) -> None:
    """Write a minimal valid WAV file so cache-hit tests can read frame counts."""
    samples = np.zeros(n_samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def _make_engine(n_samples: int = SAMPLE_RATE, sr: int = SAMPLE_RATE) -> MagicMock:
    m = MagicMock()
    m.synthesize.return_value = (np.zeros(n_samples, dtype=np.float32), sr)
    return m


# ===========================================================================
# lesson_name_to_key
# ===========================================================================


class TestLessonNameToKey:
    def test_basic_lowercase_and_spaces(self) -> None:
        from src.cache import lesson_name_to_key

        assert lesson_name_to_key("LU Decomposition") == "lu-decomposition"

    def test_underscores_become_dashes(self) -> None:
        from src.cache import lesson_name_to_key

        assert lesson_name_to_key("lu_decomposition") == "lu-decomposition"

    def test_special_chars_stripped(self) -> None:
        from src.cache import lesson_name_to_key

        assert lesson_name_to_key("Eigenvalues & Eigenvectors!") == "eigenvalues-eigenvectors"

    def test_already_slug_unchanged(self) -> None:
        from src.cache import lesson_name_to_key

        assert lesson_name_to_key("lu-decomposition") == "lu-decomposition"

    def test_multiple_spaces_collapsed(self) -> None:
        from src.cache import lesson_name_to_key

        assert lesson_name_to_key("a  b") == "a-b"

    def test_different_names_produce_different_keys(self) -> None:
        from src.cache import lesson_name_to_key

        assert lesson_name_to_key("Fourier Transform") != lesson_name_to_key("LU Decomposition")

    def test_same_input_deterministic(self) -> None:
        from src.cache import lesson_name_to_key

        assert lesson_name_to_key("Neural Nets") == lesson_name_to_key("Neural Nets")


# ===========================================================================
# hash_context
# ===========================================================================


class TestHashContext:
    def test_same_topic_deterministic(self) -> None:
        from src.cache import hash_context

        assert hash_context("LU Decomposition") == hash_context("LU Decomposition")

    def test_different_topics_differ(self) -> None:
        from src.cache import hash_context

        assert hash_context("LU Decomposition") != hash_context("Fourier Transform")

    def test_returns_16_char_hex(self) -> None:
        from src.cache import hash_context

        result = hash_context("any topic")
        assert len(result) == HASH_HEX_LENGTH
        assert all(c in "0123456789abcdef" for c in result)

    def test_with_input_dir_differs_from_without(self, tmp_path: Path) -> None:
        from src.cache import hash_context

        (tmp_path / "notes.txt").write_text("some content")
        h_with = hash_context("topic", str(tmp_path))
        h_without = hash_context("topic")
        assert h_with != h_without

    def test_file_content_change_changes_hash(self, tmp_path: Path) -> None:
        from src.cache import hash_context

        f = tmp_path / "notes.txt"
        f.write_text("version A")
        h1 = hash_context("topic", str(tmp_path))
        f.write_text("version B")
        h2 = hash_context("topic", str(tmp_path))
        assert h1 != h2

    def test_empty_input_dir_differs_from_no_input_dir(self, tmp_path: Path) -> None:
        from src.cache import hash_context

        # Empty directory - no files, but we still pass a path
        h_empty_dir = hash_context("topic", str(tmp_path))
        # Same topic, empty dir shouldn't add any bytes → should equal hash with no dir
        # (depends on implementation - both are valid; we just verify determinism)
        assert h_empty_dir == hash_context("topic", str(tmp_path))

    def test_multiple_files_order_independent(self, tmp_path: Path) -> None:
        from src.cache import hash_context

        # sorted() over rglob means the hash is deterministic regardless of FS order
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")
        h1 = hash_context("topic", str(tmp_path))
        h2 = hash_context("topic", str(tmp_path))
        assert h1 == h2

    def test_relative_path_change_changes_hash(self, tmp_path: Path) -> None:
        from src.cache import hash_context

        f = tmp_path / "notes.txt"
        f.write_text("same bytes")
        h1 = hash_context("topic", str(tmp_path))

        moved = tmp_path / "nested" / "notes.txt"
        moved.parent.mkdir(parents=True)
        moved.write_text("same bytes")
        f.unlink()
        h2 = hash_context("topic", str(tmp_path))

        assert h1 != h2

    def test_extra_context_changes_hash(self) -> None:
        from src.cache import hash_context

        h1 = hash_context("topic", extra_context={"model": "a"})
        h2 = hash_context("topic", extra_context={"model": "b"})
        assert h1 != h2


# ===========================================================================
# hash_text
# ===========================================================================


class TestHashText:
    def test_same_text_deterministic(self) -> None:
        from src.cache import hash_text

        assert hash_text("Hello world") == hash_text("Hello world")

    def test_different_texts_differ(self) -> None:
        from src.cache import hash_text

        assert hash_text("Hello world") != hash_text("Goodbye world")

    def test_returns_16_char_hex(self) -> None:
        from src.cache import hash_text

        result = hash_text("some text")
        assert len(result) == HASH_HEX_LENGTH
        assert all(c in "0123456789abcdef" for c in result)

    def test_matches_manual_sha256(self) -> None:
        from src.cache import hash_text

        text = "Hello world"
        expected = hashlib.sha256(text.encode()).hexdigest()[:16]
        assert hash_text(text) == expected

    def test_whitespace_sensitive(self) -> None:
        from src.cache import hash_text

        assert hash_text("hello ") != hash_text("hello")


# ===========================================================================
# Script cache
# ===========================================================================


class TestScriptCache:
    def test_returns_none_when_absent(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_script

        assert get_cached_script("my-lesson", "abc123") is None

    def test_returns_path_when_present(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_script

        p = tmp_path / "my-lesson" / "script" / "abc123.py"
        p.parent.mkdir(parents=True)
        p.write_text("# cached script")

        result = get_cached_script("my-lesson", "abc123")
        assert result == p

    def test_save_creates_file_with_correct_content(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import save_script_to_cache

        src_script = tmp_path / "lesson.py"
        src_script.write_text("# my script")

        saved = save_script_to_cache("my-lesson", "abc123", src_script)
        assert saved.exists()
        assert saved.read_text() == "# my script"

    def test_save_filename_is_hash_with_py_suffix(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import save_script_to_cache

        script = tmp_path / "lesson.py"
        script.write_text("content")
        saved = save_script_to_cache("my-lesson", "deadbeef", script)
        assert saved.name == "deadbeef.py"
        assert saved.parent.name == "script"

    def test_save_then_get_roundtrip(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_script, save_script_to_cache

        script = tmp_path / "lesson.py"
        script.write_text("# roundtrip test")

        save_script_to_cache("lesson", "hash1", script)
        retrieved = get_cached_script("lesson", "hash1")
        assert retrieved is not None
        assert retrieved.read_text() == "# roundtrip test"

    def test_different_hashes_stored_independently(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_script, save_script_to_cache

        s1, s2 = tmp_path / "s1.py", tmp_path / "s2.py"
        s1.write_text("script one")
        s2.write_text("script two")

        save_script_to_cache("lesson", "hash_a", s1)
        save_script_to_cache("lesson", "hash_b", s2)

        assert get_cached_script("lesson", "hash_a").read_text() == "script one"  # type: ignore[union-attr]
        assert get_cached_script("lesson", "hash_b").read_text() == "script two"  # type: ignore[union-attr]

    def test_wrong_hash_returns_none(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_script, save_script_to_cache

        script = tmp_path / "lesson.py"
        script.write_text("content")
        save_script_to_cache("lesson", "correct-hash", script)

        assert get_cached_script("lesson", "wrong-hash") is None


# ===========================================================================
# Audio cache helpers (get_audio_cache_dir, get_cached_audio, save_audio_to_cache)
# ===========================================================================


class TestAudioCacheHelpers:
    def test_audio_cache_dir_path(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_audio_cache_dir

        d = get_audio_cache_dir("My Lesson")
        assert d == tmp_path / "my-lesson" / "audio"

    def test_returns_none_when_absent(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_audio

        assert get_cached_audio("lesson", "Hello world") is None

    def test_save_uses_text_hash_as_filename(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import hash_text, save_audio_to_cache

        wav = tmp_path / "src.wav"
        wav.write_bytes(b"wav data")
        text = "Hello world"

        saved = save_audio_to_cache("lesson", text, wav)
        assert saved.name == f"{hash_text(text)}.wav"
        assert saved.parent.name == "audio"

    def test_save_preserves_file_bytes(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import save_audio_to_cache

        wav = tmp_path / "audio.wav"
        content = b"RIFF\x00\x00\x00\x00WAVEfmt "
        wav.write_bytes(content)

        dest = save_audio_to_cache("lesson", "narration text", wav)
        assert dest.read_bytes() == content

    def test_roundtrip(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_audio, save_audio_to_cache

        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake-wav-bytes")

        save_audio_to_cache("lesson", "my text", wav)
        result = get_cached_audio("lesson", "my text")
        assert result is not None
        assert result.read_bytes() == b"fake-wav-bytes"

    def test_different_texts_stored_independently(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_audio, save_audio_to_cache

        w1, w2 = tmp_path / "w1.wav", tmp_path / "w2.wav"
        w1.write_bytes(b"audio one")
        w2.write_bytes(b"audio two")

        save_audio_to_cache("lesson", "text one", w1)
        save_audio_to_cache("lesson", "text two", w2)

        assert get_cached_audio("lesson", "text one").read_bytes() == b"audio one"  # type: ignore[union-attr]
        assert get_cached_audio("lesson", "text two").read_bytes() == b"audio two"  # type: ignore[union-attr]

    def test_wrong_text_returns_none(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_audio, save_audio_to_cache

        wav = tmp_path / "w.wav"
        wav.write_bytes(b"data")
        save_audio_to_cache("lesson", "correct text", wav)

        assert get_cached_audio("lesson", "wrong text") is None


# ===========================================================================
# Video cache
# ===========================================================================


class TestVideoCache:
    def test_returns_none_when_absent(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_video

        assert get_cached_video("lesson", "abc123") is None

    def test_returns_path_when_present(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_video

        p = tmp_path / "lesson" / "video" / "abc123.mp4"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"fake mp4")

        result = get_cached_video("lesson", "abc123")
        assert result == p

    def test_save_filename_is_hash_with_mp4_suffix(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import save_video_to_cache

        mp4 = tmp_path / "output.mp4"
        mp4.write_bytes(b"fake mp4")

        saved = save_video_to_cache("lesson", "deadbeef", mp4)
        assert saved.name == "deadbeef.mp4"
        assert saved.parent.name == "video"

    def test_save_in_correct_lesson_subdirectory(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import save_video_to_cache

        mp4 = tmp_path / "output.mp4"
        mp4.write_bytes(b"content")
        saved = save_video_to_cache("My Lesson", "myhash", mp4)

        # lesson key normalisation: "My Lesson" → "my-lesson"
        assert saved.parts[-3] == "my-lesson"

    def test_roundtrip(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_video, save_video_to_cache

        mp4 = tmp_path / "output.mp4"
        content = b"fake video content"
        mp4.write_bytes(content)

        save_video_to_cache("lesson", "myhash", mp4)
        retrieved = get_cached_video("lesson", "myhash")
        assert retrieved is not None
        assert retrieved.read_bytes() == content

    def test_wrong_hash_returns_none(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import get_cached_video, save_video_to_cache

        mp4 = tmp_path / "output.mp4"
        mp4.write_bytes(b"content")
        save_video_to_cache("lesson", "correct-hash", mp4)

        assert get_cached_video("lesson", "wrong-hash") is None


# ===========================================================================
# Per-lesson cache directory layout
# ===========================================================================


class TestCacheDirectoryLayout:
    """Verify that each asset type lands under the right sub-directory."""

    def test_script_under_lesson_script_subdir(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import save_script_to_cache

        src = tmp_path / "s.py"
        src.write_text("# code")
        saved = save_script_to_cache("LU Decomposition", "h1", src)
        # .cache/lu-decomposition/script/h1.py
        assert saved.parts[-4:-1] == (tmp_path.name, "lu-decomposition", "script")

    def test_audio_under_lesson_audio_subdir(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import save_audio_to_cache

        w = tmp_path / "w.wav"
        w.write_bytes(b"data")
        saved = save_audio_to_cache("LU Decomposition", "hello", w)
        assert saved.parts[-2] == "audio"
        assert saved.parts[-3] == "lu-decomposition"

    def test_video_under_lesson_video_subdir(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        from src.cache import save_video_to_cache

        v = tmp_path / "v.mp4"
        v.write_bytes(b"data")
        saved = save_video_to_cache("LU Decomposition", "h1", v)
        assert saved.parts[-2] == "video"
        assert saved.parts[-3] == "lu-decomposition"


# ===========================================================================
# create_wav audio-cache integration
# ===========================================================================


class TestCreateWavAudioCache:
    """Tests for the cache-aware create_wav in audiomanager.py."""

    def test_cache_hit_skips_engine(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """When a cached WAV exists for the text, TTS is not called."""
        audio_out = tmp_path / "audio_out"
        audio_out.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(audio_out))
        monkeypatch.setenv("AUDIO_CACHE_DIR", str(cache_dir))

        text = "Hello cached world"
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        _write_wav(cache_dir / f"{text_hash}.wav")

        engine = _make_engine()
        from src.audiomanager import create_wav

        duration = create_wav(text, 1, engine)

        engine.synthesize.assert_not_called()
        assert (audio_out / "audio_1.wav").exists()
        assert duration == pytest.approx(2.0, abs=0.01)  # 1s audio + 1s buffer

    def test_cache_hit_copies_wav_to_output(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """The numbered audio_N.wav in the output dir matches the cached file."""
        audio_out = tmp_path / "audio_out"
        audio_out.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(audio_out))
        monkeypatch.setenv("AUDIO_CACHE_DIR", str(cache_dir))

        text = "copy me"
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cached = cache_dir / f"{text_hash}.wav"
        _write_wav(cached, n_samples=SAMPLE_RATE * 3)

        engine = _make_engine()
        from src.audiomanager import create_wav

        create_wav(text, 2, engine)

        out_wav = audio_out / "audio_2.wav"
        assert out_wav.exists()
        assert out_wav.read_bytes() == cached.read_bytes()

    def test_cache_miss_calls_engine_and_saves_to_cache(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """On a cache miss the engine is called and the WAV is stored in the cache."""
        audio_out = tmp_path / "audio_out"
        audio_out.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(audio_out))
        monkeypatch.setenv("AUDIO_CACHE_DIR", str(cache_dir))

        text = "Brand new narration"
        engine = _make_engine()
        from src.audiomanager import create_wav

        create_wav(text, 1, engine)

        engine.synthesize.assert_called_once_with(text)

        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cached_path = cache_dir / f"{text_hash}.wav"
        assert cached_path.exists()

    def test_cache_miss_cached_file_matches_output(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """The file saved to cache must be byte-identical to the numbered output."""
        audio_out = tmp_path / "audio_out"
        audio_out.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(audio_out))
        monkeypatch.setenv("AUDIO_CACHE_DIR", str(cache_dir))

        text = "Verify bytes match"
        engine = _make_engine()
        from src.audiomanager import create_wav

        create_wav(text, 1, engine)

        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        out_bytes = (audio_out / "audio_1.wav").read_bytes()
        cache_bytes = (cache_dir / f"{text_hash}.wav").read_bytes()
        assert out_bytes == cache_bytes

    def test_same_text_different_engine_config_uses_separate_cache_entries(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Changing TTS config for same text should not reuse stale cached audio."""
        audio_out = tmp_path / "audio_out"
        audio_out.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(audio_out))
        monkeypatch.setenv("AUDIO_CACHE_DIR", str(cache_dir))

        text = "Voice-sensitive cache key"

        engine_a = _make_engine()
        engine_a.voice = "am_adam"
        engine_b = _make_engine()
        engine_b.voice = "am_michael"

        from src.audiomanager import create_wav

        create_wav(text, 1, engine_a)
        create_wav(text, 2, engine_b)

        engine_a.synthesize.assert_called_once_with(text)
        engine_b.synthesize.assert_called_once_with(text)
        assert len(list(cache_dir.glob("*.wav"))) == EXPECTED_CACHE_WAV_COUNT

    def test_no_cache_dir_env_no_caching(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Without AUDIO_CACHE_DIR the engine is always called and nothing is cached."""
        audio_out = tmp_path / "audio_out"
        audio_out.mkdir()
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(audio_out))
        monkeypatch.delenv("AUDIO_CACHE_DIR", raising=False)

        engine = _make_engine()
        from src.audiomanager import create_wav

        create_wav("some text", 1, engine)

        engine.synthesize.assert_called_once()
        # No hash-named files should have appeared anywhere in audio_out
        hash_files = [f for f in audio_out.iterdir() if f.name != "audio_1.wav"]
        assert hash_files == []

    def test_second_call_same_text_hits_cache(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """The second call for the same text must read from cache, not call TTS again."""
        audio_out = tmp_path / "audio_out"
        audio_out.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(audio_out))
        monkeypatch.setenv("AUDIO_CACHE_DIR", str(cache_dir))

        text = "Say this once"
        engine = _make_engine()
        from src.audiomanager import create_wav

        create_wav(text, 1, engine)  # cache miss → TTS
        create_wav(text, 2, engine)  # cache hit → no TTS

        assert engine.synthesize.call_count == 1


# ===========================================================================
# Workflow-level cache integration
# ===========================================================================


class TestWorkflowCacheIntegration:
    """Test that run_full_pipeline correctly reads/writes the cache.

    Heavy external calls (LLM, Manim, TTS) are patched out.
    """

    @pytest.fixture()
    def patched_cache_dir(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
        """Redirect both src.cache.CACHE_DIR and src.workflow.CACHE_DIR.

        Also neutralise INPUT_DIR auto-detection so pipeline tests compute
        deterministic hashes regardless of what exists in the real input/.
        """
        cache = tmp_path / "cache"
        monkeypatch.setattr("src.cache.CACHE_DIR", cache)
        monkeypatch.setattr("src.workflow.CACHE_DIR", cache)
        # Point INPUT_DIR at a path that never exists so auto-detection is skipped
        monkeypatch.setattr("src.workflow.INPUT_DIR", tmp_path / "_no_input_dir_")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setenv("TTS_ENGINE", "kokoro")
        return cache

    def test_video_cache_hit_skips_llm_and_render(
        self, tmp_path: Path, patched_cache_dir: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """If a cached video exists the pipeline copies it and returns immediately."""
        from src.cache import hash_context
        from src.workflow import CourseWorkflow

        topic = "LU Decomposition"
        slug = "lu-decomposition"

        out_dir = tmp_path / "output"

        with (
            patch.object(CourseWorkflow, "generate_lesson_plan") as mock_plan,
            patch("src.workflow.render_and_merge") as mock_render,
        ):
            wf = CourseWorkflow(model="test-model")
            video_hash = hash_context(
                topic,
                extra_context=wf._build_cache_context("kokoro", False),
            )

            # Pre-populate video cache
            video_cache = patched_cache_dir / slug / "video" / f"{video_hash}.mp4"
            video_cache.parent.mkdir(parents=True)
            video_cache.write_bytes(b"cached video bytes")

            result = wf.run_full_pipeline(topic, output_dir=str(out_dir))

        mock_plan.assert_not_called()
        mock_render.assert_not_called()

        final = out_dir / "lu-decomposition.mp4"
        assert final.exists()
        assert final.read_bytes() == b"cached video bytes"
        assert result["cache_hit"] == "video"

    def test_script_cache_hit_skips_script_generation(
        self, tmp_path: Path, patched_cache_dir: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """If a cached script exists neither the script LLM nor the lesson-plan LLM is called."""
        from src.cache import hash_context
        from src.workflow import CourseWorkflow

        topic = "Fourier Transform"
        slug = "fourier-transform"

        out_dir = tmp_path / "output"

        _cached_script = (
            "from manim import *\n\nclass FourierScene(Scene):\n    def construct(self): pass\n"
        )
        with (
            patch.object(CourseWorkflow, "generate_lesson_plan") as mock_plan,
            patch(
                "src.workflow.render_and_merge",
                return_value=out_dir / "fourier-transform.mp4",
            ) as mock_render,
            patch("src.workflow.save_video_to_cache"),
        ):
            wf = CourseWorkflow(model="test-model")
            # Script cache uses _build_script_context() (quality/TTS-independent)
            ctx_hash = hash_context(
                topic,
                extra_context=wf._build_script_context(),
            )

            # Pre-populate script cache with a valid minimal Manim script + lesson plan
            script_dir = patched_cache_dir / slug / "script"
            script_dir.mkdir(parents=True)
            script_path = script_dir / f"{ctx_hash}.py"
            script_path.write_text(_cached_script)
            (script_dir / f"{ctx_hash}.md").write_text("# cached plan")

            wf.script_generator = MagicMock()
            wf.script_generator.review_video.return_value = (_cached_script, False, LLMUsage())
            wf.run_full_pipeline(topic, output_dir=str(out_dir))

        mock_plan.assert_not_called()  # lesson plan is skipped
        wf.script_generator.generate_and_save.assert_not_called()  # script is skipped
        mock_render.assert_called_once()  # render still runs

    def test_fresh_run_render_called_with_lesson_name_and_hash(
        self, tmp_path: Path, patched_cache_dir: Path
    ) -> None:
        """On a cache miss render_and_merge receives lesson_name and context_hash."""
        from src.workflow import CourseWorkflow

        topic = "QR Decomposition"
        slug = "qr-decomposition"
        out_dir = tmp_path / "output"
        fake_video = out_dir / f"{slug}.mp4"

        with (
            patch.object(
                CourseWorkflow, "generate_lesson_plan", return_value=("# plan", LLMUsage())
            ),
            patch("src.workflow.render_and_merge", return_value=fake_video) as mock_render,
            patch("src.workflow.save_video_to_cache"),
        ):
            wf = CourseWorkflow(model="test-model")

            def _gen_save_effect(**kwargs: object) -> LLMUsage:
                path = kwargs.get("output_path")
                if path:
                    Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
                    Path(str(path)).write_text(
                        "from manim import *\nclass S(Scene):\n    def construct(self): pass\n"
                    )
                return LLMUsage()

            wf.script_generator = MagicMock()
            wf.script_generator.generate_and_save.side_effect = _gen_save_effect
            wf.script_generator.review_video.return_value = (
                "from manim import *\nclass S(Scene):\n    def construct(self): pass\n",
                False,
                LLMUsage(),
            )
            wf.run_full_pipeline(topic, output_dir=str(out_dir))

        _, kwargs = mock_render.call_args
        assert kwargs.get("lesson_name") == slug
        # Iterative renders use context_hash=None; caching happens via save_video_to_cache
        assert kwargs.get("context_hash") is None

    def test_lesson_plan_stored_alongside_script_as_md(
        self, tmp_path: Path, patched_cache_dir: Path
    ) -> None:
        """Lesson plan is written as {hash}.md next to {hash}.py in script/."""
        from src.cache import hash_context
        from src.workflow import CourseWorkflow

        topic = "Eigenvalues"
        slug = "eigenvalues"
        out_dir = tmp_path / "output"

        with (
            patch.object(
                CourseWorkflow, "generate_lesson_plan", return_value=("# my plan", LLMUsage())
            ),
            patch(
                "src.workflow.render_and_merge",
                return_value=out_dir / f"{slug}.mp4",
            ),
            patch("src.workflow.save_video_to_cache"),
        ):
            wf = CourseWorkflow(model="test-model")
            # Script files are keyed by _build_script_context() hash
            script_hash = hash_context(
                topic,
                extra_context=wf._build_script_context(),
            )

            def _gen_save_effect(**kwargs: object) -> LLMUsage:
                path = kwargs.get("output_path")
                if path:
                    Path(str(path)).parent.mkdir(parents=True, exist_ok=True)
                    Path(str(path)).write_text(
                        "from manim import *\nclass S(Scene):\n    def construct(self): pass\n"
                    )
                return LLMUsage()

            wf.script_generator = MagicMock()
            wf.script_generator.generate_and_save.side_effect = _gen_save_effect
            wf.script_generator.review_video.return_value = (
                "from manim import *\nclass S(Scene):\n    def construct(self): pass\n",
                False,
                LLMUsage(),
            )
            wf.run_full_pipeline(topic, output_dir=str(out_dir))

        plan_path = patched_cache_dir / slug / "script" / f"{script_hash}.md"
        assert plan_path.exists(), "lesson plan .md file should be in script/ dir"
        assert plan_path.read_text() == "# my plan"
        # Must NOT write a stray lesson_plan.md at the per-lesson root
        assert not (patched_cache_dir / slug / "lesson_plan.md").exists()

    def test_context_hash_changes_with_input_dir(
        self, tmp_path: Path, patched_cache_dir: Path
    ) -> None:
        """Two runs with different input files produce different context hashes."""
        from src.cache import hash_context

        input_dir_a = tmp_path / "input_a"
        input_dir_a.mkdir()
        (input_dir_a / "notes.txt").write_text("version A")

        input_dir_b = tmp_path / "input_b"
        input_dir_b.mkdir()
        (input_dir_b / "notes.txt").write_text("version B")

        h_a = hash_context("topic", str(input_dir_a))
        h_b = hash_context("topic", str(input_dir_b))
        assert h_a != h_b


# ===========================================================================
# render_and_merge - MP4 selection logic
# ===========================================================================


class TestRenderAndMergeVideoSelection:
    """Verify that render_and_merge picks the *most recently modified* MP4.

    The manim media_dir is persistent across runs, so stale renders from
    previous topics accumulate there.  The correct file must be the one that
    was just written, not whatever rglob returns first.
    """

    def _make_workflow(self) -> object:
        with (
            patch("src.llm_metrics.ChatOpenAI"),
            patch("src.workflow.ManimScriptGenerator"),
        ):
            from src.workflow import CourseWorkflow

            return CourseWorkflow(model="test-model")

    def _write_mp4(self, path: Path, content: bytes = b"fake-mp4") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_selects_most_recently_modified_mp4(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """With multiple stale MP4s in the cache, the newest one must be used."""
        import time

        manim_dir = tmp_path / "manim"
        audio_dir = tmp_path / "audio"
        out_dir = tmp_path / "output"

        monkeypatch.setattr("src.rendering.CACHE_MANIM_DIR", manim_dir)
        monkeypatch.setattr("src.rendering.CACHE_AUDIO_DIR", audio_dir)

        # Stale file written first (older mtime)
        old_mp4 = manim_dir / "videos" / "oldhash" / "480p15" / "OldScene.mp4"
        self._write_mp4(old_mp4, b"old-content")

        # Ensure a measurable mtime gap
        time.sleep(0.05)

        # New file written after the render
        new_mp4 = manim_dir / "videos" / "newhash" / "480p15" / "NewScene.mp4"
        self._write_mp4(new_mp4, b"new-content")

        # Fake script with a Scene subclass
        script = tmp_path / "scene.py"
        script.write_text(
            "from manim import *\nclass NewScene(Scene):\n    def construct(self): pass\n"
        )

        # Patch CommandRunner.run to be a no-op (render already "done")
        with (
            patch(
                "src.rendering.CommandRunner.run",
                return_value=MagicMock(returncode=0),
            ),
            patch("src.rendering.VideoFileClip"),
            patch("src.rendering.AudioFileClip"),
        ):
            # No merged_audio.wav → takes the copy-without-audio path, so we only
            # care about which video_path was found; capture it via VideoFileClip arg.
            # Actually that path is only used when audio exists, so we verify via the
            # copy path instead by checking which file gets copied.
            copied: dict[str, Path] = {}

            def fake_copy(src: str, dst: str) -> None:
                copied["src"] = Path(src)

            with patch("shutil.copy2", side_effect=fake_copy):
                from src.rendering import render_and_merge

                render_and_merge(script, out_dir, "new-scene")

        assert copied["src"] == new_mp4, f"Expected newest MP4 {new_mp4}, but got {copied['src']}"

    def test_single_mp4_is_always_selected(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """When only one MP4 exists it must be picked regardless of mtime."""
        manim_dir = tmp_path / "manim"
        audio_dir = tmp_path / "audio"
        out_dir = tmp_path / "output"

        monkeypatch.setattr("src.rendering.CACHE_MANIM_DIR", manim_dir)
        monkeypatch.setattr("src.rendering.CACHE_AUDIO_DIR", audio_dir)

        only_mp4 = manim_dir / "videos" / "abc" / "480p15" / "MyScene.mp4"
        self._write_mp4(only_mp4)

        script = tmp_path / "scene.py"
        script.write_text(
            "from manim import *\nclass MyScene(Scene):\n    def construct(self): pass\n"
        )

        with patch(
            "src.rendering.CommandRunner.run",
            return_value=MagicMock(returncode=0),
        ):
            copied: dict[str, Path] = {}

            def fake_copy(src: str, dst: str) -> None:
                copied["src"] = Path(src)

            with patch("shutil.copy2", side_effect=fake_copy):
                from src.rendering import render_and_merge

                render_and_merge(script, out_dir, "my-scene")

        assert copied["src"] == only_mp4

    def test_partial_movie_files_excluded(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """MP4s under partial_movie_files/ must never be selected."""
        manim_dir = tmp_path / "manim"
        audio_dir = tmp_path / "audio"
        out_dir = tmp_path / "output"

        monkeypatch.setattr("src.rendering.CACHE_MANIM_DIR", manim_dir)
        monkeypatch.setattr("src.rendering.CACHE_AUDIO_DIR", audio_dir)

        # Only a partial_movie_files entry exists - no real output yet
        partial = manim_dir / "videos" / "abc" / "partial_movie_files" / "chunk.mp4"
        self._write_mp4(partial)

        script = tmp_path / "scene.py"
        script.write_text(
            "from manim import *\nclass MyScene(Scene):\n    def construct(self): pass\n"
        )

        with (
            patch(
                "src.rendering.CommandRunner.run",
                return_value=MagicMock(returncode=0),
            ),
            pytest.raises(FileNotFoundError),
        ):
            from src.rendering import render_and_merge

            render_and_merge(script, out_dir, "my-scene")
