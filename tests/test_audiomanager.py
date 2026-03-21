"""
Tests for audiomanager.py and the TTS engine abstraction.

Run with:
    uv run pytest tests/test_audiomanager.py -v

Integration tests (marked with @pytest.mark.integration) download the model
on first run -- skip with:
    uv run pytest tests/test_audiomanager.py -v -m "not integration"
"""

import os
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import MonkeyPatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000  # Qwen3-TTS default sampling rate
ONE_SECOND = np.zeros(SAMPLE_RATE, dtype=np.float32)

# PLR2004 constants
WAV_CHANNELS = 1
WAV_SAMPLE_WIDTH = 2
INT16_MAX = 32767
INT16_MIN = -32767
PIPER_DEFAULT_SR = 22050
PIPER_ALT_SR = 16000
EXPECTED_SAY_COUNT = 2  # number of say() calls in test_say_increments_index


def _make_mock_engine(audio: np.ndarray | None = None, sr: int = SAMPLE_RATE) -> MagicMock:
    """Return a mock TTSEngine whose synthesize() returns (audio, sr)."""
    if audio is None:
        audio = ONE_SECOND
    mock = MagicMock()
    mock.synthesize.return_value = (audio, sr)
    return mock


# ---------------------------------------------------------------------------
# Unit tests - no model download, no patching needed
# ---------------------------------------------------------------------------


class TestCreateWav:
    def test_writes_wav_file(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        fake_audio = np.linspace(-0.5, 0.5, SAMPLE_RATE, dtype=np.float32)
        engine = _make_mock_engine(fake_audio)
        from src.audiomanager import create_wav

        create_wav("Hello world", 1, engine)
        wav_path = tmp_path / "audio_1.wav"
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == WAV_CHANNELS
            assert wf.getsampwidth() == WAV_SAMPLE_WIDTH
            assert wf.getframerate() == SAMPLE_RATE
            frames = wf.readframes(wf.getnframes())
        assert len(np.frombuffer(frames, dtype=np.int16)) == SAMPLE_RATE

    def test_returns_duration_plus_buffer(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        engine = _make_mock_engine(audio)
        from src.audiomanager import create_wav

        duration = create_wav("Two seconds", 1, engine)
        assert duration == pytest.approx(3.0, abs=0.01)

    def test_audio_clipped_to_int16_range(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        engine = _make_mock_engine(audio, sr=3)
        from src.audiomanager import create_wav

        create_wav("Clipping", 1, engine)
        with wave.open(str(tmp_path / "audio_1.wav"), "rb") as wf:
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        assert samples[0] == INT16_MAX
        assert samples[1] == INT16_MIN

    def test_incremental_file_naming(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        engine = _make_mock_engine()
        from src.audiomanager import create_wav

        create_wav("First", 1, engine)
        create_wav("Second", 2, engine)
        create_wav("Third", 3, engine)
        assert (tmp_path / "audio_1.wav").exists()
        assert (tmp_path / "audio_2.wav").exists()
        assert (tmp_path / "audio_3.wav").exists()


class TestQwenTTSEngine:
    def test_passes_speaker_and_language_to_model(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        mock_raw_model = MagicMock()
        mock_raw_model.model.tts_model_type = "custom_voice"
        mock_raw_model.generate_custom_voice.return_value = ([ONE_SECOND], SAMPLE_RATE)
        from src.tts.qwen import QwenTTSEngine

        engine = QwenTTSEngine(speaker="Aiden", language="English")
        with patch.object(engine, "_load_model", return_value=mock_raw_model):
            engine.synthesize("Speaker test")
        mock_raw_model.generate_custom_voice.assert_called_once_with(
            text="Speaker test",
            language="English",
            speaker="Aiden",
            temperature=0.7,
        )

    def test_returns_cpu_when_no_accelerator(self, monkeypatch: MonkeyPatch) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        from src.tts.qwen import QwenTTSEngine

        assert QwenTTSEngine._device_map() == "cpu"

    def test_returns_cuda_when_available(self, monkeypatch: MonkeyPatch) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        from src.tts.qwen import QwenTTSEngine

        assert QwenTTSEngine._device_map() == "cuda:0"

    def test_from_env_reads_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("QWEN_TTS_MODEL", "Qwen/some-model")
        monkeypatch.setenv("QWEN_TTS_SPEAKER", "Vivian")
        monkeypatch.setenv("QWEN_TTS_LANGUAGE", "Chinese")
        from src.tts.qwen import QwenTTSEngine

        engine = QwenTTSEngine.from_env()
        assert engine.model_id == "Qwen/some-model"
        assert engine.speaker == "Vivian"
        assert engine.language == "Chinese"

    def test_dtype_is_bfloat16_on_mps(self, monkeypatch: MonkeyPatch) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        else:
            pytest.skip("torch.backends.mps not present on this build")
        from src.tts.qwen import QwenTTSEngine

        assert QwenTTSEngine._dtype() == torch.bfloat16

    def test_dtype_is_float32_on_cpu(self, monkeypatch: MonkeyPatch) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        from src.tts.qwen import QwenTTSEngine

        assert QwenTTSEngine._dtype() == torch.float32

    def test_synthesize_converts_torch_tensor_to_numpy(self) -> None:
        """Model may return a torch.Tensor; synthesize() must convert it."""
        import torch

        tensor_audio = torch.zeros(SAMPLE_RATE, dtype=torch.float32)
        mock_raw_model = MagicMock()
        mock_raw_model.model.tts_model_type = "custom_voice"
        mock_raw_model.generate_custom_voice.return_value = (
            [tensor_audio],
            SAMPLE_RATE,
        )
        from src.tts.qwen import QwenTTSEngine

        engine = QwenTTSEngine()
        with patch.object(engine, "_load_model", return_value=mock_raw_model):
            audio, sr = engine.synthesize("Tensor test")

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == SAMPLE_RATE

    def test_synthesize_collapses_2d_to_mono(self) -> None:
        """If the model returns a 2-D array, synthesize() takes the first channel."""
        stereo = np.ones((2, SAMPLE_RATE), dtype=np.float32) * 0.5
        mock_raw_model = MagicMock()
        mock_raw_model.model.tts_model_type = "custom_voice"
        mock_raw_model.generate_custom_voice.return_value = ([stereo], SAMPLE_RATE)
        from src.tts.qwen import QwenTTSEngine

        engine = QwenTTSEngine()
        with patch.object(engine, "_load_model", return_value=mock_raw_model):
            audio, _ = engine.synthesize("Stereo test")

        assert audio.ndim == 1
        assert len(audio) == SAMPLE_RATE


class TestPiperTTSEngine:
    def _make_mock_voice(self, sr: int = PIPER_DEFAULT_SR, n_frames: int = PIPER_DEFAULT_SR) -> MagicMock:
        """Return a mock PiperVoice that writes known PCM to a wave file handle."""

        def _fake_synthesize(text: str, wav_file: wave.Wave_write) -> None:
            frames = np.zeros(n_frames, dtype=np.int16).tobytes()
            wav_file.setnchannels(WAV_CHANNELS)
            wav_file.setsampwidth(WAV_SAMPLE_WIDTH)
            wav_file.setframerate(sr)
            wav_file.writeframes(frames)

        mock_voice = MagicMock()
        mock_voice.synthesize.side_effect = _fake_synthesize
        return mock_voice

    def test_synthesize_returns_float32_mono_array(self) -> None:
        from src.tts.piper import PiperTTSEngine

        engine = PiperTTSEngine(model_path="/fake/model.onnx")
        with patch.object(engine, "_load_voice", return_value=self._make_mock_voice()):
            audio, sr = engine.synthesize("Hello")

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert sr == PIPER_DEFAULT_SR
        assert len(audio) == PIPER_DEFAULT_SR

    def test_synthesize_normalizes_to_float(self) -> None:
        """int16 max (32767) must map to ~1.0 in float32."""

        def _peak_synthesize(text: str, wav_file: wave.Wave_write) -> None:
            frames = np.array([INT16_MAX, -32768, 0], dtype=np.int16).tobytes()
            wav_file.setnchannels(WAV_CHANNELS)
            wav_file.setsampwidth(WAV_SAMPLE_WIDTH)
            wav_file.setframerate(PIPER_DEFAULT_SR)
            wav_file.writeframes(frames)

        mock_voice = MagicMock()
        mock_voice.synthesize.side_effect = _peak_synthesize
        from src.tts.piper import PiperTTSEngine

        engine = PiperTTSEngine(model_path="/fake/model.onnx")
        with patch.object(engine, "_load_voice", return_value=mock_voice):
            audio, _ = engine.synthesize("Normalize test")

        assert audio[0] == pytest.approx(1.0, abs=1e-4)
        assert audio[1] == pytest.approx(-32768 / 32767.0, abs=1e-4)
        assert audio[2] == pytest.approx(0.0, abs=1e-6)

    def test_sample_rate_passed_through(self) -> None:
        from src.tts.piper import PiperTTSEngine

        engine = PiperTTSEngine(model_path="/fake/model.onnx")
        with patch.object(engine, "_load_voice", return_value=self._make_mock_voice(sr=PIPER_ALT_SR)):
            _, sr = engine.synthesize("Rate test")

        assert sr == PIPER_ALT_SR

    def test_from_env_reads_piper_model_env_var(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("PIPER_MODEL", "/custom/model.onnx")
        from src.tts.piper import PiperTTSEngine

        engine = PiperTTSEngine.from_env()
        assert engine.model_path == "/custom/model.onnx"

    def test_voice_loaded_lazily(self) -> None:
        """_voice is None at construction; _load_voice() is never called early."""
        from src.tts.piper import PiperTTSEngine

        engine = PiperTTSEngine(model_path="/fake/model.onnx")
        assert engine._voice is None


class TestAudioManager:
    def _make_scene(self, time: float = 0.0) -> MagicMock:
        scene = MagicMock()
        scene.renderer.time = time
        return scene

    def test_say_increments_index(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        engine = _make_mock_engine()
        from src.audiomanager import AudioManager

        mgr = AudioManager(self._make_scene(), engine=engine)
        mgr.say("Hello")
        mgr.say("World")
        assert mgr.i == EXPECTED_SAY_COUNT

    def test_say_records_time_and_duration(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        scene = self._make_scene(time=1.5)
        engine = _make_mock_engine()
        from src.audiomanager import AudioManager

        mgr = AudioManager(scene, engine=engine)
        mgr.say("Test")
        assert mgr.times[0] == pytest.approx(1.5)
        assert len(mgr.audio_durations) == 1

    def test_done_say_waits_for_remaining_audio(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        scene = self._make_scene(time=0.0)
        audio = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
        engine = _make_mock_engine(audio)
        from src.audiomanager import AudioManager

        mgr = AudioManager(scene, engine=engine)
        mgr.say("This is a sufficiently long sentence")  # 6 words → 6s limit > 3s audio
        scene.renderer.time = 1.0
        mgr.done_say()
        scene.wait.assert_called_once()
        assert scene.wait.call_args[0][0] == pytest.approx(3.0, abs=0.1)

    def test_done_say_skips_wait_when_already_finished(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        scene = self._make_scene(time=0.0)
        engine = _make_mock_engine()
        from src.audiomanager import AudioManager

        mgr = AudioManager(scene, engine=engine)
        mgr.say("Short")
        scene.renderer.time = 999.0
        mgr.done_say()
        scene.wait.assert_not_called()

    def test_done_say_before_say_is_noop(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        engine = _make_mock_engine()
        from src.audiomanager import AudioManager

        scene = self._make_scene()
        mgr = AudioManager(scene, engine=engine)
        mgr.done_say()
        scene.wait.assert_not_called()

    def test_merge_audio_creates_merged_file(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        scene = self._make_scene(time=0.0)
        engine = _make_mock_engine()
        from src.audiomanager import AudioManager

        mgr = AudioManager(scene, engine=engine)
        mgr.say("One")
        scene.renderer.time = 2.0
        mgr.say("Two")
        mgr.merge_audio()
        merged = tmp_path / "merged_audio.wav"
        assert merged.exists()
        with wave.open(str(merged), "rb") as wf:
            assert wf.getnchannels() == WAV_CHANNELS
            assert wf.getsampwidth() == WAV_SAMPLE_WIDTH
            assert wf.getframerate() == SAMPLE_RATE

    def test_merge_audio_no_files_is_noop(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
        engine = _make_mock_engine()
        from src.audiomanager import AudioManager

        mgr = AudioManager(self._make_scene(), engine=engine)
        mgr.merge_audio()
        assert not (tmp_path / "merged_audio.wav").exists()


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_integration_piper_create_wav(tmp_path: Path) -> None:
    """
    End-to-end: loads the local Piper model and synthesises speech.

    Requires models/en_US-ryan-high.onnx to be present.
    Run with:
        uv run pytest tests/test_audiomanager.py -v -m integration
    """
    from src.paths import PIPER_DEFAULT_MODEL

    if not PIPER_DEFAULT_MODEL.exists():
        pytest.skip(f"Piper model not found at {PIPER_DEFAULT_MODEL}")

    os.environ["AUDIO_OUTPUT_DIR"] = str(tmp_path)
    from src.audiomanager import create_wav
    from src.tts.piper import PiperTTSEngine

    engine = PiperTTSEngine.from_env()
    duration = create_wav("This is an integration test for Piper TTS.", 1, engine)
    wav_path = tmp_path / "audio_1.wav"
    assert wav_path.exists()
    assert duration > 1.0
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == WAV_CHANNELS
        assert wf.getsampwidth() == WAV_SAMPLE_WIDTH
        assert wf.getnframes() > 0


@pytest.mark.integration
def test_integration_create_wav(tmp_path: Path) -> None:
    """
    End-to-end: loads Qwen3-TTS and synthesises speech.

    Run with:
        uv run pytest tests/test_audiomanager.py -v -m integration

    Requires qwen-tts installed and sufficient RAM/VRAM.
    """
    os.environ["AUDIO_OUTPUT_DIR"] = str(tmp_path)
    from src.audiomanager import create_wav
    from src.tts.qwen import QwenTTSEngine

    engine = QwenTTSEngine.from_env()
    duration = create_wav("This is an integration test for Qwen3-TTS.", 1, engine)
    wav_path = tmp_path / "audio_1.wav"
    assert wav_path.exists()
    assert duration > 1.0
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == WAV_CHANNELS
        assert wf.getsampwidth() == WAV_SAMPLE_WIDTH
        assert wf.getnframes() > 0
