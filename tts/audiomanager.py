import wave

import numpy as np
from manim import Scene
from piper.voice import PiperVoice

MODEL_PATH = "tts/models/en_US-ryan-high.onnx"
CONFIG_PATH = "tts/models/en_US-ryan-high.onnx.json"
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


def create_wav(text_to_speak: str, i: int) -> float:
    voice = PiperVoice.load(MODEL_PATH, CONFIG_PATH, use_cuda=True)

    total_bytes_written = 0
    with wave.open(f"media/audio_{i}.wav", "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(voice.config.sample_rate)

        for chunk in voice.synthesize(text_to_speak):
            wav_file.writeframes(chunk.audio_int16_bytes)
            total_bytes_written += len(chunk.audio_int16_bytes)

    total_frames = total_bytes_written // (CHANNELS * SAMPLE_WIDTH)
    return (
        total_frames / voice.config.sample_rate
    ) + 1  # Add 0.5 seconds buffer for safety


class AudioManager:
    def __init__(self, scene: Scene):
        self.i: int = 0
        self.scene = scene
        self.times: list[float] = []
        self.audio_durations: list[float] = []

    def say(self, text):
        print(f"AudioManager: {text} at {self.scene.renderer.time:.2f} seconds")
        self.i += 1
        self.times.append(self.scene.renderer.time)
        # Say time and log how long the audio will be
        duration = create_wav(text, self.i)
        self.audio_durations.append(duration)
        print(f"AudioManager: Audio duration is {duration:.2f} seconds")

    def done_say(self):
        print("AudioManager: Done saying text")
        time_since_started = self.scene.renderer.time - self.times[-1]
        print(
            f"AudioManager: Time since started saying text: {time_since_started:.2f} seconds"
        )
        to_sleep = self.audio_durations[-1] - time_since_started
        if to_sleep > 0:
            print(f"AudioManager: Sleeping for {to_sleep:.2f} seconds")
            self.scene.wait(to_sleep)

    def merge_audio(self):
        if self.i == 0:
            print("AudioManager: No audio files to merge")
            return

        # Read the first audio file to get format info
        with wave.open("media/audio_1.wav", "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()

        # Calculate total duration needed
        last_time = self.times[-1]
        last_duration = self.audio_durations[-1]
        total_duration = last_time + last_duration
        total_frames = int(total_duration * sample_rate) + sample_rate  # Add buffer

        # Create output buffer filled with silence
        audio_data = np.zeros(total_frames, dtype=np.int16)

        # Merge all audio files
        for audio_idx in range(1, self.i + 1):
            with wave.open(f"media/audio_{audio_idx}.wav", "rb") as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_chunk = np.frombuffer(frames, dtype=np.int16)

                # Calculate where to place this audio
                start_time = self.times[audio_idx - 1]
                start_frame = int(start_time * sample_rate)

                # Write audio to the correct position
                end_frame = start_frame + len(audio_chunk)
                audio_data[start_frame:end_frame] = audio_chunk

        # Write merged audio to file
        with wave.open("media/merged_audio.wav", "wb") as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        print(f"AudioManager: Merged {self.i} audio files to media/merged_audio.wav")
