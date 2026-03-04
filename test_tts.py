#!/usr/bin/env python3
"""
Quick test script to generate a sample audio clip using Qwen3-TTS.
The audio will be saved to .cache/audio/test_audio.wav
"""

import time
import torch

print("=" * 60)
print("Qwen3-TTS Test Script")
print("=" * 60)
print()

# Check device
print("1. Checking compute device...")
if torch.cuda.is_available():
    device = "cuda:0"
    print(f"   ✓ Using CUDA: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS (Apple Silicon)")
    print("Note: MPS can be slow on first run due to compilation")
else:
    device = "cpu"
    print("Using CPU (this will be slow)")
print()

from src.audiomanager import create_wav  # noqa: E402
from src.tts.qwen import QwenTTSEngine  # noqa: E402

print("2. Loading Qwen3-TTS model...")
print()
start = time.time()
engine = QwenTTSEngine.from_env()
engine._load_model()
load_time = time.time() - start
print(f"Model loaded in {load_time:.1f}s")
print()

print("3. Generating audio...")
gen_start = time.time()
duration = create_wav(
    "Hello! This is a test of the Qwen3 text to speech system. The quick brown fox jumps over the lazy dog.",
    999,
    engine,
)
gen_time = time.time() - gen_start

print(f"Audio generated in {gen_time:.1f}s")
print()

print("=" * 60)
print("SUCCESS!")
print("=" * 60)
print(f"Duration: {duration:.2f}s")
print("File: .cache/audio/audio_999.wav")
print()
print(f"Performance: {duration / gen_time:.2f}x realtime")
