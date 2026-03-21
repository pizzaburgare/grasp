#!/usr/bin/env python3
"""
Quick test script to generate a sample audio clip.
The audio will be saved to .cache/audio/audio_999.wav

Select the engine with the TTS_ENGINE env var (qwen | piper | kokoro).
Defaults to kokoro.
"""

import os
import sys
import time

from src.settings import DEFAULT_TTS_ENGINE
from src.tts import available_engines

ENGINES = available_engines()
engine_name = os.environ.get("TTS_ENGINE", DEFAULT_TTS_ENGINE).lower()

if engine_name not in ENGINES:
    print(f"Unknown engine '{engine_name}'. Choose one of: {', '.join(ENGINES)}")
    sys.exit(1)

print("=" * 60)
print(f"TTS Test Script  [{engine_name}]")
print("=" * 60)
print()

# Device info is only relevant for torch-based engines
if engine_name in ("qwen",):
    import torch

    print("1. Checking compute device...")
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        print("Note: MPS can be slow on first run due to compilation")
    else:
        print("Using CPU (this will be slow)")
    print()

from src.audiomanager import create_wav  # noqa: E402
from src.tts import get_default_engine  # noqa: E402

os.environ["TTS_ENGINE"] = engine_name

print(f"2. Loading {engine_name} model...")
print()
start = time.time()
engine = get_default_engine()
# Eagerly load the model if the engine exposes a load helper
for loader in ("_load_pipeline", "_load_model", "_load_voice"):
    if callable(getattr(engine, loader, None)):
        getattr(engine, loader)()
        break
load_time = time.time() - start
print(f"Model loaded in {load_time:.1f}s")
print()

print("3. Generating audio...")
gen_start = time.time()
duration = create_wav(
    "Hello! This is a test of the text to speech system. The quick brown fox jumps over the lazy dog.",
    999,
    engine,
)
gen_time = time.time() - gen_start

print(f"Audio generated in {gen_time:.1f}s")
print()

print("=" * 60)
print("SUCCESS!")
print("=" * 60)
print(f"Engine:   {engine_name}")
print(f"Duration: {duration:.2f}s")
print("File:     .cache/audio/audio_999.wav")
print()
print(f"Performance: {duration / gen_time:.2f}x realtime")
