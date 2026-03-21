"""
generate_lesson.py - AI Course Lesson Generator

Usage:
  uv run lesson "LU Decomposition"
  uv run lesson "Fourier Transform" --input-dir ./slides --model google/gemini-3.1-pro-preview
  uv run lesson "QR Decomposition" --final
  uv run lesson "IEEE Double precicion" --input-dir ./courses/FMNF05/IEEE --final --skip-review
"""

import sys
from pathlib import Path

# Make project root importable when running directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import main

if __name__ == "__main__":
    main()
