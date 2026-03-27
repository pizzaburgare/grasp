"""Batch preprocessing pipeline for raw course materials.

This module walks an input directory and converts supported files to markdown
in a mirrored output directory while tracking aggregated LLM usage cost.
"""

import shutil
from pathlib import Path

from src.llm_metrics import LLMUsage, accumulate_llm_usage
from src.preprocessing.process_images import image_to_md_llm
from src.preprocessing.process_pdf import convert_pdf_to_md
from src.preprocessing.process_video import mp4_to_text


def _already_processed(dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Skipping (already exists): {dest}")
        return True
    return False


def _process_single_file(
    file_path: Path,
    dest: Path,
    suffix: str,
    local: bool,
    total_usage: LLMUsage,
) -> None:
    if suffix in {".md", ".txt"}:
        # If markdown or text, copy as is
        shutil.copy2(file_path, dest)
        print(f"Copied: {file_path} -> {dest}")
        return

    if suffix == ".mp4":
        # If mp4, transcribe audio and save as txt with timestamps
        usage = mp4_to_text(str(file_path), str(dest))
        if usage is not None and usage.cost_usd is not None and usage.cost_usd > 0:
            print(f"Processing video: {file_path} -> {dest} (LLM cost: ${usage.cost_usd:.4f})")
        accumulate_llm_usage(total_usage, usage)
        print(f"Processing video: {file_path} -> {dest}")
        return

    if suffix == ".pdf":
        # No TOC: process whole PDF normally
        usage = convert_pdf_to_md(str(file_path), str(dest), local=local)
        if usage is not None and usage.cost_usd is not None and usage.cost_usd > 0:
            print(f"Processing PDF: {file_path} -> {dest} (LLM cost: ${usage.cost_usd:.4f})")
        accumulate_llm_usage(total_usage, usage)
        return

    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}:
        # If image, convert to markdown using image_to_md_llm
        usage = image_to_md_llm(str(file_path), str(dest))
        if usage is not None and usage.cost_usd is not None and usage.cost_usd > 0:
            print(f"Processing Image: {file_path} -> {dest} (LLM cost: ${usage.cost_usd:.4f})")
        accumulate_llm_usage(total_usage, usage)
        return

    print(f"Unsupported file type (skipping): {file_path}")


def batch_process(
    input_dir: Path,
    output_dir: Path,
    local: bool = False,
) -> LLMUsage:
    """Process all supported files under ``input_dir`` into ``output_dir``.

    Supported file types are copied or converted to markdown/text depending on
    type, and the total LLM usage cost is accumulated and returned.
    """
    input_root = input_dir
    output_root = output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    total_usage = LLMUsage(cost_usd=0.0)

    for file_path in input_root.rglob("*"):
        if not file_path.is_file():
            continue

        relative = file_path.relative_to(input_root)
        suffix = file_path.suffix.lower()

        dest = output_root / relative.with_suffix(".md")
        if _already_processed(dest):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)

        _process_single_file(file_path, dest, suffix, local, total_usage)

    if total_usage.cost_usd is None:
        print("Batch processing complete. Total LLM cost: n/a")
    else:
        print(f"Batch processing complete. Total LLM cost: ${total_usage.cost_usd:.4f}")
    return total_usage


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch process course files.")
    parser.add_argument(
        "directory",
        type=Path,
        help="Course directory (must contain a 'raw' subdirectory)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Use local model for PDF conversion",
    )
    args = parser.parse_args()

    INPUT_DIR = args.directory / "raw"
    OUTPUT_DIR = args.directory / "processed"

    print(f"Input:  {INPUT_DIR.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")

    batch_process(INPUT_DIR, OUTPUT_DIR, local=args.local)
