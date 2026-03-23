import shutil
from pathlib import Path
from typing import Any

from src.preprocessing.process_images import image_to_md_llm
from src.preprocessing.process_pdf import convert_pdf_to_md
from src.preprocessing.process_video import mp4_to_text


def _already_processed(dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Skipping (already exists): {dest}")
        return True
    return False


def _build_text_parts(processed_dir: Path) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    files = sorted(f for f in processed_dir.rglob("*") if f.is_file() and not f.name.startswith("."))

    for file in files:
        rel = file.relative_to(processed_dir)
        text = file.read_text(errors="replace")
        parts.append({"type": "text", "text": f"--- File: {rel} ---\n{text}"})

    return parts


def batch_process(
    input_dir: Path,
    output_dir: Path,
    local: bool = False,
) -> tuple[float, list[dict[str, Any]]]:
    input_root = input_dir
    output_root = output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    total_cost = 0.0

    for file_path in input_root.rglob("*"):
        if not file_path.is_file():
            continue

        relative = file_path.relative_to(input_root)
        suffix = file_path.suffix.lower()

        dest = output_root / relative.with_suffix(".md")
        if _already_processed(dest):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)

        if suffix in {".md", ".txt"}:
            # If markdown or text, copy as is
            shutil.copy2(file_path, dest)
            print(f"Copied: {file_path} -> {dest}")

        elif suffix == ".mp4":
            # If mp4, transcribe audio and save as txt with timestamps
            cost = mp4_to_text(str(file_path), str(dest))
            if cost > 0:
                print(f"Processing video: {file_path} -> {dest} (LLM cost: ${cost:.4f})")
            total_cost += cost
            print(f"Processing video: {file_path} -> {dest}")

        elif suffix == ".pdf":
            # No TOC: process whole PDF normally
            cost = convert_pdf_to_md(str(file_path), str(dest), local=local)
            if cost > 0:
                print(f"Processing PDF: {file_path} -> {dest} (LLM cost: ${cost:.4f})")
            total_cost += cost
        elif suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}:
            # If image, convert to markdown using image_to_md_llm
            cost = image_to_md_llm(str(file_path), str(dest))
            if cost > 0:
                print(f"Processing Image: {file_path} -> {dest} (LLM cost: ${cost:.4f})")
            total_cost += cost
        else:
            print(f"Unsupported file type (skipping): {file_path}")

    print(f"Batch processing complete. Total LLM cost: ${total_cost:.4f}")
    return total_cost, _build_text_parts(output_root)


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
