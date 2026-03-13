import os
import shutil
import tempfile
from pathlib import Path

from extract_from_pdf import extract_topic_pdf, get_toc_topics, safe_topic_name
from process_pdf import convert_pdf_to_md
from process_video import mp4_to_text

from preprocessing.process_images import image_to_md_llm


def _already_processed(dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Skipping (already exists): {dest}")
        return True
    return False


def batch_process(input_dir: Path, output_dir: Path, local: bool = False):
    input_root = input_dir
    output_root = output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    total_cost = 0.0

    for file_path in input_root.rglob("*"):
        if not file_path.is_file():
            continue

        relative = file_path.relative_to(input_root)
        suffix = file_path.suffix.lower()

        if suffix == ".md":
            # If markdown, copy as is
            dest = output_root / relative
            if _already_processed(dest):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            print(f"Copied: {file_path} -> {dest}")

        if suffix == ".txt":
            # If text, copy as is
            dest = output_root / relative
            if _already_processed(dest):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            print(f"Copied: {file_path} -> {dest}")

        if suffix == ".mp4":
            # If mp4, transcribe audio and save as txt with timestamps
            dest = output_root / relative.with_suffix(".txt")
            if _already_processed(dest):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            cost = mp4_to_text(str(file_path), str(dest))
            if cost > 0:
                print(
                    f"Processing video: {file_path} -> {dest} (LLM cost: ${cost:.4f})"
                )
            total_cost += cost
            print(f"Processing video: {file_path} -> {dest}")

        elif suffix == ".pdf":
            topics = get_toc_topics(str(file_path))
            if topics:
                # Multi-topic PDF: extract and process each topic separately
                for i, topic in enumerate(topics):
                    safe = safe_topic_name(topic["title"])
                    dest = (
                        output_root / relative.parent / f"{file_path.stem}__{safe}.md"
                    )
                    if _already_processed(dest):
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
                    os.close(tmp_fd)
                    try:
                        extract_topic_pdf(str(file_path), i, tmp_path)
                        cost = convert_pdf_to_md(tmp_path, str(dest), local=local)
                        if cost > 0:
                            print(
                                f"Processing PDF topic '{topic['title']}': {file_path} -> {dest} (LLM cost: ${cost:.4f})"
                            )
                        total_cost += cost
                    finally:
                        os.unlink(tmp_path)
            else:
                # No TOC: process whole PDF normally
                dest = output_root / relative.with_suffix(".md")
                if _already_processed(dest):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                cost = convert_pdf_to_md(str(file_path), str(dest), local=local)
                if cost > 0:
                    print(
                        f"Processing PDF: {file_path} -> {dest} (LLM cost: ${cost:.4f})"
                    )
                total_cost += cost
        elif suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}:
            # If image, convert to markdown using image_to_md_llm
            dest = output_root / relative.with_suffix(".md")
            if _already_processed(dest):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            cost = image_to_md_llm(str(file_path), str(dest))
            if cost > 0:
                print(
                    f"Processing Image: {file_path} -> {dest} (LLM cost: ${cost:.4f})"
                )
            total_cost += cost
        else:
            print(f"Unsupported file type (skipping): {file_path}")

    print(f"Batch processing complete. Total LLM cost: ${total_cost:.4f}")


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
