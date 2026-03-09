import shutil
from pathlib import Path

from process_pdf import convert_pdf_to_md
from process_video import mp4_to_text


def batch_process(input_dir: Path, output_dir: Path, local: bool = False):
    input_root = input_dir
    output_root = output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    for file_path in input_root.rglob("*"):
        if not file_path.is_file():
            continue

        relative = file_path.relative_to(input_root)
        suffix = file_path.suffix.lower()

        if suffix == ".md":
            # If markdown, copy as is
            dest = output_root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            print(f"Copied: {file_path} -> {dest}")

        if suffix == ".txt":
            # If text, copy as is
            dest = output_root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest)
            print(f"Copied: {file_path} -> {dest}")

        if suffix == ".mp4":
            # If mp4, transcribe audio and save as txt with timestamps
            dest = output_root / relative.with_suffix(".txt")
            dest.parent.mkdir(parents=True, exist_ok=True)
            mp4_to_text(str(file_path), str(dest))
            print(f"Processing video: {file_path} -> {dest}")

        elif suffix == ".pdf":
            # If pdf, convert to markdown using markitdown
            dest = output_root / relative.with_suffix(".md")
            dest.parent.mkdir(parents=True, exist_ok=True)
            convert_pdf_to_md(str(file_path), str(dest), local=local)


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

    batch_process(INPUT_DIR, OUTPUT_DIR, local=args.local)
