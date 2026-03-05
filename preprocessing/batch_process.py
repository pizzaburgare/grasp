import shutil
from pathlib import Path

from process_pdf import clean_pdf_conversion
from process_video import mp4_to_text


def batch_process(input_dir: str, output_dir: str):
    input_root = Path(input_dir)
    output_root = Path(output_dir)

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
            clean_pdf_conversion(str(file_path), str(dest))


if __name__ == "__main__":
    INPUT_DIR = "rag_source/test/raw"
    OUTPUT_DIR = "rag_source/test/processed"
    batch_process(INPUT_DIR, OUTPUT_DIR)
