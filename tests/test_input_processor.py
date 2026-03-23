"""
Tests for src/input_processor.py

Verifies that each supported file type is correctly converted into
LangChain-compatible content parts without requiring real models or
external services.

Run with:
    uv run pytest tests/test_input_processor.py -v
"""

from pathlib import Path
from unittest.mock import patch

import pytest

# PLR2004 constants
PDF_PAGE_COUNT_TWO = 2
PDF_TOTAL_PAGES_FOUR = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_parts(parts: list[dict]) -> list[dict]:
    return [p for p in parts if p["type"] == "image_url"]


def _text_parts(parts: list[dict]) -> list[dict]:
    return [p for p in parts if p["type"] == "text"]


# ===========================================================================
# process_input_dir  - PDF handling
# ===========================================================================


class TestProcessInputDirPdf:
    """PDF files must contribute image_url parts (one per page)."""

    def test_pdf_pages_become_image_url_parts(self, tmp_path: Path) -> None:
        """Each rendered page must appear as an image_url part."""
        pdf = tmp_path / "lecture.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")  # file just needs to exist

        fake_uris = [
            "data:image/jpeg;base64,page1",
            "data:image/jpeg;base64,page2",
        ]

        with patch("src.input_processor.extract_pdf_pages", return_value=fake_uris):
            from src.input_processor import process_input_dir

            parts = process_input_dir(tmp_path)

        imgs = _image_parts(parts)
        assert len(imgs) == PDF_PAGE_COUNT_TWO
        assert imgs[0]["image_url"]["url"] == fake_uris[0]
        assert imgs[1]["image_url"]["url"] == fake_uris[1]

    def test_pdf_caption_text_part_included(self, tmp_path: Path) -> None:
        """A text part describing the PDF (with page count) must precede its images."""
        pdf = tmp_path / "slides.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        with patch(
            "src.input_processor.extract_pdf_pages",
            return_value=["data:image/jpeg;base64,x"],
        ):
            from src.input_processor import process_input_dir

            parts = process_input_dir(tmp_path)

        texts = _text_parts(parts)
        assert any("slides.pdf" in p["text"] for p in texts)
        assert any("1 page" in p["text"] for p in texts)

    def test_empty_pdf_produces_no_image_parts(self, tmp_path: Path) -> None:
        """A PDF that renders no pages should contribute no image_url parts."""
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("src.input_processor.extract_pdf_pages", return_value=[]):
            from src.input_processor import process_input_dir

            parts = process_input_dir(tmp_path)

        assert _image_parts(parts) == []

    def test_multiple_pdfs_all_pages_included(self, tmp_path: Path) -> None:
        """Pages from all PDFs in the directory must be present."""
        (tmp_path / "a.pdf").write_bytes(b"%PDF fake")
        (tmp_path / "b.pdf").write_bytes(b"%PDF fake")

        def fake_extract(path: Path, **kwargs: object) -> list[str]:
            # Return different page counts per file
            return ["data:image/jpeg;base64,x"] * (1 if "a" in path.name else 3)

        with patch("src.input_processor.extract_pdf_pages", side_effect=fake_extract):
            from src.input_processor import process_input_dir

            parts = process_input_dir(tmp_path)

        assert len(_image_parts(parts)) == PDF_TOTAL_PAGES_FOUR  # 1 + 3


# ===========================================================================
# process_input_dir  - mixed file types
# ===========================================================================


class TestProcessInputDirMixed:
    def test_text_file_included_as_text_part(self, tmp_path: Path) -> None:
        txt = tmp_path / "notes.txt"
        txt.write_text("Some notes here")

        from src.input_processor import process_input_dir

        parts = process_input_dir(tmp_path)

        texts = _text_parts(parts)
        assert any("notes.txt" in p["text"] for p in texts)
        assert any("Some notes here" in p["text"] for p in texts)

    def test_unsupported_extension_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "data.xyz").write_bytes(b"binary")

        from src.input_processor import process_input_dir

        parts = process_input_dir(tmp_path)

        assert parts == []

    def test_hidden_files_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".DS_Store").write_bytes(b"mac junk")
        (tmp_path / "notes.txt").write_text("visible")

        from src.input_processor import process_input_dir

        parts = process_input_dir(tmp_path)

        assert not any(".DS_Store" in str(p) for p in parts)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        from src.input_processor import process_input_dir

        parts = process_input_dir(tmp_path)

        assert parts == []

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        from src.input_processor import process_input_dir

        with pytest.raises(FileNotFoundError):
            process_input_dir(tmp_path / "does_not_exist")
