import re
import sys
from pathlib import Path

import fitz  # type: ignore

TOC_LEVEL_1 = 1
MIN_ARGS = 2


def get_toc_topics(pdf_path: Path | str) -> list[dict]:
    """Returns a list of Level 1 topics: [{'title': str, 'page': int}]"""
    with fitz.open(pdf_path) as doc:
        return [{"title": t[1], "page": t[2] - 1} for t in doc.get_toc() if t[0] == TOC_LEVEL_1]


def safe_topic_name(title: str) -> str:
    """Returns a filesystem-safe version of a topic title."""
    return re.sub(r'[\\/*?:"<>|]', "", title).strip().replace(" ", "_")


def extract_topic_pdf(pdf_path: str, topic_idx: int, output_pdf_path: str) -> None:
    """Extracts pages for the topic at topic_idx into output_pdf_path."""
    topics = get_toc_topics(pdf_path)
    if topic_idx >= len(topics):
        raise IndexError(f"Topic index {topic_idx} out of range ({len(topics)} topics)")
    with fitz.open(pdf_path) as doc:
        start = topics[topic_idx]["page"]
        end = topics[topic_idx + 1]["page"] if topic_idx + 1 < len(topics) else len(doc)
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
        new_doc.save(output_pdf_path)
        new_doc.close()


def extract_topic(pdf_path: Path | str, target_topic: str) -> None:
    topics = get_toc_topics(pdf_path)
    idx = next(
        (i for i, t in enumerate(topics) if target_topic.lower() in t["title"].lower()),
        None,
    )

    if idx is None:
        print(f"❌ Topic '{target_topic}' not found.")
        return

    with fitz.open(pdf_path) as doc:
        start, end = (
            topics[idx]["page"],
            (topics[idx + 1]["page"] if idx + 1 < len(topics) else len(doc)),
        )
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start, to_page=end - 1)

        safe_name = re.sub(r'[\\/*?:"<>|]', "", topics[idx]["title"]).strip()
        new_doc.save(f"{safe_name}.pdf")
        new_doc.close()
        print(f"✅ Saved: {safe_name}.pdf (Pages {start + 1}-{end})")


if __name__ == "__main__":
    FILE = "courses/FMNF05/lectures/slides.pdf"
    if len(sys.argv) < MIN_ARGS:
        print(
            "Usage:\n uv run preprocessing/extract_from_pdf.py --get-topics"
            "uv run preprocessing/extract_from_pdf.py <topic_keyword>"
        )
    elif sys.argv[1] == "--get-topics":
        for t in get_toc_topics(FILE):
            print(f'- "{t["title"]}"')
    else:
        extract_topic(FILE, sys.argv[1])
