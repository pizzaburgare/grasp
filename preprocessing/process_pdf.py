import re

from markitdown import MarkItDown


def clean_pdf_conversion(input_path, output_path):
    """
    Converts a PDF to Markdown and strips out CID tags,
    excessive formatting characters, and redundant whitespace.
    """
    try:
        # Initialize converter and process file
        md = MarkItDown()
        result = md.convert(input_path)
        content = result.markdown

        # 1. Remove CID tags: (cid:123)
        content = re.sub(r"\(cid:\d+\)", "", content)

        # 2. Remove specific formatting characters like pipes and dashes
        # Added [ ] to the class if you want to add more chars later
        content = re.sub(r"\||-", "", content)

        # 3. Clean up horizontal whitespace (multiple spaces -> single space)
        # We use [^\S\r\n] to target spaces/tabs but NOT newlines
        content = re.sub(r"[^\S\r\n]+", " ", content)

        # 4. Clean up vertical whitespace (3+ newlines -> 2 newlines)
        # This preserves paragraph breaks while removing excessive gaps
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Successfully cleaned and saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")


# Usage
if __name__ == "__main__":
    INPUT_FILE = "slides.pdf"
    OUTPUT_FILE = "output.md"
    clean_pdf_conversion(INPUT_FILE, OUTPUT_FILE)
