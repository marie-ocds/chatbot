"""
Utility functions for text processing and PDF handling.
"""
import re
import fitz


def clean_text(text):
    """
    Clean text extracted from PDF by:
    - Removing line breaks in the middle of sentences
    - Adding spaces around em dashes if missing
    - Removing multiple spaces
    """
    # Add spaces around em dashes (—) and en dashes (–) if missing
    text = re.sub(r'(?<! )(—|–)(?! )', r' \1 ', text)
    text = re.sub(r'(?<! )(—|–) ', r' \1 ', text)
    text = re.sub(r' (—|–)(?! )', r' \1 ', text)

    # Remove line breaks in the middle of sentences
    text = re.sub(r'([a-z]) \n', r'\1 ', text)
    text = re.sub(r'([,;]) \n', r'\1 ', text)

    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Remove multiple consecutive line breaks (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_pdf_text(pdf_path):
    """
    Extract all text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        str: Combined text from all pages
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()

    return " ".join(pages)


def extract_chapters(text):
    """
    Extract chapter names and titles from cleaned text.

    Args:
        text: Cleaned text content

    Returns:
        dict: Mapping of chapter index to chapter title
    """
    clean_lines = text.split("\n")
    upper_lines = [line for line in clean_lines if line.isupper()]

    chapters = {
        i + 1: upper_lines[j + 1]
        for i, j in enumerate(
            idx for idx, e in enumerate(upper_lines) if "CHAPTER" in e
        )
    }

    return chapters