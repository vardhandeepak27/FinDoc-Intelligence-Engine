import fitz


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF.
    """

    document = fitz.open(pdf_path)

    pages = []

    for page in document:
        pages.append(page.get_text())

    document.close()

    return "\n".join(pages)