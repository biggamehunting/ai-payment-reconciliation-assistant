import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF, including text contained in tables.
    """

    doc = fitz.open(file_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        if text.strip():
            pages.append(
                f"\n--- Page {page_number} ---\n"
                f"{text.strip()}"
            )

    doc.close()

    return "\n".join(pages)