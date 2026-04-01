"""PDF loading module.

Uses pymupdf to extract text from PDF files and returns
Document objects.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from src.document import Document


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Load a PDF file and return a list of Documents.

    Each page becomes a separate Document with page number metadata.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of Document objects, one per page.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents: list[Document] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_path),
                            "page": page_num + 1,
                            "total_pages": len(doc),
                        },
                    )
                )
    return documents
