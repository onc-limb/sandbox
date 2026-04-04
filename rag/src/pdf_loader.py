from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from src.document import Document


class PdfLoader:

    def load(self, pdf_path: str | Path) -> list[Document]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        llama_docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
        return [
            Document(page_content=doc.text, metadata=doc.metadata)
            for doc in llama_docs
            if doc.text.strip()
        ]
