from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from src.document import Document

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown"}


class DocumentLoader:

    def load(self, file_path: str | Path) -> list[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {ext}. Supported: .pdf, .md, .markdown"
            )

        llama_docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
        return [
            Document(page_content=doc.text, metadata=doc.metadata)
            for doc in llama_docs
            if doc.text.strip()
        ]
