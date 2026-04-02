"""Chunking module for splitting documents into fixed-size chunks."""

from __future__ import annotations

from src.document import Document

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200


def split_documents(
    documents: list[Document],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into fixed-size chunks with overlap.

    Args:
        documents: Source documents to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        List of chunked Documents with inherited metadata.
    """
    chunks: list[Document] = []
    for doc in documents:
        text = doc.page_content
        if len(text) <= chunk_size:
            chunks.append(doc)
            continue
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(Document(page_content=chunk_text, metadata=dict(doc.metadata)))
            start += chunk_size - chunk_overlap
    return chunks
