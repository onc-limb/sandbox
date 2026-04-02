"""Retriever module for high-level document search.

Provides a retrieve function that wraps vector_store.similarity_search
with a rerank placeholder.
"""

from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from src.document import Document
from src.vector_store import similarity_search


def retrieve(
    collection: chromadb.Collection,
    query: str,
    embeddings_model: SentenceTransformer,
    k: int = 4,
) -> list[Document]:
    """Retrieve documents relevant to the query.

    Args:
        collection: ChromaDB collection to search.
        query: Search query text.
        embeddings_model: SentenceTransformer model for encoding.
        k: Number of results to return.

    Returns:
        List of relevant Documents.
    """
    documents = similarity_search(collection, query, embeddings_model, k=k)

    # TODO: rerank処理をここに追加

    return documents
