"""Vector store module.

Provides ChromaDB-based vector store with persistence,
document indexing, and similarity search using chromadb directly.
"""

from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import Config, get_config
from src.document import Document


def create_vector_store(
    config: Config | None = None,
) -> chromadb.Collection:
    """Create or load a persistent ChromaDB collection.

    Args:
        config: Application config. Uses default if None.

    Returns:
        ChromaDB Collection instance.
    """
    if config is None:
        config = get_config()

    client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    return client.get_or_create_collection(name=config.chroma_collection_name)


def add_documents(
    collection: chromadb.Collection,
    documents: list[Document],
    embeddings_model: SentenceTransformer,
) -> None:
    """Add documents to the ChromaDB collection.

    Args:
        collection: Target ChromaDB collection.
        documents: Documents to add.
        embeddings_model: SentenceTransformer model for encoding.
    """
    if not documents:
        return

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [f"doc_{i}" for i in range(len(documents))]

    embeddings = embeddings_model.encode(
        texts, normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def similarity_search(
    collection: chromadb.Collection,
    query: str,
    embeddings_model: SentenceTransformer,
    k: int = 4,
) -> list[Document]:
    """Search for documents similar to the query.

    Args:
        collection: ChromaDB collection to search.
        query: Search query text.
        embeddings_model: SentenceTransformer model for encoding.
        k: Number of results to return.

    Returns:
        List of most similar Documents.
    """
    query_embedding = embeddings_model.encode(
        [query], normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    documents: list[Document] = []
    if results["documents"]:
        for i, text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            documents.append(Document(page_content=text, metadata=metadata))

    return documents
