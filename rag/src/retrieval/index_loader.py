from __future__ import annotations

import logging

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import Config
from src.embeddings import Embedder

logger = logging.getLogger(__name__)


def load_index(config: Config, embedder: Embedder) -> VectorStoreIndex:
    """既存の ChromaDB インデックスをロードして VectorStoreIndex を返す."""
    chroma_client = chromadb.PersistentClient(path=config.chroma_persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(
        name=config.chroma_collection_name,
    )

    if chroma_collection.count() == 0:
        raise ValueError(
            f"コレクション '{config.chroma_collection_name}' は空です。"
            f"先にインデックスを構築してください。"
        )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embedder.model,
    )

    logger.info("インデックスをロードしました (collection=%s)", config.chroma_collection_name)
    return index
