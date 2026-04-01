"""Embedding model module.

Uses sentence-transformers SentenceTransformer directly
for BAAI/bge-m3 embeddings.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.config import Config, get_config


def create_embeddings(
    config: Config | None = None,
) -> SentenceTransformer:
    """Create a SentenceTransformer instance for bge-m3.

    Args:
        config: Application config. Uses default if None.

    Returns:
        SentenceTransformer model that produces normalized embeddings
        via its encode() method.
    """
    if config is None:
        config = get_config()

    return SentenceTransformer(config.embedding_model_name)
