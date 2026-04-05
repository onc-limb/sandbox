"""Document preprocessing and VectorDB ingestion command."""

from __future__ import annotations

import argparse
import logging

from src.config import Config, get_config
from src.embeddings import Embedder
from src.indexer import Indexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    config_defaults = Config.__dataclass_fields__
    parser = argparse.ArgumentParser(
        description="Ingest documents into VectorDB",
    )
    parser.add_argument(
        "--doc",
        default=None,
        help=f"Document file path (PDF or Markdown) (default: {config_defaults['doc_path'].default})",
    )
    parser.add_argument(
        "--version",
        default="default",
        help="Index version name (default: default)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size (default: 1024)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap (default: 200)",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=f"Embedding model (default: {config_defaults['embedding_model_name'].default})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index already exists",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Build config with only specified overrides
    overrides: dict = {
        "index_version": args.version,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    if args.doc is not None:
        overrides["doc_path"] = args.doc
    if args.embedding_model is not None:
        overrides["embedding_model_name"] = args.embedding_model
    config = get_config(**overrides)

    logger.info("Document: %s", config.doc_path)
    logger.info("Version: %s", args.version)
    logger.info("Chunk size: %d, overlap: %d", config.chunk_size, config.chunk_overlap)
    logger.info("Embedding model: %s", config.embedding_model_name)
    logger.info("Persist dir: %s", config.chroma_persist_dir)

    # Initialize and build
    embedder = Embedder(config)
    indexer = Indexer(config, embedder)

    logger.info("Building index (force=%s)...", args.force)
    indexer.build_index(config.doc_path, force_rebuild=args.force)

    # Log chunk count from the chroma collection
    chunk_count = indexer._chroma_collection.count()
    logger.info("Done. Chunks in DB: %d", chunk_count)
    logger.info("Saved to: %s", config.chroma_persist_dir)


if __name__ == "__main__":
    main()
