from __future__ import annotations

import logging

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.ingestion.chunker import Chunker
from src.config import Config
from src.embeddings import Embedder
from src.ingestion.document_loader import DocumentLoader

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, config: Config, embedder: Embedder) -> None:
        self._config = config
        self._embedder = embedder
        self._loader = DocumentLoader()
        self._chunker = Chunker(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        chroma_client = chromadb.PersistentClient(path=config.chroma_persist_dir)
        self._chroma_collection = chroma_client.get_or_create_collection(
            name=config.chroma_collection_name,
        )
        self._vector_store = ChromaVectorStore(
            chroma_collection=self._chroma_collection
        )
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
        )
        self._index: VectorStoreIndex | None = None

    def build_index(self, doc_path: str, *, force_rebuild: bool = False) -> None:
        if not force_rebuild and self._chroma_collection.count() > 0:
            logger.info("Existing index found, loading from vector store")
            self._index = VectorStoreIndex.from_vector_store(
                self._vector_store,
                embed_model=self._embedder.model,
            )
            return

        documents = self._loader.load(doc_path)
        if not documents:
            logger.warning("No documents found in %s", doc_path)
            return

        chunks = self._chunker.split(documents)
        nodes = [
            TextNode(text=chunk.page_content, metadata=chunk.metadata)
            for chunk in chunks
        ]
        self._index = VectorStoreIndex(
            nodes,
            storage_context=self._storage_context,
            embed_model=self._embedder.model,
        )

    def get_full_text(self, doc_path: str) -> str:
        documents = self._loader.load(doc_path)
        return "\n\n".join(doc.page_content for doc in documents)

    @property
    def index(self) -> VectorStoreIndex:
        if self._index is None:
            raise ValueError("build_index() を先に呼んでください")
        return self._index
