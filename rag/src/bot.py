"""Unified Bot module.

Integrates PDF loading, chunking, vector indexing, retrieval,
prompt building, and LLM chat into a single UnifiedBot class.
Supports both RAG-only and RAG + full-text modes.
"""

from __future__ import annotations

import logging

from src.chunker import split_documents
from src.config import Config
from src.embeddings import create_embeddings
from src.llm_client import chat
from src.pdf_loader import load_pdf
from src.prompt_builder import build_prompt
from src.retriever import retrieve
from src.vector_store import add_documents, create_vector_store

logger = logging.getLogger(__name__)


class UnifiedBot:
    """Unified RAG bot that supports both RAG-only and RAG + full-text modes.

    Args:
        config: Application configuration.
        include_full_text: When True, stores full PDF text and passes it
            to the prompt builder for comprehensive answers.
    """

    def __init__(self, config: Config, include_full_text: bool = False) -> None:
        self._config = config
        self._include_full_text = include_full_text
        self._full_text: str = ""

        self._embeddings = create_embeddings(config)
        self._collection = create_vector_store(config)

    def build_index(self) -> None:
        """Build vector index from PDF.

        Pipeline: pdf_loader → chunker → embeddings (via vector_store) → vector_store
        """
        documents = load_pdf(self._config.pdf_path)
        if not documents:
            logger.warning("No documents extracted from PDF.")
            return

        # include_full_text mode: store concatenated full text
        if self._include_full_text:
            self._full_text = "\n\n".join(
                doc.page_content for doc in documents
            )
            logger.info("Full text stored (%d pages).", len(documents))

        existing_count = self._collection.count()
        if existing_count > 0:
            logger.info(
                "ChromaDB already has %d documents. Skipping indexing.",
                existing_count,
            )
            return

        chunks = split_documents(documents)
        logger.info("Split into %d chunks. Indexing...", len(chunks))

        add_documents(self._collection, chunks, self._embeddings)
        logger.info("Indexing complete. %d chunks stored.", len(chunks))

    def answer(self, query: str) -> str:
        """Retrieve relevant chunks and generate an answer.

        Pipeline: embeddings (via retriever) → retriever (→ rerank) → prompt_builder → llm_client
        """
        results = retrieve(self._collection, query, self._embeddings)

        prompt = build_prompt(
            query=query,
            rag_results=results,
            include_full_text=self._include_full_text,
            full_text=self._full_text,
        )

        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        return chat(messages, config=self._config)

    def answer_fulltext_only(self, query: str) -> str:
        """Generate an answer using full text only, skipping RAG retrieval.

        Args:
            query: User's question.

        Returns:
            LLM-generated answer based on full text.

        Raises:
            ValueError: If include_full_text is False.
        """
        if not self._include_full_text:
            raise ValueError("answer_fulltext_only requires include_full_text=True")

        prompt = build_prompt(
            query=query,
            rag_results=[],
            include_full_text=True,
            full_text=self._full_text,
        )
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        return chat(messages, config=self._config)
