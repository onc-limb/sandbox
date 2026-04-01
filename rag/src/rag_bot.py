"""RAG Bot CLI application.

Loads PDF documents, builds a ChromaDB vector index, and provides
an interactive CLI for question-answering using Gemini via LiteLLM.
"""

from __future__ import annotations

import logging
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import Config, get_config
from src.document import Document
from src.embeddings import create_embeddings
from src.llm_client import chat
from src.pdf_loader import load_pdf
from src.vector_store import add_documents, create_vector_store, similarity_search

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200
_TOP_K = 4

_SYSTEM_PROMPT = (
    "以下のコンテキストを参考に、質問に回答してください。"
    "コンテキストに情報がない場合は、その旨を伝えてください。"
)


def _split_documents(
    documents: list[Document],
    chunk_size: int = _CHUNK_SIZE,
    chunk_overlap: int = _CHUNK_OVERLAP,
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


def _format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string."""
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] (p.{page})\n{doc.page_content}")
    return "\n\n".join(parts)


class RAGBot:
    """RAG-based question answering bot.

    Encapsulates index building, retrieval, and answer generation.
    Reuses embeddings and vector store instances across queries.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._embeddings: SentenceTransformer = create_embeddings(config)
        self._store: chromadb.Collection = create_vector_store(config)

    def build_index(self) -> None:
        """Build vector index from PDF if not already populated."""
        existing_count = self._store.count()
        if existing_count > 0:
            logger.info(
                "ChromaDB already has %d documents. Skipping indexing.",
                existing_count,
            )
            return

        logger.info("Loading PDF: %s", self._config.pdf_path)
        documents = load_pdf(self._config.pdf_path)
        if not documents:
            logger.warning("No documents extracted from PDF.")
            return

        chunks = _split_documents(documents)
        logger.info("Split into %d chunks. Indexing...", len(chunks))

        add_documents(self._store, chunks, self._embeddings)
        logger.info("Indexing complete. %d chunks stored.", len(chunks))

    def answer(self, query: str) -> str:
        """Retrieve relevant chunks and generate an answer."""
        results = similarity_search(
            self._store, query, self._embeddings, k=_TOP_K,
        )
        if not results:
            return "関連する情報が見つかりませんでした。"

        context = _format_context(results)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"コンテキスト:\n{context}\n\n質問: {query}"},
        ]
        return chat(messages, config=self._config)


def _run_cli(bot: RAGBot) -> None:
    """Run the interactive CLI loop."""
    print("RAG Bot ready. Type your question (quit/exit to stop).")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye.")
            break

        try:
            answer = bot.answer(user_input)
            print(f"\n{answer}")
        except Exception:
            logger.exception("Error while answering question")
            print("エラーが発生しました。ログを確認してください。")


def main() -> None:
    """Entry point for the RAG Bot CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        config = get_config(
            pdf_path="oreilly-978-4-8144-0138-3e.pdf",
        )
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    bot = RAGBot(config)
    bot.build_index()
    _run_cli(bot)


if __name__ == "__main__":
    main()
