"""Hybrid RAG + Full-text bot CLI application.

Combines vector similarity search (RAG) with full PDF text
to provide comprehensive answers using Gemini via LiteLLM.
"""

from __future__ import annotations

import logging
import sys

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

_PROMPT_TEMPLATE = (
    "以下はドキュメントの全文です:\n{full_text}\n\n"
    "以下は質問に特に関連する部分です:\n{rag_context}\n\n"
    "質問: {question}\n\n"
    "上記の情報を基に、質問に回答してください。"
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
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Documents with inherited metadata.
    """
    chunks: list[Document] = []
    for doc in documents:
        text = doc.page_content
        if len(text) <= chunk_size:
            chunks.append(Document(page_content=text, metadata=dict(doc.metadata)))
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(
                Document(page_content=chunk_text, metadata=dict(doc.metadata))
            )
            start += chunk_size - chunk_overlap

    return chunks


def _build_full_text(documents: list[Document]) -> str:
    """Concatenate all page documents into a single full text string."""
    return "\n\n".join(doc.page_content for doc in documents)


def _format_rag_context(docs: list[Document]) -> str:
    """Format retrieved documents into a context string."""
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] (p.{page})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_prompt(full_text: str, search_results: list[Document], question: str) -> str:
    """Build the hybrid prompt combining full text and RAG results."""
    rag_context = _format_rag_context(search_results)
    return _PROMPT_TEMPLATE.format(
        full_text=full_text,
        rag_context=rag_context,
        question=question,
    )


def _build_index(config: Config, embeddings_model: object, collection: object, pages: list[Document]) -> None:
    """Build vector index from PDF pages if ChromaDB is not already populated."""
    existing_count = collection.count()
    if existing_count > 0:
        logger.info("ChromaDB already has %d documents. Skipping indexing.", existing_count)
        return

    if not pages:
        logger.warning("No documents to index.")
        return

    chunks = _split_documents(pages)
    logger.info("Split into %d chunks. Indexing...", len(chunks))

    add_documents(collection, chunks, embeddings_model)
    logger.info("Indexing complete. %d chunks stored.", len(chunks))


def _run_cli(config: Config, full_text: str, embeddings_model: object, collection: object) -> None:
    """Run the interactive CLI loop."""
    print("\nHybrid RAG Bot ready. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye.")
            break

        try:
            results = similarity_search(collection, user_input, embeddings_model, k=_TOP_K)
            if not results:
                logger.warning("No similar documents found for query: %s", user_input)

            prompt = _build_prompt(full_text, results, user_input)
            messages = [
                {"role": "system", "content": "あなたはドキュメントに基づいて質問に回答するアシスタントです。"},
                {"role": "user", "content": prompt},
            ]
            answer = chat(messages, config)
            print(f"\nAnswer: {answer}\n")
        except Exception:
            logger.exception("Error while answering question")
            print("エラーが発生しました。ログを確認してください。")


def main() -> None:
    """Entry point for the Hybrid RAG Bot CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        config = get_config(pdf_path="oreilly-978-4-8144-0138-3e.pdf")
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    logger.info("Loading PDF full text: %s", config.pdf_path)
    pages = load_pdf(config.pdf_path)
    if not pages:
        logger.error("PDF has no readable content: %s", config.pdf_path)
        sys.exit(1)

    full_text = _build_full_text(pages)
    logger.info("Loaded %d pages as full text.", len(pages))

    embeddings_model = create_embeddings(config)
    collection = create_vector_store(config)

    _build_index(config, embeddings_model, collection, pages)
    _run_cli(config, full_text, embeddings_model, collection)


if __name__ == "__main__":
    main()
