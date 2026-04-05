"""Centralized option definitions for models and search methods."""

LLM_MODELS: list[str] = [
    "gemini/gemini-3.1-flash-lite-preview",
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-2.5-pro",
]

EMBEDDING_MODELS: list[str] = [
    "BAAI/bge-m3",
    "BAAI/bge-large-en-v1.5",
    "intfloat/multilingual-e5-large",
]

RERANK_MODELS: list[str] = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
]

SEARCH_METHODS: list[str] = ["rag", "fulltext", "hybrid"]
