"""Configuration management module.

Loads settings from .env file and environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Config:
    """Application configuration.

    Loads GEMINI_API_KEY from environment variables.
    Model names and paths are configurable with sensible defaults.
    """

    gemini_api_key: str = field(repr=False, default="")
    llm_model_name: str = "gemini/gemini-3-flash-preview"
    embedding_model_name: str = "BAAI/bge-m3"
    chroma_persist_dir: str = str(_PROJECT_ROOT / "chroma_db")
    chroma_collection_name: str = "rag_collection"
    doc_path: str = str(_PROJECT_ROOT / "docs" / "oreilly-978-4-8144-0138-3e.pdf")
    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 4
    index_version: str = "default"
    chunk_size: int = 1024
    chunk_overlap: int = 200

    def __post_init__(self) -> None:
        if not self.gemini_api_key:
            key = os.environ.get("GEMINI_API_KEY", "")
            object.__setattr__(self, "gemini_api_key", key)
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Set it in .env or as an environment variable."
            )
        persist_dir = str(Path(self.chroma_persist_dir) / self.index_version)
        object.__setattr__(self, "chroma_persist_dir", persist_dir)


def get_config(**overrides: str) -> Config:
    """Create a Config instance with optional overrides."""
    return Config(**overrides)
