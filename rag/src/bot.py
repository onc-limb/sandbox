from __future__ import annotations

import logging

from src.config import Config
from src.embeddings import Embedder
from src.indexer import Indexer
from src.llm_client import LlmClient
from src.prompt_builder import PromptBuilder
from src.searcher import Searcher

logger = logging.getLogger(__name__)


class UnifiedBot:
    def __init__(self, config: Config, include_full_text: bool = False) -> None:
        self._config = config
        self._include_full_text = include_full_text
        self._full_text: str = ""
        embedder = Embedder(config)
        self._indexer = Indexer(config, embedder)
        self._searcher: Searcher | None = None
        self._prompt_builder = PromptBuilder()
        self._llm_client = LlmClient(config)

    def build_index(self) -> None:
        self._indexer.build_index(self._config.doc_path)
        if self._include_full_text:
            self._full_text = self._indexer.get_full_text(self._config.doc_path)
        self._searcher = Searcher(
            self._indexer.index,
            rerank_model=self._config.rerank_model_name,
            rerank_top_n=self._config.rerank_top_n,
        )

    def answer(self, query: str) -> str:
        if self._searcher is None:
            raise ValueError("build_index() を先に呼んでください")
        results = self._searcher.search(query)
        prompt = self._prompt_builder.build(
            query=query,
            rag_results=results,
            include_full_text=self._include_full_text,
            full_text=self._full_text,
        )
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        return self._llm_client.chat(messages)

    def answer_fulltext_only(self, query: str) -> str:
        if self._searcher is None:
            raise ValueError("build_index() を先に呼んでください")
        if not self._include_full_text:
            raise ValueError("answer_fulltext_only requires include_full_text=True")
        prompt = self._prompt_builder.build(
            query=query,
            rag_results=[],
            include_full_text=True,
            full_text=self._full_text,
        )
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        return self._llm_client.chat(messages)
