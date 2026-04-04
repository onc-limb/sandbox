from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import QueryBundle
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from src.document import Document


class Searcher:

    def __init__(
        self,
        index: VectorStoreIndex,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        rerank_top_n: int = 4,
    ) -> None:
        self._retriever = index.as_retriever(similarity_top_k=rerank_top_n * 2)
        self._reranker = FlagEmbeddingReranker(model=rerank_model, top_n=rerank_top_n)

    def search(self, query: str, k: int = 4) -> list[Document]:
        nodes = self._retriever.retrieve(query)
        query_bundle = QueryBundle(query_str=query)
        reranked = self._reranker.postprocess_nodes(nodes, query_bundle=query_bundle)
        return [
            Document(page_content=node.text, metadata=node.metadata)
            for node in reranked
        ]
