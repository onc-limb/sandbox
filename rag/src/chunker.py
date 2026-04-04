from __future__ import annotations

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from src.document import Document


class Chunker:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200) -> None:
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, documents: list[Document]) -> list[Document]:
        nodes = [
            TextNode(text=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        split_nodes = self._splitter.get_nodes_from_documents(nodes)
        return [
            Document(page_content=node.text, metadata=node.metadata)
            for node in split_nodes
        ]
