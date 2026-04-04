from __future__ import annotations

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import Config


class Embedder:
    def __init__(self, config: Config) -> None:
        self._embed_model = HuggingFaceEmbedding(model_name=config.embedding_model_name)

    @property
    def model(self) -> HuggingFaceEmbedding:
        return self._embed_model

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._embed_model.get_text_embedding_batch(texts)
