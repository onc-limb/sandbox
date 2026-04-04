from __future__ import annotations

from src.document import Document


class PromptBuilder:

    def _format_rag_context(self, docs: list[Document]) -> str:
        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            parts.append(f"[{i}] (p.{page})\n{doc.page_content}")
        return "\n\n".join(parts)

    def build(
        self,
        query: str,
        rag_results: list[Document],
        include_full_text: bool = False,
        full_text: str | None = None,
    ) -> dict[str, str]:
        if include_full_text:
            system = "あなたはドキュメントに基づいて質問に回答するアシスタントです。"
            if rag_results:
                rag_context = self._format_rag_context(rag_results)
                user = (
                    f"以下はドキュメントの全文です:\n{full_text}\n\n"
                    f"以下は質問に特に関連する部分です:\n{rag_context}\n\n"
                    f"質問: {query}\n\n"
                    f"上記の情報を基に、質問に回答してください。"
                )
            else:
                user = (
                    f"以下はドキュメントの全文です:\n{full_text}\n\n"
                    f"質問: {query}\n\n"
                    f"上記の情報を基に、質問に回答してください。"
                )
        else:
            rag_context = self._format_rag_context(rag_results)
            system = (
                "以下のコンテキストを参考に、質問に回答してください。"
                "コンテキストに情報がない場合は、その旨を伝えてください。"
            )
            user = f"コンテキスト:\n{rag_context}\n\n質問: {query}"

        return {"system": system, "user": user}
