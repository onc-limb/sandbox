"""Agentic RAG bot using LlamaIndex AgentWorkflow.

The agent autonomously decides search strategies, reformulates queries,
and asks the user clarifying questions when needed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from llama_index.core.agent import AgentWorkflow
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.litellm import LiteLLM

from src.config import Config
from src.embeddings import Embedder
from src.ingestion.document_loader import DocumentLoader
from src.retrieval.index_loader import load_index
from src.retrieval.searcher import Searcher

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """\
あなたはドキュメントに基づいて質問に回答するインテリジェントなアシスタントです。

## 行動指針
1. まず質問を分析し、回答に必要な情報を考えてください。
2. 質問が曖昧な場合は、ask_user_clarification ツールでユーザーに確認してください。
3. search_knowledge_base ツールで関連情報を検索してください。
4. 検索結果が不十分な場合は、クエリを言い換えたり分解したりして再検索してください。
5. 十分な情報が集まったら、検索結果に基づいて回答を生成してください。
6. ドキュメントに情報がない場合は、その旨を正直に伝えてください。

## 注意事項
- 一度の検索で十分な情報が得られなくても諦めず、別の角度から検索してください。
- ユーザーの質問が複数のトピックにまたがる場合、トピックごとに分けて検索してください。
- 推測で回答せず、必ずツールで得た情報に基づいて回答してください。
"""


class AgenticBot:
    """AgentWorkflow-based RAG bot that autonomously searches and asks questions."""

    def __init__(
        self,
        config: Config,
        user_input_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._config = config
        self._searcher: Searcher | None = None
        self._full_text: str = ""
        self._embedder = Embedder(config)
        self._user_input_fn = user_input_fn or self._default_input
        self._workflow: AgentWorkflow | None = None
        self._memory = ChatMemoryBuffer.from_defaults(token_limit=8192)

    @staticmethod
    def _default_input(question: str) -> str:
        print(f"\n[エージェントからの質問] {question}")
        return input("> ").strip()

    def load_index(self) -> None:
        """Load vector index and build agent with tools."""
        index = load_index(self._config, self._embedder)

        loader = DocumentLoader()
        self._full_text = "\n\n".join(
            doc.page_content for doc in loader.load(self._config.doc_path)
        )

        self._searcher = Searcher(
            index,
            rerank_model=self._config.rerank_model_name,
            rerank_top_n=self._config.rerank_top_n,
        )

        self._workflow = self._build_workflow()
        logger.info("Agentic RAG の準備完了")

    def _build_workflow(self) -> AgentWorkflow:
        llm = LiteLLM(
            model=self._config.llm_model_name,
            api_key=self._config.gemini_api_key,
            temperature=1.0,
        )

        search_tool = FunctionTool.from_defaults(
            fn=self._search_knowledge_base,
            name="search_knowledge_base",
            description=(
                "ナレッジベースをベクトル検索するツール。"
                "検索クエリを渡すと、関連するドキュメントの断片を返します。"
                "異なるクエリで複数回呼び出すことができます。"
                "質問そのままではなく、検索に適したキーワードやフレーズを使ってください。"
            ),
        )

        ask_tool = FunctionTool.from_defaults(
            fn=self._ask_user_clarification,
            name="ask_user_clarification",
            description=(
                "ユーザーに追加情報や明確化を求めるツール。"
                "質問が曖昧な場合や、正確に回答するために追加の文脈が必要な場合に使用してください。"
                "具体的で分かりやすい質問を渡してください。"
            ),
        )

        read_tool = FunctionTool.from_defaults(
            fn=self._read_full_document,
            name="read_full_document",
            description=(
                "ドキュメントの全文を読むツール。"
                "ベクトル検索では見つからない情報を探す場合や、"
                "ドキュメント全体の構成を把握したい場合に使用してください。"
                "ドキュメントが長い場合があるので、search_knowledge_base を先に試してください。"
            ),
        )

        return AgentWorkflow.from_tools_or_functions(
            tools_or_functions=[search_tool, ask_tool, read_tool],
            llm=llm,
            system_prompt=AGENT_SYSTEM_PROMPT,
            verbose=True,
        )

    def _search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base with the given query."""
        if self._searcher is None:
            return "エラー: インデックスが読み込まれていません。"

        results = self._searcher.search(query)
        if not results:
            return "検索結果が見つかりませんでした。別のクエリで試してください。"

        parts: list[str] = []
        for i, doc in enumerate(results, 1):
            page = doc.metadata.get("page", "?")
            parts.append(f"[結果{i}] (p.{page})\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _ask_user_clarification(self, question: str) -> str:
        """Ask the user a clarifying question."""
        return self._user_input_fn(question)

    def _read_full_document(self) -> str:
        """Read the full document text."""
        if not self._full_text:
            return "ドキュメントが読み込まれていません。"
        return self._full_text

    def chat(self, query: str) -> str:
        """Send a query to the agent and return the response."""
        if self._workflow is None:
            raise ValueError("load_index() を先に呼んでください")

        result = asyncio.run(self._run(query))
        return result

    async def _run(self, query: str) -> str:
        handler = self._workflow.run(user_msg=query, memory=self._memory)
        result = await handler
        return str(result)

    def reset(self) -> None:
        """Reset conversation history."""
        self._memory.reset()
