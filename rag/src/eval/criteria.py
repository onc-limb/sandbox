"""Evaluation criteria definitions for RAG assessment.

Customize DEFAULT_CRITERIA to change what is evaluated and how.
Each Criterion has a name (used as JSON key) and a description
(shown to the judge LLM to define the scoring dimension).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    name: str
    description: str


DEFAULT_CRITERIA: list[Criterion] = [
    Criterion(
        name="relevance",
        description="回答は質問に対して適切に答えているか？",
    ),
    Criterion(
        name="faithfulness",
        description="回答はソースコンテキストに忠実か？（ハルシネーションがないか）",
    ),
    Criterion(
        name="completeness",
        description="回答は質問の重要な側面をすべてカバーしているか？",
    ),
    Criterion(
        name="conciseness",
        description="回答は不要な情報なく簡潔にまとめられているか？",
    ),
]

_PROMPT_TEMPLATE = """\
あなたはRAG（Retrieval-Augmented Generation）システムの回答を評価する公正な審査員です。

## 入力
- **質問**: ユーザーの質問
- **コンテキスト**: システムに提供された検索ソースドキュメント
- **回答**: システムが生成した回答

## 評価基準
各基準を1（最低）〜5（最高）で採点してください：

{criteria_section}

## 出力形式
以下の構造のJSONオブジェクトのみを返してください（他のテキストは不要）：
```json
{json_schema}
```

## 評価対象

**質問:**
{question}

**コンテキスト:**
{context}

**回答:**
{answer}
"""


def build_eval_prompt(
    criteria: list[Criterion],
    question: str,
    context: str,
    answer: str,
) -> str:
    """Build the evaluation prompt for the given criteria and inputs."""
    criteria_section = "\n".join(
        f"{i}. **{c.name}**: {c.description}"
        for i, c in enumerate(criteria, 1)
    )
    json_schema = (
        "{\n"
        + ",\n".join(
            f'  "{c.name}": {{"score": <1-5>, "reason": "<簡潔な理由>"}}'
            for c in criteria
        )
        + "\n}"
    )
    return _PROMPT_TEMPLATE.format(
        criteria_section=criteria_section,
        json_schema=json_schema,
        question=question,
        context=context,
        answer=answer,
    )
