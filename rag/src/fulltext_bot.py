"""Full-text system prompt bot.

Loads an entire PDF into the system prompt and answers questions
using LiteLLM via plain Python. No RAG/vector search is used.
"""

from __future__ import annotations

from src.config import get_config
from src.llm_client import chat
from src.pdf_loader import load_pdf


def _build_system_prompt(full_text: str) -> str:
    """Build a system prompt with the full document text embedded."""
    return (
        "以下のドキュメントの内容に基づいて質問に回答してください。\n\n"
        f"{full_text}"
    )


def _load_full_text(pdf_path: str) -> str:
    """Load all pages from a PDF and concatenate their text."""
    documents = load_pdf(pdf_path)
    if not documents:
        raise ValueError(f"PDF contains no extractable text: {pdf_path}")
    return "\n".join(doc.page_content for doc in documents)


def _run_cli_loop(system_prompt: str, config: object) -> None:
    """Run the interactive CLI loop."""
    print("全文システムプロンプトボットを起動しました。")
    print("質問を入力してください（quit/exit で終了）\n")

    while True:
        try:
            user_input = input("質問> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("終了します。")
            break

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        answer = chat(messages, config)
        print(f"\n回答: {answer}\n")


def main() -> None:
    """Entry point for the full-text bot."""
    config = get_config(pdf_path="oreilly-978-4-8144-0138-3e.pdf")
    full_text = _load_full_text(config.pdf_path)
    system_prompt = _build_system_prompt(full_text)
    _run_cli_loop(system_prompt, config)


if __name__ == "__main__":
    main()
