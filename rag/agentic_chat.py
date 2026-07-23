"""Agentic RAG チャット CLI.

エージェントが自律的に検索戦略を決定し、
必要に応じてユーザーに質問を返しながら回答を生成します。
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import get_config
from src.options import LLM_MODELS
from src.retrieval.agentic_bot import AgenticBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def select_version() -> str:
    """対話式でインデックスバージョンを選択する."""
    chroma_dir = Path("chroma_db")
    if not chroma_dir.exists():
        print("\n警告: chroma_db/ が存在しません。先に ingest.py を実行してください。")
        sys.exit(1)

    versions = sorted(d.name for d in chroma_dir.iterdir() if d.is_dir())
    if not versions:
        print("\n警告: chroma_db/ にバージョンが存在しません。先に ingest.py を実行してください。")
        sys.exit(1)

    print("\nインデックスバージョンを選択してください:")
    for i, v in enumerate(versions, 1):
        print(f"{i}. {v}")
    while True:
        choice = input("> ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(versions):
                return versions[idx]
        except ValueError:
            pass
        print(f"1〜{len(versions)} の番号を入力してください")


def select_model() -> str:
    """対話式で LLM モデルを選択する."""
    print("\nLLMモデルを選択してください:")
    for i, m in enumerate(LLM_MODELS, 1):
        label = " (default)" if i == 1 else ""
        print(f"{i}. {m}{label}")
    print(f"{len(LLM_MODELS) + 1}. カスタム入力")
    while True:
        choice = input("> ").strip()
        if choice == "" or choice == "1":
            return LLM_MODELS[0]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(LLM_MODELS):
                return LLM_MODELS[idx]
            if idx == len(LLM_MODELS):
                custom = input("モデル名を入力: ").strip()
                if custom:
                    return custom
                print("モデル名を入力してください")
                continue
        except ValueError:
            pass
        print(f"1〜{len(LLM_MODELS) + 1} の番号を入力してください")


def parse_args() -> argparse.Namespace:
    """CLI 引数をパースする."""
    parser = argparse.ArgumentParser(description="Agentic RAG チャット")
    parser.add_argument("--model", help="LLM モデル名")
    parser.add_argument("--version", help="使用するインデックスバージョン")
    parser.add_argument("--doc", help="ドキュメントファイルパス (PDF or Markdown)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="エージェントの思考過程を表示する (デフォルト: 有効)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="エージェントの思考過程を非表示にする",
    )
    return parser.parse_args()


def main() -> None:
    """メイン処理."""
    args = parse_args()

    version = args.version if args.version else select_version()
    model = args.model if args.model else select_model()

    config_overrides = {
        "llm_model_name": model,
        "index_version": version,
    }
    if args.doc:
        config_overrides["doc_path"] = args.doc

    config = get_config(**config_overrides)

    bot = AgenticBot(config)

    print("\nインデックスをロード中...")
    bot.load_index()

    print("\n=== Agentic RAG Chat ===")
    print(f"モデル: {model}")
    print(f"バージョン: {version}")
    print("\n質問を入力してください（exit で終了, reset で会話リセット）:")

    while True:
        try:
            question = input("\n[あなた] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if question in ("exit", "quit", "q"):
            print("終了します。")
            break

        if question == "reset":
            bot.reset()
            print("会話履歴をリセットしました。")
            continue

        if not question:
            continue

        try:
            answer = bot.chat(question)
            print(f"\n[回答]\n{answer}")
        except Exception as e:
            logger.error("エラー: %s", e)
            print(f"\nエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
