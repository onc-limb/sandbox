"""対話型 RAG チャット CLI."""

import argparse
import logging
import sys
import time
from pathlib import Path

from src.bot import UnifiedBot
from src.config import get_config
from src.options import LLM_MODELS, SEARCH_METHODS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 8
RETRY_BASE_WAIT = 30


def retry_on_unavailable(func, *args, **kwargs):
    """503/429 エラー時に指数的待機でリトライする."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            err_type = type(e).__name__
            is_retryable = (
                "503" in err_str
                or "429" in err_str
                or "ServiceUnavailable" in err_type
                or "RateLimitError" in err_type
            )
            if is_retryable:
                wait = RETRY_BASE_WAIT * attempt
                logger.warning(
                    "API エラー (試行 %d/%d)。%d秒後にリトライ... [%s]",
                    attempt,
                    MAX_RETRIES,
                    wait,
                    err_type,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"{MAX_RETRIES}回リトライしたが失敗")


def select_method() -> str:
    """対話式で検索方式を選択する."""
    print("\n検索方式を選択してください:")
    for i, m in enumerate(SEARCH_METHODS, 1):
        print(f"{i}. {m}")
    while True:
        choice = input("> ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(SEARCH_METHODS):
                return SEARCH_METHODS[idx]
        except ValueError:
            pass
        print(f"1〜{len(SEARCH_METHODS)} の番号を入力してください")


def select_version() -> str:
    """対話式でインデックスバージョンを選択する."""
    chroma_dir = Path("chroma_db")
    if not chroma_dir.exists():
        print("\n警告: chroma_db/ が存在しません。先に ingest.py を実行してください。")
        sys.exit(1)

    versions = sorted(
        [d.name for d in chroma_dir.iterdir() if d.is_dir()]
    )
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
    parser = argparse.ArgumentParser(description="対話型 RAG チャット")
    parser.add_argument("--model", help="LLM モデル名")
    parser.add_argument(
        "--method",
        choices=["rag", "fulltext", "hybrid"],
        help="検索方式",
    )
    parser.add_argument("--version", help="使用するインデックスバージョン")
    parser.add_argument("--doc", help="ドキュメントファイルパス (PDF or Markdown)")
    return parser.parse_args()


def main() -> None:
    """メイン処理."""
    args = parse_args()

    method = args.method if args.method else select_method()
    version = args.version if args.version else select_version()
    model = args.model if args.model else select_model()

    config_overrides = {
        "llm_model_name": model,
        "index_version": version,
    }
    if args.doc:
        config_overrides["doc_path"] = args.doc

    config = get_config(**config_overrides)

    include_full_text = method in ("fulltext", "hybrid")
    bot = UnifiedBot(config, include_full_text=include_full_text)

    print("\nインデックスをロード中...")
    bot.load_index()

    print("\n=== RAG Chat ===")
    print(f"モデル: {model}")
    print(f"方式: {method}")
    print(f"バージョン: {version}")
    print("\n質問を入力してください（exit で終了）:")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if question in ("exit", "quit", "q"):
            print("終了します。")
            break

        if not question:
            continue

        try:
            if method == "fulltext":
                answer = retry_on_unavailable(bot.answer_fulltext_only, question)
            else:
                answer = retry_on_unavailable(bot.answer, question)
        except Exception as e:
            print(f"\nエラー: {e}")
            continue

        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
