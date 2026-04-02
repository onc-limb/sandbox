"""3つのボット（RAG / 全文 / ハイブリッド）の回答を比較するスクリプト."""

import logging
import time

from pathlib import Path

from src.bot import UnifiedBot
from src.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

QUESTION = "今後のAI時代におけるRAGの活用場面について教えて"
PDF_PATH = "oreilly-978-4-8144-0138-3e.pdf"
OUTPUT_DIR = Path("output")

MAX_RETRIES = 8
RETRY_BASE_WAIT = 30  # seconds


def retry_on_unavailable(func, *args, **kwargs):
    """503エラー時にリトライする."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            err_type = type(e).__name__
            is_retryable = (
                "503" in err_str or "429" in err_str
                or "ServiceUnavailable" in err_type
                or "RateLimitError" in err_type
            )
            if is_retryable:
                wait = RETRY_BASE_WAIT * attempt
                logger.warning("API エラー (試行 %d/%d)。%d秒後にリトライ... [%s]", attempt, MAX_RETRIES, wait, err_type)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"{MAX_RETRIES}回リトライしたが失敗")


def run_rag_bot(question: str) -> str:
    """RAGモード（検索結果のみ）で回答を取得する."""
    logger.info("RAGBot: 開始")
    config = get_config(pdf_path=PDF_PATH)
    bot = UnifiedBot(config, include_full_text=False)
    bot.build_index()
    answer = bot.answer(question)
    logger.info("RAGBot: 完了")
    return answer


def run_fulltext_bot(question: str) -> str:
    """全文モード（RAG結果なし、全文のみ）で回答を取得する."""
    logger.info("FulltextBot: 開始")
    config = get_config(pdf_path=PDF_PATH)
    bot = UnifiedBot(config, include_full_text=True)
    bot.build_index()
    answer = bot.answer_fulltext_only(question)
    logger.info("FulltextBot: 完了")
    return answer


def run_hybrid_bot(question: str) -> str:
    """ハイブリッドモード（全文＋RAG検索結果）で回答を取得する."""
    logger.info("HybridBot: 開始")
    config = get_config(pdf_path=PDF_PATH)
    bot = UnifiedBot(config, include_full_text=True)
    bot.build_index()
    answer = bot.answer(question)
    logger.info("HybridBot: 完了")
    return answer


def save_result(filepath: Path, bot_name: str, question: str, answer: str) -> None:
    """回答を Markdown ファイルに保存する."""
    content = f"# {bot_name} の回答\n\n## 質問\n{question}\n\n## 回答\n{answer}\n"
    filepath.write_text(content, encoding="utf-8")
    logger.info("保存: %s", filepath)


def save_comparison(
    output_dir: Path,
    question: str,
    rag_answer: str,
    fulltext_answer: str,
    hybrid_answer: str,
) -> None:
    """3つの回答を比較する Markdown を生成する."""
    content = f"""# ボット比較結果

## 質問
{question}

---

## 方式の違い

| 方式 | 概要 |
|------|------|
| RAG | PDF をチャンク分割しベクトル検索で関連部分のみ取得して回答を生成 |
| 全文 | PDF 全文をシステムプロンプトに含めて回答を生成 |
| ハイブリッド | 全文 + ベクトル検索の関連チャンクを組み合わせて回答を生成 |

---

## RAGBot の回答

{rag_answer}

---

## 全文ボットの回答

{fulltext_answer}

---

## ハイブリッドボットの回答

{hybrid_answer}
"""
    filepath = output_dir / "comparison.md"
    filepath.write_text(content, encoding="utf-8")
    logger.info("保存: %s", filepath)


def load_cached_answer(filepath: Path) -> str | None:
    """既に保存済みの回答があれば読み込む."""
    if filepath.exists():
        text = filepath.read_text(encoding="utf-8")
        marker = "## 回答\n"
        idx = text.find(marker)
        if idx != -1:
            answer = text[idx + len(marker):].strip()
            if answer:
                logger.info("キャッシュ利用: %s", filepath)
                return answer
    return None


def main() -> None:
    """メイン処理."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("出力ディレクトリ: %s", OUTPUT_DIR.resolve())

    # RAGBot
    rag_path = OUTPUT_DIR / "rag_bot_result.md"
    rag_answer = load_cached_answer(rag_path)
    if rag_answer is None:
        rag_answer = retry_on_unavailable(run_rag_bot, QUESTION)
        save_result(rag_path, "RAGBot", QUESTION, rag_answer)

    # FulltextBot（レートリミット回避のため待機）
    logger.info("レートリミット回避のため60秒待機...")
    time.sleep(60)
    fulltext_path = OUTPUT_DIR / "fulltext_bot_result.md"
    fulltext_answer = load_cached_answer(fulltext_path)
    if fulltext_answer is None:
        fulltext_answer = retry_on_unavailable(run_fulltext_bot, QUESTION)
        save_result(fulltext_path, "全文ボット", QUESTION, fulltext_answer)

    # HybridBot（レートリミット回避のため待機）
    logger.info("レートリミット回避のため60秒待機...")
    time.sleep(60)
    hybrid_path = OUTPUT_DIR / "hybrid_bot_result.md"
    hybrid_answer = load_cached_answer(hybrid_path)
    if hybrid_answer is None:
        hybrid_answer = retry_on_unavailable(run_hybrid_bot, QUESTION)
        save_result(hybrid_path, "ハイブリッドボット", QUESTION, hybrid_answer)

    save_comparison(OUTPUT_DIR, QUESTION, rag_answer, fulltext_answer, hybrid_answer)

    logger.info("全ての比較が完了しました")


if __name__ == "__main__":
    main()
