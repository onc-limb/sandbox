"""CLI付き評価実行スクリプト.

各method（rag/fulltext/hybrid）で質問に回答し、Evaluatorで評価して結果を保存する。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.retrieval.bot import UnifiedBot
from src.config import Config, get_config
from src.options import LLM_MODELS, SEARCH_METHODS
from src.eval.evaluator import Evaluator
from src.eval.criteria import DEFAULT_CRITERIA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 8
RETRY_BASE_WAIT = 30  # seconds
METHOD_WAIT = 60  # seconds between methods
METHODS = tuple(SEARCH_METHODS)
CRITERIA = tuple(c.name for c in DEFAULT_CRITERIA)

OUTPUT_DIR = Path("output")


def retry_on_unavailable(func, *args, **kwargs):
    """503/429エラー時に指数的待機でリトライする."""
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
                or "Failed to parse LLM evaluation response" in err_str
            )
            if is_retryable:
                wait = RETRY_BASE_WAIT * attempt
                logger.warning(
                    "API エラー (試行 %d/%d)。%d秒後にリトライ... [%s]",
                    attempt, MAX_RETRIES, wait, err_type,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"{MAX_RETRIES}回リトライしたが失敗")


def next_experiment_id() -> str:
    """output/ 内の既存 exp_* から次のIDを3桁ゼロ埋めで返す."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    existing = [
        d.name for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and re.match(r"^exp_\d{3,}$", d.name)
    ]
    if not existing:
        return "001"
    max_num = max(int(name.split("_")[1]) for name in existing)
    return f"{max_num + 1:03d}"


def generate_answer(
    config: Config, method: str, question: str,
) -> tuple[str, str]:
    """指定methodで回答を生成し、(answer, context) を返す."""
    if method == "rag":
        bot = UnifiedBot(config, include_full_text=False)
        bot.load_index()
        results = bot._searcher.search(question)
        context = "\n\n".join(doc.page_content for doc in results)
        answer = retry_on_unavailable(bot.answer, question)
    elif method == "fulltext":
        bot = UnifiedBot(config, include_full_text=True)
        bot.load_index()
        context = bot._full_text
        answer = retry_on_unavailable(bot.answer_fulltext_only, question)
    elif method == "hybrid":
        bot = UnifiedBot(config, include_full_text=True)
        bot.load_index()
        results = bot._searcher.search(question)
        context = bot._full_text + "\n\n" + "\n\n".join(
            doc.page_content for doc in results
        )
        answer = retry_on_unavailable(bot.answer, question)
    else:
        raise ValueError(f"Unknown method: {method}")
    return answer, context


def build_summary(results: list[dict], methods: list[str]) -> str:
    """method毎の平均スコアと各質問のスコア一覧表をMarkdownで生成する."""
    lines = ["# Evaluation Summary\n"]

    # method毎の平均スコア
    lines.append("## Average Scores by Method\n")
    lines.append("| Method | Relevance | Faithfulness | Completeness | Conciseness | Average |")
    lines.append("|--------|-----------|--------------|--------------|-------------|---------|")
    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        if not method_results:
            continue
        avgs = {}
        for c in CRITERIA:
            scores = [r["scores"][c]["score"] for r in method_results]
            avgs[c] = sum(scores) / len(scores)
        overall = sum(avgs.values()) / len(avgs)
        lines.append(
            f"| {method} | {avgs['relevance']:.2f} | {avgs['faithfulness']:.2f} "
            f"| {avgs['completeness']:.2f} | {avgs['conciseness']:.2f} | {overall:.2f} |"
        )

    # 各質問のスコア一覧
    lines.append("\n## Scores by Question\n")
    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        if not method_results:
            continue
        lines.append(f"### {method}\n")
        lines.append("| Question ID | Relevance | Faithfulness | Completeness | Conciseness |")
        lines.append("|-------------|-----------|--------------|--------------|-------------|")
        for r in method_results:
            s = r["scores"]
            lines.append(
                f"| {r['question_id']} | {s['relevance']['score']} "
                f"| {s['faithfulness']['score']} | {s['completeness']['score']} "
                f"| {s['conciseness']['score']} |"
            )
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """CLIフラグをパースする."""
    config_defaults = Config.__dataclass_fields__
    parser = argparse.ArgumentParser(description="RAG evaluation runner")
    parser.add_argument(
        "--model",
        default=None,
        help=f"LLM model name (default: {config_defaults['llm_model_name'].default})",
    )
    parser.add_argument(
        "--method",
        default="all",
        choices=[*SEARCH_METHODS, "all"],
        help="Evaluation method (default: all)",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment ID (default: auto-increment)",
    )
    parser.add_argument(
        "--questions",
        default="questions/default.json",
        help="Path to questions JSON file (default: questions/default.json)",
    )
    parser.add_argument(
        "--eval-model",
        default=None,
        help="LLM model for evaluation (default: same as --model)",
    )
    parser.add_argument(
        "--doc",
        default=None,
        help=f"Document file path (PDF or Markdown) (default: {config_defaults['doc_path'].default})",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Index version directory in chroma_db/ (default: 'default')",
    )
    return parser.parse_args()


def _list_versions() -> list[str]:
    """chroma_db/ 内のバージョンディレクトリ一覧を返す."""
    chroma_root = Path(Config.__dataclass_fields__["chroma_persist_dir"].default)
    if not chroma_root.exists():
        return ["default"]
    dirs = sorted(
        d.name for d in chroma_root.iterdir()
        if d.is_dir()
    )
    return dirs if dirs else ["default"]


def _select(prompt: str, options: list[str], *, allow_custom: bool = False) -> str:
    """番号選択式の対話入力。allow_custom=Trueなら最後にカスタム入力を追加."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    if allow_custom:
        print(f"  {len(options) + 1}. カスタム入力")
    while True:
        raw = input("番号を入力: ").strip()
        if not raw.isdigit():
            print("数字を入力してください。")
            continue
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1]
        if allow_custom and idx == len(options) + 1:
            return input("値を入力: ").strip()
        print(f"1〜{len(options) + (1 if allow_custom else 0)} の範囲で入力してください。")


def _interactive_select(args: argparse.Namespace) -> argparse.Namespace:
    """未指定のオプションを対話式で選択させる.

    全オプションが指定済みならスキップする。
    """
    all_set = (
        _was_explicitly_set("method")
        and _was_explicitly_set("version")
        and _was_explicitly_set("model")
        and _was_explicitly_set("eval_model")
    )
    if all_set:
        return args

    # method
    if not _was_explicitly_set("method"):
        args.method = _select(
            "評価メソッドを選択:",
            ["rag", "fulltext", "hybrid", "all"],
        )

    # version
    if args.version is None:
        versions = _list_versions()
        args.version = _select("インデックスバージョンを選択:", versions)

    # model
    if args.model is None:
        args.model = _select(
            "回答生成モデルを選択:",
            LLM_MODELS,
            allow_custom=True,
        )

    # eval-model
    if args.eval_model is None:
        choice = _select(
            "評価モデルを選択:",
            [f"回答生成モデルと同じ ({args.model})", "別モデルを入力"],
        )
        if choice.startswith("回答生成モデルと同じ"):
            args.eval_model = args.model
        else:
            args.eval_model = input("評価モデル名を入力: ").strip()

    return args


def _was_explicitly_set(dest: str) -> bool:
    """CLIフラグが明示的に指定されたかをsys.argvから判定する."""
    flag_map = {
        "method": ("--method",),
        "version": ("--version",),
        "model": ("--model",),
        "eval_model": ("--eval-model",),
    }
    flags = flag_map.get(dest, ())
    return any(arg in sys.argv for arg in flags)


def main() -> None:
    """メイン処理."""
    args = parse_args()

    # 対話式選択（未指定オプションがある場合）
    args = _interactive_select(args)

    # Config構築
    config_overrides = {}
    if args.model:
        config_overrides["llm_model_name"] = args.model
    if args.doc:
        config_overrides["doc_path"] = args.doc
    if args.version:
        config_overrides["index_version"] = args.version
    config = get_config(**config_overrides)

    # 実験ID
    exp_id = args.experiment_id or next_experiment_id()
    exp_dir = OUTPUT_DIR / f"exp_{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("実験ディレクトリ: %s", exp_dir)

    # 質問読み込み
    questions_path = Path(args.questions)
    if not questions_path.exists():
        logger.error("質問ファイルが見つかりません: %s", questions_path)
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    with questions_path.open(encoding="utf-8") as f:
        questions = json.load(f)
    logger.info("質問数: %d", len(questions))

    # method決定
    methods = list(METHODS) if args.method == "all" else [args.method]

    # eval_model
    eval_model = args.eval_model if args.eval_model else (args.model or None)

    # Evaluator
    evaluator = Evaluator(config, eval_model=eval_model)

    # metadata保存
    metadata = {
        "experiment_id": exp_id,
        "model": config.llm_model_name,
        "eval_model": eval_model or config.llm_model_name,
        "methods": methods,
        "questions_file": str(questions_path),
        "index_version": config.index_version,
        "doc_path": config.doc_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(questions),
    }
    (exp_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # 実行
    all_results = []
    for method_idx, method in enumerate(methods):
        if method_idx > 0:
            logger.info("レートリミット回避のため%d秒待機...", METHOD_WAIT)
            time.sleep(METHOD_WAIT)

        logger.info("=== Method: %s ===", method)
        for q in questions:
            qid = q["id"]
            question = q["question"]
            logger.info("[%s] %s: 回答生成中...", method, qid)

            answer, context = retry_on_unavailable(
                generate_answer, config, method, question,
            )

            logger.info("[%s] %s: 評価中...", method, qid)
            scores = retry_on_unavailable(
                evaluator.evaluate, question, answer, context,
            )

            result_entry = {
                "question_id": qid,
                "question": question,
                "method": method,
                "answer": answer,
                "scores": scores,
            }
            all_results.append(result_entry)
            logger.info(
                "[%s] %s: scores=%s",
                method, qid,
                {c: scores[c]["score"] for c in CRITERIA},
            )

    # results.json 保存
    (exp_dir / "results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # summary.md 保存
    summary = build_summary(all_results, methods)
    (exp_dir / "summary.md").write_text(summary, encoding="utf-8")

    logger.info("完了: %s", exp_dir)


if __name__ == "__main__":
    main()
