# RAG アプリケーション 設計ドキュメント

## 概要

ユーザーの質問に対して、固定の日本語ドキュメントセット（図・表を含む）から回答を生成するアプリケーション。
検索元ドキュメントは増減しない（静的）。

---

## 技術スタック

| コンポーネント | 選定 |
|-------------|------|
| 言語 | Python 3.11+ |
| パッケージ管理 | uv |
| LLM | Gemini 2.5 Flash（Google AI Studio 無料枠） |
| 埋め込みモデル | `BAAI/bge-m3`（sentence-transformers でローカル実行） |
| ベクトルDB | ChromaDB（ローカル永続化） |

---

## アーキテクチャ

### アプローチ選択: RAG vs 全文システムプロンプト

Gemini 2.5 Flash はコンテキスト長が非常に大きいため、まず全文投入でベースラインを測る。

| ドキュメント総量 | アプローチ |
|---------------|----------|
| 小〜中規模 | 全文システムプロンプト（シンプル・確実） |
| 大規模 | RAG |

> **検証戦略**: まず全文投入で実装 → ベースライン品質を確認 → 必要に応じて RAG に移行

### RAG パイプライン

```
[インデックス構築（初回のみ）]
docs/ のファイル群
  → チャンク分割（見出し単位 or 固定サイズ）
  → bge-m3 でベクトル化（ローカル）
  → ChromaDB に永続化

[クエリ時]
ユーザー質問
  → bge-m3 でベクトル化（ローカル）
  → ChromaDB で類似検索（Top-K チャンク取得）
  → Gemini 2.5 Flash に渡す
  → 回答
```

### 全文システムプロンプト パイプライン

```
[クエリ時]
docs/ の全ファイルを読み込み
  → システムプロンプトに全文挿入
  → Gemini 2.5 Flash に渡す
  → 回答
```

---

## 図・表の扱い

| ドキュメント形式 | 表 | 図 |
|---------------|----|----|
| Markdown (.md) | テキストとして扱える（Gemini は MD テーブルを解釈） | alt テキストのみ。画像自体は別途渡す必要あり |
| PDF | `pymupdf` + `pdfplumber` でテキスト抽出。レイアウト再現は限定的 | 画像として抽出 → Gemini Vision で解釈可能 |

---

## ディレクトリ構成

```
rag/
├── design.md
├── pyproject.toml           # uv で管理
├── .env                     # GOOGLE_API_KEY（git 管理外）
├── .gitignore
├── docs/                    # 検索対象ドキュメント（固定）
│   └── *.md / *.pdf
├── chroma_db/               # ChromaDB 永続化（自動生成・git 管理外）
├── notebooks/
│   ├── 01_fulltext.ipynb    # 全文システムプロンプト実験
│   └── 02_rag.ipynb         # RAGパイプライン実験
└── src/
    ├── full_context.py      # 全文投入アプローチ
    └── rag_pipeline.py      # RAGアプローチ
```

---

## セットアップ

### API キー取得

1. [Google AI Studio](https://aistudio.google.com/) で無料 API キーを発行
2. `.env` に記述:

```env
GOOGLE_API_KEY=AIza...
```

### 依存ライブラリ

| ライブラリ | 用途 |
|----------|------|
| `google-genai` | Gemini 2.5 Flash API クライアント（新 SDK） |
| `sentence-transformers` | bge-m3 ローカル実行 |
| `chromadb` | ローカルベクトルDB |
| `langchain` | RAGパイプライン構築 |
| `langchain-google-genai` | LangChain × Gemini |
| `langchain-community` | ChromaDB / ドキュメントローダー連携 |
| `pymupdf` | PDF 読み込み（PDF ドキュメントの場合） |
| `pdfplumber` | PDF の表抽出（必要な場合） |
| `python-dotenv` | `.env` 読み込み |
| `jupyter` | 検証・可視化 |

```bash
uv init --no-readme
uv add google-genai sentence-transformers chromadb \
        langchain langchain-google-genai langchain-community \
        pymupdf pdfplumber python-dotenv jupyter
```

---

## 検証ステップ

1. `docs/` にドキュメントを配置
2. ドキュメント総量を確認してアプローチを決定
3. `01_fulltext.ipynb` で全文投入 → ベースライン品質を確認
4. `02_rag.ipynb` で bge-m3 インデックス構築 → ChromaDB 検索 → Gemini で回答
5. 同じ質問セットで精度・速度・コストを比較

---

## 未決事項

- ドキュメントのフォーマット（Markdown / PDF / HTML）が確定したらローダーを決定する
- チャンク分割戦略（サイズ・オーバーラップ）はドキュメント構造次第
- 図が重要な情報源の場合、Gemini Vision による画像解釈も検討
