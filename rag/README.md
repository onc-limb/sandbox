# RAG Evaluation System

PDF ドキュメントをベクトル DB に取り込み、RAG チャット・評価を行うシステム。

## セットアップ

```bash
uv sync
```

`.env` ファイルを作成し、Gemini API キーを設定する。

```
GEMINI_API_KEY=your_api_key_here
```

## コマンド一覧

### 1. `ingest.py` — PDF の取り込み

PDF をチャンク分割してベクトル DB（ChromaDB）に保存する。チャットや評価の前に必ず実行する。

```bash
uv run python ingest.py
```

**オプション:**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--pdf PATH` | `docs/oreilly-978-4-8144-0138-3e.pdf` | 取り込む PDF ファイルのパス |
| `--version NAME` | `default` | インデックスのバージョン名（`chroma_db/<NAME>/` に保存） |
| `--chunk-size N` | `1024` | チャンクサイズ（トークン数） |
| `--chunk-overlap N` | `200` | チャンク間のオーバーラップ |
| `--embedding-model NAME` | `BAAI/bge-m3` | 埋め込みモデル名 |
| `--force` | `false` | インデックスが存在してもリビルドする |

**例:**

```bash
# デフォルト設定で取り込み
uv run python ingest.py

# 別バージョンとして取り込み（チャンクサイズ変更）
uv run python ingest.py --version v2 --chunk-size 512 --chunk-overlap 100

# 別の PDF を強制リビルド
uv run python ingest.py --pdf docs/other.pdf --version custom --force
```

---

### 2. `chat.py` — 対話型 RAG チャット

インデックスを使って質問に対話的に回答する。

```bash
uv run python chat.py
```

引数を省略すると、起動時に対話形式で検索方式・バージョン・モデルを選択できる。

**オプション:**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--method {rag,fulltext,hybrid}` | 対話選択 | 検索方式 |
| `--version NAME` | 対話選択 | 使用するインデックスのバージョン |
| `--model NAME` | 対話選択 | LLM モデル名 |
| `--pdf PATH` | config デフォルト | PDF ファイルパス |

**検索方式:**

| 方式 | 説明 |
|---|---|
| `rag` | ベクトル検索のみ |
| `fulltext` | 全文テキストをコンテキストとして使用 |
| `hybrid` | 全文 + ベクトル検索の組み合わせ |

**例:**

```bash
# 対話式で設定を選択しながら起動
uv run python chat.py

# オプションをすべて指定して起動
uv run python chat.py --method hybrid --version default --model gemini/gemini-2.5-pro
```

チャット中は `exit` / `quit` / `q` で終了。

---

### 3. `evaluate.py` — RAG 評価の実行

質問セットを使って各検索方式の回答品質を評価し、結果を `output/exp_XXX/` に保存する。

```bash
uv run python evaluate.py
```

引数を省略すると、起動時に対話形式でメソッド・バージョン・モデルを選択できる。

**オプション:**

| オプション | デフォルト | 説明 |
|---|---|---|
| `--method {rag,fulltext,hybrid,all}` | 対話選択 | 評価する検索方式（`all` で全方式） |
| `--version NAME` | 対話選択 | 使用するインデックスのバージョン |
| `--model NAME` | 対話選択 | 回答生成に使う LLM モデル名 |
| `--eval-model NAME` | `--model` と同じ | 評価に使う LLM モデル名 |
| `--questions PATH` | `questions/default.json` | 質問セット JSON ファイルのパス |
| `--experiment-id ID` | 自動採番 | 実験 ID（`output/exp_<ID>/` に保存） |
| `--pdf PATH` | config デフォルト | PDF ファイルパス |

**例:**

```bash
# 対話式で設定を選択しながら実行
uv run python evaluate.py

# 全メソッドを一括評価
uv run python evaluate.py --method all --version default --model gemini/gemini-2.5-pro

# 特定の質問セットで rag のみ評価
uv run python evaluate.py --method rag --questions questions/custom.json

# 回答生成と評価で別モデルを使用
uv run python evaluate.py --method hybrid --model gemini/gemini-3-flash-preview --eval-model gemini/gemini-2.5-pro
```

**出力ファイル:**

評価結果は `output/exp_<ID>/` ディレクトリに以下のファイルとして保存される。

| ファイル | 内容 |
|---|---|
| `metadata.json` | 実験設定（モデル、バージョン、質問数など） |
| `results.json` | 全質問・全メソッドの回答とスコア |
| `summary.md` | メソッド別平均スコア表（Markdown） |

**評価指標:**

| 指標 | 説明 |
|---|---|
| `relevance` | 質問に対する回答の関連性 |
| `faithfulness` | コンテキストへの忠実度 |
| `completeness` | 回答の網羅性 |
| `conciseness` | 回答の簡潔さ |

---

## 質問セット

`questions/` ディレクトリに JSON 形式で管理する。

```json
[
  {
    "id": "q001",
    "question": "質問文",
    "reference_keywords": ["キーワード1", "キーワード2"]
  }
]
```

デフォルトは `questions/default.json`。`--questions` オプションで別ファイルを指定できる。
