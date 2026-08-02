# Rust 学習（バッチ処理 3 題材）

[`../specs/elixir-rust-batch-subjects.md`](../specs/elixir-rust-batch-subjects.md) を題材にした
Rust の学習用ディレクトリ。Elixir 側（`../elixir/`）と同じ仕様を実装し、書き味を比較する。

- ツールチェーン: rustup 管理の Rust（mise にも登録済み）。`cargo --version` で確認。
- 題材ごとに `cargo new` で独立プロジェクトを作る（Elixir 側の mix プロジェクト構成と対応）。

## 進め方（全体）

| 順番 | 題材 | ディレクトリ | 心臓部 | 目安 |
|---|---|---|---|---|
| 1 | 単語頻度カウント | `word_freq/` | 所有権の基礎・コレクション・イテレータ | 1〜2 セッション |
| 2 | 並行データ収集 | `fetcher/`（未作成） | tokio・Result 連鎖・エラー型設計 | 3〜4 セッション |
| 3 | ログのストリームETL | `log_etl/`（未作成） | Iterator の遅延評価・借用でコピー回避 | 2〜3 セッション |

各題材の仕様・完了の定義・評価チェックリストは spec を正とする（ここには書き写さない）。
題材が終わるたびに spec 付録の形式で比較メモを残す（`notes/rust-<バージョン>.md` を想定）。

## Step 0: 足慣らしの前に（30分）

- `cargo run` / `cargo check` / `cargo clippy` / `cargo fmt` / `cargo test` の 5 コマンドを覚える。
  普段のループは `cargo check`（速い）→ 動かすときだけ `cargo run`。
- エディタに rust-analyzer を入れる。コンパイルエラーは `rustc --explain E0382` のように
  エラーコードで解説が読める。
- 参考書は The Rust Programming Language（通称 the book, 日本語版あり）。
  通読せず、下の対応表の章だけ都度読む。

## Step 1: word_freq（題材1）

**先に読む章**: the book 4章（所有権）・6章（enum と match）・8章（コレクション）・9章（エラー処理）

**Rust 側の学習ポイント**（spec「見るポイント」の具体化）:

- `String` と `&str` の使い分け。分割した単語をどちらで持つか迷うところが学びどころ
- `HashMap` の集計は `*map.entry(word).or_insert(0) += 1` のイディオム
- `Vec<(String, u32)>` にして `sort_by_key` / `sort_by` で降順ソート
- ファイル読み込み〜main の返り値まで `Result` + `?` 演算子で通す
  （`fn main() -> Result<(), Box<dyn Error>>` から始めてよい）
- CLI 引数はまず `std::env::args` だけでやってみる（チェックリストに「標準機能で足りたか」
  という設問があるので、clap に逃げる前に一度素で書く）

**やる順**（一気に書かず、動くものを段階的に）:

1. ファイルパス固定・ハードコードで「読んで分割して数えて上位10件表示」まで
2. 正規化（小文字化・記号除去）を足す → `char::is_alphanumeric` あたりで完結するか試す
3. CLI 引数（パス・上位N件・ストップワード除外フラグ）を足す
4. エラーハンドリング（ファイル不在・空ファイル）を整えて異常終了コードを返す
5. spec の「完了の定義」を満たしたら評価チェックリストを埋める

入力テキストは Elixir 側 `../elixir/word_freq/priv/` の生成物を流用してよい
（同じ入力なら出力を突き合わせて答え合わせできる）。

## Step 2: fetcher（題材2）

**先に読む章**: the book 10章（ジェネリクス・トレイト）・13章（クロージャとイテレータ）、
Tokio 公式チュートリアル（tokio.rs/tokio/tutorial）の最初の数章

- 依存: `tokio`（full feature でよい）・`reqwest`・`serde`/`serde_json`・`thiserror`
- 並行制御は `futures::stream::iter(...).buffer_unordered(n)` か `JoinSet` + `Semaphore` の
  2 通りがある。両方書いて比較すると Elixir の `Task.async_stream` との対応が見えやすい
- タイムアウトは `tokio::time::timeout`、バックオフはまず自前ループで書く
- カスタムエラー型（`thiserror`）で「タイムアウト/HTTPエラー/パース失敗」を enum にするのが
  この題材の Rust 的な山場
- モックサーバーは同じ workspace 内に `axum` で別 bin を立てる
  （`/ok` `/slow?ms=` `/error` `/flaky` の 4 エンドポイント程度）

## Step 3: log_etl（題材3）

**先に読む章**: the book 13章の後半（イテレータの遅延評価）を再読

- `BufReader::lines()` で 1 行ずつ処理し、一括 `read_to_string` 版と比較する
- メモリ計測は macOS なら `/usr/bin/time -l ./target/release/log_etl ...`
  （`maximum resident set size` を見る。必ず `--release` ビルドで比較）
- パースは最初は `split(',')` の手書きで。借用（`&str` のまま処理してどこで `String` に
  するか）を意識するのが spec チェックリストの狙い。慣れたら `csv` + `serde` 版と比較
- 複数軸の集計は「HashMap を複数持つ struct + `update(&mut self, record)`」の形が素直

## 学習の運用ルール

- **コードは自分で書く**。Claude には「コンパイルエラーの解説」「書いたコードのレビュー」
  「Elixir だとどう書くかの対比」を頼み、実装の丸投げはしない
- 詰まって 30 分進まなかったら答えを聞いてよい。ただしその箇所はメモに残す
- 各題材の完了時に `cargo clippy` を通し、警告の意味を一つずつ理解してから直す
