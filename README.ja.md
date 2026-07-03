[English](README.md) | **日本語**

# engram — AIエージェント用 人間型記憶基盤(MCPサーバー)

Claude Code / Codex / Antigravity(Gemini CLI)が共有する永続記憶。
使うほど思い出しやすくなり、使わない記憶は沈むが消えない — 人間の記憶と同じ性質を持つ。

---

## クイックスタート(受け取った方へ)

### 方法1: 一発インストール

Windows(PowerShell):

```powershell
irm https://raw.githubusercontent.com/ricoaiproject-cmd/engram/main/install.ps1 | iex
```

macOS / Linux:

```bash
curl -LsSf https://raw.githubusercontent.com/ricoaiproject-cmd/engram/main/install.sh | sh
```

これ1行で uv のインストール・engram のインストール・セットアップウィザードまで実行されます。
(macOS は git が必要です。無い場合は先に `xcode-select --install` を実行してください)

### 方法2: 3コマンドで手動インストール

Windows(PowerShell):

```powershell
# 1. uv をインストール(既にある場合はスキップ)
irm https://astral.sh/uv/install.ps1 | iex

# 2. engram をインストール
uv tool install --python 3.12 git+https://github.com/ricoaiproject-cmd/engram.git

# 3. セットアップウィザードを実行
engram setup
```

macOS / Linux:

```bash
# 1. uv をインストール(既にある場合はスキップ)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. engram をインストール(uv 管理の Python を強制 — 下の注記参照)
UV_PYTHON_PREFERENCE=only-managed uv tool install --python 3.12 git+https://github.com/ricoaiproject-cmd/engram.git

# 3. セットアップウィザードを実行
engram setup
```

> なぜ uv 管理の Python? engram は SQLite の拡張ロード(sqlite-vec が使用)に
> 対応した Python を必要とします。macOS 標準や python.org の Python は非対応で、
> しかも存在すると uv はそちらを優先してしまうため、
> `UV_PYTHON_PREFERENCE=only-managed` で uv 管理の Python を強制します
> (install.sh は自動で設定します)。`engram doctor` に確認項目があります。

セットアップウィザードが以下を自動で行います:
- 設定ファイル(`~/.engram/config.toml`)の作成
- 記憶フォルダの初期化
- 埋め込みモデルのダウンロード(初回のみ、約500MB)
- Claude Code / Codex / Antigravity への自動登録
- フック(自動符号化・自発的想起)の登録(Claude Code)

---

## インストール後の使い方

### エージェントに話しかけるだけ

engram のすべての操作はエージェントが自動的に行います。
ユーザーは普通に会話するだけで記憶が蓄積・活用されます。

### 最初にオンボーディングインタビューを受ける(推奨)

エージェントに次のように依頼してください:

```
~/.engram/ONBOARDING.md を読んで、私にインタビューして。
```

仕事の流儀・好み・背景情報を初期登録することで、すぐに記憶が活きはじめます。

### 環境の確認

```powershell
engram doctor
```

Python バージョン・設定ファイル・モデルキャッシュ・埋め込み実行系(ONNX / torch)・
インストール健全性(pip 再インストール失敗の残骸 `~ngram` 等による import 不能の検知)・
各エージェントへの登録状況を `[OK]` / `[NG]` / `[--]` で一覧表示するほか、
FTS5(SQLite の全文検索拡張)が読み込めているか(欠けるとキーワード検索が
黙って劣化するため)と、`data_dir/perf/perf_log.jsonl`(下記)に記録された
直近の MCP ツール呼び出し・起動時間の要約行も表示します。
「なんとなく遅い」を勘ではなくデータで診断できます。

### 再セットアップ(新しいエージェントをインストールした後など)

```powershell
engram setup
```

何度実行しても安全(冪等)です。未登録のエージェントだけが追加されます。

### 登録先のエージェントを選ぶ

複数のエージェントが入っていても、engram を繋ぎたいものだけを選べます。

```powershell
# Claude Code だけに登録する
engram setup --agents claude

# Claude Code と Codex の両方に登録する
engram setup --agents claude,codex
```

有効な名前: `claude` / `codex` / `gemini`(`antigravity` は `gemini` の別名)。
`--agents` を省略した場合、対話モード(インタラクティブ)では検出されたエージェントが一覧表示され、番号で選べます。Enter を押すとすべてに登録されます。`--non-interactive` では従来どおり検出された全エージェントに自動登録されます。

### ONNX による起動高速化(v0.6 の新機能)

一度だけ実行してください:

```powershell
engram export-onnx
```

埋め込みモデルを ONNX に変換します(追加の依存は不要 — 変換はインストール済みの
torch が一度だけ担当)。以後サーバーは自動的に ONNX を使い(`embed_backend=auto`)、
起動が 12〜24秒(torch import)から **約2秒** になります。MCP クライアント側の
タイムアウト調整も不要になります。

安全装置: 変換時に torch 経路と ONNX 経路で同じサンプル文(ModernBERT の
スライディングウィンドウ境界を越える長文を含む)を埋め込み、コサイン類似の
最小値が 0.999 を下回るモデルはインストールを拒否します — 分布がずれた
埋め込み空間を黙って採用すると、既存の `index.db` に対する recall が静かに
壊れるためです。

`config.toml` の `embed_backend`(または環境変数 `ENGRAM_EMBED_BACKEND`)で
実行系を選べます: `auto`(既定。ONNX 生成済みならそれを使い、無ければ torch)/
`onnx`(強制。未生成ならエラー)/ `torch`(フォールバックを強制)。

### 起動モード(`ENGRAM_PRELOAD`)

ONNX 実行系では先読みが軽く、既定の `blocking` のままで調整は不要です。
以下の数字は **torch フォールバック**(ONNX 未生成)時のもので、この設計に
至った理由の記録として残しています。環境変数 `ENGRAM_PRELOAD` でロードの
タイミングを選べます。

| 値 | 挙動 |
|---|---|
| `blocking`(既定) | ハンドシェイク前にメインスレッドでモデルを読み込む(warm 12〜24秒 / cold 50秒超)。接続後の `recall` は常に即応答。クライアント側の MCP 起動タイムアウトを120秒以上に延長すること(Claude Code なら `MCP_TIMEOUT=120000`)。 |
| `background` | ハンドシェイクに即応答し、モデルは裏スレッドで読み込む。**注意:** Windows では asyncio イベントループ稼働中の別スレッド torch import が病的に遅く(実測: メインスレッド約20秒 → 別スレッド約184秒)、初回 `recall` がクライアントのツールタイムアウトを超えうる。起動タイムアウトを一切延ばせない場合のみの妥協案。 |
| `off` | 先読みしない。初回ツール呼び出し時に遅延ロードする(`background` と同じ遅いスレッド import 経路を踏む)。 |

起動時に engram が接続失敗する場合は、`background` に切り替えるのではなく、クライアント側の MCP 起動タイムアウトを延長してください(Claude Code なら `MCP_TIMEOUT=120000`)。`background` への切り替えは「見える起動タイムアウト」を「初回 recall の3分ハング」に変換するだけです。

---

## もっと記憶らしく(v0.3 の新機能)

### 自動符号化 — セッションが勝手に記憶になる

Claude Code のセッションが終わると、フックがその回のやり取りを要約して
episode 記憶として自動保存します(`engram setup` でフックも自動登録)。
remember を呼び忘れても「昨日何をしたか」が残ります。
`config.toml` で `auto_encode = false` にすると無効化できます。

### 自発的想起 — 記憶のほうから思い出す

あなたが何か発言するたび、フックが軽量検索(埋め込みモデルは使わない高速経路)で
関連記憶を探します。動作モードは `config.toml` の `surface_mode` で切替:

| モード | 動作 |
|---|---|
| `shadow`(初期値) | 差し込まず「差し込むならこれ」をログに記録(様子見・調整用) |
| `active` | 関連が強い記憶をエージェントの文脈に実際に差し込む |
| `off` | 何もしない |

ログは `~/.engram/surface/surface_log.jsonl`。しばらく shadow で観察し、
浮上候補が妥当になってから `active` に切り替える運用を推奨します。
`engram surface "テキスト"` で何が浮上するかを手動確認できます。

調整パラメータ: `surface_threshold`(浮上スコア閾値、初期値 0.45)/
`surface_min_relevance`(関連度の最低ライン、初期値 0.25 — どれだけ重要な
記憶でも発話と無関係なら浮上させないゲート)/
`surface_max_items`(1回の最大件数、初期値 2)。

### 記憶の部屋 — 仕事と個人の文脈分離

記憶に `room`(部屋)ラベルが付きます。`config.toml` でフォルダと部屋を
対応付けると、作業ディレクトリから自動判定されます:

```toml
[room_paths]
'C:/Users/you/work-projects' = 'work'
'C:/Users/you/personal' = 'personal'
```

- 対応付けのないフォルダ・既存の記憶はすべて `common`(共通)
- recall は「現在の部屋 + common」だけを検索(`room="*"` で全部屋横断)
- 自動符号化・自発的想起も部屋を尊重するため、仕事の記憶が個人の文脈に
  紛れ込まない(その逆も)

### マシン間での記憶共有

記憶の Markdown ストアはクラウド同期フォルダ等に置いて複数マシンで共有できますが、
`index.db` はマシンごとにローカルです。そのため、あるマシンで書いた記憶は、別マシンが
索引するまでそのマシンの recall に出てきません。MCP サーバーは起動時にこれを検知します
(`config.toml` の `startup_index_check`): `auto`(既定)は Markdown と index の乖離を
検知すると自動 reindex、`warn` は警告ログのみ、`off` は無効。手動で `engram reindex` も可。

---

## 記憶の仕組み

### 設計の核心

埋め込み(意味の位置)は固定し、**活性度という別軸**で検索順位を変調します。
意味=どこにあるか、活性度=どれだけ思い出しやすいか、を分離しています。

### 人間の記憶と同じ性質

- **使うほど思い出しやすくなる** — ACT-R 活性化モデル。エージェントが使うたびに自動強化
- **使わない記憶は沈むが消えない** — べき乗則の減衰。deep recall で連想リンクを辿れば必ず到達
- **印象的な文脈の記憶は深く刻まれる** — importance による初期符号化ブースト + 低減衰 = フラッシュバルブ記憶
- **訂正された誤りは最も深く刻まれる** — correct ツール。間違えた経験ごと記録 = ハイパーコレクション効果

### 記憶力学の要点

- 活性度: `B = ln(Σ w_j·(now−t_j)^(−d_i))` をシグモイドで 0..1 に正規化。アクセスログから都度計算
- `d_i = clamp(0.5 − 0.2·(imp−5)/5, 0.3, 0.6)` — importance が高いほど忘れにくい
- create イベント重み `1 + 2·(imp/10)` — 重大な記憶は生まれた時から強い
- recall されただけ: weight 0.3 / 実際に役立った(reinforce): weight 1.0×strength
- 同時に reinforce された記憶同士は co_recall リンクで結合(ヘッブ則)し、deep recall の拡散活性化で辿れる連想網が育つ

### 検索

ベクトル近傍(Ruri-v3 埋め込み) + BM25 全文検索を RRF で統合し、
`0.6·関連度 + 0.25·活性度 + 0.15·重要度` で再ランクします。

#### ハイブリッド検索の改善: 完全一致トークンが埋もれなくなった

候補の relevance は、FTS ヒットをベクトル類似度のスケールへ無理やり写像する
のではなく、2つの検索路の結果を素直に合成するようになりました。ベクトル
ヒットはそのままコサイン類似度を、FTS ヒットは BM25 から直接導いた字句関連度
`1 - exp(bm25)`(`bm25 >= 0` のときは 0)を採り、両方にヒットした id は
大きい方を採用します。記憶 ID・ファイルパス・エラーコードなどの希少で
決定的な完全一致は bm25 が大きく負の値になり `lex` がほぼ 1.0 まで浮上する
ため、Ruri-v3 のコサイン類似度が 0.8〜0.87 に圧縮されがちな値域を越えられます。
旧実装では FTS のみのヒットに「候補中最小のベクトル類似度」を一律で割り当てて
いたため、明らかに正解であっても完全一致の結果が順位表の最下位に沈んでいました。

### 記憶の種類

| type | 内容 |
|---|---|
| knowledge | 知見・問題の解法・ツールの使い方 |
| preference | ユーザーの好み・流儀・指示の傾向 |
| project | 仕事の目的・制約・経緯・背景 |
| episode | セッションでやったことの要約 |

### ファイル構成

```
~/.engram/
  config.toml        設定ファイル(engram setup が生成)
  index.db           SQLite インデックス(Markdown から reindex で再構築可能)
  MEMORY_PROTOCOL.md エージェント運用指示(各エージェントの指示ファイルに組み込まれる)
  ONBOARDING.md      初期インタビュー台本
  surface/           自発的想起のログとセッション状態
  hooks.log          フックの動作記録
  consolidation_state.json  統合候補クラスタ数と最終促し時刻(統合の自動促し用)
  perf/perf_log.jsonl       MCP ツール呼び出し・起動時間のログ(perf_log = true 時)

<memories_dir>/      記憶の正本 Markdown(Obsidian でそのまま開ける・編集可)
  knowledge/
  preferences/
  projects/
  episodes/YYYY/MM/
  _trash/
```

`memories_dir` はデフォルト `~/.engram/memories` ですが、Google Drive や OneDrive の
同期フォルダを指定することでバックアップと複数デバイス共有が可能です。
SQLite インデックスは常にローカルに置かれるので同期競合リスクはありません。

### MCP ツール一覧

| ツール | いつ使う |
|---|---|
| `recall(query, mode, limit, type, room)` | タスク開始時。fast=通常 / deep=連想リンク・cold層・episodeまで探索 / exhaustive=活性度を無視し関連度のみで全件総当たり(沈んだ記憶の掘り起こし) |
| `remember(content, type, importance, tags, related_ids, room)` | 知見・好み・文脈・出来事を得た時。importance 1-10 は文脈の重大さで採点 |
| `reinforce(ids, strength)` | タスク完了時、実際に役立った記憶を報告(定着の栄養) |
| `correct(id, corrected_content, reason)` | 記憶が誤っていた時。forget ではなくこれ(誤りの経験ごと深く刻む) |
| `link` / `forget` / `stats` / `reindex` | 補助操作 |
| `consolidation_candidates` / `mark_consolidated` | 統合(下記) |

### 統合(睡眠に相当する処理)

古い episode 記憶をクラスタ化して知識へ昇華します。サーバーは候補を返すだけで、
要約は LLM(エージェント)が行います。夜間実行の例:

```powershell
claude -p "engram の consolidation_candidates を呼び、各クラスタを要約して remember(type=knowledge または project, related_ids=元episode)し、mark_consolidated で完了させて。最後に stats を報告して。"
```

#### 統合の自動促し(consolidation nudge)

上記の cron 的な夜間実行に加えて、engram はスケジュールジョブなしにエージェント
自身へ統合を促します:

1. **SessionEnd** が統合候補クラスタ数(`consolidation_candidates` 相当)を
   数え、`data_dir/consolidation_state.json` に記録します。
2. **UserPromptSubmit**(自発的想起と同じ軽量フック)が次のセッションでこの
   状態を確認し、クラスタが十分に溜まっていて前回の促しから十分な時間が
   経っていれば、`additionalContext` として
   「consolidation_candidates → remember → mark_consolidated を区切りの
   良いところで実行してほしい」という促し文を差し込みます。この促しは
   `surface_mode = "off"` でも発火します(自発的想起とは独立した仕組みです)。

`config.toml` (または `ENGRAM_*` 環境変数) の3つの設定で制御します:

| 設定 | 既定値 | 意味 |
|---|---|---|
| `consolidate_nudge` | `true` | 促し機能全体のスイッチ |
| `consolidate_nudge_min_clusters` | `3` | 促す最小クラスタ数 |
| `consolidate_nudge_interval_days` | `7.0` | 促しの最短間隔(日) |

---

## 開発者向け

このリポジトリで開発する場合のセットアップ(リポジトリのルートで実行):

```powershell
# 仮想環境(開発者用。配布時は uv tool install を使う)
python -m venv "$env:USERPROFILE\.engram\venv"
& "$env:USERPROFILE\.engram\venv\Scripts\python.exe" -m pip install -e ".[dev]"
```

### テストと検証

```powershell
$py = "$env:USERPROFILE\.engram\venv\Scripts\python.exe"

& $py -m pytest                          # 全テスト
& $py -m pytest tests\test_setup.py -q   # セットアップ関連のみ
& $py scripts\simulate.py                # アクセスパターン模擬(30日)
& $py scripts\check_mcp_e2e.py           # MCP E2E 確認
```

### 環境診断・CLI(動作確認用)

```powershell
$engram = "$env:USERPROFILE\.engram\venv\Scripts\engram.exe"

& $engram doctor
& $engram remember "本文" --type knowledge --importance 7
& $engram recall "クエリ" --deep
& $engram surface "発話テキスト"
& $engram stats
```

### プロジェクト構成

```
src/engram/
  config.py        設定(既定値 < config.toml < 環境変数)+ 部屋の解決
  engine.py        記憶エンジン本体
  store.py         Markdown 正本ストア
  db.py            SQLite インデックス(sqlite-vec + FTS5)
  dynamics.py      ACT-R 活性化モデル
  embedder.py      RuriEmbedder / FakeEmbedder
  server.py        MCP サーバー(stdio)
  cli.py           CLI エントリーポイント
  setup.py         セットアップウィザード & doctor & フック登録
  hooks.py         フック入口(自動符号化・自発的想起)
  transcript.py    transcript の決定的要約(自動符号化)
  surface.py       自発的想起の軽量検索路(モデル不使用)
  templates/       MEMORY_PROTOCOL.md / ONBOARDING.md
tests/
  test_setup.py    セットアップ純粋ロジックのテスト
  test_config.py   設定優先順位のテスト
  test_store.py    Markdown ストアのテスト
  test_db.py       DB 操作のテスト
  test_engine.py   エンジンのテスト
  test_room.py     記憶の部屋のテスト
  test_surface.py  自発的想起のテスト
  test_transcript.py  transcript 要約のテスト
  test_hooks.py    フックとフック登録のテスト
  test_integration.py  統合テスト
```

---

## アンインストール

```powershell
# 1. 各エージェントから登録を削除
claude mcp remove engram

# 1b. ~/.claude/settings.json の hooks から engram の項目
#     (SessionEnd / UserPromptSubmit の "engram hook ...")を手動削除
# 2. ~/.claude/CLAUDE.md から engram のブロックを手動削除
# 3. ~/.codex/config.toml から [mcp_servers.engram] ブロックを手動削除
# 4. ~/.gemini/config/mcp_config.json から engram エントリを手動削除

# 5. engram 本体をアンインストール
uv tool uninstall engram

# 6. データを削除する場合(記憶・設定・モデルキャッシュ)
Remove-Item -Recurse -Force "$env:USERPROFILE\.engram"
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--cl-nagoya--ruri*"
```

macOS / Linux の手順5-6:

```bash
uv tool uninstall engram
rm -rf ~/.engram
rm -rf ~/.cache/huggingface/hub/models--cl-nagoya--ruri*
```
