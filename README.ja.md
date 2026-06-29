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

Python バージョン・設定ファイル・モデルキャッシュ・各エージェントへの登録状況を
`[OK]` / `[NG]` / `[--]` で一覧表示します。

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
