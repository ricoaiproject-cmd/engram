[English version → engram-global](https://github.com/ricoaiproject-cmd/engram-global)

# engram — AIエージェント用 人間型記憶基盤(MCPサーバー)

Claude Code / Codex / Gemini CLI / Antigravity が共有する永続記憶(どのエージェントで何が動くかは「どのエージェントで何が動くか」の表を参照)。
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
- 埋め込みモデルのダウンロード(初回のみ、約530MB)と、起動を速くする ONNX 形式への変換
- 入っているエージェント(Claude Code / Codex / Gemini CLI / Antigravity)への MCP サーバーの登録と、記憶の使い方
  (記憶プロトコル)の追記
- フック(会話の終わりの自動記録・発言ごとの自発的想起)の登録。Claude Code・Codex・Gemini CLI のそれぞれに登録します
  (Codex は、Codex を開いて `/hooks` で engram のフックを承認すると動き始めます)
- 夜の整理(毎晩 2:30)の自動実行の登録と、DMN(自発思考、15分毎)の登録(DMN はローカル LLM があるときだけ)。
  止めるには `engram night --uninstall` / `engram dmn --uninstall`
- 次の3つは、対話のセットアップで1回だけ聞きます(既定はどれも「いいえ」):
  - 夜の整理の要約に Claude Code / Codex を使うか(使うと利用枠を毎晩少し使い、記憶の文章を Anthropic / OpenAI へ送ります。
    既定はパソコンの中だけで動くローカル LLM)
  - 仮説サイクル(記憶から仮説を立てて Web で確かめる)を毎日回すか(Claude Code か Codex を使います)
  - 媒体の棚(画像・動画・文書を言葉で探す)を入れるか(数GBの追加導入が要ります)

最後の画面に「止めてある機能・使えない機能」と、それぞれをオンにする1行が表示されます。

### どのエージェントで何が動くか

engram の機能は、MCP(エージェントが外部の道具を使う共通の仕組み)とフック(会話の区切りで engram を呼ぶ仕組み)と
OS の予約(毎晩・毎日の自動実行)で動きます。どのエージェントでも、MCP の初期化の案内で記憶の使い方が伝わり、
`recall` の応答で記憶整理などの促しが届きます。

| 機能 | Claude Code | Codex | Gemini CLI | Antigravity |
|---|---|---|---|---|
| 記憶する・思い出す(MCP の道具) | ○ | ○ | ○ | ○ |
| 記憶の使い方の案内(指示ファイル・MCP の案内) | ○ CLAUDE.md | ○ AGENTS.md | ○ GEMINI.md | ○ GEMINI.md(※1) |
| 会話の自動記録(フック) | ○ 会話の終わり | ○ 会話の終わり(※4) | ○ 会話の終わり(※2) | ○ 応答が終わるたびに増えた分(※5) |
| 発言ごとの自発的想起・縁・着手前の促し(フック) | ○ | ○(※4) | ○ | ×(※5) |
| 記憶整理・スキル化の促し | ○ フックと MCP | ○ フックと MCP | ○ フックと MCP | ○ MCP |
| 夜の整理(沈黙・常連抑制) | ○ | ○ | ○ | ○ |
| 夜の整理の要約 | ローカル LLM か、同意すれば Claude Code | ローカル LLM か、同意すれば Codex | ローカル LLM | ローカル LLM |
| DMN(自発思考) | ローカル LLM があれば | 同左 | 同左 | 同左 |
| 仮説サイクルの毎日の実行 | 同意すれば ○ | 同意すれば ○(※3) | ×(手で回すのみ) | ×(手で回すのみ) |
| 媒体の棚 | ○ | ○ | ○ | ○ |

- ※1 Antigravity は、MCP の設定を `~/.gemini/config/mcp_config.json` から、指示ファイルを `~/.gemini/GEMINI.md`
  などから読みます(Antigravity の公式資料の docs/mcp と docs/rules)。engram はこの2つに書きます。
- ※2 Gemini CLI は会話の終わりのフックの完了を待たないので、engram は入力を受け取ってすぐ返し、要約は裏の
  別のプロセスで行います。Gemini CLI は設定 `tools.enableHooks` が `true` のときだけフックを動かすので、
  engram はフックを登録するときにこれをオンにします。
- ※3 Codex での仮説サイクルは、MCP の道具と Web 検索を非対話で使う設定(`codex exec`)の実機確認がまだです。
- ※4 Codex は、新しく足したフックを、利用者が Codex の中で `/hooks` を開いて承認するまで動かしません
  (Codex の安全の仕組みなので、engram は承認を代わりに行いません)。セットアップの画面に案内が出ます。
  会話の終わりのフックは Codex が3秒で打ち切るため、要約は裏の別のプロセスで行います。
- ※5 Antigravity のフックは `~/.gemini/config/hooks.json` に「engram」という名前で入ります(ほかの名前の項目には
  触れません)。応答が終わるたびに呼ばれる Stop で、手が空いたとき(サブエージェントを待っていないとき)だけ、
  前回からの増えた分を記録します。発言ごとの想起は未対応です。Antigravity には「話しかけたとき」のイベントが無く、
  近い PreInvocation(モデルを呼ぶ前)は利用者の発言を受け取らず、文脈へ差し込む項目(injectSteps)の中身の形も
  公式の資料で確定できないためです(形を間違えると Antigravity の動きを妨げうるので、確かめられるまで入れません)。
  記憶の使い方の案内と、`recall` の応答に添える促しは届きます。

フックが無い環境(Codex でフックを承認していない人など)では、記憶の保存と想起はエージェントが
`recall` / `remember` を呼んだときだけになり、会話が自動で記憶になることはありません。

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

### ONNX による起動高速化

v0.11 からは `engram setup`(1行インストーラが最後に自動実行します)が
ONNX 変換まで自動で行うため、**手動での操作は不要**です。変換済みなら
スキップされます(冪等)。埋め込みモデルを変更した後などに再変換したい
場合だけ、手動コマンドを使ってください:

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

既定は `auto` です(v0.10.0〜): ONNX モデルが生成済みなら `background`
(ハンドシェイク即応答)、torch フォールバック環境なら `blocking` を自動選択
するため、通常は調整不要です。MCP の起動タイムアウト設定が効かないクライアント
(実例: Codex Desktop 26.707)でもそのまま接続できます。

| 値 | 挙動 |
|---|---|
| `auto`(既定) | ONNX 生成済みなら `background`、無ければ `blocking`。 |
| `blocking` | ハンドシェイク前にメインスレッドでモデルを読み込む。接続後の `recall` は常に即応答。torch 経路では warm 12〜24秒 / cold 50秒超かかるため、クライアント側の MCP 起動タイムアウトを120秒以上に延長すること(Claude Code なら `MCP_TIMEOUT=120000`)。 |
| `background` | ハンドシェイクに即応答し、モデルは裏スレッドで読み込む。ONNX 経路では安全(初回ツールが読み込み完了を数秒待つだけ。実測: 別スレッドでも約5秒)。**torch 経路では非推奨:** Windows では asyncio イベントループ稼働中の別スレッド torch import が病的に遅く(実測: メインスレッド約20秒 → 別スレッド約184秒)、初回 `recall` がクライアントのツールタイムアウトを超えうる。 |
| `off` | 先読みしない。初回ツール呼び出し時に遅延ロードする。 |

torch フォールバック環境で起動時に接続失敗する場合は、`background` に切り替えるのではなく、クライアント側の MCP 起動タイムアウトを延長するか、`engram export-onnx` で ONNX モデルを生成してください(torch での `background` は「見える起動タイムアウト」を「初回 recall の3分ハング」に変換するだけです)。

### 多重プロセスとメモリ(`ENGRAM_IDLE_UNLOAD_SEC`)(v0.12.0〜)

MCP クライアントは engram を**1プロセスだけ**起動するとは限りません。実例:
Codex Desktop(26.810)は実行ホストごとに stdio MCP プロセスを起動し、使い
終わっても残留させるため、約1.1GBの ONNX モデルがプロセス数ぶん積み上がり、
16GB 級 PC では空きメモリが枯渇して PC 全体がカクつきます(実機で4プロセス
同時を確認)。v0.12.0 で二段の対策を入れました:

1. **Codex 登録時の先読み抑止** — `engram setup` が `~/.codex/config.toml` の
   engram ブロックに `ENGRAM_PRELOAD = "off"` を書き込みます(既存ユーザーも
   setup 再実行だけで追記されます。手動設定済みの値には触れません)。
   これで「使っていないプロセス」はモデルを読み込みません。
2. **アイドル解放** — `ENGRAM_IDLE_UNLOAD_SEC` 秒(既定600 = 10分)ツールが
   呼ばれなければ、モデルをメモリから解放します。実際にツールを使った
   プロセスが残留しても、後からメモリが返ります。次回ツール呼び出し時に
   自動で再ロードされ(ONNX で数秒)、記憶は一切失われません。`0` で無効。
   対象は ONNX 経路のみ(torch 経路は再ロードが遅い病理があるため対象外)。

---

## もっと記憶らしく(v0.3 の新機能)

### 自動符号化 — セッションが勝手に記憶になる

会話(セッション)が終わると、フックがその回のやり取りを要約して episode 記憶として自動保存します
(`engram setup` が Claude Code・Codex・Gemini CLI にフックを登録します。対応は上の表を参照)。
remember を呼び忘れても「昨日何をしたか」が残ります。
`config.toml` で `auto_encode = false` にすると無効化できます。

### 自発的想起 — 記憶のほうから思い出す

あなたが何か発言するたび、フック(Claude Code・Codex は UserPromptSubmit、Gemini CLI は BeforeAgent)が
軽量検索で関連記憶を探します。動作モードは `config.toml` の `surface_mode` で切替:

| モード | 動作 |
|---|---|
| `active`(初期値) | 関連が強い記憶をエージェントの文脈に実際に差し込む |
| `shadow` | 差し込まず「差し込むならこれ」をログに記録(様子見・調整用) |
| `off` | 何もしない |

ログは `~/.engram/surface/surface_log.jsonl`。無関係な記憶が浮上する場合は
`surface_min_relevance` を上げるか、`shadow` に切り替えてログだけで観察できます。
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

短いクエリも字句検索の対象になりました(v0.7.1)。FTS5 の trigram トークナイザは
3文字未満の語を扱えないため、従来は「会議」のような日本語の2文字語が字句検索から
完全に漏れ、3文字未満のトークンを含む複合クエリは AND 条件ごと0件になっていました。
現在は3文字以上のトークンだけで MATCH 式を組み、3文字以上のトークンが1つも無い
クエリは LIKE 部分一致にフォールバックして IDF ベースの擬似スコア
(`lex = N/(N+df)`: 希少語ほど高い)を与えます。

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
| `read(ids)` | 自発的想起の「縁」(id と要旨1行だけ差し込まれた記憶)から、本文を読みたいものを選んで取る |
| `link` / `forget` / `stats` / `reindex` | 補助操作 |
| `consolidation_candidates` / `mark_consolidated` | 統合(下記) |
| `skill_candidates` | 同じ形の作業の episode が溜まったとき、スキル(手順書)に切り出す候補を見る |
| `hypothesis_seeds` / `hypothesize` / `hypotheses` / `resolve_hypothesis` | 仮説サイクル(下記「仮説サイクル」) |
| `remember_file(path, note, room)` / `media_status` | 媒体の棚(下記「媒体の棚」。`engram media setup` でオン)。探すときは `recall(modality="image")` など |

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
3. フックが無い環境(Antigravity や、Codex でフックを承認していない人)にも届くよう、MCP サーバーも
   プロセスの最初の `recall` の応答に同じ促しを `notice` として添えます。候補数の記録が1日より古ければ、
   そのときに数え直します。促しの間隔はフックと共通です(二重には出ません)。

`config.toml` (または `ENGRAM_*` 環境変数) の3つの設定で制御します:

| 設定 | 既定値 | 意味 |
|---|---|---|
| `consolidate_nudge` | `true` | 促し機能全体のスイッチ |
| `consolidate_nudge_min_clusters` | `3` | 促す最小クラスタ数 |
| `consolidate_nudge_interval_days` | `7.0` | 促しの最短間隔(日) |

---

## 代謝層 — 記憶で育つ(v0.13〜v0.23)

v0.3 までは「記憶を持つ」ところまででした。v0.13 以降は、人間の睡眠・無意識に相当する
**代謝ループ**(書く → 眠って圧縮・忘却 → 知恵 → 連想)を engram 自身が回します。
賢さは蔵書量ではなく、この回転量から生まれる、という設計方針です。

| コマンド | 何をするか | いつ動くか |
|---|---|---|
| `engram night` | 眠り一式: ①統合(LLM) → ②沈黙・常連抑制 → ③蒸留(LLM) → ④追加処理 `night.d/` | 毎晩 2:30(`engram setup` が登録) |
| `engram sleep --commit` | ②だけ(純 Python・モデル非依存) | night の中で自動実行 |
| `engram dmn` | DMN(自発思考): 入力の無い時間に世界の差分を見て「ふと浮かんだこと」を日記と記憶に残す | 15分毎(ローカル LLM があれば `engram setup` が登録) |
| `engram metabolism` | 記憶側の物差し: 浮上命中率・書き込み量・hot在庫・統合の滞留 | いつでも |
| `engram rsi` | ループ側の物差し: 自律反復/日・提案採用率・DMN朝送り率 | いつでも |

`engram setup`(1行インストーラが最後に自動で実行します)が、眠りを OS のスケジューラ(Windows は
タスクスケジューラ、ほかは crontab)に登録するので、入れた日の夜から回り始めます。DMN は考える係がローカル LLM
だけなので、ローカル LLM があるときだけ登録します(無ければ最後の画面に入れ方が出ます)。Windows では、PC が
眠っていて実行時刻を逃した回も、起きたあとで実行します。**止めるには
`engram night --uninstall` / `engram dmn --uninstall`**(止めたことは設定に残り、setup をやり直しても戻りません)。
同じ名前(`EngramSleepNight` / `EngramDMN`)で自前の仕組みを登録している場合や、別のホームの engram が
登録したものは、上書きも削除もしません(置き換えるなら `--force-schedule`)。同じ PC に2つ入れるときは、
設定の `schedule_task_suffix` でタスク名を分けてください。

### 眠り(`engram night`)

```powershell
engram night --dry-run     # 何も書かず段取りだけ見る
engram night               # 1晩ぶん実行
engram night --install     # 毎晩の自動実行を登録し直す(setup が済ませているので普通は不要。--time で時刻)
engram night --uninstall   # 毎晩の自動実行を止める
```

1. **統合(LLM)** — `~/.engram/sleep_agents.txt` の優先順でエージェント CLI(例: `claude -p`、`codex exec`)を試し、
   `~/.engram/sleep_prompt.md` の手順で episode クラスタを knowledge に要約させます。特別行 `local-llm` は
   engram 内蔵のローカル LLM 統合(Ollama 互換 API、既定 `gemma4:e2b`)です。**全滅した夜は翌セッションの
   ナッジに自動縮退**するので、LLM が1つも無くても壊れません。モデルを替えるときはコードでなくこのファイルを編集します。
   **既定で使うのは、パソコンの中だけで動くローカル LLM だけです。** `claude -p` や `codex exec` を使うと、
   あなたの利用枠が毎晩少し減り、記憶の文章が Anthropic / OpenAI へ送られます。そのため、**本人が同意した
   CLI だけを使います**。`engram setup` を対話で実行すると、見つかった CLI ごとに1回だけ聞きます(既定は N)。
   答えは設定(`night_allow_claude` / `night_allow_codex`)に残り、setup をやり直しても聞き直しません。
   あとから変えるときは次の1行です:

   ```powershell
   engram night --allow claude    # 夜の整理の要約に Claude Code を使う(codex も同じ)
   engram night --deny claude     # 使わない
   ```

   対話できない導入(`--non-interactive` など)では聞かずに使わず、最後の画面に上の1行を表示します。
   `sleep_agents.txt` を自分で書き換えて CLI の行を指定している人は、その指定に従います(同梱の雛形と有効な行が
   違えば、書き換えたとみなします)。`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` が環境にあると従量課金になりうるため、
   同意があってもその CLI は使いません(使うなら `night_agents_allow_api_key = true`)。手元に無いコマンドの行は飛ばします。
2. **沈黙・常連抑制** — 使われない hot 記憶を cold へ(削除はしない)。reinforce されないまま浮上を繰り返す
   記憶は一定期間黙らせます。統合を沈黙より**先**に置くのは、統合の材料が消化前に沈むのを防ぐためです。
3. **蒸留(LLM)** — 直近48時間の自動符号化 episode から「後日の別の作業でも役立つ知見」だけを抽出し、
   knowledge / preference として覚えます。手がかり語(後でその知見を必要とする人が打ちそうな言葉)を本文に
   入れるので、言い換えでも想起できます。会話中の remember はエージェントの心がけ頼みで漏れるため、
   「打ち漏らしても毎晩必ず拾われる」床の保証です。ローカル LLM が無ければスキップします。
4. **追加処理** — `~/.engram/night.d/*.py` を順に実行します。利用者固有の後処理(自前フォルダを読む学習など)
   はここに置きます。engram 本体には利用者の間取りを焼き込みません。

ログ: `~/.engram/sleep/night_log.txt`(1行=時刻+出来事)、`night_agent_log.md`(LLM 部分の明細)。

### DMN(`engram dmn`)

人間の脳は入力が無い時間に Default Mode Network が動き、記憶を眺め直して連想します。

- **理由駆動で起きる**: 前回から世界に差分(新しい記憶が3件以上、`dmn_watch` に書いた監視先の更新、眠りの実行)が
  あれば起きます。差分ゼロなら LLM を呼ばず終了(課金ゼロ・ノイズ最小)。
- **無理由でも薄く起きる**: 黄金比の加法列(低食い違い列)で位相を進め、平均1時間に1回だけ。乱数だと発火が固まるためです。
- **考える係はローカル LLM**、道具の操作は決定的 Python。想起の組み立て(クエリ生成)にも LLM を入れません。
- **自分の思考で記憶を温めない**(reinforce しない)。DMN が読んだ記憶を温めると「富者益富」の自己ループになります。
- 出力は日記(`~/.engram/dmn/diary/日付.md`、`dmn_diary_dir` で変更可)と episode(importance 3)。思いつきは知識では
  ないので episode。外れは代謝が沈めます。

```toml
# ~/.engram/config.toml
dmn_watch = ["~/notes/daily", "~/projects/TASKS.md"]   # 「世界の変化」として見張る場所
dmn_diary_dir = "~/notes/dmn"                           # 日記の置き場(省略時は ~/.engram/dmn/diary)
llm_model = "gemma4:e2b"                                # 眠り・蒸留・DMN 共通のローカル LLM
```

```powershell
engram dmn --force --dry-run   # いま1回考えさせて表示だけ(書かない)
engram dmn --uninstall         # 15分毎の自動実行を止める(戻すときは --install)
```

DMN の考える係はローカル LLM なので、Ollama が無い PC では起きても何もせずに終わります
(料金も外部への送信もありません)。使うには Ollama を入れて `ollama pull gemma4:e2b` です。

### 回転計(`engram rsi`)

RSI(再帰的自己改善)は「回転数 × 1回転あたりの改善率」。`metabolism` が記憶側の物差しなら、`rsi` はループ側の
物差しです。眠りの実行夜数・DMN の起床と日記・蒸留で増えた知見を「自律反復/日」として数え、任意で提案台帳
(`rsi_ledger_path`。Markdown 表、1列目=ID・3列目=状態)の採用率も出します。

### 設計メモ

- 忘却は**削除ではなく沈黙**(hot → cold)。人間の忘却もほとんどはアクセスの喪失です。
- 記憶が温まるのは「読まれた」ときではなく「**使ったと申告された**」とき(reinforce・暗黙の使用検出)。
  読んだだけで温めると、参照され続けた記憶が冷えずに居座ります。
- 想起の経路(surface・DMN のクエリ)に LLM を入れない。毎ターンの追加往復を排し、モデルが替わっても挙動が変わらないようにするためです。

---

## 仮説サイクル — 記憶から仮説を立てて確かめる(v0.25)

記憶は、ためるだけでは新しいことを言いません。仮説サイクルは、近すぎず遠すぎない2つの記憶を
組み合わせて「両方を合わせると言えそうで、どちらにも書いていないこと」を仮説として書き、期限が
来たら確かめ、結果をまた記憶に戻す仕組みです。

### しくみ

役割を分けています。仮説を考えて、Web などで確かめるのはエージェント(LLM)の仕事です。
engram は「覚えておく係」と「思い出させる係」に徹し、LLM を使わない決まった処理だけを持ちます。

1. **種選び**(`hypothesis_seeds`): 縁の薄い記憶の組を返します。過去の当たり率で組の選び方を重み付けします。
2. **書く**(`hypothesize`): 仮説を episode 記憶として書き、台帳に1行足します。
   **外れたと分かる条件(`falsify_if`)が書けない仮説は受け付けません。**
3. **一覧**(`hypotheses`): 未決・期限が来たもの・判定済みを返します。未決の仮説には、
   書いた後に入った近い記憶を「証拠候補」として添えます。
4. **判定**(`resolve_hypothesis`): 「当たり」なら knowledge を新しく作り、元の記憶と結びます。
   「外れ」なら `correct` と同じく、誤りごと刻みます。「判定不能」なら期限を延ばして再挑戦します
   (上限 `hyp_max_attempts` 回)。

決まりごとが3つあります。書いた日と同じ日には判定できません(自分で立てて自分で丸を付けると、
当たりばかりになるため)。仮説が元の記憶を参照しただけでは、その記憶を温めません(温めるのは「当たり」の
判定が付いたときだけ)。モデルが下した判定には `judge` を残すので、人が後から `correct` で直せます。

台帳は `<記憶フォルダ>/_hypotheses/ledger.json` です。記憶の Markdown と同じ場所に置くので、
記憶フォルダを同期していれば、ほかのマシンとも共有されます。

### 使い方

エージェントからは上の MCP ツールで、手元からは CLI で同じことができます:

```powershell
engram hyp seeds --n 3                         # 種になる記憶の組を見る
engram hyp add "主張" --sources ID1,ID2 --falsify "外れたと分かる条件" --test "確かめ方" --due 3
engram hyp list --status due                   # 期限が来て、今日判定してよいもの
engram hyp resolve ID 当たり --evidence "根拠の要約" --urls "https://..."
engram hyp stats                               # 答え合わせ率・当たり率
```

### 毎日回す

engram 自身は仮説を考えないので、毎日1回エージェントの CLI に回してもらいます。Web で調べものをし、記憶の一部を
CLI の会社と検索先へ送り、利用枠を使うので、**本人が選んだときだけ回します**。対話の `engram setup` が、見つかった
Claude Code / Codex について1回だけ聞きます(既定は回さない)。あとから決めるときは次の1行です:

```powershell
engram hyp-run --use claude    # 毎日 7:30 に Claude Code で回す(codex も可)。OS のスケジューラに登録する
engram hyp-run --stop          # 回さない
engram hyp-run                 # いま1回回す(選んだ CLI で)
```

Claude Code では engram の道具と Web 検索・Web 取得だけを許して `claude -p` で、Codex では `codex exec`
(Web 検索を live)で回します。Codex で MCP の道具が非対話でどう承認されるかは、まだ実機で確かめていません。
Gemini CLI と Antigravity では、エージェントに「engram の仮説サイクルを1回回して」と頼めば同じことができます
(毎日の自動実行は未対応)。

検索語の注意を入れているのは、種になる記憶の本文には個人の事情が入りうるためです。既定では、種は
共通の部屋(`common`)の knowledge と project からだけ選びます。

### 設定(`~/.engram/config.toml`)

```toml
hyp_default_due_days = 3                  # 期限の既定(書いた日 + N 日)
hyp_seed_rooms = ["common"]               # 種を選ぶ部屋。仕事の部屋は既定で入れない
hyp_seed_types = ["knowledge", "project"] # 好み(preference)は Web で確かめられないので既定で外す
hyp_max_attempts = 2                      # 判定不能のときの再挑戦の上限
```

---

## 媒体の棚 — 画像・動画・文書を言葉で探す(v0.24)

文章の記憶とは別に、画像・動画・文書(PDF・Office 文書)・音声を「見た目・音」で索引し、
「前に作ったあの図」「あの画面写真」のように言葉で探せるようにする棚です。
**数GBの追加導入(torch など)が要るため、セットアップでは入れずに止めてあります。**
オンにするのは次の1行です(セットアップの最後にも表示されます):

```powershell
engram media setup   # 環境を作り、有効にし、対象フォルダ(未設定なら画像フォルダ)と毎晩の差分索引を登録する
```

### 何に頼るか

重い部品を本体に入れないため、索引係(`engram-media`)は**本体とは別の Python 仮想環境**で動きます。
本体はローカル(127.0.0.1)の通信で索引係に頼むだけです。

| 要るもの | 説明 |
|---|---|
| 別の仮想環境 | `engram media setup` が作り、`engram[media]`(torch・sentence-transformers 6.1 以上・PyMuPDF・PyAV など)を入れます。数分かかり、数GBを使います |
| 埋め込みモデル | 既定 `google/embeddinggemma-2`。初回に Hugging Face から自動でダウンロードします。GPU が無くても CPU で動きます |
| LibreOffice(任意) | PowerPoint・Word などを PDF にしてから索引するのに使います。無ければ Office 文書は索引されません(PDF・画像・動画は索引されます) |
| ローカル LLM(任意) | 画像に短い説明文を付けるのに、Ollama 互換の画像対応モデルを使います(`media_caption`)。無ければ説明文なしで索引します |
| Google の認証(任意) | メール添付の取り込みを使うときだけ(下記) |

### 入れ方

1. `engram media setup` を実行します。別の仮想環境と部品を入れ(数分・数GB)、`media_enabled = true` にし、
   対象フォルダが未設定なら `~/Pictures/Screenshots`(無ければ `~/Pictures`)を足し、毎晩の眠り(night.d)に
   差分索引を置きます。

2. 対象フォルダを変えるときは設定に書きます(フォルダごとに部屋も決めます):

   ```toml
   # ~/.engram/config.toml
   media_roots = { "~/Pictures/Screenshots" = "common", "~/Documents/work" = "work" }
   media_kinds = ["image", "video", "document"]   # 音声は精度が大分類止まりのため既定で外す
   ```

   すぐ索引したいときは `engram media index --wait`(初回は件数に応じて時間がかかります)。何もしなくても、
   その夜の眠りで差分が索引されます。

3. 探します: エージェントからは `recall(query, modality="image")`(`"video"` `"document"` `"media"` も可)、
   手元からは `engram media find "青い表紙の企画書"`。ファイル1件をすぐ覚えさせるときは
   `engram media remember <パス>` か MCP の `remember_file` です。ファイル本体は動かさず、場所を覚えるだけです。

発言の中に「画面」「スクショ」「図」「PDF」などの手がかり語があるときだけ、関連しそうな画像・文書を
自発的想起に1〜3件添えます(`media_surface_mode`。`off` で止められます)。

### メール添付(任意)

Gmail の添付ファイルを、読み取り専用で棚に取り込めます。メールの送信・変更・削除は一切しません。

1. 索引係の環境に `engram[mail]` を追加で入れ、Google Cloud で OAuth クライアント(デスクトップ)を作ります。
   索引係の環境は `~/.engram/media/venv` にあります(Windows の例):

   ```powershell
   & "$HOME\.engram\media\venv\Scripts\python.exe" -m pip install "engram[mail] @ git+https://github.com/ricoaiproject-cmd/engram.git"
   ```
2. 設定にアカウントごとの区画を書きます:

   ```toml
   [media_mail.personal]
   client_secret = "~/.engram/media/mail/client_secret.json"
   token         = "~/.engram/media/mail/token_personal.json"
   query         = "has:attachment newer_than:30d -category:promotions"
   room          = "common"
   save_dir      = "~/Documents/attachments"   # 省略時は ~/.engram/media/inbox/<名前>(同期なし)
   ```

3. 初回だけブラウザで許可し、以後は `--auth` を外して実行します。コマンドは索引係の環境の中にあります:

   ```powershell
   & "$HOME\.engram\media\venv\Scripts\engram-media.exe" mail-fetch --account personal --auth
   ```

### 撤去

```powershell
engram media uninstall                 # 索引係を止め、夜間処理から外し、media_enabled = false にする(索引と記憶本体は残る)
engram media uninstall --purge-index   # 媒体の索引も消す
engram media uninstall --purge-env     # 仮想環境・モデル・索引フォルダも消す
```

---

## 機能ごとに要るもの

engram の基本(記憶する・思い出す・自発的想起)は、本体だけで動きます。それより先の機能は、
次のものがあるときに働き、無ければ静かに縮退します。

| 機能 | 要るもの | 無いとき |
|---|---|---|
| 記憶する・思い出す(MCP・CLI) | 本体のみ。初回に埋め込みモデル(約530MB)をダウンロード | — |
| 自動符号化・自発的想起・初回発話の想起 | フック(`engram setup` が Claude Code・Codex・Gemini CLI に登録。Codex は `/hooks` で承認) | エージェントが自分で `recall` / `remember` するだけになる |
| 記憶整理・スキル化の促し | フックか MCP(どちらでも届く) | — |
| 埋め込みの常駐プロセス | なし(フックや MCP サーバーが自動で起動する) | 各セッションが自分でモデルを読み込む |
| 夜の整理の要約(`engram night`) | ローカル LLM(既定)。同意した人は `claude -p` / `codex exec` も(利用枠を使い、記憶の文章を送る。`engram night --allow claude`) | 翌セッションの促し(フックか MCP の `recall` の応答)に切り替わる |
| 夜の整理の蒸留・DMN・画像の説明文 | ローカル LLM(Ollama 互換。既定 `gemma4:e2b`) | その処理だけ飛ばす。DMN は登録もしない |
| 毎晩・15分毎の自動実行 | なし(`engram setup` が Windows はタスクスケジューラ、ほかは crontab に登録する) | 止めた人(`--uninstall`・`engram setup --no-schedule`)は手で実行したときだけ動く |
| 仮説サイクルの毎日の実行 | 本人が選んだ Claude Code か Codex(`engram hyp-run --use claude`) | 種選びと台帳は使えるが、判定は手で頼んだときだけ進む |
| 媒体の棚 | 別の仮想環境(数GB)。Office 文書には LibreOffice。説明文にはローカル LLM | 数GBの導入が要るため、対話のセットアップで聞いて「はい」のときだけ入れる。あとからは `engram media setup` |
| メール添付 | `engram[mail]` と Google の OAuth クライアント | 使わない |

セットアップの最後に「止めてある機能・使えない機能」と、それぞれをオンにする1行が表示されます。

説明・メッセージ・LLM への指示文は日本語です(英語版は [engram-global](https://github.com/ricoaiproject-cmd/engram-global))。

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
  models.py        記憶の型
  onnx_export.py   埋め込みモデルの ONNX 変換
  perf.py          起動時間・ツール呼び出しの記録
  server.py        MCP サーバー(stdio)。初期化の案内と、recall に添える促し
  cli.py           CLI エントリーポイント(__main__.py で python -m engram も可)
  setup.py         セットアップウィザード & doctor & 各エージェントへの登録・フック登録・同意の確認
  hooks.py         フック入口(自動符号化・自発的想起・促し)
  transcript.py    transcript の決定的要約(自動符号化)
  surface.py       自発的想起の軽量検索路(モデル不使用)
  daemon.py        埋め込みモデルを共用する常駐プロセス
  sleep.py         眠り(沈黙・常連抑制・レポート)
  night.py         夜の整理一式(統合・蒸留・追加処理)と、使う CLI の同意
  dmn.py           DMN(自発思考)
  localllm.py      ローカル LLM(Ollama 互換)の薄いクライアント
  metabolism.py    記憶側の物差し
  rsi.py           ループ側の物差し(回転計)
  hypothesis.py    仮説サイクル(仮説・判定・種選び)
  hyp_run.py       仮説サイクルを選んだ CLI で1回回す(毎日の自動実行の本体)
  scheduler.py     OS スケジューラへの登録(タスクスケジューラ・crontab)
  media.py         媒体の棚の本体側(索引係への問い合わせ・導入・撤去)
  media_worker.py  媒体の棚の索引係(別の仮想環境で動く)
  media_schema.py  媒体の棚の表
  media_mail.py    メール添付の取り込み(読み取り専用)
  templates/       MEMORY_PROTOCOL.md / ONBOARDING.md / sleep_agents.txt / sleep_prompt.md
tests/             各モジュールのテスト(test_*.py)
```

---

## トラブルシューティング

### Codex で engram が「有効」なのに起動時に接続タイムアウトする

engram は起動のたびに約530MBの埋め込みモデルを読み込み、記憶フォルダを確認
します。これが Codex の MCP 初期接続タイムアウト既定値(30秒)を超えることが
あります — 特に PC 再起動直後、ウイルス対策ソフトのスキャン中、記憶フォルダ
がクラウド同期ドライブ(Google ドライブ・OneDrive 等)上にある場合。記憶
データは無事で、最初の接続だけが間に合っていない状態です。

新しいバージョンの `engram setup` は起動待ち時間を自動で書き込みます。古い
バージョンで登録した場合は、`engram setup --agents codex` を再実行するか、
`~/.codex/config.toml` の engram ブロックに次の1行を追記してください:

```toml
[mcp_servers.engram]
command = "..."                # そのまま
startup_timeout_sec = 120.0    # この行を追記
```

その後、Codex を完全に再起動してください(ウィンドウを閉じるだけでなく
終了→起動)。

### 起動時に `No module named 'mcp.server.fastmcp'` で止まる

engram は MCP(AI エージェントと道具をつなぐ規格)の公式 Python 部品 `mcp` を使います。この部品の
2.x では、engram が使う `FastMCP` が `MCPServer` に名前を変えたため、v0.12.0 以前の engram に
mcp 2.x が入ると、起動したところでこのエラーになります。v0.12.1 からは依存の指定を `mcp>=1.2,<2` に
しているので、新しく入れ直せば 1.x が選ばれて直ります。

```powershell
uv tool install --python 3.12 --force git+https://github.com/ricoaiproject-cmd/engram.git
```

engram は今も mcp 1.x(1.9 以上)で動かしています。mcp 2.x では動きません(依存の指定で 2.x が入らないように
してあるので、通常の入れ方で困ることはありません)。

### 同じ PC に engram を2つ入れる(試験用の導入など)

engram は、埋め込みモデルを抱える常駐プロセス(`engram daemon`)と、媒体の棚の
索引係を、127.0.0.1 の決まった番号で待ち受けさせます。v0.25.2 からは、この番号を
記憶の置き場(`data_dir`。既定 `~/.engram`)の場所から決めるので、置き場が違えば
番号も別になります。さらに、常駐プロセスはどの応答にも自分の置き場を名乗り、
窓口は名乗りが自分と一致する相手の応答しか使いません。番号がたまたま重なっても、
別の engram を止めたり、その検索結果を混ぜたりはしません。

番号を自分で決めたいときは `~/.engram/config.toml` に書きます:

```toml
daemon_port = 61873   # 常駐プロセス(0 で空き番号。単一起動の保証は効かない)
media_port  = 61874   # 索引係(省略時は daemon_port + 1)
```

### Windows でファイルが見つからない(`FileNotFoundError`)と言われる

Windows には、パスの長さがおよそ260文字を超えるとファイルを開けない制限があります。
記憶のファイル名には本文の頭が入るため、記憶の置き場や `HF_HOME`(モデルの置き場)を
深いフォルダに置くと、この制限にかかることがあります。浅い場所に置くか、Windows の
長いパスの許可(LongPathsEnabled)を有効にしてください。

---

## アップデート

インストールと同じ1行コマンドをもう一度実行するだけで、最新版に上書き更新されます:

```powershell
irm https://raw.githubusercontent.com/ricoaiproject-cmd/engram/main/install.ps1 | iex
```

macOS / Linux:

```bash
curl -LsSf https://raw.githubusercontent.com/ricoaiproject-cmd/engram/main/install.sh | sh
```

`uv tool upgrade engram` でも同じです(こちらは本体の更新のみ)。

- 1行版は更新後にセットアップも自動で走るため、ONNX 未変換の環境なら
  変換まで済み、起動が約2秒になります(変換済みならスキップ)
- 記憶・設定(`~/.engram` と記憶フォルダ)はそのまま残ります。消えません
- 更新後は、engram を使う各エージェント(Claude Code 等)を再起動してください(MCP サーバーの再接続に必要)

### v0.12.x から v0.25.5 へ上げる

v0.12 から v0.25 までの間に、眠り・DMN・仮説サイクル・媒体の棚・埋め込みの常駐プロセスが加わりました。
記憶のファイルと索引はそのまま使えます。上げるときにすることは次のとおりです。

1. 上の1行コマンドを実行します(セットアップも自動で走ります)。終わったら `engram doctor` で、
   使っているエージェントの行が `[OK]` になっていることを確かめます。
2. engram を使う各エージェントを再起動します。
3. セットアップが、夜の整理(毎晩)と DMN(15分毎)を自動実行に登録します。止めたいものは
   `engram night --uninstall` / `engram dmn --uninstall` で止められます。媒体の棚だけは数GBの追加導入が
   要るので止めてあり、`engram media setup` の1行でオンになります(セットアップの最後にも表示されます)。
4. 夜の整理の要約に Claude Code や Codex を使うかは、セットアップが1回だけ聞きます(既定は使わない)。
   v0.25.3 は手元の CLI を自動で使っていましたが、v0.25.4 に上げると、聞かれて「使う」と答えるまでは
   ローカル LLM だけになります。`sleep_agents.txt` を自分で書き換えて CLI を指定している場合は、その指定に従います。
5. v0.25.5 からは、Codex と Gemini CLI にもフックを登録します。Codex は、Codex を開いて `/hooks` で engram の
   フックを承認すると動き始めます。Gemini CLI には `~/.gemini/settings.json` にも MCP サーバーを登録します
   (Gemini CLI はここを読みます)。仮説サイクルを毎日回すか、媒体の棚を入れるかも、対話のセットアップで1回だけ聞きます。
   DMN は、ローカル LLM があるときだけ登録するようになりました。

v0.25.1 以前の常駐プロセスについて: 古い常駐プロセス(番号 51873)は、自分がどの記憶の係かを名乗らない
ため、v0.25.2 からは別の engram として扱われ、使われも止められもしません。v0.25.2 は記憶の置き場から
決まる番号で、自分の常駐プロセスを立て直します。古いほうは無操作6時間で自分から終わります。すぐ片付けたい
ときは PC を再起動してください。

媒体の棚を使っている場合は、索引係も同じ版にそろえるため、`engram media setup` をもう一度実行してください。
そろうまでは、本体が索引係を別の engram とみなすので、媒体の検索は休みます。

---

## アンインストール

```powershell
# 1. 自動実行(夜の整理・DMN・仮説サイクル)と媒体の棚の登録を外す(本体を消す前に)
engram night --uninstall
engram dmn --uninstall
engram hyp-run --stop
engram media uninstall

# 2. 各エージェントのフックから engram の項目だけを外す(Claude Code・Codex・Gemini CLI・Antigravity。
#    利用者が自分で入れたフックは残る)
engram setup --remove-hooks

# 3. 各エージェントから MCP の登録と記憶プロトコルを外す
claude mcp remove engram
# 3a. Claude Code: ~/.claude/CLAUDE.md の「# 記憶プロトコル(engram)」のブロックを手動削除
# 3b. Codex: ~/.codex/config.toml の [mcp_servers.engram] ブロックと、~/.codex/AGENTS.md の
#     「# 記憶プロトコル(engram)」のブロックを手動削除
# 3c. Gemini CLI / Antigravity: ~/.gemini/settings.json の mcpServers.engram、~/.gemini/config/mcp_config.json の
#     engram、~/.gemini/GEMINI.md の「# 記憶プロトコル(engram)」のブロックを手動削除

# 4. engram 本体をアンインストール
uv tool uninstall engram

# 5. データを削除する場合(記憶・設定・モデルキャッシュ)
Remove-Item -Recurse -Force "$env:USERPROFILE\.engram"
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--cl-nagoya--ruri*"
```

macOS / Linux の手順4-5:

```bash
uv tool uninstall engram
rm -rf ~/.engram
rm -rf ~/.cache/huggingface/hub/models--cl-nagoya--ruri*
```
