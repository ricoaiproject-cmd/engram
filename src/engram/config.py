"""設定。優先順位: 環境変数 > ~/.engram/config.toml > 既定値。

配布可能にするため、コードの設置場所には一切依存しない。
記憶の保存先などユーザーごとの値は config.toml に永続化する(engram setup が生成)。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


def _engram_home() -> Path:
    # SQLite を Google Drive 等の同期下に置くとロック破損リスクがあるためローカル側。
    # %LOCALAPPDATA% は MSIX サンドボックスで仮想化されることがあるため ~/.engram を使う
    return Path(os.environ.get("ENGRAM_HOME", str(Path.home() / ".engram")))


@dataclass(frozen=True)
class Settings:
    # --- 配置 ---
    memories_dir: Path = field(
        default_factory=lambda: _engram_home() / "memories"
    )
    data_dir: Path = field(default_factory=_engram_home)

    # --- 埋め込み(Ruri-v3 はプレフィックス必須)---
    embed_model: str = "cl-nagoya/ruri-v3-130m"
    # 実行系の選択。"auto" は export-onnx 済みなら ONNX(軽量・起動1〜2秒)、
    # 未生成なら torch(sentence-transformers、import が重い)へフォールバック。
    # "onnx" / "torch" で強制。詳細は embedder.make_embedder を参照。
    embed_backend: str = "auto"          # "auto" | "onnx" | "torch"
    query_prefix: str = "検索クエリ: "
    doc_prefix: str = "検索文書: "

    # --- 再ランクの重み(関連度を支配的にして富者益富ループを防ぐ)---
    w_relevance: float = 0.6
    w_activation: float = 0.25
    w_importance: float = 0.15
    # 関連度の候補内 min-max 正規化(dynamics.normalize_relevances)の床。
    # コサインの値域圧縮(Ruri で 0.8〜0.87)で関連度の弁別力が潰れ、活性度の
    # 下駄が順位を支配する逆転への対策。候補間の spread がこの床未満のときは
    # 増幅せず、実際の差に比例して関連度の寄与を縮める。0 に近づけるほど
    # 微小差でも全力で増幅する(非推奨)。順位付け専用で、報告値・しきい値は生値。
    relevance_norm_floor: float = 0.10

    # --- アクセスイベントの重み ---
    recall_hit_weight: float = 0.3   # recall で返却されただけ(弱い強化)
    # 暗黙の使用(申告なしで使われた形跡): 浮上させた記憶の特徴語がその後の
    # アシスタント発話に現れていたら弱く記録する。reinforce は申告制なので
    # 「役立ったが報告されなかった」を拾えない物差しの穴を埋める
    implicit_use_weight: float = 0.5
    implicit_use_detect: bool = True
    implicit_use_min_terms: int = 2   # 特徴語がこの数以上一致したら使用とみなす
    reinforce_weight: float = 1.0    # 「実際に役立った」報告(強い強化)
    reinforce_strength_max: float = 3.0
    create_alpha: float = 2.0        # 初期符号化ブースト: w = 1 + α·(importance/10)

    # --- 減衰(フラッシュバルブ記憶: 高 importance ほど減衰が遅い)---
    decay_base: float = 0.5
    decay_spread: float = 0.2        # d_i = clamp(base − spread·(imp−5)/5, min, max)
    decay_min: float = 0.3
    decay_max: float = 0.6
    min_elapsed_seconds: float = 60.0  # t^-d の発散防止(直後アクセスの経過時間下限)

    # --- 検索 ---
    candidate_k: int = 50            # ベクトル/FTS 各候補数
    rrf_k: int = 60                  # Reciprocal Rank Fusion の定数
    # これ以上の cos 類似は重複とみなす。Ruri-v3 は同一話題の短い日本語文
    # 同士でも cos が高めに出るため、0.92 では別事実が併合された実例があり
    # 0.95 に調整(2026-06-11)
    dup_threshold: float = 0.95
    deep_score_threshold: float = 0.35  # fast の最高スコアがこれ未満なら deep を自動発動
    hop_decay: float = 0.7           # 拡散活性化のホップ毎減衰
    max_hops: int = 2

    # --- 網羅検索(exhaustive: 沈んだ記憶の掘り起こし)---
    # fast/deep は活性度を加味するため、長く使われず沈んだ記憶は関連が高くても
    # 最終スコアで埋もれる(「忘れない記憶」なのに想起できない)。exhaustive は
    # 活性度を無視し関連度のみで全 tier/type を総当たりするので、沈んだ記憶も
    # 意味的に近ければ必ず浮上する。
    exhaustive_min_relevance: float = 0.30    # これ未満の関連度は返さない(ノイズ防止のゲート)
    exhaustive_score_threshold: float = 0.30  # deep の最高スコアがこれ未満なら exhaustive を自動発動

    # --- 起動時インデックス同期(マルチマシン共有対策)---
    # 記憶 Markdown は共有(例: Google Drive)でも index.db はマシンごとローカルなため、
    # 他マシンが書いた記憶が index に無く recall に出ない盲点が生じる。MCP サーバー起動時に
    # Markdown(非trash)件数と index(active)件数の乖離を検知して:
    #   "auto" = 自動 reindex / "warn" = stderr に警告 / "off" = 何もしない
    startup_index_check: str = "auto"   # "auto" | "warn" | "off"

    # --- ヘッブ結合 ---
    colink_increment: float = 0.1
    colink_max: float = 1.0

    # --- 訂正(ハイパーコレクション効果)---
    correction_min_importance: int = 7

    # --- 統合(睡眠)---
    consolidate_min_age_days: int = 14
    consolidate_cluster_sim: float = 0.75

    # --- 統合の自動促し(consolidation nudge)---
    # engram は LLM を持たないため要約(統合の本体)はエージェントにしかできない。
    # SessionEnd で候補クラスタ数を棚卸しし、閾値超過なら UserPromptSubmit が
    # additionalContext でエージェントに統合作業を促す(hooks.py 参照)。
    consolidate_nudge: bool = True
    consolidate_nudge_min_clusters: int = 3     # このクラスタ数以上で促す
    consolidate_nudge_interval_days: float = 7.0  # 促しの最短間隔

    # --- スキル化候補(skill nudge)---
    # consolidation(統合)と相似形の機構。「同じ形の作業」を記録した episode が
    # 一定件数(三度ルール)以上クラスタを成したら、手順として切り出す(スキル化)
    # 価値があると見て、エージェントにユーザーへの提案を促す。consolidation と
    # 違い、直近の繰り返し作業こそ対象にしたいため年齢フィルタは掛けない。
    skill_nudge: bool = True
    skill_min_count: int = 3            # 1クラスタこの件数以上でスキル化候補(三度ルール)
    # episode の手順類似とみなすコサイン閾値。Ruri はコサインが 0.8 前後に
    # 圧縮されるため、0.80 では実データ(86 episode)の大半が1つの巨大クラスタに
    # 併合された。0.85 で初めて「同じ形の作業」(動画公開作業・決算転記・起案
    # 作成など)が別クラスタに分離することを実測で確認(2026-07-06 校正)
    skill_cluster_sim: float = 0.85
    skill_nudge_min_clusters: int = 1   # このクラスタ数以上で促す(1つでも繰り返し作業があれば提案価値がある)
    skill_nudge_interval_days: float = 7.0  # 促しの最短間隔
    # スキル自律作成モード: True のときナッジは「承認を待たず自動作成してよい」
    # 指示になる(自動生成枠 auto-* に隔離し、使われないものは代謝が引退させる)。
    # 既定 False=提案止まり(作成は人間の承認後)。ユーザー方針で切り替える。
    skill_autonomy: bool = False

    # --- 観測性(perf ログ)---
    # MCP ツール呼び出しと起動(preload)の所要時間を data_dir/perf/perf_log.jsonl
    # に記録する。「調子が悪い」をデータで診断するための常時計測(軽量)。
    perf_log: bool = True

    # --- 自動符号化(セッション終了フック)---
    auto_encode: bool = True             # SessionEnd フックでの自動 episode 保存
    auto_episode_importance: int = 3     # 自動要約 episode の importance(粗い記録なので低め)
    auto_encode_min_chars: int = 16      # ユーザー発言の合計がこれ未満のセッションは記録しない
    # 符号化の粒度(「代謝する脳」プラン Phase 3): 長いセッションはユーザー発言
    # この件数ごとに区切って複数 episode にする(1セッション1要約だと長丁場の
    # 中盤が失われるため)。0 以下で従来の1セッション1episodeに戻る
    encode_segment_user_turns: int = 12

    # --- 自発的想起(surface)---
    surface_mode: str = "active"         # off | shadow(ログのみ) | active(文脈に差し込む)
    surface_threshold: float = 0.45      # このスコア以上の記憶だけ浮上する
    # 関連度の最低ライン(ゲート)。活性度・重要度がいくら高くても、発話との
    # 字句関連がこれ未満なら浮上しない。よく使う重要な記憶が無関係な文脈に
    # 出しゃばる事故を防ぐ(実データで閾値ぎりぎりまで来た実例があった)
    surface_min_relevance: float = 0.25
    # 沈黙(cold)した記憶の再点火: surface は cold も走査するが、浮上は
    # 「強い手がかり」(この関連度以上)のときだけ。人間のサイレント・エングラム
    # が強い刺激でだけ再点火するのと同型。cold はスコアゲートを使わない
    # (活性度ゼロでスコアが構造的に届かないため。関連度のみで判定)
    surface_cold_min_relevance: float = 0.45
    # 選別の2段構え(R8.8.27): まず関連度だけで top-K に絞って「土俵」を作り、
    # その中で活性度・重要度を混ぜて並べる。加重和で最初にカットすると、
    # 関連度が高い新しい記憶が高活性の常連に枠ごと押し出されるため
    surface_candidate_k: int = 20
    # 意味の網(R8.8.27): 埋め込み常駐デーモンが生きていれば、字句バイグラムに
    # 加えて意味検索の候補も土俵に載せる。デーモン不在なら黙って字句だけで動く。
    # 「毎発話に埋め込みは重い」の1〜2秒はモデルのロード代であって推論代では
    # ないため、常駐させれば毎発話でも数十msで済む
    surface_semantic: bool = True
    surface_semantic_k: int = 30
    # コサインの引き伸ばし(実測: 無関係でも0.78前後、強い一致で0.92前後)。
    # 生値のまま字句関連度と混ぜると全候補が同じ値に見えて弁別力が消える
    semantic_cos_floor: float = 0.78
    semantic_cos_ceiling: float = 0.95
    surface_autostart_daemon: bool = True
    # 常駐デーモンの待受ポート。決まった値にすることで bind の失敗が
    # 「既に動いている」の判定になる(単一インスタンス保証)。
    # None(既定)= data_dir の場所から決まる番号(daemon_port_effective)。
    # 同じPCに2つめの engram(試験用の導入など)を入れても番号がぶつからない。
    # 全員共通の固定値(旧既定 51873)だと、試験用の環境から出した stop が
    # 本物のデーモンに届いて止める事故が起きた(R8.10.30)。
    # 0 を指定すると空きポートを自動で取る(テスト用・排他は効かない)
    daemon_port: int | None = None
    # 無操作でデーモンが終わるまでの時間。短いと長い作業の途中で消えて、
    # 意味の網が黙って字句だけに戻る(実測: 1時間だと作業中に落ちた)
    daemon_idle_hours: float = 6.0
    # MCP サーバー(セッションごとに1プロセス)の埋め込みをデーモンに頼むか
    # (R8.10.23)。True ならモデルはデーモン1つだけが持ち、各 MCP は持たない。
    # デーモンが使えないときは自動で手元の計算に戻る。False で従来どおり
    embed_via_daemon: bool = True
    # surface 専用の重み(R8.8.27)。importance は実態として「手書きか機械書きか」
    # の署名になっており(手動remember=7〜10、夜間蒸留=4〜6に固定)、保護(沈黙の
    # 免除)と競争力(浮上の下駄)を同じ変数に担わせると、hot が古参の寡頭制へ
    # 漸近して新参が永久に浮上できない。順位付けからは外し、保護だけに使う。
    # 重要度に充てていた配分は関連度へ回し、合計1.0を保つ(合計が1を割ると
    # 全スコアが押し下げられ、閾値が実質的に厳しくなる)
    surface_w_relevance: float = 0.75
    surface_w_activation: float = 0.25
    surface_w_importance: float = 0.0
    # 探索枠: 選ばれたのが古参(高importance)ばかりのとき、1枠を若い記憶へ譲る。
    # 新規記憶に浮上→reinforce の機会を与えないと代謝ループが新参に閉じる
    surface_explore_slot: bool = True
    surface_explore_max_importance: int = 6
    surface_max_items: int = 2           # 1プロンプトで差し込む最大件数
    # --- 縁の層(R8.10.11 ユーザー「人間はもっといろいろ想起する」)---
    # 意識に上る分(surface_max_items・本文つき)とは別に、意識の縁に弱く浮かんだ記憶を
    # 「id + 要旨1行」だけで多めに添える。読まなければ費用はほぼゼロ。必要なものだけ
    # read(ids) で本文を引く(引かれた回数が縁の効き目の物差し)。
    # 出どころは2つ: 検索で惜しかったもの(near)と、浮上した記憶から一歩つながったもの(link)
    surface_fringe_items: int = 10
    surface_fringe_min_relevance: float = 0.10
    surface_fringe_gist_chars: int = 48
    surface_fringe_links: bool = True
    surface_min_prompt_chars: int = 8    # これより短い発言では想起しない

    # --- 眠り(sleep: 夜間の保守。「代謝する脳」プラン R8.8.21)---
    # 沈黙 = 使われない hot 記憶の cold 降格(削除ではない。deep recall で到達可)。
    # ユーザーの訂正・流儀(feedback/preference/user)と高 importance は保護する。
    # 沈黙の判定方式(R8.8.27): "age" = 固定日数 / "pressure" = 在庫圧力。
    # 人間の忘却は経過時間ではなく干渉(新しい記憶との競合)で進むため、
    # hot 在庫が目標を超えたぶんだけ活性度の低い順に沈める方が同型。
    # 書き込み量が変わっても定数の手直しが要らない
    sleep_mode: str = "age"                # "age" | "pressure"
    sleep_hot_target: int = 900            # pressure 時の hot 在庫の目標件数
    sleep_pressure_max_demote: int = 120   # 1晩に沈める上限(急変を避ける)
    sleep_demote_days: float = 90.0        # この日数使われなかった hot は cold へ
    sleep_demote_max_importance: int = 7   # これを超える importance は降格しない(ageモード)
    # pressure モード専用の保護2つ。importance 8 まで恒久保護にすると hot が
    # 降格不能な古参で埋まり(実測 R8.9.27: hot 940件中934件が保護で不動)、
    # 圧力の全量が生後1日の新参にかかる硬直が起きるため、保護は別格(9+)のみ
    sleep_pressure_shield_importance: int = 9  # これ以上は pressure でも沈めない
    sleep_pressure_min_age_days: float = 3.0   # 生後この日数は沈めない(新参の猶予)
    sleep_exempt_types: tuple = ("feedback", "preference", "user")
    # 常連抑制 = reinforce されないまま浮上を繰り返す記憶を surface から黙らせる
    # (実測: ノイズ-15%・失う当たり78件中6件。R8.8.21 シミュレーション)
    sleep_suppress_misses: int = 3         # 無 reinforce 浮上がこの回数続いたら抑制
    sleep_suppress_days: float = 14.0      # 抑制期間(初回)
    # 抑制の漸増(間隔効果と同型): 繰り返し外す記憶ほど長く黙らせる。
    # 14 → 28 → 56日。上限を超えては伸ばさない
    sleep_suppress_escalate: bool = True
    sleep_suppress_max_days: float = 56.0
    sleep_suppress_window_days: float = 45.0  # surface ログを遡る窓
    sleep_hit_window_hours: float = 24.0   # 浮上→reinforce を「当たり」と数える窓

    # --- 着手前 recall の促し(recall nudge)---
    # 「タスク開始時に recall する」という記憶プロトコルの責務は、モデルの
    # 自発性に任せると世代交代で静かに守られなくなる(2026-08 実測: モデルに
    # よってセッション実施率に 74% / 48% の開きが出た)。セッション最初の
    # 実質的な発話で一度だけ促しを差し込み、想起の起点をモデル非依存にする。
    recall_nudge: bool = True

    # --- 初回発話の recall 代行(first prompt recall。「代謝する脳」プラン Phase 2)---
    # recall_nudge は「促し」でありモデルの遵守に依存する(モデル非依存にならない)。
    # 代行は、セッション初回の実質発話そのものをクエリに recall を実行して結果を
    # 注入する(8/12 の SessionStart 代行の初回発話版)。失敗時はナッジへ縮退。
    # 代償: 初回発話に数秒(埋め込みモデルのロード)が乗る。
    first_prompt_recall: bool = True
    first_prompt_recall_limit: int = 4

    # --- 自分の思考をキューにする連想(assistant cue。Phase 2)---
    # 想起のきっかけが「ユーザー発話」に限られる問題への対策。
    # "active" = 毎発話時に、直前のアシスタント発話(transcript の末尾)を
    #            キューに surface し、浮上した記憶を文脈へ注入する(本番)。
    # "shadow" = SessionEnd にログだけ残す(観察)。 "off" = 何もしない。
    # ユーザー方針(R8.8.21)「体感もテストにしたいから本番で」により既定 active。
    assistant_cue_mode: str = "active"
    assistant_cue_max_chars: int = 1500

    # --- ローカル LLM(眠りの統合・蒸留・DMN の「文章を生成する係」。Ollama 互換 API)---
    # engram 本体は LLM を持たない。到達できなければ各処理は静かに縮退する。
    llm_url: str = "http://localhost:11434/api/chat"
    llm_model: str = "gemma4:e2b"
    llm_timeout_sec: int = 600

    # --- 夜間の眠り一式(engram night)---
    night_agent_timeout_sec: int = 1200     # 外部エージェント CLI・night.d の1本あたり上限
    # ANTHROPIC_API_KEY / OPENAI_API_KEY が環境にあるとき claude / codex を夜に使うか
    # (従量課金になりうるため既定は使わない。定額の利用枠でログインしていれば鍵は不要)
    night_agents_allow_api_key: bool = False
    # 夜の整理の要約に、入れた人の claude / codex を使うかの本人の同意。
    # None=まだ聞いていない(使わない) / True=使う / False=使わない。
    # 使うと利用枠が毎晩減り、記憶の文章が Anthropic / OpenAI へ送られるため、既定は使わない。
    # setup が対話で1回聞き、`engram night --allow claude` / `--deny claude` で変えられる
    night_allow_claude: bool | None = None
    night_allow_codex: bool | None = None
    distill_window_hours: float = 48.0      # 蒸留の材料にする episode の新しさ
    distill_max_episodes: int = 10          # 1晩に蒸留する episode 数の上限
    # スキル自律作成の置き場と、使われない auto スキルの引退日数
    skills_dir: str = "~/.claude/skills"
    auto_skill_retire_days: int = 30

    # --- 自動実行(OS スケジューラ)---
    # engram setup が夜の整理(毎晩)と DMN(15分毎)を登録する。止めた人の意思は
    # `engram night --uninstall` / `engram dmn --uninstall` がここに false で残し、
    # setup を再実行しても勝手に戻さない。この1行を false にしても止められる
    schedule_night: bool = True
    schedule_dmn: bool = True
    # 同じ PC に engram を2つ入れるとき、タスク名(EngramSleepNight 等)の後ろに付ける語
    schedule_task_suffix: str = ""

    # --- DMN(自発思考。engram dmn)---
    # dmn_watch: 「世界の変化」として見張るファイル/フォルダ(更新されたら起きる理由になる)。
    # 例: 夜勤の報告フォルダ・タスク台帳・日誌。利用者の間取りはここに書き、本体には焼き込まない。
    dmn_watch: list = field(default_factory=list)
    dmn_diary_dir: str | None = None        # 日記の置き場(既定 data_dir/dmn/diary)
    dmn_p_fire: float = 0.25                # 無理由ティックの起床確率(15分毎→約1時間に1回)
    dmn_min_gap_min: int = 20               # 起床の最短間隔(分)
    dmn_max_per_day: int = 12               # 暴発防止
    dmn_new_memory_threshold: int = 3       # これ以上の新規記憶で「理由あり」
    dmn_importance: int = 3                 # 思いつき(episode)の importance

    # --- RSI ループ回転計(engram rsi)---
    # 提案台帳(Markdown 表。1列目=ID、3列目=状態)の場所と「採用済み」と見なす状態欄のパターン
    rsi_ledger_path: str | None = None
    rsi_ledger_done_pattern: str = r"配備済み|完了|発効|done|shipped|adopted"

    # --- 仮説サイクル(engram 0.25。hypothesis.py)---
    # 仮説を立てて確かめるのはモデル(ルーティン)、engram は仮説・判定・元の記憶を結んで持つ
    hyp_default_due_days: int = 3           # 期限の既定(書いた日+N日。同日判定は禁止)
    hyp_importance: int = 4                 # 仮説(episode)の importance
    hyp_result_importance: int = 6          # 当たりから作る knowledge の importance
    hyp_min_age_hours: float = 12.0         # 書いてからこれだけ経つまで判定しない(日付が変わることも必要)
    hyp_max_attempts: int = 2              # 判定不能でも再挑戦する回数の上限
    hyp_retry_days: int = 7                 # 判定不能のとき期限を延ばす日数
    hyp_evidence_min_cos: float = 0.80      # 証拠候補とみなす近さ(Ruri は同話題で 0.8 台)
    # 種選び: 外部(Web)で確かめるため、既定は共通の部屋だけ。仕事の部屋は入れない
    hyp_seed_rooms: list = field(default_factory=lambda: ["common"])
    # 好み(preference)は本人についての記憶で Web では確かめられないため既定では入れない。
    # 沈んだ(cold)記憶も種にする=忘れかけた知識どうしを組み合わせ直す
    hyp_seed_types: list = field(default_factory=lambda: ["knowledge", "project"])
    hyp_seed_tiers: list = field(default_factory=lambda: ["hot", "cold"])
    hyp_seed_band: list = field(default_factory=lambda: [0.55, 0.85])  # 類似度の分位の帯
    hyp_seed_chars: int = 500               # 種として返す本文の長さ

    # --- 記憶の部屋(仕事/個人の文脈分離)---
    # {ディレクトリのプレフィックス: 部屋名}。最長一致で判定、該当なしは "common"
    room_paths: dict = field(default_factory=dict)

    # --- 媒体の記憶(画像・音声・動画・文書。engram 0.24)---
    # 文章の記憶とは別の棚。見た目・音の埋め込み(EmbeddingGemma 2)は別 venv の
    # 索引係(engram-media)が作り、index.db の media_* 表に書く。本体は TCP で
    # 索引係に問い合わせるだけで重い部品を import しない。media_enabled=false なら
    # 本体側の入口(recall の modality、CLI の media)は何もしない
    media_enabled: bool = False
    # 索引係を動かす Python(別 venv)。None なら data_dir/media/venv の既定位置
    media_python: str | None = None
    # {フォルダ: 部屋名}。room_paths と同じ型。ここに書いたフォルダ配下が索引の対象
    media_roots: dict = field(default_factory=dict)
    # 取り込む種別。音声は精度が大分類止まりのため既定では入れない
    media_kinds: list = field(default_factory=lambda: ["image", "video", "document"])
    media_min_bytes: int = 20_000           # これ未満の画像(アイコン・ロゴ)は取らない
    # 走査で飛ばすフォルダ名(fnmatch のパターン。パスのどの階層でも一致したら配下ごと除外)。
    # 既定は汎用の雑音だけ。制作途中の書き出し物(review/thumb/render_parts…)のような利用者固有の
    # 除外は config.toml で足す。除外に変えたフォルダの既存行は次の索引で落ちる
    media_exclude_dirs: list = field(default_factory=lambda: [
        "node_modules", ".git", "__pycache__", ".cache", "venv", ".venv", ".claude", ".agent",
        "$RECYCLE.BIN", "System Volume Information", ".Trash*"])
    media_dup_threshold: float = 0.98       # これ以上のコサインはほぼ同一→代表に畳む
    # 索引係の待受。None(既定)= 本体デーモンの番号 + 1(media_port_effective)
    media_port: int | None = None
    media_caption: bool = True              # ローカルLLM(vision)で短い説明文を付ける
    media_caption_model: str | None = None  # None なら llm_model を使う
    media_cold_days: float = 45.0           # これだけ使われなければ cold へ(消さない)
    media_find_timeout_sec: float = 15.0    # 索引係の起動待ちの上限(初回はモデル読込40秒)
    media_video_segment_sec: float = 60.0   # 動画は約1分ごとに1段(1埋め込み≒32コマ/1fps)
    media_audio_segment_sec: float = 300.0  # 音声は約5分ごとに1段
    media_doc_max_pages: int = 20           # 文書は先頭この頁数まで見た目を埋め込む(以降は文字だけ)
    # メール添付の取り込み {名前: {token, client_secret, query, room, save_dir}}。読むだけ
    media_mail: dict = field(default_factory=dict)
    # 媒体の自発的想起(文章の surface と同じ発話フックで、関連しそうな画像・文書を1件添える)。
    # off | shadow(ログのみ) | active(文脈に差し込む)。見た目のコサインは 0.6〜0.9 に圧縮されるため
    # 床は高めに置き、無関係な絵が毎回浮かぶ雑音を避ける。索引係が寝ていれば起こすだけで今回は出さない
    media_surface_mode: str = "active"
    # 見た目のコサインだけでは「画像の話をしていない発言」(0.73〜0.80)と本当の当たり(0.75〜0.85)が
    # 分かれない(R8.10.11 実測)。そこで発言に「画面・スクショ・図・PDF…」のような手がかり語が
    # あるときだけ出す。語は fnmatch ではなく部分一致。空にすると手がかり語の門を外す(雑音注意)
    media_surface_min_cos: float = 0.74
    # 件数の上限で切らず「1位との差」で切る(ユーザー指摘 R8.10.11「なんで1件?」)。1位に近い
    # (差が media_surface_margin 以内の)ものだけを最大 max_items 件。1枚だけ抜けていれば1枚、
    # 横並び(同じ画面の別状態・シリーズのスライド)なら並んだ分が出る
    media_surface_max_items: int = 3
    media_surface_margin: float = 0.02
    media_surface_min_prompt_chars: int = 12
    media_surface_cues: list = field(default_factory=lambda: [
        "画面", "スクショ", "スクリーンショット", "画像", "写真", "図", "イラスト", "スライド", "ポスター", "サムネ",
        "チラシ", "資料", "PDF", "pdf", "文書", "ファイル", "表紙", "ページ", "動画", "キャプチャ", "見た目", "デザイン",
        "絵", "素材", "キー画像", "背景", "ロゴ", "アイコン", "image", "screenshot", "photo", "slide", "diagram",
        "poster", "thumbnail", "document", "figure"])

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def media_python_path(self) -> Path:
        if self.media_python:
            return Path(self.media_python).expanduser()
        import os as _os
        sub = "Scripts/python.exe" if _os.name == "nt" else "bin/python"
        return self.media_dir / "venv" / sub

    @property
    def home_key(self) -> str:
        """この engram の身元(data_dir の正規化した絶対パス)。

        デーモン・索引係は応答にこれを名乗り、窓口は自分と一致する相手の
        応答しか使わない。番号が同じでも、別の記憶を持つ相手を止めたり
        その検索結果を混ぜたりしないため。
        """
        return os.path.normcase(str(Path(self.data_dir).expanduser().resolve()))

    @property
    def daemon_port_effective(self) -> int:
        """実際に使う待受ポート。0 は「空きポートを取る」(テスト用)。"""
        if self.daemon_port is not None:
            return int(self.daemon_port)
        import zlib
        # 52000〜59998 の偶数。奇数側(+1)は索引係が使う
        return 52000 + (zlib.crc32(self.home_key.encode("utf-8")) % 4000) * 2

    @property
    def media_port_effective(self) -> int:
        if self.media_port is not None:
            return int(self.media_port)
        base = self.daemon_port_effective
        return base + 1 if base else 0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def onnx_model_dir(self) -> Path:
        """export-onnx が生成する ONNX モデルの置き場所(モデル名ごとに分離)。"""
        return self.data_dir / "onnx" / self.embed_model.replace("/", "--")


def onnx_model_ready(model_dir: Path) -> bool:
    """export-onnx 済みの ONNX モデル一式が揃っているか(ファイル存在チェックのみ)。

    embedder.OnnxRuriEmbedder.is_available の実体。server 起動直後の
    ENGRAM_PRELOAD=auto 判定でも使うため、numpy 等の重い import を伴う
    embedder モジュールではなく、軽量なこのモジュールに置く。
    """
    d = Path(model_dir)
    return (
        (d / "model.onnx").is_file()
        and (d / "tokenizer.json").is_file()
        and (d / "meta.json").is_file()
    )


def config_path() -> Path:
    """ユーザー設定ファイルの場所(engram setup が生成する)。"""
    return _engram_home() / "config.toml"


def _load_config_file() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}  # 壊れた設定ファイルで起動不能にしない(既定値で動かす)


_PATH_FIELDS = {"memories_dir", "data_dir"}
_ENV_OVERRIDES = {
    "memories_dir": "ENGRAM_MEMORIES_DIR",
    "data_dir": "ENGRAM_DATA_DIR",
    "embed_model": "ENGRAM_EMBED_MODEL",
    "embed_backend": "ENGRAM_EMBED_BACKEND",
}


def get_settings() -> Settings:
    """既定値 < config.toml < 環境変数 の順で上書きして Settings を返す。"""
    overrides: dict = {}
    file_conf = _load_config_file()
    valid = {f.name: f.type for f in fields(Settings)}

    for key, value in file_conf.items():
        if key not in valid:
            continue
        overrides[key] = Path(value) if key in _PATH_FIELDS else value

    for key, env_name in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value:
            overrides[key] = Path(value) if key in _PATH_FIELDS else value

    return Settings(**overrides)


def resolve_room(cwd: str | Path | None, room_paths: dict) -> str:
    """作業ディレクトリから記憶の部屋を決める。

    room_paths のキー(ディレクトリのプレフィックス)と最長一致で照合する。
    区切り文字(\\ と /)と大文字小文字の差は吸収する。該当なしは "common"。
    """
    if not cwd or not room_paths:
        return "common"
    norm = str(cwd).replace("\\", "/").casefold().rstrip("/")
    best_len = -1
    best_room = "common"
    for prefix, room in room_paths.items():
        p = str(prefix).replace("\\", "/").casefold().rstrip("/")
        if not p:
            continue
        if (norm == p or norm.startswith(p + "/")) and len(p) > best_len:
            best_len = len(p)
            best_room = str(room)
    return best_room
