"""MCP サーバー(担当: Agent B)。

mcp 公式 SDK の FastMCP(stdio)で engine の操作をツールとして公開する。

    from mcp.server.fastmcp import FastMCP

実装要件:
- エンジンは初回ツール呼び出し時に遅延構築(build_engine)。起動を速く保つ。
- 各ツールは engine の同名メソッドへ委譲し、dict をそのまま返す。
- docstring がそのままツール説明になるので、エージェントが「いつ呼ぶべきか」を
  判断できる文面にする(日本語でよい)。
- ツール: remember / recall / reinforce / correct / link / forget /
         consolidation_candidates / mark_consolidated / skill_candidates /
         reindex / stats
- 引数は engine のシグネチャに合わせる(now は公開しない)。
- main() でstdio実行: mcp.run()。
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import perf
from .config import resolve_room
from .engine import MemoryEngine, build_engine

# 初期化の応答で渡す案内。Claude Code・Codex・Gemini CLI はこれをモデルに渡すので、
# フックや指示ファイルが無い環境でも記憶の使い方が届く
_INSTRUCTIONS = (
    "engram は人間型の永続記憶です。作業に着手する前に、その主題で recall を呼んで過去の知見・"
    "利用者の流儀・経緯を確かめてください。苦労して解決した原因と解法、重要な判断とその理由、"
    "利用者の好みや訂正は、その場で remember してください(1記憶=1事実)。recall した記憶のうち"
    "実際に役立ったものは reinforce し、誤っていた記憶は forget ではなく correct で直してください。"
    "recall の応答に notice があれば、それは記憶の保守(整理・スキル化)の促しなので、作業の区切りで対応してください。"
    "パスワード・API キー・個人情報は記憶しないでください。"
)

mcp = FastMCP("engram", instructions=_INSTRUCTIONS)

# 記憶整理・スキル化の促しは、フックの無い環境(Codex・Gemini でフック未承認の人、
# Antigravity など)にも届くよう、プロセスの最初の recall の応答にも添える
_nudge_checked = False
_nudge_lock = threading.Lock()


def _mcp_nudges(engine: "MemoryEngine") -> list[str]:
    """このプロセスで1回だけ、時期の来た促しを返す(間隔の管理はフックと共通の状態ファイル)。"""
    global _nudge_checked
    with _nudge_lock:
        if _nudge_checked:
            return []
        _nudge_checked = True
    import time as _time

    from .hooks import (_consolidation_nudge, _read_consolidation_state, _skill_nudge,
                        _write_consolidation_state)

    settings = engine.settings
    try:
        state = _read_consolidation_state(settings)
        # 候補数の棚卸しは本来セッション終了のフックが行う。フックが無い環境のため、
        # 1日以上古ければここで数え直す
        if _time.time() - float(state.get("checked_at", 0) or 0) > 86400:
            upd: dict = {"checked_at": _time.time()}
            if settings.consolidate_nudge:
                upd["clusters"] = len(engine.consolidation_candidates().get("clusters", []))
            if settings.skill_nudge:
                upd["skill_clusters"] = len(engine.skill_candidates().get("clusters", []))
            _write_consolidation_state(settings, upd)
        return [n for n in (_consolidation_nudge(settings), _skill_nudge(settings)) if n]
    except Exception:
        return []

# モジュールレベル遅延シングルトン(初回ツール呼び出しで構築)
_engine: MemoryEngine | None = None
_engine_lock = threading.Lock()

# サーバーの既定の部屋。プロセス起動時の作業ディレクトリ(=エージェントを
# 起動したプロジェクト)から config の room_paths で決まる
_room: str | None = None

# 最後にツールが呼ばれた(またはモデルが先読みされた)時刻。アイドル解放の
# 判定に使う。CPython では float 代入はアトミックなのでロック不要
_last_activity: float = time.monotonic()


def _touch() -> None:
    """アイドル解放タイマーをリセットする(ツール呼び出し・先読み完了時)。"""
    global _last_activity
    _last_activity = time.monotonic()


# 裏で埋め込みを使う長い処理(起動時 reindex・reindex ツール)の実行数。
# 0 より大きい間はアイドル解放しない。
# 背景(R8.10.23 実測): 起動時 reindex は _touch() を通らないため、600 秒を
# 過ぎると解放スレッドが毎分モデルを捨て、reindex は次の1件のために ONNX
# セッションを作り直す(全コアを使う重い処理)。読み込み↔解放の空回りで
# reindex が終わらず、2プロセスで PC 全体の CPU の 72% を焼き続けた。
_busy_count = 0
_busy_lock = threading.Lock()


@contextlib.contextmanager
def _busy():
    """この間はアイドル解放を止める。抜けたらそこからアイドル計測を始める。"""
    global _busy_count
    with _busy_lock:
        _busy_count += 1
    try:
        yield
    finally:
        with _busy_lock:
            _busy_count -= 1
        _touch()


@contextlib.contextmanager
def _startup_index_lock(data_dir: Path | None):
    """起動時の索引点検を同じマシンの1プロセスだけに通すロック(待たない)。

    index.db はマシンに1つで全プロセスが共有するため、点検と reindex は誰か
    1プロセスがやれば足りる。取れたら True、他プロセスが点検中なら False を
    yield する。ロックは OS のファイルロックなので、持ち主のプロセスが落ちれば
    自動で外れる(古いロックが残って永久に点検されない事故は起きない)。
    data_dir が無い・ロックファイルを作れない場合はロック無しで通す(従来どおり)。
    """
    if data_dir is None:
        yield True
        return
    try:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        fh = open(Path(data_dir) / "startup_index.lock", "a+b")
    except OSError:
        yield True
        return
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        fh.close()


def _get_engine() -> MemoryEngine:
    """エンジンを遅延構築して返す(スレッド安全)。"""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = build_engine(embedder=_make_server_embedder())
            _run_startup_index_check(_engine)
    return _engine


def _make_server_embedder():
    """MCP サーバー用の埋め込み部品を選ぶ。

    ONNX 経路で embed_via_daemon が有効なら、計算を共有デーモンに頼む窓口
    (DaemonEmbedder)にする。モデルはデーモン1つだけが持ち、セッションの数だけ
    積み上がらない。torch 経路・無効化時・ENGRAM_DISABLE_DAEMON(テスト)の
    ときは従来どおり手元の実行系を返す。
    """
    from .config import get_settings
    from .embedder import DaemonEmbedder, OnnxRuriEmbedder, make_embedder

    settings = get_settings()
    local = make_embedder(settings)
    if (
        getattr(settings, "embed_via_daemon", True)
        and isinstance(local, OnnxRuriEmbedder)
        and not os.environ.get("ENGRAM_DISABLE_DAEMON")
    ):
        return DaemonEmbedder(settings, local)
    return local


def _timed(name: str):
    """perf.timed への薄い委譲(settings は _get_engine 経由で取得しキャッシュ)。

    FastMCP はツール関数のシグネチャを検査してスキーマを作るため、ツール関数
    自体をデコレータで包むのではなく、各ツールの本体内で `with _timed(...):`
    のように使う(関数のシグネチャ・docstring には一切触れない)。
    全ツールがここを通るので、アイドル解放タイマーのリセットも兼ねる。
    """
    _touch()
    return perf.timed(_get_engine().settings, "tool", name)


def _run_startup_index_check(engine: MemoryEngine) -> None:
    """起動時に Markdown と index の乖離を検知し、auto なら reindex / warn なら警告。

    記憶 Markdown は共有(例: Google Drive)でも index.db はマシンごとローカルなため、
    他マシンが書いた記憶が index に取り込まれず recall に出ない盲点が生じる。これを
    起動時に解消する。失敗してもエンジン提供は止めない(記憶基盤の可用性を最優先)。
    """
    import sys

    try:
        mode = getattr(engine.settings, "startup_index_check", "auto")
        data_dir = getattr(engine.settings, "data_dir", None)
        with _busy(), _startup_index_lock(data_dir) as acquired:
            if not acquired:
                print(
                    "engram: startup index check skipped "
                    "(another engram process on this machine is checking)",
                    file=sys.stderr,
                )
                return
            res = engine.check_index_freshness(mode=mode)
        action = res.get("action")
        if action == "reindexed":
            print(
                f"engram: index out of sync (markdown={res.get('markdown')} "
                f"index={res.get('index')}) -> reindexed {res.get('reindex')}",
                file=sys.stderr,
            )
        elif action == "warn":
            print(
                f"engram: WARNING memory index out of sync "
                f"(markdown={res.get('markdown')} vs index={res.get('index')}). "
                f"Run 'engram reindex' to sync this machine.",
                file=sys.stderr,
            )
    except Exception as e:
        print(f"engram: startup index check skipped: {e}", file=sys.stderr)


def _default_room() -> str:
    """サーバー起動ディレクトリから記憶の部屋を解決する(初回のみ計算)。"""
    global _room
    if _room is None:
        try:
            settings = _get_engine().settings
            _room = resolve_room(Path.cwd(), settings.room_paths)
        except Exception:
            _room = "common"
    return _room


@mcp.tool()
def remember(
    content: str,
    type: str,
    importance: int,
    tags: list[str] | None = None,
    source: str = "unknown",
    related_ids: list[str] | None = None,
    room: str | None = None,
    valence: float = 0.0,
) -> dict:
    """新しい記憶を保存する。

    タスク中に重要な情報(事実・好み・プロジェクト状況・出来事)を発見したとき呼ぶ。
    importance は 1〜10 で文脈の重要度を自己採点する。
    valence は -1..1 の感情価(負=避けろ/失敗・訂正、正=近づけ/成功。省略可)。
    既存の類似記憶(cos が dup_threshold=0.95 以上)がある場合は重複強化して返す。
    room は通常指定不要(作業ディレクトリから自動判定)。どの文脈でも使う
    普遍的な記憶だけ room="common" を明示する。
    """
    engine = _get_engine()
    with _timed("remember"):
        return engine.remember(
            content=content,
            type=type,
            importance=importance,
            tags=tags,
            source=source,
            related_ids=related_ids,
            room=room if room is not None else _default_room(),
            valence=valence,
        )


@mcp.tool()
def recall(
    query: str,
    mode: str = "fast",
    limit: int = 5,
    type: str | None = None,
    record_hits: bool = True,
    room: str | None = None,
    modality: str | None = None,
) -> dict:
    """記憶を検索して返す。

    タスク開始時に必ず呼ぶ。関連する過去の知識・好み・プロジェクト状況を想起できる。
    mode="fast" は tier=hot のみ高速検索。スコアが低いと自動で deep に切り替わる。
    mode="deep" は cold/superseded/episode も含め連想リンクを辿って広く探す。
    mode="exhaustive" は活性度を無視し関連度のみで全記憶を総当たりする。
    「確かに記録したはずなのに fast/deep で出ない」沈んだ記憶を掘り起こす最終手段。
    room は通常指定不要(現在の部屋+共通だけを検索する)。room="*" で全部屋を
    横断検索できるが、仕事/個人の分離を壊さないよう必要時のみ使うこと。
    modality を "image" / "video" / "audio" / "document" / "media"(全種別) にすると、
    文章の記憶ではなく媒体の棚(画像・音声・動画・文書ファイルを見た目・音で索引した別の棚)を
    言葉で探す。結果の各 hit に path(ファイルの場所)・kind・caption(説明文)・page が付く。
    「前に作ったあの図解」「あの画面写真」「◯◯から届いた見積書のPDF」のような、ファイルを
    探す問いに使う。索引係が寝ていると初回は起動待ち(最大15秒)で空振りすることがある。
    """
    engine = _get_engine()
    with _timed("recall"):
        result = engine.recall(
            query=query,
            mode=mode,
            limit=limit,
            type=type,
            record_hits=record_hits,
            modality=modality,
            room=room if room is not None else _default_room(),
        )
    if record_hits and isinstance(result, dict):
        notes = _mcp_nudges(engine)
        if notes:
            result["notice"] = "\n\n".join(notes)
    return result


@mcp.tool()
def read(ids: list[str]) -> dict:
    """id を指定して記憶の本文を読む。

    自発的想起の「縁」(id と要旨1行だけが差し込まれる弱い想起)から、今の話に効きそうな
    ものを選んで本文を取るための入口。読んだ記憶には弱い使用記録(recall_hit)が付く。
    役立ったら reinforce で申告する(read だけでは強い強化にならない)。
    """
    engine = _get_engine()
    with _timed("read"):
        return engine.read(ids=ids)


@mcp.tool()
def reinforce(
    ids: list[str],
    strength: float = 1.0,
) -> dict:
    """実際に役立った記憶の使用を報告する。

    タスク完了時、実際に役立った記憶の id を報告する。
    強化された記憶は次回の recall で上位に浮上しやすくなる。
    同時に複数 id を渡すと、それらの記憶が共起リンクで結ばれる(ヘッブ則)。
    strength は 0.1〜3.0 で強化の強さを指定できる。
    """
    engine = _get_engine()
    with _timed("reinforce"):
        return engine.reinforce(ids=ids, strength=strength)


@mcp.tool()
def correct(
    id: str,
    corrected_content: str,
    reason: str,
    source: str = "unknown",
) -> dict:
    """記憶が誤っていたとき forget ではなくこれを使う。

    旧記憶を superseded(訂正済み)に降格し、訂正の経緯を記録した新記憶を作成する。
    誤りを明示的に記録することで、同じ間違いの繰り返しを防ぐ(ハイパーコレクション効果)。
    """
    engine = _get_engine()
    with _timed("correct"):
        return engine.correct(
            id=id,
            corrected_content=corrected_content,
            reason=reason,
            source=source,
        )


@mcp.tool()
def link(src: str, dst: str) -> dict:
    """2つの記憶に explicit リンクを張る。

    関連する記憶を手動で結びたいときに呼ぶ。
    deep recall でリンクを辿って連想的に想起できるようになる。
    """
    engine = _get_engine()
    with _timed("link"):
        return engine.link(src=src, dst=dst)


@mcp.tool()
def forget(id: str) -> dict:
    """記憶をソフト削除する(ゴミ箱へ移動)。

    不要になった記憶を検索対象から外したいときに呼ぶ。
    物理削除ではなくゴミ箱移動なので、誤削除の場合は復元できる。
    誤りを訂正したい場合は forget ではなく correct を使うこと。
    """
    engine = _get_engine()
    with _timed("forget"):
        return engine.forget(id=id)


@mcp.tool()
def consolidation_candidates() -> dict:
    """統合候補の episode クラスタを返す。

    就寝前(セッション終了時)に呼び、類似する古い episode をまとめて
    知識・プロジェクト記憶として圧縮する候補を提示する。
    LLM が要約を生成し、mark_consolidated で統合を完了する。
    """
    engine = _get_engine()
    with _timed("consolidation_candidates"):
        return engine.consolidation_candidates()


@mcp.tool()
def mark_consolidated(episode_ids: list[str], new_memory_id: str) -> dict:
    """統合完了を記録する。

    consolidation_candidates で提示されたクラスタを LLM が要約し、
    remember で新記憶を作成した後にこれを呼ぶ。
    元の episode は cold(長期保存)に降格し、derived_from リンクで繋がれる。
    スキル化候補(skill_candidates)を整理したときも、対象 episode を
    このツールで cold に降格して使う。
    """
    engine = _get_engine()
    with _timed("mark_consolidated"):
        result = engine.mark_consolidated(
            episode_ids=episode_ids,
            new_memory_id=new_memory_id,
        )
    # 統合完了でクラスタ数が変わるので促し用の状態を即時更新する。
    # 怠ると次の session-end まで古いクラスタ数のまま促し続ける
    # (consolidation・スキル化候補の両方を更新する)
    try:
        import time

        from .hooks import _write_consolidation_state

        n = len(engine.consolidation_candidates().get("clusters", []))
        n_skill = len(engine.skill_candidates().get("clusters", []))
        _write_consolidation_state(
            engine.settings,
            {
                "clusters": n,
                "skill_clusters": n_skill,
                "checked_at": time.time(),
            },
        )
    except Exception:
        pass
    return result


@mcp.tool()
def skill_candidates() -> dict:
    """スキル化候補の episode クラスタを返す。

    同じ形の作業(手順)を記録した episode が3件以上(既定値。三度ルール)
    似たクラスタを成しているとき、その手順を再利用可能なスキル(手順書。
    Claude Code なら SKILL.md 等)として切り出す価値があるかを判断する
    材料として使う。consolidation_candidates と違い年齢フィルタはない
    (直近の繰り返し作業こそ対象)。
    クラスタが見つかっても、スキル化するかどうかは必ずユーザーに提案して
    承認を得ること。勝手に作成・配備しない。
    採用・見送りが決まったら remember(type=knowledge)で経緯を記録し、
    mark_consolidated(episode_ids, new_memory_id) で元 episode を整理する。
    """
    engine = _get_engine()
    with _timed("skill_candidates"):
        return engine.skill_candidates()


@mcp.tool()
def reindex() -> dict:
    """Markdown ファイルから DB インデックスを再構築する。

    手動でファイルを編集した後や、DB が壊れた疑いがあるときに呼ぶ。
    差異のある記憶だけ再埋め込みするため、全件再構築より高速。
    """
    engine = _get_engine()
    with _timed("reindex"), _busy():
        return engine.reindex()


@mcp.tool()
def stats() -> dict:
    """記憶の統計情報を返す。

    記憶の件数(type別・tier別)、アクセスイベント数、リンク数などを確認できる。
    """
    engine = _get_engine()
    with _timed("stats"):
        return engine.stats()


@mcp.tool()
def hypothesis_seeds(n: int = 3) -> dict:
    """仮説の種になる記憶の組を返す(仮説サイクルの1歩目)。

    近すぎず遠すぎない記憶の組を選ぶ。過去に当たりやすかった組み合わせの種類ほど選ばれやすい。
    返った組を読み、2つを組み合わせると言えそうなこと(仮説)を考えて hypothesize で書く。
    既定では共通の部屋の記憶だけを使う(Web で確かめるため、仕事の記憶は入れない)。
    """
    engine = _get_engine()
    with _timed("hypothesis_seeds"):
        from . import hypothesis as _h

        return _h.hypothesis_seeds(engine, n=n)


@mcp.tool()
def hypothesize(
    claim: str,
    source_ids: list[str],
    falsify_if: str,
    test_plan: str = "",
    due: str | None = None,
    source: str = "unknown",
    origin: str = "",
) -> dict:
    """記憶の組み合わせから立てた仮説を書く(未決の仮説として持ち続け、期限に確かめる)。

    claim は確かめられる形の主張。source_ids は元にした記憶 ID(2件以上)。
    falsify_if は「外れたと分かる条件」で、書けない仮説は受け付けない。
    test_plan は確かめ方(どんな記事・論文・公式資料で確かめるか)。due は期限
    (YYYY-MM-DD か日数。既定3日後)。書いた日には判定できない(自己採点の防止)。
    """
    engine = _get_engine()
    with _timed("hypothesize"):
        from . import hypothesis as _h

        return _h.hypothesize(engine, claim, source_ids, falsify_if, test_plan,
                              due=due, source=source, origin=origin)


@mcp.tool()
def hypotheses(status: str = "open", limit: int = 20) -> dict:
    """仮説の一覧。status は open(未決)/ due(期限が来て今日判定してよいもの)/ resolved / all。

    未決の仮説には evidence_candidates(仮説の後に入った、意味の近い記憶)が付く。
    summary に答え合わせ率・当たり率・種類の組ごとの成績が入る。
    """
    engine = _get_engine()
    with _timed("hypotheses"):
        from . import hypothesis as _h

        return _h.list_hypotheses(engine, status=status, limit=limit)


@mcp.tool()
def resolve_hypothesis(
    id: str,
    verdict: str,
    evidence: str,
    evidence_urls: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    note: str = "",
    judge: str = "model",
    source: str = "unknown",
) -> dict:
    """仮説に判定を付ける。verdict は 当たり / 外れ / 判定不能。

    evidence は根拠の要約、evidence_urls は確かめた記事・論文の URL。
    当たり → 検証済みの知見(knowledge)を新しく作り、仮説と元の記憶に結ぶ(元の記憶を温める)。
    外れ → correct と同じく誤りごと深く刻む。判定不能 → 期限を延ばして再挑戦(上限で打ち切り)。
    judge はモデルの判定なら "model"(既定)。ユーザーの判断なら "user"。
    """
    engine = _get_engine()
    with _timed("resolve_hypothesis"):
        from . import hypothesis as _h

        return _h.resolve(engine, id, verdict, evidence, evidence_urls=evidence_urls,
                          evidence_ids=evidence_ids, note=note, judge=judge, source=source)


@mcp.tool()
def remember_file(path: str, note: str | None = None, room: str | None = None) -> dict:
    """ファイル1件(画像・音声・動画・PDF/Office 文書)を媒体の棚に今すぐ覚える。

    「このファイル覚えて」「この画像をあとで探せるようにして」と頼まれたとき、または作業中に
    作った成果物(スライド画像・図解・録音)を後から言葉で探せるようにしたいときに呼ぶ。
    フォルダ単位の索引(夜間の差分)を待たずに1件だけ即時に索引する。note は人が付けた短い
    説明(あれば説明文として一緒に覚える)。room は省略時に作業ディレクトリから判定。
    ファイル本体は動かさない(場所を覚えるだけ)。見つけるときは recall(modality=...)。
    """
    engine = _get_engine()
    with _timed("remember_file"):
        from . import media as _media

        res = _media.index_path(engine.settings, path, room=room or _default_room(), source="manual",
                                source_meta={"note": note} if note else None)
        if res.get("ok") and note:
            for it in res.get("items") or []:
                if it and not it.get("caption"):
                    try:
                        from .media_schema import MediaIndex

                        MediaIndex.open(engine.settings.db_path).set_caption(it["id"], note)
                    except Exception:
                        pass
        return res


@mcp.tool()
def media_status() -> dict:
    """媒体の棚(画像・音声・動画・文書の索引)の状態。索引係の稼働・件数・対象フォルダ・索引作成の進み具合。"""
    engine = _get_engine()
    with _timed("media_status"):
        from . import media as _media

        return _media.status(engine.settings)


def _preload() -> None:
    """埋め込みモデルを先読みする(失敗しても起動は続行)。preload 全体の所要時間を計測する。"""
    import sys
    import time

    # settings は build_engine 前でも config だけから取れるので、エンジン構築
    # 自体が失敗しても計測できるようにここで先に用意する
    from .config import get_settings

    settings = get_settings()
    start = time.perf_counter()
    ok = True
    try:
        engine = _get_engine()
        engine.embedder.embed_query("ウォームアップ")
        # 先読み完了からアイドル計測を始める(ツールが一度も呼ばれない
        # プロセスは idle_sec 後にモデルが解放される — それが狙い)
        _touch()
        print("engram: engine preloaded", file=sys.stderr)
    except Exception as e:  # 失敗しても起動は続け、初回ツール呼び出しで再試行
        ok = False
        print(f"engram: preload failed, will retry lazily: {e}", file=sys.stderr)
    finally:
        if settings.perf_log:
            ms = (time.perf_counter() - start) * 1000.0
            perf.append_perf(
                settings,
                {"ts": time.time(), "kind": "preload", "name": "preload", "ms": ms, "ok": ok},
            )


def _resolve_idle_unload_sec(raw: str | None) -> float:
    """ENGRAM_IDLE_UNLOAD_SEC の値を解決する(純粋関数・テスト対象)。

    既定 600 秒(10分)。0 以下で無効。数値でない値は既定に落とす。
    """
    default = 600.0
    if raw is None or not raw.strip():
        return default
    try:
        sec = float(raw.strip())
    except ValueError:
        return default
    return max(0.0, sec)


def _maybe_idle_unload(idle_sec: float) -> bool:
    """アイドル判定1回分。解放を実行したら True(テスト対象)。

    解放するのは unload() を持つ embedder(=ONNX 経路)だけ。torch 経路は
    再ロードが Windows で3分級になる病理(main() のコメント参照)があるため
    対象外。エンジン未構築(モデル未ロード)なら何もしない — 解放のために
    エンジンを構築しては本末転倒。
    """
    import gc
    import sys

    engine = _engine  # _get_engine() は使わない(構築を誘発するため)
    if engine is None:
        return False
    embedder = engine.embedder
    unload = getattr(embedder, "unload", None)
    if unload is None or not getattr(embedder, "loaded", False):
        return False
    if _busy_count > 0:  # 裏の reindex 中は使用中扱い(空回り防止。_busy 参照)
        return False
    idle = time.monotonic() - _last_activity
    if idle < idle_sec:
        return False
    if not unload():
        return False
    gc.collect()  # ORT セッションのアリーナを即時返却させる
    print(
        f"engram: idle {int(idle)}s >= {int(idle_sec)}s, "
        "embedding model unloaded (reloads on next tool call)",
        file=sys.stderr,
    )
    try:
        if engine.settings.perf_log:
            perf.append_perf(
                engine.settings,
                {
                    "ts": time.time(),
                    "kind": "unload",
                    "name": "idle_unload",
                    "ms": idle * 1000.0,
                    "ok": True,
                },
            )
    except Exception:
        pass  # 記録は最善努力
    return True


def _idle_unload_loop(idle_sec: float) -> None:
    """アイドル時に埋め込みモデルをメモリから解放するデーモンループ。

    背景: stdio MCP は常駐プロセスであり、Codex のように1クライアントが
    実行ホストごとに MCP プロセスを起動・残留させる環境では、約1.1GBの
    ONNX モデルがプロセス数ぶん積み上がる(実機: 26.810 で4プロセス同時、
    16GB PC の空きが約1GBまで低下)。ENGRAM_PRELOAD=off は「使うまで
    ロードしない」だが、実際にツールを使ったプロセスのロード分は残る。
    このループが「使い終わったら返す」の側を受け持つ。
    """
    # 起床間隔は idle_sec より細かく(最大60秒)。解放は最悪 idle_sec+間隔 後
    interval = min(60.0, max(1.0, idle_sec / 4))
    while True:
        time.sleep(interval)
        try:
            _maybe_idle_unload(idle_sec)
        except Exception:
            # 解放は最善努力。失敗してもサーバー本体は止めない
            continue


def _resolve_preload_mode(raw: str | None, onnx_ready: bool) -> str:
    """ENGRAM_PRELOAD の値を実際の先読み方式に解決する(純粋関数・テスト対象)。

    blocking / background / off の明示値はそのまま。auto(既定)および未知の値は、
    ONNX モデルが生成済みなら background、torch フォールバック環境なら blocking。
    根拠は main() 冒頭のコメントを参照。
    """
    mode = (raw or "auto").strip().lower()
    if mode in ("blocking", "background", "off"):
        return mode
    return "background" if onnx_ready else "blocking"


def main() -> None:
    """stdio MCP サーバーを起動する。"""
    # 【v0.6.0〜】既定の実行系は ONNX(embed_backend=auto + export-onnx 済み)。
    # 以下の torch 病理の記録は、ONNX 未生成のフォールバック環境に戻ると
    # 何が起きるかの記録として残す。
    #
    # torch / sentence_transformers の import は非常に重い(実測: import だけで
    # cold 50秒超)。これを mcp.run() の前にメインスレッドで実行すると initialize
    # ハンドシェイクがその間ブロックされ、起動タイムアウトの短い MCP クライアント
    # は接続を打ち切ってしまう(実例: Antigravity IDE の "context canceled"、
    # Claude Code の起動タイムアウト既定30秒によるコールド起動時の断続的な不通)。
    #
    # 一方で、mcp.run() のイベントループ稼働中に「別スレッドで」この import を行うと
    # Windows では桁違いに遅くなる(実測: メインスレッド 12〜24秒 → デーモン/ワーカー
    # スレッド 約184秒。2026-07-02 の Claude Code MCP ログ2セッションで再現)。
    # background 先読みはハンドシェイクこそ1.5秒で返すが、初回 recall がこの遅い
    # ロードを待って 180 秒級になり、クライアントのツールタイムアウトに化ける。
    # イベントループ停止中のスレッド import は 6 秒で終わる(単体では再現しない)
    # ため、ループとの GIL/DLL ローダー競合とみられる。教訓: 重い import は
    # イベントループが動き出す前にメインスレッドで済ませるのが唯一速い経路。
    #
    # 【v0.10.0〜】上記の病理が ONNX 経路には無いことを実測で確認した
    # (2026-07-12: イベントループ稼働中のデーモンスレッドでの embedder ロードが
    # 4.9秒 — メインスレッド 4.7秒とほぼ同じ。184秒級への劣化は torch の
    # DLL ローダー固有)。そこで既定を auto にし、ONNX モデルが生成済みなら
    # background(ハンドシェイク即応答・初回ツールは最大でも数秒待ち)、
    # torch フォールバック環境では従来どおり blocking を選ぶ。
    # 背景: 起動タイムアウトを延ばす設定が効かない MCP クライアントが実在する
    # (実例: Codex Desktop 26.707 は startup_timeout_sec=120 を認識しつつ
    # blocking の十数秒を待たず、初期化未完了のまま張り付いた)。
    #
    # ENGRAM_PRELOAD で先読み方式を明示できる:
    #   auto (既定)      — ONNX 生成済みなら background、無ければ blocking。
    #   blocking        — 起動時にメインスレッドで先読み。ハンドシェイクは import
    #                     完了まで待つ(torch 経路では warm 12〜24秒 / cold 50秒超の
    #                     ため、クライアント側の MCP 起動タイムアウトを 120 秒以上に
    #                     すること。Claude Code は settings.json の env で
    #                     MCP_TIMEOUT=120000)。接続後の recall は常に即応答する。
    #   background      — 先読みをデーモンスレッドに回し、mcp.run() を即実行する。
    #                     ONNX 経路では安全(初回ツールが先読み完了を数秒待つだけ)。
    #                     torch 経路では上記の病理により Windows で初回 recall が
    #                     3 分級になりうるので明示指定は非推奨。
    #   off             — 先読みしない。初回ツール呼び出し時に遅延ロードする
    #                     (torch 経路では background と同じ病理を踏む)。
    # FastMCP は同期ツールをワーカースレッドで実行するため、_get_engine と
    # RuriEmbedder._load はロックで多重ロードを防いでいる。
    from .config import get_settings, onnx_model_ready

    mode = _resolve_preload_mode(
        os.environ.get("ENGRAM_PRELOAD"),
        onnx_model_ready(get_settings().onnx_model_dir),
    )
    if mode == "blocking":
        _preload()
    elif mode == "background":
        threading.Thread(
            target=_preload, name="engram-preload", daemon=True
        ).start()

    # ENGRAM_IDLE_UNLOAD_SEC 秒(既定600)ツールが呼ばれなければ埋め込み
    # モデルを解放する(0以下で無効)。詳細は _idle_unload_loop の docstring
    idle_sec = _resolve_idle_unload_sec(os.environ.get("ENGRAM_IDLE_UNLOAD_SEC"))
    if idle_sec > 0:
        threading.Thread(
            target=_idle_unload_loop,
            args=(idle_sec,),
            name="engram-idle-unload",
            daemon=True,
        ).start()

    mcp.run()
