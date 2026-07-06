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

import os
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import perf
from .config import resolve_room
from .engine import MemoryEngine, build_engine

mcp = FastMCP("engram")

# モジュールレベル遅延シングルトン(初回ツール呼び出しで構築)
_engine: MemoryEngine | None = None
_engine_lock = threading.Lock()

# サーバーの既定の部屋。プロセス起動時の作業ディレクトリ(=エージェントを
# 起動したプロジェクト)から config の room_paths で決まる
_room: str | None = None


def _get_engine() -> MemoryEngine:
    """エンジンを遅延構築して返す(スレッド安全)。"""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = build_engine()
            _run_startup_index_check(_engine)
    return _engine


def _timed(name: str):
    """perf.timed への薄い委譲(settings は _get_engine 経由で取得しキャッシュ)。

    FastMCP はツール関数のシグネチャを検査してスキーマを作るため、ツール関数
    自体をデコレータで包むのではなく、各ツールの本体内で `with _timed(...):`
    のように使う(関数のシグネチャ・docstring には一切触れない)。
    """
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
) -> dict:
    """新しい記憶を保存する。

    タスク中に重要な情報(事実・好み・プロジェクト状況・出来事)を発見したとき呼ぶ。
    importance は 1〜10 で文脈の重要度を自己採点する。
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
        )


@mcp.tool()
def recall(
    query: str,
    mode: str = "fast",
    limit: int = 5,
    type: str | None = None,
    record_hits: bool = True,
    room: str | None = None,
) -> dict:
    """記憶を検索して返す。

    タスク開始時に必ず呼ぶ。関連する過去の知識・好み・プロジェクト状況を想起できる。
    mode="fast" は tier=hot のみ高速検索。スコアが低いと自動で deep に切り替わる。
    mode="deep" は cold/superseded/episode も含め連想リンクを辿って広く探す。
    mode="exhaustive" は活性度を無視し関連度のみで全記憶を総当たりする。
    「確かに記録したはずなのに fast/deep で出ない」沈んだ記憶を掘り起こす最終手段。
    room は通常指定不要(現在の部屋+共通だけを検索する)。room="*" で全部屋を
    横断検索できるが、仕事/個人の分離を壊さないよう必要時のみ使うこと。
    """
    engine = _get_engine()
    with _timed("recall"):
        return engine.recall(
            query=query,
            mode=mode,
            limit=limit,
            type=type,
            record_hits=record_hits,
            room=room if room is not None else _default_room(),
        )


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
    with _timed("reindex"):
        return engine.reindex()


@mcp.tool()
def stats() -> dict:
    """記憶の統計情報を返す。

    記憶の件数(type別・tier別)、アクセスイベント数、リンク数などを確認できる。
    """
    engine = _get_engine()
    with _timed("stats"):
        return engine.stats()


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


def main() -> None:
    """stdio MCP サーバーを起動する。"""
    # 【v0.6.0〜】既定の実行系は ONNX(embed_backend=auto + export-onnx 済み)で、
    # import+モデルロードは1〜2秒。以下の病理は torch 経路(ONNX 未生成の
    # フォールバック)にのみ該当するが、そこに戻ると何が起きるかの記録として残す。
    # blocking 既定は ONNX でも無害(先読みが軽いだけ)なので変更しない。
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
    # ENGRAM_PRELOAD で先読み方式を選ぶ:
    #   blocking (既定)  — 起動時にメインスレッドで先読み。ハンドシェイクは import
    #                     完了まで待つ(warm 12〜24秒 / cold 50秒超)ため、クライアント
    #                     側の MCP 起動タイムアウトを 120 秒以上にすること(Claude
    #                     Code は settings.json の env で MCP_TIMEOUT=120000)。
    #                     接続後の recall は常に即応答する。
    #   background      — 先読みをデーモンスレッドに回し、mcp.run() を即実行する。
    #                     ハンドシェイクは即応答するが、上記の病理により Windows では
    #                     初回 recall が 3 分級になりうる。起動タイムアウトを一切
    #                     延ばせないクライアント向けの妥協案。
    #   off             — 先読みしない。初回ツール呼び出し時に遅延ロードする
    #                     (background と同じ病理を踏む)。
    # FastMCP は同期ツールをワーカースレッドで実行するため、_get_engine と
    # RuriEmbedder._load はロックで多重ロードを防いでいる。
    mode = os.environ.get("ENGRAM_PRELOAD", "blocking").strip().lower()
    if mode == "blocking":
        _preload()
    elif mode != "off":
        threading.Thread(
            target=_preload, name="engram-preload", daemon=True
        ).start()

    mcp.run()
