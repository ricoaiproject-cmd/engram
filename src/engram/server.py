"""MCP サーバー(担当: Agent B)。

mcp 公式 SDK の FastMCP(stdio)で engine の操作をツールとして公開する。

    from mcp.server.fastmcp import FastMCP

実装要件:
- エンジンは初回ツール呼び出し時に遅延構築(build_engine)。起動を速く保つ。
- 各ツールは engine の同名メソッドへ委譲し、dict をそのまま返す。
- docstring がそのままツール説明になるので、エージェントが「いつ呼ぶべきか」を
  判断できる文面にする(日本語でよい)。
- ツール: remember / recall / reinforce / correct / link / forget /
         consolidation_candidates / mark_consolidated / reindex / stats
- 引数は engine のシグネチャに合わせる(now は公開しない)。
- main() でstdio実行: mcp.run()。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

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
    既存の類似記憶(cos ≥ 0.92)がある場合は重複強化して返す。
    room は通常指定不要(作業ディレクトリから自動判定)。どの文脈でも使う
    普遍的な記憶だけ room="common" を明示する。
    """
    engine = _get_engine()
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
    return engine.link(src=src, dst=dst)


@mcp.tool()
def forget(id: str) -> dict:
    """記憶をソフト削除する(ゴミ箱へ移動)。

    不要になった記憶を検索対象から外したいときに呼ぶ。
    物理削除ではなくゴミ箱移動なので、誤削除の場合は復元できる。
    誤りを訂正したい場合は forget ではなく correct を使うこと。
    """
    engine = _get_engine()
    return engine.forget(id=id)


@mcp.tool()
def consolidation_candidates() -> dict:
    """統合候補の episode クラスタを返す。

    就寝前(セッション終了時)に呼び、類似する古い episode をまとめて
    知識・プロジェクト記憶として圧縮する候補を提示する。
    LLM が要約を生成し、mark_consolidated で統合を完了する。
    """
    engine = _get_engine()
    return engine.consolidation_candidates()


@mcp.tool()
def mark_consolidated(episode_ids: list[str], new_memory_id: str) -> dict:
    """統合完了を記録する。

    consolidation_candidates で提示されたクラスタを LLM が要約し、
    remember で新記憶を作成した後にこれを呼ぶ。
    元の episode は cold(長期保存)に降格し、derived_from リンクで繋がれる。
    """
    engine = _get_engine()
    return engine.mark_consolidated(
        episode_ids=episode_ids,
        new_memory_id=new_memory_id,
    )


@mcp.tool()
def reindex() -> dict:
    """Markdown ファイルから DB インデックスを再構築する。

    手動でファイルを編集した後や、DB が壊れた疑いがあるときに呼ぶ。
    差異のある記憶だけ再埋め込みするため、全件再構築より高速。
    """
    engine = _get_engine()
    return engine.reindex()


@mcp.tool()
def stats() -> dict:
    """記憶の統計情報を返す。

    記憶の件数(type別・tier別)、アクセスイベント数、リンク数などを確認できる。
    """
    engine = _get_engine()
    return engine.stats()


def _preload() -> None:
    """埋め込みモデルを先読みする(失敗しても起動は続行)。"""
    import sys

    try:
        engine = _get_engine()
        engine.embedder.embed_query("ウォームアップ")
        print("engram: engine preloaded", file=sys.stderr)
    except Exception as e:  # 失敗しても起動は続け、初回ツール呼び出しで再試行
        print(f"engram: preload failed, will retry lazily: {e}", file=sys.stderr)


def main() -> None:
    """stdio MCP サーバーを起動する。"""
    # torch / sentence_transformers の import は非常に重い(実測: import だけで
    # cold 50秒超)。これを mcp.run() の前にメインスレッドで実行すると initialize
    # ハンドシェイクがその間ブロックされ、起動タイムアウトの短い MCP クライアント
    # (例: Antigravity IDE)は接続を "context canceled" で打ち切ってしまう。
    #
    # ENGRAM_PRELOAD で先読み方式を選ぶ:
    #   blocking (既定) — 従来どおり起動時にメインスレッドで先読み。初回 recall は
    #                     速いが、ハンドシェイクは import 完了まで待つ。
    #   background      — 先読みをデーモンスレッドに回し、mcp.run() を即実行する。
    #                     ハンドシェイクは即応答し、モデルは裏で準備される。初回
    #                     recall はモデル準備完了までロック待ちになる。
    #   off             — 先読みしない。初回ツール呼び出し時に遅延ロードする。
    # FastMCP は同期ツールをワーカースレッドで実行するため、_get_engine と
    # RuriEmbedder._load はロックで多重ロードを防いでいる。
    mode = os.environ.get("ENGRAM_PRELOAD", "blocking").strip().lower()
    if mode == "background":
        threading.Thread(
            target=_preload, name="engram-preload", daemon=True
        ).start()
    elif mode != "off":
        _preload()

    mcp.run()
