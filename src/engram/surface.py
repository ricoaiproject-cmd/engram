"""自発的想起(②プロアクティブメモリ)の高速検索路。

エージェントに聞かれなくても、ユーザーの発話に関連する記憶を見つけて
文脈に差し込む。UserPromptSubmit フックから毎プロンプト呼ばれるため、
埋め込みモデル(torch)も numpy も sqlite-vec もロードしない:

- 関連度: IDF 重み付き文字バイグラムの包含度(発話の語が記憶にどれだけ
  含まれるか)。日本語は形態素解析なしでバイグラムが実用的に効く
- 活性度: dynamics.activation_norm(ACT-R、math のみ)
- 最終スコア: recall と同じ加重和(dynamics.final_score)

DB へは読み取りのみ(recall_hit イベントも記録しない)。浮上した記憶が
実際に役立ったかの判断と reinforce はエージェント側の責務。

モード(config.toml の surface_mode):
- off:    何もしない
- shadow: 差し込まず、「差し込むならこれを出していた」をログに残す(調整用)
- active: 閾値を超えた記憶を実際に文脈へ差し込む

shadow/active とも全候補をログ(surface/surface_log.jsonl)に記録するので、
後から閾値の妥当性を検証できる。
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

from . import dynamics
from .config import Settings

_SESSION_ID_SAFE = re.compile(r"[^A-Za-z0-9_-]")
_STATE_MAX_AGE_SECONDS = 7 * 86400
_LOG_ROTATE_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# 字句関連度(IDF 重み付きバイグラム包含度)
# ---------------------------------------------------------------------------

def bigrams(text: str) -> set[str]:
    """NFKC 正規化 + casefold した文字バイグラム集合。

    空白で区切った断片ごとに作る(単語境界を跨ぐバイグラムを作らない)。
    1文字の断片はそのまま1グラムとして扱う。
    """
    norm = unicodedata.normalize("NFKC", text).casefold()
    grams: set[str] = set()
    for seg in norm.split():
        if len(seg) == 1:
            grams.add(seg)
            continue
        for i in range(len(seg) - 1):
            grams.add(seg[i : i + 2])
    return grams


def lexical_scores(prompt: str, docs: list[str]) -> list[float]:
    """各文書について、発話バイグラムの IDF 重み付き包含度(0..1)を返す。

    score = Σ_{g ∈ P∩D} idf(g) / Σ_{g ∈ P} idf(g)
    ありふれたバイグラム(です・ます等)は文書頻度が高く idf が小さいので、
    内容語の一致が支配的になる。
    """
    p_grams = bigrams(prompt)
    doc_grams = [bigrams(d) for d in docs]
    n = len(docs)
    if not p_grams or n == 0:
        return [0.0] * n

    df: dict[str, int] = {}
    for grams in doc_grams:
        for g in grams & p_grams:  # 発話に出るグラムだけ数えれば十分
            df[g] = df.get(g, 0) + 1

    idf = {g: math.log(1.0 + n / (1.0 + df.get(g, 0))) for g in p_grams}
    denom = sum(idf.values())
    if denom <= 0:
        return [0.0] * n

    scores: list[float] = []
    for grams in doc_grams:
        num = sum(idf[g] for g in p_grams & grams)
        scores.append(num / denom)
    return scores


# ---------------------------------------------------------------------------
# DB 読み取り(軽量経路: sqlite3 標準モジュールのみ)
# ---------------------------------------------------------------------------

def _fetch_candidates(db_path: Path, rooms: list[str] | None) -> list[dict]:
    """tier=hot の全記憶(本文付き)を返す。DB が無ければ空。

    vec_memories(sqlite-vec)には触れないため拡張ロード不要。FTS5 は
    SQLite 標準なので fts_memories の読み取りは問題ない。
    """
    if not Path(db_path).is_file():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=2000")
        # episode は除外する。自発的想起は軽量な fast 相当であり、
        # (1) episode は生の体験ログでノイズになりやすく、deep recall で
        #     連想的に辿るのが本来の使い道(recall の fast も episode を除外)
        # (2) 自動符号化(①)はユーザー発言を要約 episode に保存するため、
        #     除外しないと「似た発言→自分の言葉を含む episode が完全一致で
        #     浮上」という相互作用でオウム返し状態になる(実ログで確認)
        # 知見・好み・プロジェクトに昇華された記憶だけを浮上させる
        sql = (
            "SELECT m.id, m.type, m.importance, m.room, m.created_at, "
            "       f.content AS content "
            "FROM memories m JOIN fts_memories f ON f.memory_id = m.id "
            "WHERE m.tier = 'hot' AND m.type != 'episode'"
        )
        params: list = []
        if rooms:
            ph = ",".join("?" * len(rooms))
            sql += f" AND m.room IN ({ph})"
            params.extend(rooms)
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        ph = ",".join("?" * len(ids))
        events: dict[str, list[tuple[float, float]]] = {i: [] for i in ids}
        for row in conn.execute(
            f"SELECT memory_id, ts, weight FROM access_events "
            f"WHERE memory_id IN ({ph}) ORDER BY ts",
            ids,
        ):
            events[row["memory_id"]].append((row["ts"], row["weight"]))
        for r in rows:
            r["events"] = events[r["id"]]
        return rows
    except sqlite3.Error:
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# セッション状態とログ
# ---------------------------------------------------------------------------

def _surface_dir(settings: Settings) -> Path:
    return settings.data_dir / "surface"

def _state_path(settings: Settings, session_id: str) -> Path:
    safe = _SESSION_ID_SAFE.sub("_", session_id) or "unknown"
    return _surface_dir(settings) / f"session-{safe}.json"


def _load_surfaced_ids(settings: Settings, session_id: str) -> set[str]:
    try:
        data = json.loads(
            _state_path(settings, session_id).read_text(encoding="utf-8")
        )
        return set(data.get("surfaced", []))
    except Exception:
        return set()


def _save_surfaced_ids(settings: Settings, session_id: str,
                       ids: set[str], now: float) -> None:
    d = _surface_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    _state_path(settings, session_id).write_text(
        json.dumps({"surfaced": sorted(ids)}, ensure_ascii=False),
        encoding="utf-8",
    )
    # ついでに古いセッション状態を掃除(数が小さいので毎回でも安い)
    try:
        for f in d.glob("session-*.json"):
            if now - f.stat().st_mtime > _STATE_MAX_AGE_SECONDS:
                f.unlink(missing_ok=True)
    except OSError:
        pass


def _append_log(settings: Settings, entry: dict) -> None:
    d = _surface_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    log = d / "surface_log.jsonl"
    try:
        if log.is_file() and log.stat().st_size > _LOG_ROTATE_BYTES:
            log.replace(log.with_suffix(".jsonl.old"))
    except OSError:
        pass
    # 入力に不正なサロゲート等が紛れてもログ書き込みでは落とさない
    with log.open("a", encoding="utf-8", errors="replace") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------

def run_surface(
    prompt: str,
    *,
    settings: Settings,
    room: str = "common",
    session_id: str = "unknown",
    mode: str | None = None,
    now: float | None = None,
    dry_run: bool = False,
) -> dict:
    """発話に関連する記憶を探し、モードに応じてログ・状態を更新する。

    dry_run=True は手動確認(engram surface コマンド)用で、ログも状態も
    書かず候補だけ返す。
    返り値: {"mode", "room", "candidates": [...], "surfaced": [...]}
    candidates は上位5件(スコア降順)、surfaced は閾値・重複抑制を通過して
    実際に差し込む(または shadow で差し込んだことにする)記憶の id リスト。
    """
    ts = now if now is not None else time.time()
    mode = mode or settings.surface_mode
    rooms = sorted({room, "common"})

    result: dict = {"mode": mode, "room": room, "candidates": [],
                    "surfaced": []}
    if mode == "off":
        return result

    rows = _fetch_candidates(settings.db_path, rooms)
    if not rows:
        return result

    lex = lexical_scores(prompt, [r["content"] for r in rows])

    scored: list[tuple[float, float, float, dict]] = []
    for r, rel in zip(rows, lex):
        d = dynamics.decay_rate(r["importance"])
        act = dynamics.activation_norm(
            r["events"], ts, d, min_elapsed=settings.min_elapsed_seconds
        )
        score = dynamics.final_score(
            rel, act, r["importance"],
            w_relevance=settings.w_relevance,
            w_activation=settings.w_activation,
            w_importance=settings.w_importance,
        )
        scored.append((score, rel, act, r))
    scored.sort(key=lambda t: t[0], reverse=True)

    top = scored[:5]
    result["candidates"] = [
        {
            "id": r["id"],
            "score": round(score, 4),
            "relevance": round(rel, 4),
            "activation": round(act, 4),
            "importance": r["importance"],
            "type": r["type"],
            "room": r["room"],
            "content": r["content"],
        }
        for score, rel, act, r in top
    ]

    # 閾値 + 関連度ゲート + 同一セッション内の再浮上抑制
    already = set() if dry_run else _load_surfaced_ids(settings, session_id)
    surfaced: list[dict] = []
    for cand in result["candidates"]:
        if len(surfaced) >= settings.surface_max_items:
            break
        if cand["score"] < settings.surface_threshold:
            continue
        if cand["relevance"] < settings.surface_min_relevance:
            continue
        if cand["id"] in already:
            continue
        surfaced.append(cand)
    result["surfaced"] = [c["id"] for c in surfaced]
    result["surfaced_items"] = surfaced

    if dry_run:
        return result

    # shadow でも active と同じ状態更新をする(影モードを本番の忠実な
    # シミュレーションにするため)。ログは調整・監査の生データ
    if surfaced:
        _save_surfaced_ids(
            settings, session_id, already | set(result["surfaced"]), ts
        )
    _append_log(settings, {
        "ts": ts,
        "session_id": session_id,
        "mode": mode,
        "room": room,
        "prompt": " ".join(prompt.split())[:120],
        "candidates": [
            {k: (v[:80] if k == "content" else v) for k, v in c.items()}
            for c in result["candidates"]
        ],
        "surfaced": result["surfaced"],
    })
    return result


def format_context(surfaced_items: list[dict]) -> str:
    """active モードで文脈に差し込むテキストを組み立てる。"""
    lines = [
        "(engram 自発的想起)以下はあなたの記憶基盤から自動的に浮上した、"
        "今の発言に関連する可能性のある記憶です。",
        "",
    ]
    for c in surfaced_items:
        content = " ".join(c["content"].split())
        if len(content) > 300:
            content = content[:299] + "…"
        lines.append(f"- [{c['id']}] ({c['type']}/{c['room']}) {content}")
    lines += [
        "",
        "実際に役立った場合のみ engram の reinforce にこの id を渡して定着させて"
        "ください。内容が誤っていれば correct を、無関係なら無視してください。",
    ]
    return "\n".join(lines)
