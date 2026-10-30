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
        # cold(沈黙)も走査に含める。浮上のゲートは呼び出し元で tier 別に
        # 掛ける(cold は「強い手がかり」のときだけ再点火する)
        sql = (
            "SELECT m.id, m.type, m.importance, m.room, m.created_at, "
            "       m.tier, f.content AS content "
            "FROM memories m JOIN fts_memories f ON f.memory_id = m.id "
            "WHERE m.tier IN ('hot', 'cold') AND m.type != 'episode'"
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


def _load_fringe_ids(settings: Settings, session_id: str) -> set[str]:
    try:
        data = json.loads(_state_path(settings, session_id).read_text(encoding="utf-8"))
        return set(data.get("fringe", []))
    except Exception:
        return set()


def _save_state(settings: Settings, session_id: str, *, surfaced: set[str], fringe: set[str]) -> None:
    d = _surface_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    _state_path(settings, session_id).write_text(
        json.dumps({"surfaced": sorted(surfaced), "fringe": sorted(fringe)}, ensure_ascii=False),
        encoding="utf-8",
    )


def _fetch_neighbors(db_path: Path, ids: list[str]) -> list[str]:
    """浮上した記憶から一歩つながった記憶の id(共起・明示・派生のリンク)。軽量 sqlite 直読み。"""
    if not ids or not Path(db_path).is_file():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA busy_timeout=2000")
        ph = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT src, dst, weight FROM links WHERE (src IN ({ph}) OR dst IN ({ph})) "
            f"AND kind IN ('co_recall', 'explicit', 'derived_from') ORDER BY weight DESC LIMIT 200",
            ids + ids).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    seen = set(ids)
    out: list[str] = []
    for src, dst, _w in rows:
        for x in (src, dst):
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def _load_suppressed(settings: Settings, now: float) -> set[str]:
    """sleep(夜間の保守)が生成した常連抑制表を読む。無ければ空。

    「reinforce されないまま浮上を繰り返す記憶」を一定期間黙らせる仕組み。
    重い判定は夜の sleep 側で済ませ、ここでは JSON を読むだけ(毎プロンプトの
    コストを増やさない)。
    """
    try:
        data = json.loads(
            (_surface_dir(settings) / "suppression.json").read_text(
                encoding="utf-8"
            )
        )
        return {
            mid
            for mid, item in (data.get("suppressed") or {}).items()
            if float(item.get("until", 0)) > now
        }
    except Exception:
        return set()


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
    log_extra: dict | None = None,
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

    # 意味の網(デーモンが生きているときだけ)。字句の関連度と、意味検索の
    # コサインの高い方を採る。言い換え・同義語で字句が外れても拾えるように
    # するのが目的で、デーモン不在なら何も起きない(字句だけで動く)
    if settings.surface_semantic:
        try:
            from . import daemon as _daemon

            sem = _daemon.semantic_candidates(
                settings, prompt, k=settings.surface_semantic_k, rooms=rooms
            )
            if not sem and settings.surface_autostart_daemon and not dry_run:
                # デーモンが落ちていたら起こす(長い作業の途中で無操作終了する
                # ため、セッション初回だけでは足りない)。多重起動は3段で守る:
                # 90秒の冷却・固定ポートの排他・import 可否(結果はキャッシュ)。
                # 守りが無いまま毎発話で起こして40プロセスの暴走を招いた反省
                _daemon.ensure_running(settings)
            if sem:
                lo = settings.semantic_cos_floor
                span = max(settings.semantic_cos_ceiling - lo, 1e-6)

                def _rescale(cos: float) -> float:
                    # 床(無関係でも出る値)を0、天井を1へ写す
                    return max(0.0, min(1.0, (cos - lo) / span))

                lex = [max(rel, _rescale(sem.get(r["id"], 0.0)))
                       for r, rel in zip(rows, lex)]
                result["semantic"] = len(sem)
        except Exception:
            pass  # 意味の網は「あれば使う」もの。落ちても字句で動く

    # 1段目: 関連度だけで候補を絞る(土俵を「意味の近さ」で作る)。
    # 加重和で先にカットすると、関連度が高い新しい記憶が高活性の常連に
    # 枠ごと押し出される(実測: 蒸留記憶114件の候補入り0回)
    pre = sorted(zip(rows, lex), key=lambda pair: pair[1], reverse=True)
    pre = [pair for pair in pre if pair[1] > 0][:settings.surface_candidate_k]
    if not pre:
        return result

    # 2段目: 候補内で関連度を正規化してから加重和で並べる。
    # 字句包含度は実測 median 0.128 と強く圧縮されており、生値のままだと
    # 名目の重み(0.6)が効かず活性度が順位を支配する(engine が recall で
    # 既に採っている対策を surface にも適用する)
    rel_norm = dynamics.normalize_relevances(
        {r["id"]: rel for r, rel in pre},
        floor=settings.relevance_norm_floor,
    )

    scored: list[tuple[float, float, float, float, dict]] = []
    for r, rel in pre:
        d = dynamics.decay_rate(r["importance"])
        act = dynamics.activation_norm(
            r["events"], ts, d, min_elapsed=settings.min_elapsed_seconds
        )
        # surface では重要度を順位付けに使わない(既定 0.0)。importance は
        # 「手書きか機械書きか」の署名になっており、下駄にすると新参が
        # 永久に浮上できない。保護(沈黙の免除)には引き続き使う
        kw = dict(
            w_relevance=settings.surface_w_relevance,
            w_activation=settings.surface_w_activation,
            w_importance=settings.surface_w_importance,
        )
        # 並べ替えは正規化した関連度で(常連の下駄を相対化する)
        rank_score = dynamics.final_score(
            rel_norm.get(r["id"], rel), act, r["importance"], **kw
        )
        # しきい値判定と報告値は生の関連度で(正規化は順位付け専用。候補が
        # 1件・同程度のとき正規化値は0になる設計なので、閾値に使うと
        # 「何も浮上しない」状態を作ってしまう)
        score = dynamics.final_score(rel, act, r["importance"], **kw)
        scored.append((rank_score, score, rel, act, r))
    scored.sort(key=lambda t: t[0], reverse=True)

    top = [(score, rel, act, r) for _rank, score, rel, act, r in scored[:5]]
    result["candidates"] = [
        {
            "id": r["id"],
            "score": round(score, 4),
            "relevance": round(rel, 4),
            "activation": round(act, 4),
            "importance": r["importance"],
            "type": r["type"],
            "room": r["room"],
            "tier": r["tier"],
            "content": r["content"],
        }
        for score, rel, act, r in top
    ]

    # 閾値 + 関連度ゲート + 常連抑制(sleep 生成) + 同一セッション内の再浮上抑制
    already = set() if dry_run else _load_surfaced_ids(settings, session_id)
    suppressed_now = _load_suppressed(settings, ts)
    surfaced: list[dict] = []
    skipped: list[str] = []
    for cand in result["candidates"]:
        if len(surfaced) >= settings.surface_max_items:
            break
        if cand.get("tier") == "cold":
            # 沈黙した記憶は強い手がかりのときだけ浮上(再点火)。スコアは
            # 活性度ゼロで構造的に届かないため関連度のみで判定する
            if cand["relevance"] < settings.surface_cold_min_relevance:
                continue
        else:
            if cand["score"] < settings.surface_threshold:
                continue
            if cand["relevance"] < settings.surface_min_relevance:
                continue
        if cand["id"] in suppressed_now:
            skipped.append(cand["id"])
            continue
        if cand["id"] in already:
            continue
        surfaced.append(cand)
    # 探索枠: 選ばれたのが古参(高importance)ばかりなら、最後の1枠を
    # 若い記憶へ譲る。新規記憶に浮上→reinforce の機会を与えないと、
    # 代謝ループが新参に対して永久に閉じる(多腕バンディットの exploration)
    if (settings.surface_explore_slot and len(surfaced) >= 2
            and all(c["importance"] > settings.surface_explore_max_importance
                    for c in surfaced)):
        chosen = {c["id"] for c in surfaced}
        for cand in result["candidates"]:
            if cand["id"] in chosen or cand["id"] in already:
                continue
            if cand["id"] in suppressed_now:
                continue
            if cand["importance"] > settings.surface_explore_max_importance:
                continue
            if cand["relevance"] < settings.surface_min_relevance:
                continue
            surfaced[-1] = cand
            break

    result["surfaced"] = [c["id"] for c in surfaced]
    result["surfaced_items"] = surfaced
    result["suppressed_skipped"] = skipped

    # 縁の層: 意識に上らなかったが弱く浮かんだ記憶を id+要旨1行で添える。
    # near = 検索で惜しかったもの(順位の続き)、link = 浮上した記憶から一歩つながったもの
    fringe: list[dict] = []
    n_fringe = int(getattr(settings, "surface_fringe_items", 0) or 0)
    if n_fringe > 0:
        gist_n = int(getattr(settings, "surface_fringe_gist_chars", 48))
        fr_min = float(getattr(settings, "surface_fringe_min_relevance", 0.10))
        already_fringe = set() if dry_run else _load_fringe_ids(settings, session_id)
        taken = set(result["surfaced"]) | already | already_fringe | suppressed_now
        rows_by_id = {r["id"]: r for r in rows}

        def _gist(r: dict) -> str:
            c = " ".join(str(r.get("content", "")).split())
            return c[:gist_n] + ("…" if len(c) > gist_n else "")

        for _rank, _score, rel, _act, r in scored:
            if len(fringe) >= n_fringe:
                break
            if r["id"] in taken or rel < fr_min:
                continue
            taken.add(r["id"])
            fringe.append({"id": r["id"], "type": r["type"], "tier": r["tier"], "via": "near",
                           "relevance": round(rel, 3), "gist": _gist(r)})
        if getattr(settings, "surface_fringe_links", True) and result["surfaced"] and len(fringe) < n_fringe:
            for nid in _fetch_neighbors(settings.db_path, result["surfaced"]):
                if len(fringe) >= n_fringe:
                    break
                r = rows_by_id.get(nid)
                if r is None or nid in taken:
                    continue
                taken.add(nid)
                fringe.append({"id": nid, "type": r["type"], "tier": r["tier"], "via": "link",
                               "relevance": None, "gist": _gist(r)})
    result["fringe_items"] = fringe

    if dry_run:
        return result

    # shadow でも active と同じ状態更新をする(影モードを本番の忠実な
    # シミュレーションにするため)。ログは調整・監査の生データ
    if surfaced or fringe:
        _save_state(settings, session_id,
                    surfaced=already | set(result["surfaced"]),
                    fringe=_load_fringe_ids(settings, session_id) | {f["id"] for f in fringe})
        # ついでに古いセッション状態を掃除(数が小さいので毎回でも安い)
        try:
            for f in _surface_dir(settings).glob("session-*.json"):
                if ts - f.stat().st_mtime > _STATE_MAX_AGE_SECONDS:
                    f.unlink(missing_ok=True)
        except OSError:
            pass
    _append_log(settings, {
        **(log_extra or {}),
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
        "fringe": [f["id"] for f in fringe],
        # 縁の出どころ(near=検索の続き / link=一歩の連想)。週次の点検で「どちらが引かれるか」を見る
        "fringe_via": [f["via"] for f in fringe],
        "suppressed_skipped": skipped,
        # 意味の網が効いたか(デーモンから得た候補数。無効/不在なら欠落)。
        # これが無いと「意味検索ありの浮上」と「字句のみの浮上」を後から
        # 区別できず、命中率の答え合わせができない(R8.8.28の検証で判明)
        "semantic": result.get("semantic"),
    })
    return result


def format_context(surfaced_items: list[dict], *, lead: str | None = None,
                   fringe_items: list[dict] | None = None) -> str:
    """active モードで文脈に差し込むテキストを組み立てる。

    lead を渡すと前置き文を差し替えられる(assistant cue 用)。
    fringe_items(縁の層)は id と要旨1行だけ。本文は read(ids) で引く。
    """
    lines: list[str] = []
    if surfaced_items:
        lines += [
            lead or (
                "(engram 自発的想起)以下はあなたの記憶基盤から自動的に浮上した、"
                "今の発言に関連する可能性のある記憶です。"
            ),
            "",
        ]
        for c in surfaced_items:
            content = " ".join(c["content"].split())
            if len(content) > 300:
                content = content[:299] + "…"
            mark = "沈黙からの再点火・" if c.get("tier") == "cold" else ""
            lines.append(f"- [{c['id']}] ({mark}{c['type']}/{c['room']}) {content}")
        lines += [
            "",
            "実際に役立った場合のみ engram の reinforce にこの id を渡して定着させて"
            "ください。内容が誤っていれば correct を、無関係なら無視してください。",
        ]
    if fringe_items:
        if lines:
            lines.append("")
        lines += [
            "(engram 縁の想起)意識の縁に薄く浮かんだ記憶です。要旨だけなので、今の話に効きそうな"
            "ものがあれば engram の read(ids) で本文を読んでください。読まなければ無視して構いません。",
            "",
        ]
        for f in fringe_items:
            via = "連想" if f.get("via") == "link" else "近接"
            cold = "・沈黙" if f.get("tier") == "cold" else ""
            lines.append(f"- [{f['id']}] ({via}{cold}・{f.get('type')}) {f.get('gist', '')}")
    return "\n".join(lines)
