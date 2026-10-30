"""仮説サイクル — 記憶の組み合わせから仮説を立て、確かめ、結果をまた記憶する(engram 0.25)。

分担: 仮説を考えて Web 等で確かめるのはモデル(ルーティン)の仕事。engram は
「覚えておく係」と「思い出させる係」に徹し、ここでは LLM を使わない決定的な処理だけを持つ。

    種選び   hypothesis_seeds  縁の薄い記憶の組(近すぎず遠すぎない)を返す。過去の当たり率で重み付け
    書く     hypothesize        仮説=episode 記憶(tags: hypothesis, hyp-open)+台帳の1行
    一覧     list_hypotheses    未決・期限切れ・判定済み。未決には「証拠候補」(後から入った近い記憶)を添える
    判定     resolve            当たり→knowledge を新設し derived_from で結ぶ／外れ→correct と同じく誤りごと刻む
                                ／判定不能→期限を延ばして再挑戦(上限で打ち切り)

状態の正本は台帳 <memories_dir>/_hypotheses/ledger.json(記憶 Markdown と同じく共有ドライブに置く)。
index.db はマシンごとなので、仮説の状態はそこへ置かない。

設計上の約束(仮説サイクル設計案 §5):
- 反証の条件(外れたと分かる条件)が書けない仮説は受け付けない。
- 書いた日と同じ日には判定させない(自分で立てて自分で丸を付けると当たりばかりになる)。
- 仮説が参照しただけでは元の記憶を温めない。温めるのは「当たり」の判定が付いたときだけ。
- モデルが下した判定には judge を残す。ユーザーが後で訂正したら correct がそれを上書きする。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np

VERDICT_ALIASES = {
    "当たり": "当たり", "supported": "当たり", "confirmed": "当たり", "true": "当たり",
    "外れ": "外れ", "refuted": "外れ", "falsified": "外れ", "false": "外れ",
    "判定不能": "判定不能", "inconclusive": "判定不能", "unknown": "判定不能",
}
OPEN_TAG = "hyp-open"
TAG_FOR_VERDICT = {"当たり": "hyp-当たり", "外れ": "hyp-外れ", "判定不能": "hyp-判定不能"}


# ------------------------------------------------------------------ ledger
def ledger_path(settings) -> Path:
    return Path(settings.memories_dir) / "_hypotheses" / "ledger.json"


def load_ledger(settings) -> dict:
    p = ledger_path(settings)
    if not p.is_file():
        return {"version": 1, "hypotheses": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # 壊れた台帳で止めない。ただし上書きで消さないよう退避してから空で始める
        bak = p.with_suffix(f".broken-{int(time.time())}.json")
        try:
            p.replace(bak)
        except OSError:
            pass
        return {"version": 1, "hypotheses": {}}
    data.setdefault("version", 1)
    data.setdefault("hypotheses", {})
    return data


def save_ledger(settings, data: dict) -> None:
    p = ledger_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="ledger-", suffix=".tmp", dir=str(p.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def _today(now: float | None) -> _dt.date:
    return _dt.datetime.fromtimestamp(now if now is not None else time.time()).date()


def _parse_due(due: str | int | None, today: _dt.date, default_days: int) -> str:
    if due is None or due == "":
        return (today + _dt.timedelta(days=default_days)).isoformat()
    if isinstance(due, int) or (isinstance(due, str) and due.strip().lstrip("+").isdigit()):
        return (today + _dt.timedelta(days=int(str(due).lstrip("+")))).isoformat()
    return _dt.date.fromisoformat(str(due).strip()[:10]).isoformat()


def _set_tags(engine, mem_id: str, add: list[str], remove: list[str]) -> None:
    """記憶 Markdown の tags を書き換える(本文・埋め込みは不変なので索引は触らない)。"""
    mem = engine.db.get_memory(mem_id)
    if mem is None:
        return
    try:
        rec = engine.store.read(Path(mem["path"]))
    except Exception:
        return
    tags = [t for t in rec.tags if t not in remove]
    for t in add:
        if t not in tags:
            tags.append(t)
    rec.tags = tags
    engine.store.update(rec)


# -------------------------------------------------------------- hypothesize
def hypothesize(
    engine,
    claim: str,
    source_ids: list[str],
    falsify_if: str,
    test_plan: str = "",
    *,
    due: str | int | None = None,
    importance: int | None = None,
    room: str | None = None,
    source: str = "unknown",
    origin: str = "",
    now: float | None = None,
) -> dict:
    s = engine.settings
    claim = (claim or "").strip()
    falsify_if = (falsify_if or "").strip()
    srcs = list(dict.fromkeys([x.strip() for x in (source_ids or []) if x and x.strip()]))
    if not claim:
        return {"status": "rejected", "reason": "claim(主張)が空です"}
    if len(falsify_if) < 6:
        return {"status": "rejected",
                "reason": "falsify_if(外れたと分かる条件)が書けない仮説は受け付けません"}
    if len(srcs) < 2:
        return {"status": "rejected", "reason": "元にした記憶(source_ids)は2件以上必要です"}
    rooms = set()
    for sid in srcs:
        m = engine.db.get_memory(sid)
        if m is None or m.get("tier") == "trash":
            return {"status": "rejected", "reason": f"元の記憶が見つかりません: {sid}"}
        rooms.add(m.get("room") or "common")

    ts = now if now is not None else time.time()
    today = _today(ts)
    due_s = _parse_due(due, today, s.hyp_default_due_days)
    if _dt.date.fromisoformat(due_s) <= today:
        due_s = (today + _dt.timedelta(days=1)).isoformat()  # 同日判定の禁止と揃える
    room = room or (rooms.pop() if len(rooms) == 1 else "common")

    ledger = load_ledger(s)
    # 重複は台帳で「同じ主張+同じ元の記憶」を見る。埋め込みの近さで見ると、欄の書式が
    # 同じぶん別の仮説どうしが併合されるため remember の重複検知は使わない
    for hid, h in ledger["hypotheses"].items():
        if h.get("claim") == claim and set(h.get("sources", [])) == set(srcs):
            return {"status": "duplicate", "id": hid, "reason": "同じ主張・同じ元の記憶の仮説があります"}

    content = (
        f"仮説: {claim}\n"
        f"外れたと分かる条件: {falsify_if}\n"
        f"確かめ方: {test_plan.strip() or '(未記入)'}\n"
        f"期限: {due_s}\n"
        f"元にした記憶: " + " ".join(f"[[{x}]]" for x in srcs)
    )
    res = engine._remember_direct(
        content=content, type="episode",
        importance=int(importance if importance is not None else s.hyp_importance),
        tags=["hypothesis", OPEN_TAG], source=source, related_ids=srcs, room=room, now=ts,
    )
    hid = res["id"]
    ledger["hypotheses"][hid] = {
        "claim": claim, "falsify_if": falsify_if, "test_plan": test_plan.strip(),
        "sources": srcs, "room": room, "due": due_s,
        "created": _dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
        "status": "open", "attempts": 0, "origin": origin, "source": source,
        "source_types": sorted((engine.db.get_memory(x) or {}).get("type", "?") for x in srcs),
    }
    save_ledger(s, ledger)
    return {"status": "created", "id": hid, "due": due_s, "room": room}


# -------------------------------------------------------------------- list
def _evidence_candidates(engine, hid: str, h: dict, exclude: set[str], k: int = 3) -> list[dict]:
    """仮説の後に入った記憶のうち、仮説と意味が近いもの(LLM なし・埋め込みの近さだけ)。"""
    s = engine.settings
    emb = engine.db.get_embeddings([hid]).get(hid)
    if emb is None:
        return []
    try:
        created_ts = _dt.datetime.fromisoformat(h["created"]).timestamp()
    except Exception:
        return []
    out = []
    for mid, cos in engine.db.vector_search(emb, 40, tiers=["hot", "cold"]):
        if mid in exclude or cos < s.hyp_evidence_min_cos:
            continue
        m = engine.db.get_memory(mid)
        if not m or (m.get("created_at") or 0) <= created_ts:
            continue
        out.append({"id": mid, "cos": round(float(cos), 4), "type": m.get("type")})
        if len(out) >= k:
            break
    return out


def list_hypotheses(engine, status: str = "open", *, limit: int = 20,
                    with_evidence: bool = True, now: float | None = None) -> dict:
    """status: open(未決) / due(期限が来て今日判定してよいもの) / resolved / all。"""
    s = engine.settings
    ledger = load_ledger(s)
    today = _today(now)
    hyps = ledger["hypotheses"]
    all_hyp_ids = set(hyps) | {h.get("result_id") for h in hyps.values() if h.get("result_id")}
    rows = []
    for hid, h in sorted(hyps.items(), key=lambda kv: kv[1].get("due", "")):
        st = h.get("status")
        if status == "open" and st != "open":
            continue
        if status == "resolved" and st == "open":
            continue
        if status == "due":
            if st != "open" or h.get("due", "9999") > today.isoformat():
                continue
            if h.get("created", "")[:10] >= today.isoformat():
                continue  # 書いた日には判定しない
            try:
                age_h = ((now if now is not None else time.time())
                         - _dt.datetime.fromisoformat(h["created"]).timestamp()) / 3600
            except Exception:
                age_h = 0.0
            if age_h < s.hyp_min_age_hours:
                continue
        row = {"id": hid, **h}
        if with_evidence and st == "open":
            row["evidence_candidates"] = _evidence_candidates(
                engine, hid, h, exclude=all_hyp_ids | set(h.get("sources", [])))
        rows.append(row)
        if len(rows) >= limit:
            break
    return {"status": status, "count": len(rows), "hypotheses": rows,
            "summary": ledger_stats(s, now=now)}


# ----------------------------------------------------------------- resolve
def resolve(
    engine,
    id: str,
    verdict: str,
    evidence: str,
    *,
    evidence_urls: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    note: str = "",
    judge: str = "model",
    source: str = "unknown",
    allow_same_day: bool = False,
    now: float | None = None,
) -> dict:
    s = engine.settings
    v = VERDICT_ALIASES.get((verdict or "").strip().lower(), VERDICT_ALIASES.get((verdict or "").strip()))
    if v is None:
        return {"status": "rejected", "reason": "verdict は 当たり / 外れ / 判定不能 のいずれか"}
    evidence = (evidence or "").strip()
    if not evidence:
        return {"status": "rejected", "reason": "evidence(根拠)が空です"}
    ledger = load_ledger(s)
    h = ledger["hypotheses"].get(id)
    if h is None:
        return {"status": "not_found"}
    if h.get("status") != "open":
        return {"status": "already_resolved", "verdict": h.get("verdict")}
    ts = now if now is not None else time.time()
    today = _today(ts)
    if not allow_same_day:
        try:
            age_h = (ts - _dt.datetime.fromisoformat(h["created"]).timestamp()) / 3600
        except Exception:
            age_h = 0.0
        if h.get("created", "")[:10] >= today.isoformat() or age_h < s.hyp_min_age_hours:
            return {"status": "rejected",
                    "reason": f"書いた日と同じ日、または{s.hyp_min_age_hours:g}時間以内には判定しません"
                              "(自己採点の防止)"}

    urls = [u for u in (evidence_urls or []) if u]
    ev_ids = [x for x in (evidence_ids or []) if x and engine.db.get_memory(x)]
    if v != "判定不能" and not (urls or ev_ids):
        return {"status": "rejected",
                "reason": "当たり・外れには根拠の出典(evidence_urls か evidence_ids)が必要です"}
    url_lines = "".join(f"\n- {u}" for u in urls)
    stamp = _dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    h["attempts"] = int(h.get("attempts", 0)) + 1
    h.setdefault("history", []).append(
        {"at": stamp, "verdict": v, "judge": judge, "evidence": evidence[:500], "urls": urls})
    out: dict = {"id": id, "verdict": v}

    if v == "当たり":
        content = (
            f"検証済みの知見({judge}判定): {h['claim']}\n"
            f"根拠: {evidence}" + (f"\n出典:{url_lines}" if urls else "")
            + (f"\n補足: {note}" if note else "")
            + f"\n元の仮説: [[{id}]]"
        )
        res = engine.remember(
            content=content, type="knowledge", importance=s.hyp_result_importance,
            tags=["hypothesis-result", "当たり", f"judge:{judge}"], source=source,
            related_ids=[id] + list(h.get("sources", [])) + ev_ids, room=h.get("room"),
            now=ts, valence=0.3,
        )
        rid = res.get("id")
        if rid:
            engine.db.add_link(rid, id, "derived_from", increment=1.0, max_weight=1.0)
        # 温めるのは当たりの判定が付いたときだけ(元の記憶は組み合わせとして役立った)
        live = [x for x in h.get("sources", []) if engine.db.get_memory(x)]
        if live:
            engine.reinforce(ids=live, strength=1.0, now=ts)
        _set_tags(engine, id, [TAG_FOR_VERDICT[v]], [OPEN_TAG])
        h.update(status="resolved", verdict=v, result_id=rid, resolved_at=stamp, judge=judge)
        out.update(status="resolved", result_id=rid)
    elif v == "外れ":
        res = engine.correct(
            id=id,
            corrected_content=(
                f"仮説「{h['claim']}」は外れだった({judge}判定)。\n根拠: {evidence}"
                + (f"\n出典:{url_lines}" if urls else "") + (f"\n補足: {note}" if note else "")
            ),
            reason=f"検証の結果、外れたと分かる条件「{h['falsify_if']}」に当たったため",
            source=source, now=ts,
        )
        rid = res.get("new_id")
        _set_tags(engine, id, [TAG_FOR_VERDICT[v]], [OPEN_TAG])
        if rid:
            _set_tags(engine, rid, [TAG_FOR_VERDICT[v], "hypothesis-result", f"judge:{judge}"], [OPEN_TAG])
            for x in ev_ids:
                engine.db.add_link(rid, x, "explicit", increment=1.0, max_weight=1.0)
        h.update(status="resolved", verdict=v, result_id=rid, resolved_at=stamp, judge=judge)
        out.update(status="resolved", result_id=rid)
    else:  # 判定不能
        if h["attempts"] < s.hyp_max_attempts:
            h["due"] = (today + _dt.timedelta(days=s.hyp_retry_days)).isoformat()
            out.update(status="extended", due=h["due"])
        else:
            _set_tags(engine, id, [TAG_FOR_VERDICT[v]], [OPEN_TAG])
            h.update(status="resolved", verdict=v, resolved_at=stamp, judge=judge)
            out.update(status="resolved")
    save_ledger(s, ledger)
    return out


# ------------------------------------------------------------------- seeds
def _pair_weight(stats_by_kind: dict, kind: str) -> float:
    st = stats_by_kind.get(kind, {})
    hit, n = st.get("当たり", 0), st.get("resolved", 0)
    return (hit + 1.0) / (n + 2.0)  # ラプラス平滑: データが無いうちは一様(0.5)


def hypothesis_seeds(engine, n: int = 3, *, rooms: list[str] | None = None,
                     sample: int = 400, rng: random.Random | None = None) -> dict:
    """仮説の種になる記憶の組を返す(LLM なし)。

    近すぎる組(言い換え)と遠すぎる組(無関係)を避け、候補内の類似度の中ほどの帯
    (hyp_seed_band の分位)から選ぶ。既に仮説で使った組・直接リンク済みの組は除く。
    種類の組(knowledge×project 等)ごとの過去の当たり率で重みを付ける=当たりやすい
    組み合わせを覚えて次の種選びに戻す(「育つ」部分)。
    """
    s = engine.settings
    rng = rng or random.Random()
    rooms = rooms or list(s.hyp_seed_rooms)
    ledger = load_ledger(s)
    hyps = ledger["hypotheses"]
    used_ids = set(hyps) | {h.get("result_id") for h in hyps.values()}
    used_pairs = set()
    for h in hyps.values():
        src = h.get("sources", [])
        for i in range(len(src)):
            for j in range(i + 1, len(src)):
                used_pairs.add(frozenset((src[i], src[j])))
    mems = [m for m in engine.db.all_memories(tiers=list(s.hyp_seed_tiers),
                                              types=list(s.hyp_seed_types), rooms=rooms)
            if m["id"] not in used_ids]
    if len(mems) < 2:
        return {"pairs": [], "reason": "種にできる記憶が足りません"}
    if len(mems) > sample:
        mems = rng.sample(mems, sample)
    ids = [m["id"] for m in mems]
    embs = engine.db.get_embeddings(ids)
    ids = [i for i in ids if i in embs]
    if len(ids) < 2:
        return {"pairs": [], "reason": "埋め込みがありません"}
    mat = np.stack([embs[i] for i in ids]).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    sims = mat @ mat.T
    iu = np.triu_indices(len(ids), k=1)
    vals = sims[iu]
    lo, hi = np.quantile(vals, s.hyp_seed_band[0]), np.quantile(vals, s.hyp_seed_band[1])
    meta = {m["id"]: m for m in mems}
    linked = {frozenset((a, b)) for a, b, _k, _w in engine.db.get_links(ids)}
    stats_by_kind = ledger_stats(s).get("by_kind", {})
    cand = []
    for a, b, v in zip(iu[0], iu[1], vals):
        if not (lo <= v <= hi):
            continue
        ia, ib = ids[a], ids[b]
        key = frozenset((ia, ib))
        if key in used_pairs or key in linked:
            continue
        kind = "×".join(sorted((meta[ia]["type"], meta[ib]["type"])))
        cand.append((ia, ib, float(v), kind))
    if not cand:
        return {"pairs": [], "reason": "条件に合う組がありません"}
    picked, seen = [], set()
    pool = cand[:]
    while pool and len(picked) < n:
        w = [_pair_weight(stats_by_kind, c[3]) for c in pool]
        c = rng.choices(pool, weights=w, k=1)[0]
        pool.remove(c)
        if c[0] in seen or c[1] in seen:
            continue  # 1回の種選びで同じ記憶を使い回さない
        seen.update((c[0], c[1]))
        picked.append(c)
    pairs = []
    for ia, ib, v, kind in picked:
        items = []
        for mid in (ia, ib):
            m = meta[mid]
            try:
                text = engine.store.read(Path(m["path"])).content
            except Exception:
                text = ""
            items.append({"id": mid, "type": m["type"], "room": m.get("room"),
                          "content": text[: s.hyp_seed_chars]})
        pairs.append({"cos": round(v, 4), "kind": kind, "memories": items})
    return {"pairs": pairs, "band": [round(float(lo), 4), round(float(hi), 4)], "pool": len(cand)}


# ------------------------------------------------------------------- stats
def ledger_stats(settings, *, since_ts: float | None = None, now: float | None = None) -> dict:
    """台帳だけから数える(engram rsi 用・エンジン不要)。"""
    data = load_ledger(settings)
    today = _today(now).isoformat()
    since = (_dt.datetime.fromtimestamp(since_ts).isoformat(timespec="seconds")
             if since_ts is not None else "")
    out = {"created": 0, "open": 0, "overdue": 0, "resolved": 0, "当たり": 0, "外れ": 0,
           "判定不能": 0, "resolved_by_due": 0, "answer_rate": None, "hit_rate": None,
           "by_kind": {}}
    for h in data["hypotheses"].values():
        if since and h.get("created", "") < since:
            continue
        out["created"] += 1
        kind = "×".join(h.get("source_types") or ["?"])
        bk = out["by_kind"].setdefault(kind, {"created": 0, "resolved": 0, "当たり": 0})
        bk["created"] += 1
        if h.get("status") == "open":
            out["open"] += 1
            if h.get("due", "9999") < today:
                out["overdue"] += 1
            continue
        out["resolved"] += 1
        bk["resolved"] += 1
        v = h.get("verdict")
        if v in ("当たり", "外れ", "判定不能"):
            out[v] += 1
        if v == "当たり":
            bk["当たり"] += 1
        if h.get("resolved_at", "")[:10] <= h.get("due", ""):
            out["resolved_by_due"] += 1
    if out["created"]:
        out["answer_rate"] = round(out["resolved_by_due"] / out["created"], 3)
    decided = out["当たり"] + out["外れ"]
    if decided:
        out["hit_rate"] = round(out["当たり"] / decided, 3)
    return out
