"""縁の層(R8.10.11): 要旨だけの弱い想起(近接・連想)と、read(ids) による引き上げ。"""
from __future__ import annotations

import json

from engram.config import Settings
from engram.db import IndexDB
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine
from engram.store import MarkdownStore
from engram.surface import format_context, run_surface


def _settings(tmp_path, **kw) -> Settings:
    base = dict(memories_dir=tmp_path / "memories", data_dir=tmp_path / "data",
                surface_semantic=False, surface_fringe_items=5, surface_fringe_min_relevance=0.0)
    base.update(kw)
    return Settings(**base)


def _engine(s: Settings) -> MemoryEngine:
    emb = FakeEmbedder(dim=64)
    return MemoryEngine(settings=s, store=MarkdownStore(s.memories_dir), db=IndexDB(s.db_path, emb.dim), embedder=emb)


def test_fringe_lists_near_and_link_without_duplicating_surfaced(tmp_path):
    s = _settings(tmp_path)
    eng = _engine(s)
    a = eng.remember("予算要求の書式は財務課の様式7を使う", "knowledge", 7)["id"]
    b = eng.remember("予算要求の締切は毎年10月末", "knowledge", 5)["id"]
    c = eng.remember("予算のヒアリングは11月に経理課と行う", "knowledge", 5)["id"]
    d = eng.remember("動画のサムネイルはピンク基調", "preference", 5)["id"]
    eng.link(a, d)   # 予算の記憶から、無関係に見える記憶へ明示リンク(連想の種)

    res = run_surface("来年度の予算要求の準備を始めたい", settings=s, session_id="s1")
    surfaced = set(res["surfaced"])
    fringe = res["fringe_items"]
    ids = [f["id"] for f in fringe]
    assert fringe, "縁が空"
    assert not (surfaced & set(ids)), "浮上した記憶が縁に重複している"
    assert len(ids) == len(set(ids))
    assert all(len(f["gist"]) <= s.surface_fringe_gist_chars + 1 for f in fringe)
    assert {f["via"] for f in fringe} <= {"near", "link"}
    if a in surfaced:
        assert d in ids and next(f for f in fringe if f["id"] == d)["via"] == "link"

    # 同じセッションでは同じ縁を二度出さない
    res2 = run_surface("来年度の予算要求の準備を始めたい", settings=s, session_id="s1")
    assert not (set(ids) & {f["id"] for f in res2["fringe_items"]})
    # ログに縁の id が残る
    log = (s.data_dir / "surface" / "surface_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(log[0])["fringe"] == ids

    txt = format_context(res["surfaced_items"], fringe_items=fringe)
    assert "縁の想起" in txt and "read(ids)" in txt and ids[0] in txt
    assert format_context([], fringe_items=[]) == ""


def test_fringe_disabled_when_items_zero(tmp_path):
    s = _settings(tmp_path, surface_fringe_items=0)
    eng = _engine(s)
    eng.remember("予算要求の書式は財務課の様式7を使う", "knowledge", 7)
    res = run_surface("予算要求の準備", settings=s, session_id="s0")
    assert res["fringe_items"] == []


def test_engine_read_returns_bodies_and_records(tmp_path):
    s = _settings(tmp_path)
    eng = _engine(s)
    a = eng.remember("読むための記憶の本文です", "knowledge", 5)["id"]
    r = eng.read([a, "01NOPE"])
    assert r["memories"][0]["id"] == a and "本文" in r["memories"][0]["content"]
    assert r["missing"] == ["01NOPE"]
    kinds = [e for e in eng.db.get_events(a)]
    assert kinds, "read が使用記録を残していない"
    reads = (s.data_dir / "surface" / "fringe_reads.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(reads[-1])["ids"] == [a]
