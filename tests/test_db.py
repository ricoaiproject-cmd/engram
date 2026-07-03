"""IndexDB のテスト。"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from engram.db import IndexDB
from engram.embedder import FakeEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 64


@pytest.fixture()
def db(tmp_path):
    d = IndexDB(tmp_path / "test.db", dim=DIM)
    yield d
    d.close()


@pytest.fixture()
def embedder():
    return FakeEmbedder(dim=DIM)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _upsert(db: IndexDB, embedder: FakeEmbedder, id: str, content: str,
            type: str = "knowledge", tier: str = "hot",
            importance: int = 5) -> None:
    emb = embedder.embed_docs([content])[0]
    db.upsert_memory(
        id=id,
        path=f"/memories/{id}.md",
        type=type,
        content_hash="abc",
        created_at=time.time(),
        importance=importance,
        tier=tier,
        content=content,
        embedding=emb,
    )


# ---------------------------------------------------------------------------
# スキーマ初期化
# ---------------------------------------------------------------------------

def test_schema_init(tmp_path):
    db = IndexDB(tmp_path / "db" / "test.db", dim=DIM)
    # Verify tables exist by querying them
    db._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
    db._conn.execute("SELECT COUNT(*) FROM access_events").fetchone()
    db._conn.execute("SELECT COUNT(*) FROM links").fetchone()
    db._conn.execute("SELECT COUNT(*) FROM meta").fetchone()
    db.close()


def test_parent_dir_created(tmp_path):
    nested = tmp_path / "deep" / "nested" / "db.sqlite"
    db = IndexDB(nested, dim=DIM)
    assert nested.exists()
    db.close()


# ---------------------------------------------------------------------------
# dim 不一致
# ---------------------------------------------------------------------------

def test_dim_mismatch_raises(tmp_path):
    db = IndexDB(tmp_path / "test.db", dim=64)
    db.close()
    with pytest.raises(ValueError, match="dim"):
        IndexDB(tmp_path / "test.db", dim=128)


def test_same_dim_ok(tmp_path):
    db1 = IndexDB(tmp_path / "test.db", dim=64)
    db1.close()
    db2 = IndexDB(tmp_path / "test.db", dim=64)
    db2.close()


# ---------------------------------------------------------------------------
# upsert → get
# ---------------------------------------------------------------------------

def test_upsert_and_get(db, embedder):
    _upsert(db, embedder, "id1", "hello world knowledge")
    row = db.get_memory("id1")
    assert row is not None
    assert row["id"] == "id1"
    assert row["type"] == "knowledge"
    assert row["tier"] == "hot"


def test_get_nonexistent(db):
    assert db.get_memory("nonexistent") is None


def test_upsert_updates_existing(db, embedder):
    _upsert(db, embedder, "id1", "original content", tier="hot")
    _upsert(db, embedder, "id1", "updated content", tier="cold")
    row = db.get_memory("id1")
    assert row["tier"] == "cold"


# ---------------------------------------------------------------------------
# vector_search: 類似テキストが上位 + tier/type フィルタ
# ---------------------------------------------------------------------------

def test_vector_search_similar_first(db, embedder):
    # Insert a closely related and an unrelated memory
    _upsert(db, embedder, "cat1", "cats are fluffy animals with whiskers")
    _upsert(db, embedder, "cat2", "my cat loves to sleep on the sofa every day")
    _upsert(db, embedder, "unrelated", "quantum mechanics and wave functions in physics")

    query_vec = embedder.embed_query("cats are cute pets")
    results = db.vector_search(query_vec, k=3)
    assert len(results) > 0
    ids = [r[0] for r in results]
    # cat-related memories should appear before unrelated
    assert ids.index("cat1") < ids.index("unrelated") or ids.index("cat2") < ids.index("unrelated")
    # scores should be between -1 and 1
    for _, score in results:
        assert -1.0 <= score <= 1.0


def test_vector_search_tier_filter(db, embedder):
    _upsert(db, embedder, "hot1", "hello world test", tier="hot")
    _upsert(db, embedder, "cold1", "hello world test", tier="cold")

    query_vec = embedder.embed_query("hello world")
    results = db.vector_search(query_vec, k=10, tiers=["hot"])
    ids = [r[0] for r in results]
    assert "hot1" in ids
    assert "cold1" not in ids


def test_vector_search_type_filter(db, embedder):
    _upsert(db, embedder, "ep1", "episode memory content here", type="episode")
    _upsert(db, embedder, "kn1", "knowledge memory content here", type="knowledge")

    query_vec = embedder.embed_query("memory content")
    results = db.vector_search(query_vec, k=10, types=["knowledge"])
    ids = [r[0] for r in results]
    assert "kn1" in ids
    assert "ep1" not in ids


def test_vector_search_empty_db(db, embedder):
    query_vec = embedder.embed_query("something")
    results = db.vector_search(query_vec, k=5)
    assert results == []


# ---------------------------------------------------------------------------
# keyword_search: 日本語・英語、trigram(3文字以上)
# ---------------------------------------------------------------------------

def test_keyword_search_english(db, embedder):
    _upsert(db, embedder, "en1", "Python programming language features")
    _upsert(db, embedder, "en2", "SQLite database storage engine")
    results = db.keyword_search("Python programming", k=5)
    assert len(results) > 0
    assert results[0][0] == "en1"


def test_keyword_search_japanese(db, embedder):
    _upsert(db, embedder, "jp1", "日本語のテキストを検索するテスト")
    _upsert(db, embedder, "jp2", "英語のテキストとは異なる内容")
    # trigram: at least 3 chars
    results = db.keyword_search("日本語", k=5)
    assert len(results) > 0
    assert results[0][0] == "jp1"


def test_keyword_search_short_query_no_match_returns_empty(db, embedder):
    _upsert(db, embedder, "x1", "test content here")
    # 3文字未満は LIKE フォールバック。部分一致しなければ空
    assert db.keyword_search("ab", k=5) == []
    assert db.keyword_search("a", k=5) == []


# ---------------------------------------------------------------------------
# keyword_search: 短クエリの LIKE フォールバック(trigram 非対応領域)
# ---------------------------------------------------------------------------

def test_keyword_search_short_query_like_fallback(db, embedder):
    _upsert(db, embedder, "m1", "毎週火曜の会議の議事録")
    _upsert(db, embedder, "m2", "全く関係ない内容のメモ")
    results = db.keyword_search("会議", k=5)
    assert [r[0] for r in results] == ["m1"]
    # 擬似スコアも BM25 と同じ「負値・小さいほど良い」向き
    assert results[0][1] < 0


def test_keyword_search_short_query_rare_scores_better_than_common(db, embedder):
    _upsert(db, embedder, "r1", "予算の稟議書")
    for i in range(5):
        _upsert(db, embedder, f"c{i}", f"会議メモその{i}")
    rare = db.keyword_search("稟議", k=5)
    common = db.keyword_search("会議", k=5)
    # 希少語(df=1/6)はありふれた語(df=5/6)よりスコアが良い(より負)
    assert rare[0][1] < common[0][1]


def test_keyword_search_short_query_orders_by_occurrences(db, embedder):
    _upsert(db, embedder, "many", "会議の前に会議室で会議の準備")
    _upsert(db, embedder, "once", "会議は火曜です")
    results = db.keyword_search("会議", k=5)
    assert [r[0] for r in results] == ["many", "once"]


def test_keyword_search_short_query_respects_tier_filter(db, embedder):
    _upsert(db, embedder, "h1", "会議の記録", tier="hot")
    _upsert(db, embedder, "t1", "会議の記録その二", tier="trash")
    results = db.keyword_search("会議", k=5, tiers=["hot"])
    assert [r[0] for r in results] == ["h1"]


def test_keyword_search_short_query_multi_token_and(db, embedder):
    _upsert(db, embedder, "both", "AB 棟の会議は火曜")
    _upsert(db, embedder, "one", "CD 棟の会議は水曜")
    # 全トークンが3文字未満 → LIKE の AND
    results = db.keyword_search("AB 会議", k=5)
    assert [r[0] for r in results] == ["both"]


def test_keyword_search_mixed_query_drops_short_tokens(db, embedder):
    _upsert(db, embedder, "room", "第3会議室の予約方法について")
    # 旧実装では 2文字トークン "AB" が MATCH 全体を0件にしていた。
    # 新実装は3文字以上のトークン(会議室)だけで検索する
    results = db.keyword_search("AB 会議室", k=5)
    assert [r[0] for r in results] == ["room"]


def test_keyword_search_short_query_escapes_like_wildcards(db, embedder):
    _upsert(db, embedder, "pct", "進捗は95%です")
    _upsert(db, embedder, "other", "進捗は九割五分です")
    # "%" が LIKE のワイルドカードとして解釈されないこと
    results = db.keyword_search("5%", k=5)
    assert [r[0] for r in results] == ["pct"]


def test_keyword_search_fts_error_returns_empty(db, embedder):
    # A query that could cause FTS parse errors → should not raise
    results = db.keyword_search("OR AND OR", k=5)
    # May return empty or results; should not raise
    assert isinstance(results, list)


def test_keyword_search_tier_filter(db, embedder):
    _upsert(db, embedder, "h1", "keyword search test content", tier="hot")
    _upsert(db, embedder, "c1", "keyword search test content", tier="cold")
    results = db.keyword_search("keyword search test", k=10, tiers=["cold"])
    ids = [r[0] for r in results]
    assert "c1" in ids
    assert "h1" not in ids


def test_keyword_search_type_filter(db, embedder):
    _upsert(db, embedder, "k1", "knowledge search content test", type="knowledge")
    _upsert(db, embedder, "p1", "knowledge search content test", type="preference")
    results = db.keyword_search("knowledge search content", k=10, types=["knowledge"])
    ids = [r[0] for r in results]
    assert "k1" in ids
    assert "p1" not in ids


# ---------------------------------------------------------------------------
# add_event / get_events
# ---------------------------------------------------------------------------

def test_add_and_get_events(db, embedder):
    _upsert(db, embedder, "ev1", "event test")
    now = time.time()
    db.add_event("ev1", "recall_hit", 0.3, now)
    db.add_event("ev1", "reinforce", 1.0, now + 1)

    events = db.get_events(["ev1"])
    assert "ev1" in events
    assert len(events["ev1"]) == 2
    tses = [e[0] for e in events["ev1"]]
    assert tses == sorted(tses)  # ordered by ts


def test_get_events_missing_id_empty_list(db):
    result = db.get_events(["nonexistent"])
    assert result == {"nonexistent": []}


def test_get_events_multiple_ids(db, embedder):
    _upsert(db, embedder, "a", "content a")
    _upsert(db, embedder, "b", "content b")
    db.add_event("a", "create", 1.0, 1000.0)
    db.add_event("b", "create", 1.0, 2000.0)

    result = db.get_events(["a", "b", "c"])
    assert len(result["a"]) == 1
    assert len(result["b"]) == 1
    assert result["c"] == []


# ---------------------------------------------------------------------------
# add_link: weight 加算とクランプ
# ---------------------------------------------------------------------------

def test_add_link_creates_new(db):
    db.add_link("a", "b", "explicit")
    links = db.get_links(["a"])
    assert len(links) == 1
    src, dst, kind, weight = links[0]
    assert src == "a" and dst == "b" and kind == "explicit"
    assert weight == 1.0


def test_add_link_increments_weight(db):
    db.add_link("a", "b", "co_recall", increment=0.1)
    db.add_link("a", "b", "co_recall", increment=0.1)
    db.add_link("a", "b", "co_recall", increment=0.1)
    links = db.get_links(["a"])
    assert len(links) == 1
    _, _, _, weight = links[0]
    assert abs(weight - 0.3) < 1e-9


def test_add_link_clamped(db):
    db.add_link("a", "b", "explicit", increment=0.8, max_weight=1.0)
    db.add_link("a", "b", "explicit", increment=0.8, max_weight=1.0)
    links = db.get_links(["a"])
    _, _, _, weight = links[0]
    assert weight == 1.0


def test_get_links_includes_dst(db):
    db.add_link("x", "y", "derived_from")
    # y is a dst, should appear when querying for y
    links = db.get_links(["y"])
    assert len(links) == 1
    assert links[0][0] == "x" and links[0][1] == "y"


def test_get_links_kind_filter(db):
    db.add_link("a", "b", "explicit")
    db.add_link("a", "c", "co_recall")
    links = db.get_links(["a"], kinds=["explicit"])
    assert all(l[2] == "explicit" for l in links)
    assert len(links) == 1


# ---------------------------------------------------------------------------
# get_embeddings
# ---------------------------------------------------------------------------

def test_get_embeddings(db, embedder):
    _upsert(db, embedder, "e1", "embedding test content")
    result = db.get_embeddings(["e1"])
    assert "e1" in result
    vec = result["e1"]
    assert vec.dtype == np.float32
    assert vec.shape == (DIM,)


def test_get_embeddings_missing_not_included(db, embedder):
    _upsert(db, embedder, "e1", "content")
    result = db.get_embeddings(["e1", "nonexistent"])
    assert "e1" in result
    assert "nonexistent" not in result


def test_get_embeddings_empty(db):
    assert db.get_embeddings([]) == {}


# ---------------------------------------------------------------------------
# delete_memory
# ---------------------------------------------------------------------------

def test_delete_removes_from_all_tables(db, embedder):
    _upsert(db, embedder, "del1", "to be deleted content here")
    db.add_event("del1", "create", 1.0, time.time())
    db.add_link("del1", "other", "explicit")

    db.delete_memory("del1")

    assert db.get_memory("del1") is None

    # Not in vec
    embs = db.get_embeddings(["del1"])
    assert "del1" not in embs

    # Not in fts
    results = db.keyword_search("deleted content here", k=5)
    assert not any(r[0] == "del1" for r in results)

    # Not in events
    events = db.get_events(["del1"])
    assert events["del1"] == []

    # Not in links
    links = db.get_links(["del1"])
    assert len(links) == 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def test_stats_empty(db):
    s = db.stats()
    assert s["total_memories"] == 0
    assert s["event_count"] == 0


def test_stats_counts(db, embedder):
    _upsert(db, embedder, "k1", "knowledge one", type="knowledge", tier="hot")
    _upsert(db, embedder, "k2", "knowledge two", type="knowledge", tier="cold")
    _upsert(db, embedder, "e1", "episode one", type="episode", tier="hot")
    db.add_event("k1", "create", 1.0, time.time())
    db.add_link("k1", "e1", "explicit")

    s = db.stats()
    assert s["total_memories"] == 3
    assert s["by_type"]["knowledge"] == 2
    assert s["by_type"]["episode"] == 1
    assert s["by_tier"]["hot"] == 2
    assert s["by_tier"]["cold"] == 1
    assert s["event_count"] == 1
    assert s["links_by_kind"]["explicit"] == 1


# ---------------------------------------------------------------------------
# export_events_jsonl
# ---------------------------------------------------------------------------

def test_export_events_jsonl(db, embedder, tmp_path):
    _upsert(db, embedder, "ej1", "export test content")
    now = time.time()
    db.add_event("ej1", "create", 1.0, now)
    db.add_event("ej1", "recall_hit", 0.3, now + 1)

    out = tmp_path / "out" / "events.jsonl"
    count = db.export_events_jsonl(out)
    assert count == 2
    assert out.exists()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["memory_id"] == "ej1"
    assert first["kind"] == "create"


def test_export_events_jsonl_empty(db, tmp_path):
    out = tmp_path / "events.jsonl"
    count = db.export_events_jsonl(out)
    assert count == 0
    assert out.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# set_tier / set_path / all_memories
# ---------------------------------------------------------------------------

def test_set_tier(db, embedder):
    _upsert(db, embedder, "t1", "tier test", tier="hot")
    db.set_tier("t1", "cold")
    assert db.get_memory("t1")["tier"] == "cold"


def test_set_path(db, embedder):
    _upsert(db, embedder, "p1", "path test")
    db.set_path("p1", "/new/path.md")
    assert db.get_memory("p1")["path"] == "/new/path.md"


def test_all_memories_no_filter(db, embedder):
    _upsert(db, embedder, "a1", "aaa", type="knowledge", tier="hot")
    _upsert(db, embedder, "a2", "bbb", type="episode", tier="cold")
    rows = db.all_memories()
    assert len(rows) == 2


def test_all_memories_tier_filter(db, embedder):
    _upsert(db, embedder, "a1", "aaa", tier="hot")
    _upsert(db, embedder, "a2", "bbb", tier="cold")
    rows = db.all_memories(tiers=["hot"])
    assert len(rows) == 1 and rows[0]["id"] == "a1"


def test_all_memories_type_filter(db, embedder):
    _upsert(db, embedder, "a1", "aaa", type="knowledge")
    _upsert(db, embedder, "a2", "bbb", type="episode")
    rows = db.all_memories(types=["episode"])
    assert len(rows) == 1 and rows[0]["id"] == "a2"
