"""自発的想起(surface)のテスト。軽量経路なので埋め込みモデルは使わない。"""

from __future__ import annotations

import json

from engram.config import Settings
from engram.db import IndexDB
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine
from engram.store import MarkdownStore
from engram.surface import bigrams, format_context, lexical_scores, run_surface


def _make_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        memories_dir=tmp_path / "memories",
        data_dir=tmp_path / "data",
        **overrides,
    )


def _make_engine(settings) -> MemoryEngine:
    embedder = FakeEmbedder(dim=64)
    store = MarkdownStore(settings.memories_dir)
    db = IndexDB(settings.db_path, embedder.dim)
    return MemoryEngine(settings=settings, store=store, db=db,
                        embedder=embedder)


# ---------------------------------------------------------------------------
# 字句関連度
# ---------------------------------------------------------------------------

def test_bigrams_basic():
    g = bigrams("予算 要求")
    assert "予算" in g
    assert "要求" in g
    # 空白を跨ぐバイグラムは作らない
    assert "算要" not in g


def test_lexical_scores_ranks_related_doc_higher():
    docs = [
        "予算要求の書式は財務課の様式7を使うこと",
        "動画のサムネイルはピンク基調でかわいく",
        "会議室の予約は前日までに行う",
    ]
    scores = lexical_scores("来年度の予算要求の準備を始めたい", docs)
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_lexical_scores_empty():
    assert lexical_scores("", ["doc"]) == [0.0]
    assert lexical_scores("クエリ", []) == []


# ---------------------------------------------------------------------------
# run_surface
# ---------------------------------------------------------------------------

def test_run_surface_finds_related_memory(tmp_path):
    settings = _make_settings(tmp_path, surface_threshold=0.3)
    engine = _make_engine(settings)
    engine.remember("予算要求の書式は財務課の様式7を使うこと", "knowledge", 7)
    engine.remember("動画のサムネイルはピンク基調でかわいく", "knowledge", 5)
    engine.db.close()

    result = run_surface(
        "予算要求の書式ってどうだったか",
        settings=settings, session_id="s1",
    )
    assert result["candidates"]
    top = result["candidates"][0]
    assert "予算要求" in top["content"]
    # 関連が強いものは閾値を超えて浮上する
    assert top["id"] in result["surfaced"]


def test_run_surface_excludes_episodes(tmp_path):
    # episode は自発的想起の対象外(オウム返し・ノイズ防止)
    settings = _make_settings(tmp_path, surface_threshold=0.3)
    engine = _make_engine(settings)
    engine.remember("予算要求の書式は財務課の様式7を使うこと", "knowledge", 7)
    engine.remember("予算要求の書式について相談したセッションの記録", "episode", 5)
    engine.db.close()

    result = run_surface("予算要求の書式", settings=settings, session_id="s1")
    types = {c["type"] for c in result["candidates"]}
    assert "episode" not in types
    assert result["candidates"]  # knowledge は出る


def test_run_surface_no_repeat_within_session(tmp_path):
    settings = _make_settings(tmp_path, surface_threshold=0.3)
    engine = _make_engine(settings)
    engine.remember("予算要求の書式は財務課の様式7を使うこと", "knowledge", 7)
    engine.db.close()

    r1 = run_surface("予算要求の書式について", settings=settings,
                     session_id="s1")
    assert r1["surfaced"]
    # 同一セッションでは同じ記憶を二度浮上させない
    r2 = run_surface("予算要求の書式をもう一度", settings=settings,
                     session_id="s1")
    assert r1["surfaced"][0] not in r2["surfaced"]
    # 別セッションでは再浮上できる
    r3 = run_surface("予算要求の書式について", settings=settings,
                     session_id="s2")
    assert r3["surfaced"]


def test_run_surface_respects_rooms(tmp_path):
    settings = _make_settings(tmp_path, surface_threshold=0.1)
    engine = _make_engine(settings)
    engine.remember("動画のサムネイルはピンク基調でかわいく", "knowledge", 7,
                    room="personal")
    engine.db.close()

    # work の部屋からは personal の記憶が候補にすら入らない
    result = run_surface("サムネイルの色はどうする?", settings=settings,
                         room="work", session_id="s1")
    assert all(c["room"] != "personal" for c in result["candidates"])

    # personal の部屋では出る
    result = run_surface("サムネイルの色はどうする?", settings=settings,
                         room="personal", session_id="s2")
    assert any(c["room"] == "personal" for c in result["candidates"])


def test_run_surface_writes_log_and_dry_run_does_not(tmp_path):
    settings = _make_settings(tmp_path, surface_threshold=0.3)
    engine = _make_engine(settings)
    engine.remember("予算要求の書式は財務課の様式7を使うこと", "knowledge", 7)
    engine.db.close()

    log = settings.data_dir / "surface" / "surface_log.jsonl"

    run_surface("予算要求の話", settings=settings, session_id="s1",
                dry_run=True)
    assert not log.exists()

    run_surface("予算要求の話", settings=settings, session_id="s1")
    assert log.is_file()
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["session_id"] == "s1"
    assert entry["candidates"]


def test_run_surface_off_mode(tmp_path):
    settings = _make_settings(tmp_path, surface_mode="off")
    result = run_surface("なんでもいい発言", settings=settings,
                         session_id="s1")
    assert result["candidates"] == []
    assert result["surfaced"] == []


def test_run_surface_threshold_filters(tmp_path):
    settings = _make_settings(tmp_path, surface_threshold=0.99)
    engine = _make_engine(settings)
    engine.remember("予算要求の書式は財務課の様式7を使うこと", "knowledge", 7)
    engine.db.close()

    result = run_surface("予算要求の書式", settings=settings, session_id="s1")
    assert result["candidates"]          # 候補はある
    assert result["surfaced"] == []      # が、閾値が高すぎて浮上しない


def test_run_surface_relevance_gate(tmp_path):
    # 活性度・重要度が高くても、発話と無関係な記憶は浮上しない
    settings = _make_settings(tmp_path, surface_threshold=0.3)
    engine = _make_engine(settings)
    r = engine.remember("本番データベースは絶対に削除しない", "preference", 10)
    engine.reinforce([r["id"]], strength=3.0)  # 活性度を盛る
    engine.db.close()

    result = run_surface("今日の天気はどうかな", settings=settings,
                         session_id="s1")
    assert result["surfaced"] == []


def test_run_surface_missing_db(tmp_path):
    settings = _make_settings(tmp_path)
    result = run_surface("発言", settings=settings, session_id="s1")
    assert result["candidates"] == []


def test_format_context_includes_ids_and_instruction():
    items = [{"id": "01ABC", "type": "knowledge", "room": "common",
              "content": "予算要求の書式は様式7"}]
    text = format_context(items)
    assert "01ABC" in text
    assert "reinforce" in text
    assert "様式7" in text
