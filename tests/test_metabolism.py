"""代謝レポート(metabolism)のテスト。軽量経路(素 sqlite3)で動くことを含めて検証。"""

from __future__ import annotations

import json

from engram.config import Settings
from engram.db import IndexDB
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine
from engram.metabolism import compute_metrics, format_report
from engram.store import MarkdownStore

NOW = 1_800_000_000.0
DAY = 86400.0


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


def test_compute_metrics_counts_hits_and_stock(tmp_path):
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)

    r_hit = engine.remember("役立った記憶", "knowledge", 5, now=NOW - 10 * DAY)
    r_miss = engine.remember("外れた記憶", "project", 5, now=NOW - 10 * DAY)
    r_ep = engine.remember("古いエピソード", "episode", 3, now=NOW - 30 * DAY)
    # 浮上→2時間後に reinforce(当たり)
    engine.db.add_event(r_hit["id"], "reinforce", 1.0, NOW - 3 * DAY + 7200)
    # もう1件は申告なしだが、暗黙の使用(implicit_use)の形跡がある
    engine.db.add_event(r_miss["id"], "implicit_use", 0.5, NOW - 2 * DAY + 3600)
    engine.db.close()

    d = settings.data_dir / "surface"
    d.mkdir(parents=True, exist_ok=True)
    rows = [
        {"ts": NOW - 3 * DAY, "surfaced": [r_hit["id"]], "semantic": 12},
        {"ts": NOW - 2 * DAY, "surfaced": [r_miss["id"]]},
    ]
    with (d / "surface_log.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # 統合の滞留状態
    (settings.data_dir / "consolidation_state.json").write_text(
        json.dumps({"clusters": 2, "skill_clusters": 5}), encoding="utf-8"
    )

    m = compute_metrics(settings, now=NOW, window_days=7.0)

    s = m["surface"]
    assert s["instances_total"] == 2
    assert s["hits_total"] == 1
    assert s["hit_rate_total"] == 0.5
    # 層別: 意味検索ありの浮上は当たり、字句のみは外れ
    assert s["semantic_window"] == {"instances": 1, "hits": 1, "hit_rate": 1.0}
    assert s["lexical_window"] == {"instances": 1, "hits": 0, "hit_rate": 0.0}
    # 実使用率: 申告1件 + 暗黙の使用1件 = 2/2
    assert s["use_window"] == {"hits": 2, "use_rate": 1.0}

    # 書き込み: 直近7日の create は 0(全部それ以前)、総記憶は3
    assert m["writes"]["creates_window"] == 0
    assert m["writes"]["total_memories"] == 3

    st = m["stock"]
    assert st["hot_total"] == 3
    assert st["hot_episode"] == 1
    assert st["hot_other"] == 2

    c = m["consolidation"]
    assert c["clusters"] == 2
    assert c["skill_clusters"] == 5
    # 30日前の episode は統合適齢(既定14日)に達している
    assert c["old_hot_episodes"] == 1

    # 人間可読レポートが4指標を含む
    report = format_report(m)
    assert "浮上命中率" in report
    assert "書き込み量" in report
    assert "hot在庫" in report
    assert "統合の滞留" in report


def test_compute_metrics_survives_missing_inputs(tmp_path):
    """DB もログも無い環境で落ちず、ゼロ埋めで返す。"""
    settings = _make_settings(tmp_path)
    m = compute_metrics(settings, now=NOW)
    assert m["surface"]["instances_total"] == 0
    assert m["surface"]["hit_rate_total"] is None
    assert m["writes"]["total_memories"] == 0
    assert m["stock"]["hot_total"] == 0
    format_report(m)  # 例外を出さない
