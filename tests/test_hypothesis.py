"""仮説サイクル(hypothesis.py)の統合テスト。実 db/store/engine + FakeEmbedder。"""

import random

import pytest

from engram import hypothesis as H
from engram.config import Settings
from engram.embedder import FakeEmbedder
from engram.engine import build_engine
from engram.rsi import compute, format_report

DAY = 86400.0
T0 = 1_750_000_000.0  # 2025-06-15 頃


@pytest.fixture
def engine(tmp_path):
    settings = Settings(memories_dir=tmp_path / "memories", data_dir=tmp_path / "data",
                        candidate_k=10, hyp_evidence_min_cos=0.0)
    eng = build_engine(settings, embedder=FakeEmbedder(dim=256))
    yield eng
    eng.db.close()


def _two(engine):
    a = engine.remember("夜間に記憶を整理すると翌日の想起率が上がる", type="knowledge", importance=5, now=T0)
    b = engine.remember("YouTube の動画は冒頭5秒で離脱が決まる", type="knowledge", importance=5, now=T0)
    return a["id"], b["id"]


def test_rejects_without_falsifier_or_two_sources(engine):
    a, b = _two(engine)
    assert H.hypothesize(engine, "主張", [a, b], "", now=T0)["status"] == "rejected"
    assert H.hypothesize(engine, "主張", [a], "外れたと分かる条件はこれ", now=T0)["status"] == "rejected"
    assert H.hypothesize(engine, "主張", [a, "nope"], "外れたと分かる条件はこれ", now=T0)["status"] == "rejected"


def test_cycle_supported_creates_knowledge_and_links(engine):
    a, b = _two(engine)
    r = H.hypothesize(engine, "睡眠中の記憶整理は学習動画の視聴維持にも効く", [a, b],
                      "査読論文で効果なしと示されている", "論文検索", now=T0)
    assert r["status"] == "created"
    hid = r["id"]
    assert engine.store.find_by_id(hid).tags == ["hypothesis", "hyp-open"]
    # 同じ日には判定できない
    assert H.resolve(engine, hid, "当たり", "根拠", now=T0 + 60)["status"] == "rejected"
    # 期限前は due に出ない、期限後は出る
    assert H.list_hypotheses(engine, "due", now=T0 + DAY)["count"] == 0
    assert H.list_hypotheses(engine, "due", now=T0 + 3 * DAY)["count"] == 1
    out = H.resolve(engine, hid, "supported", "論文Xで確認", evidence_urls=["https://example.org/x"],
                    now=T0 + 3 * DAY)
    assert out["status"] == "resolved" and out["verdict"] == "当たり"
    rid = out["result_id"]
    rec = engine.store.find_by_id(rid)
    assert rec.type == "knowledge" and "hypothesis-result" in rec.tags
    assert "https://example.org/x" in rec.content
    kinds = {(s, d, k) for s, d, k, _w in engine.db.get_links([rid])}
    assert (rid, hid, "derived_from") in kinds
    assert "hyp-当たり" in engine.store.find_by_id(hid).tags
    # 元の記憶が温められている(reinforce イベント)
    ev = engine.db.get_events([a])[a]
    assert len(ev) >= 2
    st = H.ledger_stats(engine.settings, now=T0 + 3 * DAY)
    assert st["当たり"] == 1 and st["hit_rate"] == 1.0 and st["answer_rate"] == 1.0
    assert H.resolve(engine, hid, "外れ", "x", now=T0 + 4 * DAY)["status"] == "already_resolved"


def test_refuted_goes_through_correct(engine):
    a, b = _two(engine)
    hid = H.hypothesize(engine, "仮説その2", [a, b], "公式資料に反対の記述がある", now=T0)["id"]
    assert H.resolve(engine, hid, "外れ", "出典なし", now=T0 + 2 * DAY)["status"] == "rejected"
    out = H.resolve(engine, hid, "外れ", "公式資料で否定", evidence_urls=["https://example.org/doc"],
                    now=T0 + 2 * DAY)
    assert out["status"] == "resolved"
    assert engine.db.get_memory(hid)["tier"] == "superseded"
    new = engine.store.find_by_id(out["result_id"])
    assert "correction" in new.tags and "hyp-外れ" in new.tags and "hyp-open" not in new.tags
    assert new.valence < 0


def test_twelve_hour_lock_across_midnight(engine):
    import datetime as dt
    late = dt.datetime(2025, 6, 15, 23, 50).timestamp()
    a, b = _two(engine)
    hid = H.hypothesize(engine, "深夜の仮説", [a, b], "反証の条件をここに書く", due="1", now=late)["id"]
    after_midnight = late + 30 * 60
    assert H.list_hypotheses(engine, "due", now=after_midnight)["count"] == 0
    assert H.resolve(engine, hid, "判定不能", "x", now=after_midnight)["status"] == "rejected"
    assert H.resolve(engine, hid, "判定不能", "x", now=late + 13 * 3600)["status"] == "extended"


def test_similar_templates_are_not_merged(engine):
    a, b = _two(engine)
    c = engine.remember("音声は口元に近いマイクほど文字起こしが正確になる", type="knowledge",
                        importance=5, now=T0)["id"]
    r1 = H.hypothesize(engine, "仮説A", [a, b], "反証の条件をここに書く", now=T0)
    r2 = H.hypothesize(engine, "仮説A", [a, c], "反証の条件をここに書く", now=T0)
    r3 = H.hypothesize(engine, "仮説A", [b, a], "反証の条件をここに書く", now=T0)
    assert r1["status"] == r2["status"] == "created" and r1["id"] != r2["id"]
    assert r3["status"] == "duplicate" and r3["id"] == r1["id"]


def test_inconclusive_extends_then_closes(engine):
    a, b = _two(engine)
    hid = H.hypothesize(engine, "仮説その3", [a, b], "反証の条件をここに書く", now=T0)["id"]
    first = H.resolve(engine, hid, "判定不能", "資料が見つからない", now=T0 + 3 * DAY)
    assert first["status"] == "extended"
    second = H.resolve(engine, hid, "判定不能", "やはり無い", now=T0 + 11 * DAY)
    assert second["status"] == "resolved"
    assert H.list_hypotheses(engine, "open", now=T0 + 11 * DAY)["count"] == 0


def test_seeds_exclude_used_pairs_and_ledger_survives_reload(engine):
    ids = [engine.remember(f"知見{i}: テーマ{i % 4}について観察した事実{i}", type="knowledge",
                           importance=5, now=T0)["id"] for i in range(12)]
    res = H.hypothesis_seeds(engine, n=3, rng=random.Random(1))
    assert res["pairs"] and all(len(p["memories"]) == 2 for p in res["pairs"])
    p = res["pairs"][0]
    pair = [m["id"] for m in p["memories"]]
    H.hypothesize(engine, "組み合わせの仮説", pair, "反証の条件をここに書く", now=T0)
    for _ in range(5):
        again = H.hypothesis_seeds(engine, n=5, rng=random.Random())
        for q in again["pairs"]:
            assert {m["id"] for m in q["memories"]} != set(pair)
    assert H.load_ledger(engine.settings)["hypotheses"]
    assert set(ids)  # noqa


def test_rsi_report_has_hypothesis_line(engine):
    a, b = _two(engine)
    H.hypothesize(engine, "仮説その4", [a, b], "反証の条件をここに書く", now=T0)
    m = compute(engine.settings, window_days=7, now=T0 + DAY)
    assert m["hypotheses"]["created"] == 1
    assert "4. 仮説サイクル" in format_report(m)
