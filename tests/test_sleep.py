"""眠り(sleep: 沈黙降格・常連抑制)のテスト。

silence は FakeEmbedder + IndexDB で実記憶を作って判定・降格を検証する。
suppression は surface ログと reinforce イベントを組み立てて検証する。
"""

from __future__ import annotations

import json

import frontmatter

from engram.config import Settings
from engram.db import IndexDB
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine
from engram.sleep import (
    compute_suppression,
    run_sleep,
    silence_candidates,
    suppression_path,
)
from engram.store import MarkdownStore
from engram.surface import run_surface

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


def _write_surface_log(settings, rows) -> None:
    d = settings.data_dir / "surface"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "surface_log.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 沈黙(silence)
# ---------------------------------------------------------------------------

def test_silence_candidates_selects_only_idle_unprotected(tmp_path):
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)

    old = NOW - 100 * DAY
    # 対象になる: 古い knowledge / importance 5
    r_idle = engine.remember("古い作業メモ その1", "knowledge", 5, now=old)
    # 保護: type=feedback(ユーザー訂正)
    r_fb = engine.remember("ユーザー訂正の記録", "feedback", 5, now=old)
    # 保護: importance 9
    r_imp = engine.remember("重大事故の教訓", "knowledge", 9, now=old)
    # 対象外: 最近作成
    r_new = engine.remember("昨日のメモ", "knowledge", 5, now=NOW - 1 * DAY)
    # 対象外: 古いが最近 reinforce された
    r_used = engine.remember("よく使う手順", "knowledge", 5, now=old)
    engine.db.add_event(r_used["id"], "reinforce", 1.0, NOW - 5 * DAY)
    engine.db.close()

    ids = {c["id"] for c in silence_candidates(settings, now=NOW)}
    assert r_idle["id"] in ids
    assert r_fb["id"] not in ids
    assert r_imp["id"] not in ids
    assert r_new["id"] not in ids
    assert r_used["id"] not in ids


def test_silence_pressure_demotes_oldest_idle_not_newborns(tmp_path):
    """在庫圧力モードの選抜: 保護は別格(imp9+)のみ・新参には生後猶予。

    importance 8 を恒久保護にすると hot が降格不能な古参で埋まり、
    圧力の全量が生後1日の新参にかかる(実測 R8.9.27: hot 940件中934件が
    保護で不動・沈むのは新参だけ)。この硬直を防ぐ選抜を検証する。
    """
    settings = _make_settings(tmp_path, sleep_mode="pressure",
                              sleep_hot_target=3)
    engine = _make_engine(settings)
    old = NOW - 100 * DAY
    # 沈む: imp8 でも放置90日超なら pressure では対象(恒久保護しない)
    r_old8 = engine.remember("放置された古参のimp8", "knowledge", 8, now=old)
    # 保護: 別格(imp9)
    r_shield = engine.remember("重大事故の教訓", "knowledge", 9, now=old)
    # 猶予: 生後1日の新参(imp5)は沈めない
    r_new = engine.remember("生まれたての知見", "knowledge", 5,
                            now=NOW - 1 * DAY)
    # 残る: 古参でも最近使われた(放置2日 < 古参の放置)
    r_used = engine.remember("よく使う手順", "knowledge", 5, now=old)
    engine.db.add_event(r_used["id"], "reinforce", 1.0, NOW - 2 * DAY)
    engine.db.close()

    # hot 4件・目標3 → 超過1件だけ沈む。放置が最長の古参imp8が選ばれる
    ids = [c["id"] for c in silence_candidates(settings, now=NOW)]
    assert ids == [r_old8["id"]]


def test_silence_pressure_no_over_no_demote(tmp_path):
    """在庫が目標以下なら、放置が長くても何も沈めない。"""
    settings = _make_settings(tmp_path, sleep_mode="pressure",
                              sleep_hot_target=10)
    engine = _make_engine(settings)
    engine.remember("放置された古い記憶", "knowledge", 5, now=NOW - 100 * DAY)
    engine.db.close()
    assert silence_candidates(settings, now=NOW) == []


def test_run_sleep_dry_run_changes_nothing(tmp_path):
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r = engine.remember("古い作業メモ dry", "knowledge", 5, now=NOW - 120 * DAY)
    engine.db.close()

    result = run_sleep(settings, now=NOW, commit=False)
    assert result["dry_run"] is True
    assert result["silence"]["candidates"] == 1
    assert result["silence"]["demoted"] == []
    assert result["report_path"] is None
    assert not suppression_path(settings).exists()

    # DB もファイルも hot のまま
    db = IndexDB(settings.db_path, 64)
    assert db.get_memory(r["id"])["tier"] == "hot"
    db.close()
    post = frontmatter.load(r["path"])
    assert post["tier"] == "hot"


def test_run_sleep_commit_demotes_db_and_file(tmp_path):
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r = engine.remember("古い作業メモ commit", "knowledge", 5, now=NOW - 120 * DAY)
    engine.db.close()

    result = run_sleep(settings, now=NOW, commit=True)
    assert result["silence"]["demoted"] == [r["id"]]
    assert result["silence"]["file_errors"] == []
    assert result["report_path"] is not None
    assert (settings.data_dir / "sleep").is_dir()

    db = IndexDB(settings.db_path, 64)
    assert db.get_memory(r["id"])["tier"] == "cold"
    db.close()
    post = frontmatter.load(r["path"])
    assert post["tier"] == "cold"
    # 抑制表も(空でも)書かれる
    assert suppression_path(settings).exists()


# ---------------------------------------------------------------------------
# 常連抑制(suppression)
# ---------------------------------------------------------------------------

def test_suppression_streak_of_three_misses(tmp_path):
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r = engine.remember("常連の記憶", "knowledge", 5, now=NOW - 30 * DAY)
    engine.db.close()
    mid = r["id"]

    _write_surface_log(settings, [
        {"ts": NOW - 3 * DAY, "surfaced": [mid]},
        {"ts": NOW - 2 * DAY, "surfaced": [mid]},
        {"ts": NOW - 1 * DAY - 3600, "surfaced": [mid]},
    ])
    data = compute_suppression(settings, now=NOW)
    assert mid in data["suppressed"]
    item = data["suppressed"][mid]
    assert item["misses"] == 3
    assert item["until"] > NOW


def test_suppression_reset_by_reinforce(tmp_path):
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r = engine.remember("たまに役立つ記憶", "knowledge", 5, now=NOW - 30 * DAY)
    mid = r["id"]
    # 2回目の浮上の直後に reinforce → ミス連続が切れる
    engine.db.add_event(mid, "reinforce", 1.0, NOW - 2 * DAY + 600)
    engine.db.close()

    _write_surface_log(settings, [
        {"ts": NOW - 3 * DAY, "surfaced": [mid]},
        {"ts": NOW - 2 * DAY, "surfaced": [mid]},
        {"ts": NOW - 1 * DAY - 3600, "surfaced": [mid]},
    ])
    data = compute_suppression(settings, now=NOW)
    assert mid not in data["suppressed"]


def test_suppression_reset_by_implicit_use(tmp_path):
    """申告(reinforce)が無くても、暗黙の使用の形跡があれば黙らせない。

    reinforce のみを見ると申告漏れの記憶を誤爆する
    (実測 R8.9.26: 抑制316件中88件=28%が実は使われていた)。
    """
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r = engine.remember("申告漏れだが使われている記憶", "knowledge", 5,
                        now=NOW - 30 * DAY)
    mid = r["id"]
    # 2回目の浮上の直後に implicit_use → ミス連続が切れる
    engine.db.add_event(mid, "implicit_use", 0.5, NOW - 2 * DAY + 600)
    engine.db.close()

    _write_surface_log(settings, [
        {"ts": NOW - 3 * DAY, "surfaced": [mid]},
        {"ts": NOW - 2 * DAY, "surfaced": [mid]},
        {"ts": NOW - 1 * DAY - 3600, "surfaced": [mid]},
    ])
    data = compute_suppression(settings, now=NOW)
    assert mid not in data["suppressed"]


def test_suppression_recent_surfacing_not_counted_as_miss(tmp_path):
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r = engine.remember("判定保留中の記憶", "knowledge", 5, now=NOW - 30 * DAY)
    engine.db.close()
    mid = r["id"]

    # 2回の古いミス + 1回のごく最近の浮上(hit_window 未満) → まだ抑制しない
    _write_surface_log(settings, [
        {"ts": NOW - 3 * DAY, "surfaced": [mid]},
        {"ts": NOW - 2 * DAY, "surfaced": [mid]},
        {"ts": NOW - 3600, "surfaced": [mid]},
    ])
    data = compute_suppression(settings, now=NOW)
    assert mid not in data["suppressed"]


def test_surface_skips_suppressed_memory(tmp_path):
    settings = _make_settings(
        tmp_path, surface_threshold=0.0, surface_min_relevance=0.0
    )
    engine = _make_engine(settings)
    r = engine.remember(
        "予算要求の書式は財務課の様式7を使うこと", "knowledge", 5, now=NOW - DAY
    )
    engine.db.close()
    mid = r["id"]

    # 抑制表に載せる
    d = settings.data_dir / "surface"
    d.mkdir(parents=True, exist_ok=True)
    (d / "suppression.json").write_text(
        json.dumps({"suppressed": {mid: {"until": NOW + DAY}}}),
        encoding="utf-8",
    )

    result = run_surface(
        "予算要求の書式を確認したい", settings=settings,
        session_id="t1", now=NOW,
    )
    assert mid not in result["surfaced"]
    assert mid in result["suppressed_skipped"]

    # 期限切れなら再び浮上できる
    (d / "suppression.json").write_text(
        json.dumps({"suppressed": {mid: {"until": NOW - 1}}}),
        encoding="utf-8",
    )
    result2 = run_surface(
        "予算要求の書式を確認したい", settings=settings,
        session_id="t2", now=NOW,
    )
    assert mid in result2["surfaced"]


def test_reinforce_repromotes_silenced_memory(tmp_path):
    """沈黙(cold)した記憶が実際に役立ったら hot へ復帰する(想起による再固定化)。"""
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r = engine.remember("沈黙からの復帰テスト", "knowledge", 5, now=NOW - 120 * DAY)

    run_sleep(settings, now=NOW, commit=True)
    assert engine.db.get_memory(r["id"])["tier"] == "cold"

    result = engine.reinforce([r["id"]], now=NOW)
    assert r["id"] in result["repromoted"]
    assert engine.db.get_memory(r["id"])["tier"] == "hot"
    post = frontmatter.load(r["path"])
    assert post["tier"] == "hot"
    engine.db.close()


def test_cold_memory_resurfaces_on_strong_cue_only(tmp_path):
    """沈黙(cold)した記憶は、強い手がかり(関連度0.5以上)のときだけ浮上する。"""
    settings = _make_settings(tmp_path)
    engine = _make_engine(settings)
    r = engine.remember(
        "予算要求の書式は財務課の様式7を使うこと", "knowledge", 5,
        now=NOW - 120 * DAY,
    )
    run_sleep(settings, now=NOW, commit=True)
    assert engine.db.get_memory(r["id"])["tier"] == "cold"
    engine.db.close()

    # 弱い手がかり(部分一致): 浮上しない
    weak = run_surface(
        "予算の話をしたいのだけど", settings=settings,
        session_id="c1", now=NOW,
    )
    assert r["id"] not in weak["surfaced"]

    # 強い手がかり(ほぼ同じ語): ふわっと浮上する
    strong = run_surface(
        "予算要求の書式は財務課のどの様式を使うんだったか", settings=settings,
        session_id="c2", now=NOW,
    )
    assert r["id"] in strong["surfaced"]
    cand = next(c for c in strong["candidates"] if c["id"] == r["id"])
    assert cand["tier"] == "cold"
