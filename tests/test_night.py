"""夜間の眠り一式(night)のテスト。外部 CLI・LLM は差し替え(runner / FakeLLM)で動かす。"""

from __future__ import annotations

import json

from engram.config import Settings
from engram.db import IndexDB
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine
from engram.night import (
    distill,
    ensure_night_files,
    format_night,
    parse_insights,
    read_agent_lines,
    run_integration,
    run_night,
    run_plugins,
)
from engram.store import MarkdownStore

NOW = 1_800_000_000.0
DAY = 86400.0


def _settings(tmp_path, **kw) -> Settings:
    return Settings(memories_dir=tmp_path / "memories", data_dir=tmp_path / "data", **kw)


def _engine(settings) -> MemoryEngine:
    emb = FakeEmbedder(dim=64)
    return MemoryEngine(settings=settings, store=MarkdownStore(settings.memories_dir),
                        db=IndexDB(settings.db_path, emb.dim), embedder=emb)


class FakeLLM:
    model = "fake"
    url = "http://fake"

    def __init__(self, reply, up=True):
        self.reply, self.up, self.calls = reply, up, 0

    def available(self, timeout: int = 10) -> bool:
        return self.up

    def chat(self, system, user, **kw):
        self.calls += 1
        return self.reply(user) if callable(self.reply) else self.reply


def test_ensure_night_files_creates_templates_once(tmp_path):
    s = _settings(tmp_path)
    agents, prompt = ensure_night_files(s)
    assert agents.is_file() and prompt.is_file()
    lines = read_agent_lines(agents)
    assert "claude -p" in lines and "local-llm" in lines
    agents.write_text("my-agent --x\n", encoding="utf-8")
    ensure_night_files(s)  # 上書きしない
    assert read_agent_lines(agents) == ["my-agent --x"]


def test_integration_tries_agents_in_order_and_stops_at_first_success(tmp_path):
    s = _settings(tmp_path)
    (s.data_dir).mkdir(parents=True)
    (s.data_dir / "sleep_agents.txt").write_text("first --a\nsecond\nthird\n", encoding="utf-8")
    calls = []

    def runner(argv):
        calls.append(argv[0])
        return 1 if argv[0] == "first" else 0

    eng = _engine(s)
    r = run_integration(eng, s, dry_run=False, llm=FakeLLM("", up=False), runner=runner)
    assert calls == ["first", "second"]
    assert r["done_by"] == "second"
    log = (s.data_dir / "sleep" / "night_log.txt").read_text(encoding="utf-8")
    assert "agent exit=1" in log and "agent OK" in log
    # プロンプト文が最後の引数として付いている
    eng.db.close()


def test_integration_all_fail_falls_back_to_nudge(tmp_path):
    s = _settings(tmp_path)
    s.data_dir.mkdir(parents=True)
    (s.data_dir / "sleep_agents.txt").write_text("a\nlocal-llm\n", encoding="utf-8")
    eng = _engine(s)
    r = run_integration(eng, s, dry_run=False, llm=FakeLLM("", up=False), runner=lambda argv: 7)
    assert r["done_by"] is None
    log = (s.data_dir / "sleep" / "night_log.txt").read_text(encoding="utf-8")
    assert "nudge fallback" in log and "agent skip: local-llm" in log
    eng.db.close()


def test_local_llm_consolidates_old_episode_cluster(tmp_path):
    s = _settings(tmp_path, consolidate_min_age_days=1)
    s.data_dir.mkdir(parents=True)
    (s.data_dir / "sleep_agents.txt").write_text("local-llm\n", encoding="utf-8")
    eng = _engine(s)
    old = NOW - 30 * DAY
    a = eng.remember("動画公開作業の記録 その一 手順は同じ", "episode", 4, now=old)
    b = eng.remember("動画公開作業の記録 その二 サムネ差し替えの手順", "episode", 4, now=old)
    # クラスタ判定(埋め込み類似)はここでは対象外なので、候補を固定して統合の機構だけを検証する
    eng.consolidation_candidates = lambda **kw: {"clusters": [{"ids": [a["id"], b["id"]],
                                                              "contents": ["記録1", "記録2"]}]}
    eng.skill_candidates = lambda **kw: {"clusters": []}
    llm = FakeLLM("これは統合要約です。" * 20)
    r = run_integration(eng, s, dry_run=False, llm=llm)
    assert r["done_by"] == "local-llm"
    assert llm.calls == 1
    mems = {m["id"]: m for m in eng.db.all_memories()}
    assert mems[a["id"]]["tier"] == "cold" and mems[b["id"]]["tier"] == "cold"
    assert any(m["type"] == "knowledge" for m in mems.values())
    assert "統合完了" in (s.data_dir / "sleep" / "night_agent_log.md").read_text(encoding="utf-8")
    eng.db.close()


def test_parse_insights_filters_and_clamps():
    text = ('前置き ```json [{"insight": "openpyxl は数式を再計算しない。値が要るなら LibreOffice で開き直す",'
            ' "kind": "knowledge", "importance": 9, "cues": ["openpyxl", "数式", "再計算"]},'
            ' {"insight": "短い", "kind": "x"}, {"insight": "ユーザーは完了報告を結果→変更点→検証結果の順で読む流儀を持つ",'
            ' "kind": "preference", "importance": 2}] ```')
    out = parse_insights(text)
    assert len(out) == 2
    assert out[0]["importance"] == 6 and out[0]["cues"] == ["openpyxl", "数式", "再計算"]
    assert out[1]["kind"] == "preference" and out[1]["importance"] == 4
    assert parse_insights("[]") == [] and parse_insights("何も") == []


def test_distill_saves_insights_and_skips_done(tmp_path):
    s = _settings(tmp_path)
    eng = _engine(s)
    ep = eng.remember("セッション記録: openpyxl の数式が空になって困った。LibreOffice で解決した。",
                      "episode", 4, now=NOW - 3600)
    reply = json.dumps([{"insight": "openpyxl は数式を再計算しない。値が要るなら LibreOffice 経由で開き直す",
                         "kind": "knowledge", "importance": 5, "cues": ["openpyxl", "数式が空"]}],
                       ensure_ascii=False)
    llm = FakeLLM(reply)
    r = distill(eng, s, llm, now=NOW)
    assert r["episodes"] == 1 and r["saved"] == 1
    kinds = [m["type"] for m in eng.db.all_memories()]
    assert "knowledge" in kinds
    # 手がかり語が本文に入る(符号化特異性)
    md = list((s.memories_dir).rglob("*.md"))
    assert any("手がかり: openpyxl / 数式が空" in p.read_text(encoding="utf-8") for p in md)
    # 2回目は処理済みでスキップ
    r2 = distill(eng, s, llm, now=NOW)
    assert r2["episodes"] == 0 and llm.calls == 1
    # LLM 不達なら何もしないで llm_down
    eng.remember("別のセッション記録", "episode", 4, now=NOW - 1800)
    r3 = distill(eng, s, FakeLLM("", up=False), now=NOW)
    assert r3["llm_down"] is True and r3["saved"] == 0
    eng.db.close()


def test_plugins_run_from_night_d(tmp_path):
    s = _settings(tmp_path)
    d = s.data_dir / "night.d"
    d.mkdir(parents=True)
    (d / "10_ok.py").write_text("import os,sys; assert os.environ['ENGRAM_HOME']; sys.exit(0)", encoding="utf-8")
    (d / "20_fail.py").write_text("import sys; sys.exit(4)", encoding="utf-8")
    out = run_plugins(s, dry_run=False)
    assert [(p["plugin"], p["rc"]) for p in out] == [("10_ok.py", 0), ("20_fail.py", 4)]
    assert all(p["rc"] is None for p in run_plugins(s, dry_run=True))


def test_run_night_dry_run_writes_nothing(tmp_path):
    s = _settings(tmp_path)
    eng = _engine(s)
    eng.remember("何かの記憶", "knowledge", 5, now=NOW - 200 * DAY)
    r = run_night(s, engine=eng, llm=FakeLLM("", up=False), dry_run=True, now=NOW)
    assert r["dry_run"] is True
    assert not (s.data_dir / "sleep" / "night_log.txt").exists()
    text = format_night(r)
    assert "dry-run" in text and "統合" in text
    eng.db.close()


def test_run_night_commit_logs_full_sequence(tmp_path):
    s = _settings(tmp_path)
    s.data_dir.mkdir(parents=True)
    (s.data_dir / "sleep_agents.txt").write_text("agent-x\n", encoding="utf-8")
    eng = _engine(s)
    r = run_night(s, engine=eng, llm=FakeLLM("", up=False), dry_run=False,
                  runner=lambda argv: 0, now=NOW)
    log = (s.data_dir / "sleep" / "night_log.txt").read_text(encoding="utf-8")
    for needle in ("=== sleep night start ===", "agent OK", "sleep --commit OK", "distill", "=== sleep night end ==="):
        assert needle in log
    assert r["sleep"]["report_path"]
    eng.db.close()
