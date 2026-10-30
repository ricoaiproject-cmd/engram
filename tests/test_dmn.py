"""DMN(自発思考)のテスト。LLM は差し替え。観測・起床判定・救済パーサ・書き込みを検証。"""

from __future__ import annotations

import json

from engram.config import Settings
from engram.db import IndexDB
from engram.dmn import GOLDEN, decide, keywords, load_state, parse_thought, run_dmn
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine
from engram.store import MarkdownStore

NOW = 1_800_000_000.0


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
        self.last_user = user
        return self.reply


THOUGHT = ("二つの記憶を並べると、検証ゲートの期待値を固定する方針と、蒸留が毎晩知見を増やす方針の間に、"
           "誰が期待値の鮮度を見るのかという空白があるように見える。これは推測。")


def test_parse_thought_json_and_broken_fallback():
    good = json.dumps({"thought": THOUGHT, "question": "期待値の鮮度は誰が見る？", "for_user": True,
                       "used_ids": ["01ABCDEFGHJKMNPQRSTVWXYZ01", "bad"]}, ensure_ascii=False)
    d = parse_thought("```json\n" + good + "\n```")
    assert d["thought"] == THOUGHT and d["for_user"] is True
    assert d["used_ids"] == ["01ABCDEFGHJKMNPQRSTVWXYZ01"]
    # 途中で切れた出力でも thought を救済する
    broken = '{\n  "thought": "' + THOUGHT + '",\n  "question": "問い',
    d2 = parse_thought(broken[0])
    assert d2 is not None and d2["thought"] == THOUGHT and d2["for_user"] is False
    assert parse_thought("ただの文章") is None


def test_keywords_without_llm():
    q = keywords("openpyxl は数式を再計算しない。LibreOffice で開き直す。openpyxl 再確認")
    words = q.split()
    assert words[0] == "openpyxl" and "LibreOffice" in words
    assert words.count("openpyxl") == 1  # 重複しない
    assert len(keywords("a " * 100 + " ".join(f"word{i}" for i in range(30))).split()) <= 12


def test_decide_reason_phase_gap_and_cap(tmp_path):
    s = _settings(tmp_path, dmn_p_fire=0.25, dmn_min_gap_min=20, dmn_max_per_day=2)
    st = {"phase": 0.0, "last_fire": 0.0, "observed": {}, "ticks": []}
    fire, why = decide(s, st, ["何か変わった"], now=NOW, force=False)
    assert fire and why == "理由あり"
    assert abs(st["phase"] - GOLDEN) < 1e-9
    # 理由なし: 位相 0.618→0.236 で起きる、次 0.854 で静か
    fire2, why2 = decide(s, st, [], now=NOW, force=False)
    assert fire2 and why2 == "位相"
    fire3, why3 = decide(s, st, [], now=NOW, force=False)
    assert not fire3 and why3 == "静か"
    # 最短間隔
    st["last_fire"] = NOW - 60
    assert decide(s, st, ["x"], now=NOW, force=False) == (False, "最短間隔内")
    # 1日の上限
    st["last_fire"] = 0.0
    st["ticks"] = [{"ts": NOW - 100, "fired": True}, {"ts": NOW - 200, "fired": True}]
    assert decide(s, st, ["x"], now=NOW, force=False) == (False, "本日の上限")
    # force は全部を飛ばす
    assert decide(s, st, [], now=NOW, force=True) == (True, "手動(--force)")


def test_run_dmn_quiet_tick_does_not_touch_llm_or_engine(tmp_path):
    s = _settings(tmp_path, dmn_p_fire=0.0)
    llm = FakeLLM("")

    class Boom:
        def __getattr__(self, name):
            raise AssertionError("engine must not be built on a quiet tick")

    r = run_dmn(s, engine=Boom(), llm=llm, now=NOW)
    assert r["fired"] is False and llm.calls == 0
    st = load_state(s)
    assert len(st["ticks"]) == 1 and st["observed"] != {}


def test_run_dmn_wakes_on_watch_change_and_writes_diary_and_episode(tmp_path):
    watched = tmp_path / "reports"
    watched.mkdir()
    (watched / "a.md").write_text("first", encoding="utf-8")
    s = _settings(tmp_path, dmn_watch=[str(watched)], dmn_p_fire=0.0, dmn_new_memory_threshold=99)
    eng = _engine(s)
    seed = eng.remember("検証ゲートの期待値はユーザー裁可なしに動かさない", "knowledge", 7, now=NOW - 5000)
    reply = json.dumps({"thought": THOUGHT, "question": "期待値の鮮度は誰が見る？", "for_user": True,
                        "used_ids": [seed["id"]]}, ensure_ascii=False)
    llm = FakeLLM(reply)
    # 1ティック目: 署名を覚えるだけ(初回は差分なし)
    r0 = run_dmn(s, engine=eng, llm=llm, now=NOW)
    assert r0["fired"] is False
    # 監視先が更新される → 理由あり
    (watched / "b.md").write_text("new report", encoding="utf-8")
    r1 = run_dmn(s, engine=eng, llm=llm, now=NOW + 3600)
    assert r1["fired"] and r1["why"] == "理由あり" and any("reports" in x for x in r1["reasons"])
    assert r1["result"] == "wrote" and r1["memory_id"]
    assert "【世界の変化】" in llm.last_user and "【種の記憶" in llm.last_user
    diary = (s.data_dir / "dmn" / "diary").glob("*.md")
    text = next(diary).read_text(encoding="utf-8")
    assert THOUGHT in text and "朝、ユーザーに伝える価値あり" in text
    mems = {m["id"]: m for m in eng.db.all_memories()}
    new = mems[r1["memory_id"]]
    assert new["type"] == "episode" and new["importance"] == s.dmn_importance
    # DMN が読んだ記憶は温めない(reinforce イベントが無い)
    kinds = [row[0] for row in eng.db._conn.execute(
        "SELECT kind FROM access_events WHERE memory_id=?", (seed["id"],))]
    assert "reinforce" not in kinds
    eng.db.close()


def test_run_dmn_llm_down_exits_2_and_dry_run_writes_nothing(tmp_path):
    s = _settings(tmp_path)
    eng = _engine(s)
    r = run_dmn(s, engine=eng, llm=FakeLLM("", up=False), force=True, now=NOW)
    assert r["exit"] == 2 and r["fired"] is False
    reply = json.dumps({"thought": THOUGHT, "question": "", "for_user": False, "used_ids": []}, ensure_ascii=False)
    r2 = run_dmn(s, engine=eng, llm=FakeLLM(reply), force=True, dry_run=True, now=NOW + 5000)
    assert r2["result"] == "dry-run" and r2["thought"] == THOUGHT
    assert not (s.data_dir / "dmn" / "diary").exists()
    assert all(m["type"] != "episode" for m in eng.db.all_memories())
    eng.db.close()
