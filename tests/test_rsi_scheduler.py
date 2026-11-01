"""RSI ループ回転計(rsi)と OS スケジューラ登録内容(scheduler)のテスト。登録は実行しない。"""

from __future__ import annotations

import json
import os

from engram import scheduler
from engram.config import Settings
from engram.rsi import compute, format_report, ledger_adoption

NOW = 1_800_000_000.0


def test_rsi_counts_from_logs_and_state(tmp_path):
    data = tmp_path / "data"
    (data / "sleep").mkdir(parents=True)
    (data / "dmn").mkdir()
    import datetime as dt
    day = dt.datetime.fromtimestamp(NOW - 3600).strftime("%Y-%m-%d %H:%M:%S")
    old = dt.datetime.fromtimestamp(NOW - 30 * 86400).strftime("%Y-%m-%d %H:%M:%S")
    (data / "sleep" / "night_log.txt").write_text(
        f"{old} === sleep night start ===\n{day} === sleep night start ===\n{day} agent OK\n", encoding="utf-8")
    (data / "sleep" / "night_agent_log.md").write_text(
        f"- {day} [distill] 蒸留まとめ: 7件の知見を保存 (model=x)\n", encoding="utf-8")
    (data / "dmn" / "state.json").write_text(json.dumps({"ticks": [
        {"ts": NOW - 100, "fired": True, "result": "wrote", "for_user": True},
        {"ts": NOW - 200, "fired": True, "result": "wrote", "for_user": False},
        {"ts": NOW - 300, "fired": False},
    ]}), encoding="utf-8")
    ledger = tmp_path / "queue.md"
    ledger.write_text("| # | 項目 | 状態 |\n|---|---|---|\n| G1 | a | **配備済み** |\n| G2 | b | 起票 |\n| G3 | c | 完了 |\n",
                      encoding="utf-8")
    s = Settings(memories_dir=tmp_path / "m", data_dir=data, rsi_ledger_path=str(ledger))
    m = compute(s, now=NOW)
    a = m["autonomous"]
    assert a["night_runs"] == 1 and a["night_integration_ok"] == 1 and a["distilled_insights"] == 7
    assert a["dmn"] == {"ticks": 3, "fired": 2, "wrote": 2, "for_user": 1, "for_user_rate": 0.5}
    assert a["loops_total"] == 3
    assert m["ledger"] == {"total": 3, "adopted": 2, "rate": 0.667}
    text = format_report(m)
    assert "66.7%" in text and "DMN日記2件" in text


def test_ledger_missing_is_empty(tmp_path):
    assert ledger_adoption(None, "x") == {"total": 0, "adopted": 0, "rate": None}
    assert ledger_adoption(tmp_path / "none.md", "x")["total"] == 0


def test_schedule_spec_builds_commands_without_running():
    night = scheduler.schedule_spec("night", night_time="03:15")
    dmn = scheduler.schedule_spec("dmn", dmn_minutes=10)
    assert "-m" in " ".join(scheduler.command_for("night")) and "engram" in " ".join(scheduler.command_for("dmn"))
    if os.name == "nt":
        assert night["install"][:4] == ["schtasks", "/Create", "/TN", "EngramSleepNight"]
        assert "T03:15:00" in night["xml"] and "<Interval>PT10M</Interval>" in dmn["xml"]
        # PC が眠っていた夜の分を、起きたあとで実行する(R8.11.1 指摘)
        assert "<StartWhenAvailable>true</StartWhenAvailable>" in night["xml"]
        assert night["uninstall"][-2:] == ["EngramSleepNight", "/F"]
        hyp = scheduler.schedule_spec("hyp")
        assert hyp["install"][3] == "EngramHypothesis" and "hyp-run" in hyp["xml"]
    else:
        assert night["cron_line"].startswith("15 3 * * *") and night["cron_line"].endswith("# engram:night")
        assert dmn["cron_line"].startswith("*/10 * * * *")
