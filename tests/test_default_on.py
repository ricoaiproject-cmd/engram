"""新しい機能を既定でオンにする仕組み(0.25.3)のテスト。本物のスケジューラは触らない。"""

from __future__ import annotations

import os
from pathlib import Path

from engram import media as media_mod
from engram import night as night_mod
from engram import scheduler
from engram import setup as setup_mod
from engram.config import Settings


class FakeScheduler:
    def __init__(self):
        self.calls = []

    def install(self, kind, **kw):
        self.calls.append((kind, kw))
        return True, "ok"


def _no_llm(monkeypatch):
    monkeypatch.setattr(setup_mod.LocalLLMProbe, "available", staticmethod(lambda s: False))
    monkeypatch.setattr(night_mod, "available_agents", lambda s: [("claude -p", "claude (見つからない)")])


def test_spec_carries_home_and_suffix(tmp_path):
    spec = scheduler.schedule_spec("night", home=tmp_path / "h", suffix="_Test")
    joined = " ".join(spec.get("install", [spec.get("cron_line", "")]))
    assert "--home" in joined and str(tmp_path / "h") in joined
    if os.name == "nt":
        assert spec["install"][3] == "EngramSleepNight_Test"
        assert spec["uninstall"][3] == "EngramSleepNight_Test"
    else:
        assert spec["marker"] == "# engram:night_Test"


def test_set_config_scalar_keeps_arrays_tables_and_comments(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("# 手書きのコメント\nmemories_dir = 'X'\ndmn_watch = [\"a\", \"b\"]\n\n"
                   "[room_paths]\n'D:/w' = 'work'\n", encoding="utf-8")
    setup_mod.set_config_scalar(cfg, "schedule_night", False)
    setup_mod.set_config_scalar(cfg, "memories_dir", "Y")
    text = cfg.read_text(encoding="utf-8")
    assert "# 手書きのコメント" in text and 'dmn_watch = ["a", "b"]' in text
    data = setup_mod.read_config_toml(cfg)
    assert data["schedule_night"] is False and data["memories_dir"] == "Y"
    assert data["room_paths"] == {"D:/w": "work"} and data["dmn_watch"] == ["a", "b"]


def test_schedule_step_registers_both_by_default(tmp_path, monkeypatch):
    _no_llm(monkeypatch)
    cfg, fake, results = tmp_path / "config.toml", FakeScheduler(), []
    notices = setup_mod.schedule_step(cfg, tmp_path, schedule=True, results=results, scheduler_mod=fake)
    assert [c[0] for c in fake.calls] == ["night", "dmn"]
    assert all(c[1]["home"] == tmp_path for c in fake.calls)
    # 止めてある・使えない機能は画面に出す(黙ってオフにしない)
    assert any("媒体の棚" in n and "engram media setup" in n for n in notices)
    assert any("Ollama" in n for n in notices)


def test_schedule_step_respects_users_stop(tmp_path, monkeypatch):
    _no_llm(monkeypatch)
    cfg, fake = tmp_path / "config.toml", FakeScheduler()
    cfg.write_text("schedule_dmn = false\n", encoding="utf-8")
    notices = setup_mod.schedule_step(cfg, tmp_path, schedule=True, results=[], scheduler_mod=fake)
    assert [c[0] for c in fake.calls] == ["night"]
    assert any("engram dmn --install" in n for n in notices)


def test_schedule_step_no_schedule_records_intent(tmp_path, monkeypatch):
    _no_llm(monkeypatch)
    cfg, fake = tmp_path / "config.toml", FakeScheduler()
    setup_mod.schedule_step(cfg, tmp_path, schedule=False, results=[], scheduler_mod=fake)
    assert fake.calls == []
    data = setup_mod.read_config_toml(cfg)
    assert data["schedule_night"] is False and data["schedule_dmn"] is False


def test_schedule_step_none_touches_nothing(tmp_path):
    fake = FakeScheduler()
    assert setup_mod.schedule_step(tmp_path / "c.toml", tmp_path, schedule=None,
                                   results=[], scheduler_mod=fake) == []
    assert fake.calls == [] and not (tmp_path / "c.toml").exists()


def test_night_skips_missing_cli_and_api_key(monkeypatch):
    s = Settings()
    assert "見つからない" in night_mod.agent_unusable_reason("engram-no-such-cli-xyz -p", s)
    monkeypatch.setattr(night_mod.shutil, "which", lambda exe: "C:/bin/" + exe)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    ok = Settings(night_allow_claude=True)
    assert "ANTHROPIC_API_KEY" in night_mod.agent_unusable_reason("claude -p", ok)
    assert night_mod.agent_unusable_reason(
        "claude -p", Settings(night_allow_claude=True, night_agents_allow_api_key=True)) is None
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert night_mod.agent_unusable_reason("claude -p", ok) is None


def test_media_enable_sets_flag_roots_and_plugin(tmp_path, monkeypatch):
    shots = tmp_path / "Pictures" / "Screenshots"
    shots.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    cfg = tmp_path / "config.toml"
    cfg.write_text("memories_dir = 'M'\n\n[room_paths]\n'D:/w' = 'work'\n", encoding="utf-8")
    s = Settings(data_dir=tmp_path / "data", memories_dir=tmp_path / "m")
    out = media_mod.enable(s, config_file=cfg)
    data = setup_mod.read_config_toml(cfg)
    assert data["media_enabled"] is True
    assert data["media_roots"] == {str(shots): "common"}
    assert data["room_paths"] == {"D:/w": "work"} and data["memories_dir"] == "M"
    assert Path(out["night_plugin"]).is_file()


def test_owner_protects_other_homes_and_custom_jobs(tmp_path, monkeypatch):
    """別のホームや利用者自前の仕組みの登録は、上書きも削除もしない(R8.11.1 の事故の再発防止)。"""
    mine, theirs = tmp_path / "mine", tmp_path / "theirs"
    regs = {"night": f"pythonw.exe -m engram night --home {theirs}",
            "dmn": "powershell.exe -File C:/tools/my_own_job.ps1"}
    monkeypatch.setattr(scheduler, "registered_command", lambda kind, suffix="": regs.get(kind))
    ran = []
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **k: ran.append(a) or None)
    assert scheduler.owner("night", mine) == "other" and scheduler.owner("dmn", mine) == "other"
    assert scheduler.owner("night", theirs) == "ours"
    ok, msg = scheduler.uninstall("night", home=mine)
    assert not ok and "外しません" in msg
    ok, msg = scheduler.install("dmn", home=mine)
    assert not ok and "上書きしません" in msg
    assert ran == []  # schtasks / crontab は一度も呼ばれていない
    regs.clear()
    assert scheduler.uninstall("night", home=mine) == (True, "登録はありません(EngramSleepNight)")


def test_schedule_step_leaves_custom_job_alone(tmp_path, monkeypatch):
    _no_llm(monkeypatch)

    class Sched(FakeScheduler):
        task_name = staticmethod(scheduler.task_name)

        def owner(self, kind, home, suffix=""):
            return "other" if kind == "night" else "none"

    fake = Sched()
    notices = setup_mod.schedule_step(tmp_path / "c.toml", tmp_path, schedule=True, results=[], scheduler_mod=fake)
    assert [c[0] for c in fake.calls] == ["dmn"]
    assert any("上書きしていません" in n for n in notices)
