"""Claude Code を入れていない人(Codex だけ・Gemini だけ)にも同じように動くか(0.25.5)のテスト。"""

from __future__ import annotations

import json
from pathlib import Path

from engram import hyp_run, scheduler
from engram import setup as setup_mod
from engram.config import Settings
from engram.night import resolve_skills_dir


def test_codex_hooks_use_same_shape_with_explicit_session_end_timeout(tmp_path):
    path = tmp_path / ".codex" / "hooks.json"
    ok, msg = setup_mod.register_codex_hooks(path, Path("C:/bin/engram.exe"))
    data = json.loads(path.read_text(encoding="utf-8"))
    ups = data["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    end = data["hooks"]["SessionEnd"][0]["hooks"][0]
    assert ok and ups["type"] == "command" and "hook user-prompt" in ups["command"]
    # Codex は SessionEnd を3秒で打ち切るので、要約は裏のプロセスに任せてすぐ返す
    assert "--detach" in end["command"] and end["timeout"] == 3
    assert setup_mod.register_codex_hooks(path, Path("C:/bin/engram.exe"))[1] == "既登録(スキップ)"


def test_gemini_hooks_use_before_agent_in_ms_and_keep_other_settings(tmp_path):
    path = tmp_path / ".gemini" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"theme": "Dark", "mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")
    ok, _ = setup_mod.register_gemini_hooks(path, Path("C:/bin/engram.exe"))
    ok2, _ = setup_mod.register_gemini_mcp(path, Path("C:/bin/engram-mcp.exe"))
    data = json.loads(path.read_text(encoding="utf-8"))
    h = data["hooks"]["BeforeAgent"][0]["hooks"][0]
    assert ok and ok2 and h["name"] == "engram" and h["timeout"] == 15_000
    assert "--event-name BeforeAgent" in h["command"]
    assert data["theme"] == "Dark" and set(data["mcpServers"]) == {"other", "engram"}
    # Gemini CLI は tools.enableHooks が true でないとフックを動かさない
    assert data["tools"]["enableHooks"] is True


def test_user_prompt_hook_reports_requested_event_name(monkeypatch, capsys):
    from engram import hooks

    monkeypatch.setattr(hooks, "_consolidation_nudge", lambda s: "整理してください")
    monkeypatch.setattr(hooks, "_skill_nudge", lambda s: None)
    hooks.run_user_prompt(json.dumps({"session_id": "s", "prompt": "こんにちは", "cwd": "."}),
                          event_name="BeforeAgent")
    out = capsys.readouterr().out.strip()
    if out:  # 差し込むものがあるときだけ JSON を1つ出す(Gemini は JSON 以外を混ぜられない)
        assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "BeforeAgent"


def test_mcp_recall_carries_nudge_once(tmp_path, monkeypatch):
    from engram import hooks, server
    from engram.embedder import FakeEmbedder
    from engram.engine import build_engine

    s = Settings(data_dir=tmp_path / "d", memories_dir=tmp_path / "m")
    eng = build_engine(s, embedder=FakeEmbedder())
    monkeypatch.setattr(server, "_get_engine", lambda: eng)
    monkeypatch.setattr(server, "_nudge_checked", False)
    monkeypatch.setattr(hooks, "_consolidation_nudge", lambda st: "(engram 記憶整理) 整理してください")
    monkeypatch.setattr(hooks, "_skill_nudge", lambda st: None)
    first = server.recall("何か")
    second = server.recall("何か")
    assert "記憶整理" in first.get("notice", "") and "notice" not in second
    assert "recall" in server._INSTRUCTIONS and "remember" in server._INSTRUCTIONS
    eng.db.close()


class Sched:
    def __init__(self):
        self.calls = []

    def owner(self, kind, home, suffix=""):
        return "none"

    def install(self, kind, **kw):
        self.calls.append(kind)
        return True, "ok"


def test_hyp_step_asks_and_registers_only_on_yes(tmp_path):
    cfg = tmp_path / "c.toml"
    which = lambda name: f"C:/bin/{name}.cmd" if name == "codex" else None  # noqa: E731
    asked = []

    def ask(q):
        asked.append(q)
        assert "Web" in q and "OpenAI" in q and "[y/N]" in q
        return "y"

    sched = Sched()
    setup_mod.hyp_step(cfg, tmp_path, interactive=True, ask=ask, sched=sched, suffix="", results=[], which=which)
    assert sched.calls == ["hyp"] and setup_mod.read_config_toml(cfg)["hyp_agent"] == "codex"
    # 2回目は聞かない
    setup_mod.hyp_step(cfg, tmp_path, interactive=True, ask=lambda q: 1 / 0, sched=sched, suffix="",
                       results=[], which=which)
    assert len(asked) == 1


def test_hyp_step_non_interactive_and_no_answer_do_nothing(tmp_path):
    cfg = tmp_path / "c.toml"
    which = lambda name: f"C:/bin/{name}.cmd" if name == "claude" else None  # noqa: E731
    sched = Sched()
    notices = setup_mod.hyp_step(cfg, tmp_path, interactive=False, ask=input, sched=sched, suffix="",
                                 results=[], which=which)
    assert sched.calls == [] and any("engram hyp-run --use claude" in n for n in notices)
    setup_mod.hyp_step(cfg, tmp_path, interactive=True, ask=lambda q: "", sched=sched, suffix="",
                       results=[], which=which)
    assert sched.calls == [] and setup_mod.read_config_toml(cfg)["hyp_agent"] == "none"


def test_hyp_run_uses_only_the_chosen_cli(tmp_path):
    calls = []
    s = Settings(data_dir=tmp_path / "d", memories_dir=tmp_path / "m")
    assert hyp_run.run(s, runner=lambda a: calls.append(a) or 0)["ran"] is False and calls == []
    s2 = Settings(data_dir=tmp_path / "d", memories_dir=tmp_path / "m", hyp_agent="codex")
    r = hyp_run.run(s2, runner=lambda a: calls.append(a) or 0)
    assert r["ran"] and calls[0][:2] == ["codex", "exec"] and "hypothesis_seeds" in calls[0][-1]


def test_media_step_asks_once(tmp_path):
    cfg = tmp_path / "c.toml"
    notices = setup_mod.media_step(cfg, interactive=True, ask=lambda q: "n", installer=lambda: 1 / 0)
    assert setup_mod.read_config_toml(cfg)["media_enabled"] is False and notices
    # 「いいえ」と答えた人には次から聞かない
    setup_mod.media_step(cfg, interactive=True, ask=lambda q: 1 / 0, installer=lambda: 1 / 0)
    cfg2 = tmp_path / "c2.toml"
    assert setup_mod.media_step(cfg2, interactive=True, ask=lambda q: "y",
                                installer=lambda: {"ok": True}) == []
    assert setup_mod.media_step(tmp_path / "c3.toml", interactive=False, ask=lambda q: 1 / 0)


def test_skills_dir_follows_installed_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    s = Settings(data_dir=tmp_path / "d", memories_dir=tmp_path / "m")
    assert resolve_skills_dir(s) == tmp_path / "d" / "skills"
    (tmp_path / ".codex").mkdir()
    assert resolve_skills_dir(s) == tmp_path / ".codex" / "skills"
    (tmp_path / ".claude").mkdir()
    assert resolve_skills_dir(s) == tmp_path / ".claude" / "skills"


def test_task_xml_escapes_and_runs_missed_jobs():
    xml = scheduler.task_xml(["C:/a b/pythonw.exe", "-m", "engram", "night", "--home", "C:/x&y"],
                             daily_at="02:30", every_minutes=None)
    assert "<StartWhenAvailable>true</StartWhenAvailable>" in xml and "C:/x&amp;y" in xml


def test_codex_transcript_is_read_like_claude(tmp_path):
    """Codex の会話記録(response_item/payload)も、会話の終わりの自動記録の材料として読める。"""
    from engram.transcript import extract_messages

    rows = [
        {"type": "session_meta", "payload": {"cwd": "C:/work/proj"}},
        {"type": "response_item", "payload": {"type": "message", "role": "developer",
                                              "content": [{"type": "input_text", "text": "system"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
                                              "content": [{"type": "input_text", "text": "# AGENTS.md instructions for x"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
                                              "content": [{"type": "input_text", "text": "ログの集計を直して"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                              "content": [{"type": "output_text", "text": "集計を直しました"}]}},
        {"type": "event_msg", "payload": {"type": "token_count"}},
    ]
    p = tmp_path / "rollout.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    m = extract_messages(p)
    assert m["user_texts"] == ["ログの集計を直して"]
    assert m["last_assistant"] == "集計を直しました" and m["cwd"] == "C:/work/proj"


def test_antigravity_stop_records_only_when_fully_idle():
    from engram.hooks import normalize_stop_input

    base = {"conversationId": "c1", "transcriptPath": "C:/t.jsonl", "workspacePaths": ["C:/w"]}
    assert normalize_stop_input({**base, "fullyIdle": False}, "antigravity") is None
    p = normalize_stop_input({**base, "fullyIdle": True}, "antigravity")
    assert p == {"session_id": "c1", "transcript_path": "C:/t.jsonl", "cwd": "C:/w", "incremental": True}


def test_antigravity_transcript_is_read(tmp_path):
    from engram.transcript import extract_messages

    rows = [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>\n報告書を直して\n</USER_REQUEST>\n<ADDITIONAL_METADATA>x</ADDITIONAL_METADATA>"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "thinking": "...", "tool_calls": []},
        {"step_index": 2, "type": "GENERIC", "content": "tool output"},
        {"step_index": 3, "type": "PLANNER_RESPONSE", "content": "直しました"},
    ]
    p = tmp_path / "transcript_full.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    m = extract_messages(p)
    assert m["user_texts"] == ["報告書を直して"] and m["last_assistant"] == "直しました"


def test_incremental_session_end_records_only_new_turns(tmp_path, monkeypatch):
    from engram import hooks

    monkeypatch.setattr(hooks, "get_settings", lambda: Settings(data_dir=tmp_path / "d", memories_dir=tmp_path / "m"))
    seen = []
    monkeypatch.setattr("engram.transcript.build_episodes",
                        lambda messages, **kw: seen.append(list(messages["user_texts"])) or [])
    t = tmp_path / "t.jsonl"

    def write(n):
        rows = [{"type": "USER_INPUT", "content": f"<USER_REQUEST>発言{i}</USER_REQUEST>"} for i in range(n)]
        t.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    payload = {"session_id": "c1", "transcript_path": str(t), "cwd": "", "incremental": True}
    write(2)
    hooks.run_session_end(json.dumps(payload))
    write(3)
    hooks.run_session_end(json.dumps(payload))
    hooks.run_session_end(json.dumps(payload))  # 増えていなければ何もしない
    assert seen == [["発言0", "発言1"], ["発言2"]]


def test_antigravity_hooks_keep_users_named_entries(tmp_path):
    path = tmp_path / "config" / "hooks.json"
    path.parent.mkdir()
    mine = {"self-verify-gate": {"enabled": False, "Stop": [{"type": "command", "command": "py gate.py stop", "timeout": 20}]}}
    path.write_text(json.dumps(mine), encoding="utf-8")
    ok, _ = setup_mod.register_antigravity_hooks(path, Path("C:/bin/engram.exe"))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert ok and data["self-verify-gate"] == mine["self-verify-gate"]
    assert data["engram"]["enabled"] is True
    assert data["engram"]["Stop"][0]["command"].startswith("C:") and "hook stop --agent antigravity" in data["engram"]["Stop"][0]["command"]
    assert setup_mod.register_antigravity_hooks(path, Path("C:/bin/engram.exe"))[1] == "既登録(スキップ)"
    assert not setup_mod.register_antigravity_hooks(path, Path("C:/Program Files/engram.exe"))[0]
    setup_mod.remove_engram_hooks(path)
    assert json.loads(path.read_text(encoding="utf-8")) == mine


def test_remove_hooks_keeps_users_own_hooks(tmp_path):
    path = tmp_path / ".codex" / "hooks.json"
    path.parent.mkdir()
    own = {"type": "command", "command": "python context_bridge.py prompt", "timeout": 10}
    path.write_text(json.dumps({"hooks": {"UserPromptSubmit": [{"hooks": [own]}],
                                          "Stop": [{"hooks": [own]}]}}), encoding="utf-8")
    setup_mod.register_codex_hooks(path, Path("C:/bin/engram.exe"))
    setup_mod.register_codex_hooks(path, Path("C:/bin/engram.exe"))  # 二重に入らない
    data = json.loads(path.read_text(encoding="utf-8"))
    cmds = [h["command"] for e in data["hooks"]["UserPromptSubmit"] for h in e["hooks"]]
    assert cmds.count(own["command"]) == 1 and sum("engram" in c for c in cmds) == 1
    ok, msg = setup_mod.remove_engram_hooks(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert ok and "2件" in msg
    assert data == {"hooks": {"UserPromptSubmit": [{"hooks": [own]}], "Stop": [{"hooks": [own]}]}}
