"""フック(①自動符号化・②自発的想起)とフック登録のテスト。"""

from __future__ import annotations

import json

import pytest

from engram.config import get_settings
from engram.db import IndexDB
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine
from engram.hooks import run_session_end, run_user_prompt
from engram.setup import (
    merge_config_toml,
    read_config_toml,
    register_claude_hooks,
    write_config_toml,
)
from engram.store import MarkdownStore


@pytest.fixture
def engram_home(tmp_path, monkeypatch):
    """ENGRAM_HOME を一時ディレクトリに向ける(設定・DB・記憶を隔離)。"""
    home = tmp_path / "engram_home"
    home.mkdir()
    monkeypatch.setenv("ENGRAM_HOME", str(home))
    monkeypatch.delenv("ENGRAM_MEMORIES_DIR", raising=False)
    monkeypatch.delenv("ENGRAM_DATA_DIR", raising=False)
    return home


def _fake_build_engine(settings=None, *, embedder=None):
    settings = settings or get_settings()
    embedder = embedder or FakeEmbedder(dim=64)
    store = MarkdownStore(settings.memories_dir)
    db = IndexDB(settings.db_path, embedder.dim)
    return MemoryEngine(settings=settings, store=store, db=db,
                        embedder=embedder)


def _write_transcript(path):
    objs = [
        {"type": "summary", "summary": "計画書の作成"},
        {"type": "user", "cwd": "C:/proj/demo",
         "message": {"role": "user",
                     "content": "来年度の委員会計画書のたたき台を作ってほしい"}},
        {"type": "assistant",
         "message": {"role": "assistant",
                     "content": [{"type": "text",
                                  "text": "完成させました。"}]}},
    ]
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# ① session-end
# ---------------------------------------------------------------------------

def test_session_end_creates_episode(engram_home, tmp_path, monkeypatch):
    import engram.engine
    monkeypatch.setattr(engram.engine, "build_engine", _fake_build_engine)

    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript)
    stdin = json.dumps({
        "session_id": "sess-1",
        "transcript_path": str(transcript),
        "cwd": "C:/proj/demo",
    })

    assert run_session_end(stdin) == 0

    settings = get_settings()
    db = IndexDB(settings.db_path, 64)
    episodes = db.all_memories(types=["episode"])
    assert len(episodes) == 1
    db.close()

    # 符号化済みとして記録される
    marks = (engram_home / "encoded_sessions.txt").read_text(encoding="utf-8")
    assert "sess-1" in marks

    # 同じセッションを二度符号化しない
    assert run_session_end(stdin) == 0
    db = IndexDB(settings.db_path, 64)
    assert len(db.all_memories(types=["episode"])) == 1
    db.close()


def test_session_end_skips_trivial_transcript(engram_home, tmp_path,
                                              monkeypatch):
    import engram.engine
    monkeypatch.setattr(engram.engine, "build_engine", _fake_build_engine)

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({"type": "user",
                    "message": {"role": "user", "content": "ok"}},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stdin = json.dumps({"session_id": "sess-2",
                        "transcript_path": str(transcript), "cwd": ""})
    assert run_session_end(stdin) == 0
    settings = get_settings()
    assert not settings.db_path.is_file() or not IndexDB(
        settings.db_path, 64
    ).all_memories(types=["episode"])


def test_session_end_never_raises_on_garbage(engram_home):
    assert run_session_end("not json at all") == 0
    assert run_session_end("{}") == 0


# ---------------------------------------------------------------------------
# ② user-prompt
# ---------------------------------------------------------------------------

def test_user_prompt_shadow_logs(engram_home, monkeypatch):
    # 記憶を1件用意
    engine = _fake_build_engine()
    engine.remember("予算要求の書式は財務課の様式7を使うこと", "knowledge", 7)
    engine.db.close()

    stdin = json.dumps({
        "session_id": "sess-3",
        "prompt": "予算要求の書式ってどうだったっけ",
        "cwd": "C:/anywhere",
    })
    assert run_user_prompt(stdin) == 0

    log = engram_home / "surface" / "surface_log.jsonl"
    assert log.is_file()
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["mode"] == "shadow"


def test_user_prompt_active_outputs_context(engram_home, monkeypatch, capsys):
    (engram_home / "config.toml").write_text(
        "surface_mode = 'active'\nsurface_threshold = 0.3\n",
        encoding="utf-8",
    )
    engine = _fake_build_engine()
    engine.remember("予算要求の書式は財務課の様式7を使うこと", "knowledge", 7)
    engine.db.close()

    stdin = json.dumps({
        "session_id": "sess-4",
        "prompt": "予算要求の書式ってどうだったっけ",
        "cwd": "C:/anywhere",
    })
    assert run_user_prompt(stdin) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert "様式7" in ctx
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"


def test_user_prompt_skips_short_and_slash(engram_home):
    assert run_user_prompt(json.dumps({
        "session_id": "s", "prompt": "短い", "cwd": ""})) == 0
    assert run_user_prompt(json.dumps({
        "session_id": "s", "prompt": "/clear して再開しよう", "cwd": ""})) == 0
    assert not (engram_home / "surface" / "surface_log.jsonl").exists()


# ---------------------------------------------------------------------------
# フック登録(setup)
# ---------------------------------------------------------------------------

def test_register_claude_hooks_creates_file(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.json"
    ok, msg = register_claude_hooks(settings_path, tmp_path / "engram.exe")
    assert ok
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    events = data["hooks"]
    assert "SessionEnd" in events
    assert "UserPromptSubmit" in events
    cmd = events["SessionEnd"][0]["hooks"][0]["command"]
    assert "engram" in cmd and "hook session-end" in cmd


def test_register_claude_hooks_idempotent(tmp_path):
    settings_path = tmp_path / "settings.json"
    exe = tmp_path / "engram.exe"
    register_claude_hooks(settings_path, exe)
    before = settings_path.read_text(encoding="utf-8")
    ok, msg = register_claude_hooks(settings_path, exe)
    assert ok
    assert "スキップ" in msg
    assert settings_path.read_text(encoding="utf-8") == before


def test_register_claude_hooks_updates_stale_path(tmp_path):
    settings_path = tmp_path / "settings.json"
    register_claude_hooks(settings_path, tmp_path / "old" / "engram.exe")
    ok, msg = register_claude_hooks(settings_path, tmp_path / "new" / "engram.exe")
    assert ok
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    cmd = data["hooks"]["SessionEnd"][0]["hooks"][0]["command"]
    assert "new" in cmd and "old" not in cmd
    # エントリが増殖していない
    assert len(data["hooks"]["SessionEnd"]) == 1


def test_register_claude_hooks_preserves_other_hooks(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "permissions": {"allow": ["Bash(ls:*)"]},
        "hooks": {
            "SessionEnd": [
                {"hooks": [{"type": "command", "command": "other-tool run"}]}
            ]
        },
    }), encoding="utf-8")
    ok, _ = register_claude_hooks(settings_path, tmp_path / "engram.exe")
    assert ok
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    cmds = [h["command"] for e in data["hooks"]["SessionEnd"]
            for h in e["hooks"]]
    assert "other-tool run" in cmds
    assert any("engram" in c for c in cmds)
    assert data["permissions"]["allow"] == ["Bash(ls:*)"]


def test_register_claude_hooks_refuses_broken_json(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{broken json", encoding="utf-8")
    ok, msg = register_claude_hooks(settings_path, tmp_path / "engram.exe")
    assert not ok
    assert settings_path.read_text(encoding="utf-8") == "{broken json"


# ---------------------------------------------------------------------------
# config.toml の dict(セクション)往復
# ---------------------------------------------------------------------------

def test_config_toml_roundtrip_with_room_paths(tmp_path):
    cfg = tmp_path / "config.toml"
    write_config_toml(cfg, {
        "memories_dir": "C:/mem",
        "room_paths": {"C:/Users/me/work": "work",
                       "H:/マイドライブ/個人": "personal"},
    })
    data = read_config_toml(cfg)
    assert data["memories_dir"] == "C:/mem"
    assert data["room_paths"]["C:/Users/me/work"] == "work"
    assert data["room_paths"]["H:/マイドライブ/個人"] == "personal"

    # merge してもセクションが保持される
    merge_config_toml(cfg, {"surface_mode": "active"})
    data = read_config_toml(cfg)
    assert data["surface_mode"] == "active"
    assert data["room_paths"]["C:/Users/me/work"] == "work"
