"""媒体の自発的想起: 床を越えた1件だけ・同じセッションで二度出さない・shadow はログのみ。"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from engram.config import Settings
from engram import media as media_core
from engram.media_worker import FakeMediaEmbedder, serve


def _touch(path: Path, size: int = 200, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([(seed + i) % 256 for i in range(size)]))
    return path


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.delenv("ENGRAM_DISABLE_DAEMON", raising=False)
    root = tmp_path / "pics"
    _touch(root / "sunset_beach.png", 200, 1)
    _touch(root / "invoice_table.png", 200, 9)
    s = Settings(memories_dir=tmp_path / "m", data_dir=tmp_path / "d", media_enabled=True, media_port=0,
                 media_min_bytes=10, media_caption=False, media_dup_threshold=1.01,
                 media_roots={str(root): "common"}, media_surface_min_cos=0.0, media_surface_min_prompt_chars=3,
                 media_surface_cues=["写真", "画像"], media_surface_margin=0.0)   # 模擬埋め込みは何でも近いので既定は1位だけ
    t = threading.Thread(target=serve, kwargs={"settings": s, "embedder": FakeMediaEmbedder(dim=32), "idle_hours": 1.0},
                         daemon=True)
    t.start()
    for _ in range(100):
        if media_core.is_alive(s):
            break
        time.sleep(0.05)
    assert media_core.start_index(s, wait=0)["ok"]
    assert media_core.wait_index(s, timeout=20, poll=0.1)["state"] == "idle"
    yield s
    media_core.stop(s)
    t.join(timeout=5)


def test_surface_active_once_per_session_and_logs(worker):
    s = worker
    r1 = media_core.surface(s, "sunset_beach の写真を探している", room="common", session_id="sess1", mode="active")
    assert len(r1["items"]) == 1 and r1["items"][0]["path"].endswith("sunset_beach.png")
    r2 = media_core.surface(s, "sunset_beach の写真を探している", room="common", session_id="sess1", mode="active")
    assert r2["items"] == [] or not r2["items"][0]["path"].endswith("sunset_beach.png")  # 二度出さない
    r3 = media_core.surface(s, "sunset_beach の写真を探している", room="common", session_id="sess2", mode="active")
    assert r3["items"] and r3["items"][0]["path"].endswith("sunset_beach.png")       # 別セッションでは出る
    log = (s.media_dir / "surface_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(log) == 3 and json.loads(log[0])["injected"]
    txt = media_core.format_surface(r1["items"])
    assert "媒体の想起" in txt and "sunset_beach.png" in txt


def test_surface_shadow_logs_only_and_floor(worker):
    s = worker
    r = media_core.surface(s, "sunset_beach の写真", session_id="s3", mode="shadow")
    assert r["items"] == []
    log = (s.media_dir / "surface_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(log[-1])["mode"] == "shadow" and json.loads(log[-1])["injected"] == []
    strict = Settings(**{**s.__dict__, "media_surface_min_cos": 0.999})
    assert media_core.surface(strict, "sunset_beach の写真", session_id="s4", mode="active")["items"] == []


def test_surface_off_or_short_prompt(tmp_path):
    s = Settings(memories_dir=tmp_path / "m", data_dir=tmp_path / "d", media_enabled=True, media_port=0,
                 media_surface_mode="off")
    assert media_core.surface(s, "長い発言でも off なら何もしない", session_id="x")["items"] == []
    s2 = Settings(memories_dir=tmp_path / "m", data_dir=tmp_path / "d", media_enabled=True, media_port=0)
    assert media_core.surface(s2, "短い", session_id="x")["items"] == []


def test_surface_requires_cue_word(worker):
    s = worker
    r = media_core.surface(s, "sunset_beach について相談がある", session_id="cue1", mode="active")
    assert r["items"] == [] and r.get("skipped") == "no cue"          # 手がかり語なし→出さない
    r2 = media_core.surface(s, "sunset_beach の写真を見たい", session_id="cue2", mode="active")
    assert r2["items"] and r2["items"][0]["path"].endswith("sunset_beach.png")


def test_hook_user_prompt_injects_media(worker, monkeypatch, capsys):
    """発話フック経由で媒体の想起が注入される(引数の取り違えの回帰・R8.10.11)。"""
    import json as _json

    from engram import hooks

    s = worker
    monkeypatch.setattr(hooks, "get_settings", lambda: Settings(**{**s.__dict__, "surface_mode": "off",
                                                                   "first_prompt_recall": False,
                                                                   "recall_nudge": False}))
    payload = _json.dumps({"prompt": "sunset_beach の写真はどこ", "session_id": "hook-sess", "cwd": str(s.data_dir)})
    assert hooks.run_user_prompt(payload) == 0
    out = capsys.readouterr().out
    assert "媒体の想起" in out and "sunset_beach.png" in out


def test_surface_margin_keeps_only_items_close_to_top(worker):
    """1位との差で切る: 離れた2位は出ない。差が0なら並んだ分が出る。"""
    s = worker
    wide = Settings(**{**s.__dict__, "media_surface_margin": 1.0, "media_surface_max_items": 3})
    r_all = media_core.surface(wide, "sunset_beach invoice_table の写真", session_id="m1", mode="active")
    assert len(r_all["items"]) >= 2
    tight = Settings(**{**s.__dict__, "media_surface_margin": 0.0, "media_surface_max_items": 3})
    r_top = media_core.surface(tight, "sunset_beach invoice_table の写真", session_id="m2", mode="active")
    assert len(r_top["items"]) == 1
