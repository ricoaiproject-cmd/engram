"""索引係への同時問い合わせ(R8.10.23 の不具合の再発防止)。

窓口が複数(ThreadingTCPServer)なので find は同時に走る。問い合わせ用の sqlite3 接続を
1本のまま同時に使うと行の形が崩れ、find が dup_count で TypeError / InterfaceError で
落ちていた(実測: 6本同時で2本が失敗)。問い合わせを直列化したことを確かめる。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from engram.config import Settings
from engram.media_worker import FakeMediaEmbedder, serve
from engram import media as media_core


def _settings(tmp_path: Path, **kw) -> Settings:
    base = dict(memories_dir=tmp_path / "memories", data_dir=tmp_path / "data", media_enabled=True,
                media_port=0, media_min_bytes=10, media_caption=False, media_dup_threshold=1.01)
    base.update(kw)
    return Settings(**base)


def _touch(path: Path, size: int = 100, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([(seed + i) % 256 for i in range(size)]))
    return path


@pytest.fixture
def worker_with_items(tmp_path, monkeypatch):
    monkeypatch.delenv("ENGRAM_DISABLE_DAEMON", raising=False)
    root = tmp_path / "pics"
    for i in range(8):
        _touch(root / f"shot_{i}.png", 200, i * 3)
    s = _settings(tmp_path, media_roots={str(root): "common"})
    t = threading.Thread(target=serve,
                         kwargs={"settings": s, "embedder": FakeMediaEmbedder(dim=32), "idle_hours": 1.0},
                         daemon=True)
    t.start()
    for _ in range(100):
        if media_core.port_file(s).is_file() and media_core.is_alive(s):
            break
        time.sleep(0.05)
    else:
        pytest.fail("索引係が起動しない")
    media_core.start_index(s, wait=0)
    media_core.wait_index(s, timeout=20, poll=0.1)
    yield s
    media_core.stop(s)
    t.join(timeout=5)


def test_concurrent_find_and_status_do_not_corrupt_rows(worker_with_items):
    s = worker_with_items
    errors: list[str] = []
    hits: list[int] = []

    def _find() -> None:
        try:
            r = media_core.find(s, "shot", k=5, kinds=["image"])
            if r.get("mode") != "media":
                errors.append(f"mode={r.get('mode')} {r}")
            else:
                hits.append(len(r["hits"]))
        except Exception as exc:  # pragma: no cover - 失敗時の手がかり
            errors.append(f"{type(exc).__name__}: {exc}")

    def _status() -> None:
        try:
            st = media_core.status(s)
            if not st["alive"] or st["stats"]["items"] != 8:
                errors.append(f"status={st}")
        except Exception as exc:  # pragma: no cover
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=_find) for _ in range(10)]
    threads += [threading.Thread(target=_status) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30)

    assert errors == []
    assert len(hits) == 10 and all(n > 0 for n in hits)


def test_find_holds_the_query_lock(tmp_path, monkeypatch):
    """問い合わせ中、別スレッドは qlock を取れないこと(直列化の担保)。"""
    from engram import media_worker

    s = _settings(tmp_path, media_roots={str(tmp_path / "pics"): "common"})
    state = media_worker._State(s, embedder=FakeMediaEmbedder(dim=32))
    other_could_take: list[bool] = []

    def _spy(*a, **kw):
        def _try() -> None:
            got = state.qlock.acquire(blocking=False)
            other_could_take.append(got)
            if got:
                state.qlock.release()

        th = threading.Thread(target=_try)
        th.start()
        th.join(timeout=5)
        return []

    monkeypatch.setattr(media_worker, "hybrid_find", _spy)
    state.find("なにか", 3, None, None, None, False)
    assert other_could_take == [False]
