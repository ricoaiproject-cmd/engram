"""媒体の棚(engram 0.24): 表・索引係・本体側の窓口のテスト。重い部品なし(FakeMediaEmbedder)。"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from engram.config import Settings
from engram.media_schema import MediaIndex, TABLES, hybrid_find
from engram.media_worker import FakeMediaEmbedder, Indexer, IndexReport, kind_of, serve, item_id
from engram import media as media_core


def _settings(tmp_path: Path, **kw) -> Settings:
    # 模擬埋め込みはパスの文字 n-gram なので、同じフォルダの別ファイルがほぼ同一(cos>0.98)に
    # 見えてしまう。近似重複の判定は専用テストでだけ有効にする(media_dup_threshold=0.98 を明示)
    base = dict(memories_dir=tmp_path / "memories", data_dir=tmp_path / "data", media_enabled=True,
                media_port=0, media_min_bytes=10, media_caption=False, media_dup_threshold=1.01)
    base.update(kw)
    return Settings(**base)


def _touch(path: Path, size: int = 100, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([(seed + i) % 256 for i in range(size)]))
    return path


# ---------------------------------------------------------------- schema

def test_schema_roundtrip_and_search(tmp_path):
    idx = MediaIndex.open(tmp_path / "i.db", dim=8, model="fake")
    assert idx.ready and idx.dim == 8 and idx.meta("model") == "fake"
    v1 = np.eye(8, dtype=np.float32)[0]
    v2 = np.eye(8, dtype=np.float32)[1]
    idx.upsert(id="a", path="/x/cat.png", root="/x", room="common", kind="image", embedding=v1,
               caption="黒い猫の写真", bytes=100, mtime=1.0, sha256="s1")
    idx.upsert(id="b", path="/x/dog.png", root="/x", room="work", kind="image", embedding=v2,
               caption="犬のイラスト", bytes=100, mtime=1.0, sha256="s2")
    assert idx.count() == 2 and idx.count(room="work") == 1
    hits = idx.vector_search(v1, 5)
    assert hits[0][0] == "a" and hits[0][1] > 0.99
    assert idx.vector_search(v1, 5, rooms=["work"])[0][0] == "b"
    assert idx.keyword_search("黒い猫", 5)[0][0] == "a"
    assert idx.keyword_search("犬", 5)[0][0] == "b"           # 短語→LIKE 経路
    res = hybrid_find(idx, qvec=v1, query="猫", k=5, record_hits=True)
    assert res[0]["id"] == "a" and res[0]["hits"] == 0      # 返却時点の行は hits 更新前
    assert idx.get("a")["hits"] == 1 and idx.get("a")["last_hit"] is not None
    # 差分用の既知ファイル表
    known = idx.known_files("/x")
    assert set(known) == {"/x/cat.png", "/x/dog.png"} and known["/x/cat.png"][0] == "s1"


def test_schema_demote_and_drop(tmp_path):
    idx = MediaIndex.open(tmp_path / "i.db", dim=4)
    now = 1_800_000_000.0
    idx.upsert(id="old", path="/o", root="/", room="common", kind="image",
               embedding=np.ones(4, dtype=np.float32), created_at=now - 100 * 86400)
    idx.upsert(id="new", path="/n", root="/", room="common", kind="image",
               embedding=np.ones(4, dtype=np.float32), created_at=now - 1 * 86400)
    assert idx.demote_cold(older_than_days=45, now=now) == 1
    assert idx.get("old")["tier"] == "cold" and idx.get("new")["tier"] == "hot"
    assert idx.vector_search(np.ones(4, dtype=np.float32), 5, tiers=["hot"]) == [("new", pytest.approx(1.0))]
    idx.record_hit(["old"], now=now)                         # 使われた瞬間に hot へ戻る
    assert idx.get("old")["tier"] == "hot"
    idx.drop_all()
    names = {r[0] for r in idx._conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert not (set(TABLES) & names)


def test_schema_fresh_db_without_dim_is_not_ready_and_dim_mismatch_raises(tmp_path):
    ro = MediaIndex.open(tmp_path / "i.db")
    assert not ro.ready and ro.vector_search(np.ones(3), 3) == [] and ro.stats() == {"ready": False, "items": 0}
    MediaIndex.open(tmp_path / "i.db", dim=8)
    with pytest.raises(ValueError):
        MediaIndex.open(tmp_path / "i.db", dim=16)


def test_dup_is_folded_in_search(tmp_path):
    idx = MediaIndex.open(tmp_path / "i.db", dim=4)
    v = np.ones(4, dtype=np.float32)
    idx.upsert(id="rep", path="/a.png", root="/", room="common", kind="image", embedding=v)
    idx.upsert(id="dup", path="/b.png", root="/", room="common", kind="image", embedding=v, dup_of="rep")
    hits = hybrid_find(idx, qvec=v, query="", k=5, record_hits=False)
    assert [h["id"] for h in hits] == ["rep"] and hits[0]["duplicates"] == 1


# ---------------------------------------------------------------- indexer

def test_indexer_diff_add_update_remove_and_gate(tmp_path):
    root = tmp_path / "pics"
    big = _touch(root / "a.png", 100, 1)
    _touch(root / "tiny.png", 5, 2)                 # min_bytes 未満 → 門で止まる
    _touch(root / "note.txt", 100, 3)               # 対象外の拡張子
    _touch(root / "node_modules" / "x.png", 100, 4) # 除外フォルダ
    s = _settings(tmp_path, media_roots={str(root): "common"})
    emb = FakeMediaEmbedder(dim=32)
    idx = MediaIndex.open(s.db_path, dim=emb.dim, model=emb.model_id)
    ix = Indexer(s, emb, idx)
    rep = ix.run()
    assert (rep.scanned, rep.added, rep.skipped, rep.removed, rep.failed) == (1, 1, 0, 0, 0)
    assert idx.count() == 1 and idx.get(item_id(str(big), 0))["room"] == "common"
    # 2回目: 変更なし → skip
    rep2 = ix.run()
    assert (rep2.added, rep2.skipped, rep2.updated) == (0, 1, 0)
    # 中身を変える(mtime を進める) → update
    time.sleep(0.01)
    big.write_bytes(b"changed" * 20)
    import os
    os.utime(big, (time.time() + 5, time.time() + 5))
    rep3 = ix.run()
    assert rep3.updated == 1
    # 消す → remove
    big.unlink()
    rep4 = ix.run()
    assert rep4.removed == 1 and idx.count() == 0


def test_indexer_marks_near_duplicates(tmp_path):
    root = tmp_path / "pics"
    _touch(root / "a.png", 100, 1)
    _touch(root / "b.png", 100, 2)

    class Const(FakeMediaEmbedder):
        def embed_doc(self, payload):
            return np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)

    s = _settings(tmp_path, media_roots={str(root): "common"}, media_dup_threshold=0.98)
    emb = Const(dim=16)
    idx = MediaIndex.open(s.db_path, dim=emb.dim)
    rep = Indexer(s, emb, idx).run()
    assert rep.added == 2 and rep.dups == 1
    assert idx.stats()["near_duplicates"] == 1
    hits = hybrid_find(idx, qvec=np.ones(16, dtype=np.float32), query="", k=5, record_hits=False)
    assert len(hits) == 1 and hits[0]["duplicates"] == 1


def test_kind_of():
    assert kind_of(Path("x.PNG")) == "image" and kind_of(Path("x.mp4")) == "video"
    assert kind_of(Path("x.wav")) == "audio" and kind_of(Path("x.docx")) == "document"
    assert kind_of(Path("x.txt")) is None


# ---------------------------------------------------------------- daemon + core client

@pytest.fixture
def running_worker(tmp_path):
    root = tmp_path / "pics"
    _touch(root / "sunset.png", 200, 7)
    _touch(root / "invoice.png", 200, 9)
    s = _settings(tmp_path, media_roots={str(root): "common"})
    emb = FakeMediaEmbedder(dim=32)
    t = threading.Thread(target=serve, kwargs={"settings": s, "embedder": emb, "idle_hours": 1.0}, daemon=True)
    t.start()
    for _ in range(100):
        if media_core.port_file(s).is_file() and media_core.is_alive(s):
            break
        time.sleep(0.05)
    else:
        pytest.fail("索引係が起動しない")
    yield s, root
    media_core.stop(s)
    t.join(timeout=5)


def test_worker_roundtrip(running_worker, monkeypatch):
    s, root = running_worker
    monkeypatch.delenv("ENGRAM_DISABLE_DAEMON", raising=False)
    assert media_core.is_alive(s)
    res = media_core.start_index(s, wait=0)
    assert res["ok"]
    prog = media_core.wait_index(s, timeout=20, poll=0.1)
    assert prog["state"] == "idle" and prog["last"]["added"] == 2
    st = media_core.status(s)
    assert st["alive"] and st["stats"]["items"] == 2 and st["stats"]["by_kind"] == {"image": 2}
    found = media_core.find(s, "sunset.png", k=3, kinds=["image"])
    assert found["mode"] == "media" and found["hits"] and found["hits"][0]["type"] == "media"
    assert found["hits"][0]["path"].endswith("sunset.png") and "kind" in found["hits"][0]
    # 1件の即時索引(別フォルダ・メール由来の出所つき)
    extra = _touch(tmp_path_of(root) / "mail" / "見積書.png", 300, 11)
    r = media_core.index_path(s, extra, room="work", source="mail",
                              source_meta={"subject": "見積書送付", "from": "someone", "date": "2026-10-01"}, wait=0)
    assert r["ok"] and r["items"][0]["room"] == "work" and r["items"][0]["source"] == "mail"
    hit = media_core.find(s, "見積書", k=3, rooms=["work", "common"])["hits"][0]
    assert "見積書送付" in hit["content"] and hit["source_meta"]["from"] == "someone"
    # 夜間処理
    night = media_core.run_nightly(s, timeout=20)
    assert night["ok"] and night["index"]["skipped"] == 2


def tmp_path_of(root: Path) -> Path:
    return root.parent


def test_find_when_disabled_or_down(tmp_path):
    s = _settings(tmp_path, media_enabled=False)
    assert media_core.find(s, "x")["error"] == "media_enabled=false"
    s2 = _settings(tmp_path, media_enabled=True)
    res = media_core.find(s2, "x", wait=0)     # ENGRAM_DISABLE_DAEMON=1 のため起動せず
    assert res["hits"] == [] and "起動していません" in res["error"]


def test_engine_recall_modality_routes_to_media(tmp_path):
    from engram.embedder import FakeEmbedder
    from engram.engine import MemoryEngine
    from engram.store import MarkdownStore
    from engram.db import IndexDB

    s = _settings(tmp_path, media_enabled=False)
    emb = FakeEmbedder(dim=64)
    eng = MemoryEngine(settings=s, store=MarkdownStore(s.memories_dir), db=IndexDB(s.db_path, emb.dim), embedder=emb)
    eng.remember("猫の写真について書いた文章", "knowledge", 5)
    text = eng.recall("猫の写真")
    assert text["hits"] and text["mode"] in ("fast", "deep", "exhaustive")
    media = eng.recall("猫の写真", modality="image")
    assert media["mode"] == "media" and media["hits"] == [] and media.get("error")


def test_install_and_uninstall_are_reversible(tmp_path):
    s = _settings(tmp_path)
    p = media_core.install_plugin(s)
    assert p.is_file() and p.name == media_core.PLUGIN_NAME and "run_nightly" in p.read_text(encoding="utf-8")
    MediaIndex.open(s.db_path, dim=4).upsert(id="a", path="/a", root="/", room="common", kind="image",
                                              embedding=np.ones(4, dtype=np.float32))
    res = media_core.uninstall(s, purge_index=True)
    assert not p.is_file() and any("媒体の表を削除" in d for d in res["done"])
    assert not MediaIndex.open(s.db_path).ready
    # 記憶本体の表には触れていない(存在しないまま)
    import sqlite3
    names = {r[0] for r in sqlite3.connect(s.db_path).execute("SELECT name FROM sqlite_master")}
    assert "memories" not in names and not (set(TABLES) & names)


def test_format_status_smoke():
    txt = media_core.format_status({"enabled": True, "alive": False, "python": "x", "python_exists": False,
                                    "roots": {"a": "common"}, "kinds": ["image"], "night_plugin": False,
                                    "stats": {"ready": False}})
    assert "無し→ engram media setup" in txt and "索引: 未作成" in txt
