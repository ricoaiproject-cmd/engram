"""media_exclude_dirs: 走査で飛ばし、除外に変わったフォルダの既存行は次の索引で落ちる。"""
from __future__ import annotations

from pathlib import Path

from engram.config import Settings
from engram.media_schema import MediaIndex
from engram.media_worker import FakeMediaEmbedder, Indexer, is_excluded


def _touch(path: Path, size: int = 100, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([(seed + i) % 256 for i in range(size)]))
    return path


def test_is_excluded_patterns():
    pats = ["node_modules", "review", "render_parts*", "qa_*"]
    assert is_excluded(("a", "review", "x.png"), pats)
    assert is_excluded(("render_parts_v7", "x.png"), pats)
    assert is_excluded(("v", "qa_stills", "x.png"), pats)
    assert not is_excluded(("assets", "x.png"), pats)
    assert not is_excluded(("review.png",), pats)   # ファイル名自体は対象外(最後の要素は見ない)


def test_exclusion_change_prunes_existing_rows(tmp_path):
    root = tmp_path / "lib"
    keep = _touch(root / "assets" / "a.png", 100, 1)
    noisy = _touch(root / "review" / "thumb_01.png", 100, 2)
    emb = FakeMediaEmbedder(dim=16)
    base = dict(memories_dir=tmp_path / "m", data_dir=tmp_path / "d", media_enabled=True, media_port=0,
                media_min_bytes=10, media_caption=False, media_dup_threshold=1.01, media_kinds=["image"],
                media_roots={str(root): "common"})
    s1 = Settings(**base)
    idx = MediaIndex.open(s1.db_path, dim=emb.dim)
    assert Indexer(s1, emb, idx).run().added == 2
    s2 = Settings(**base, media_exclude_dirs=["node_modules", "review"])
    rep = Indexer(s2, emb, idx).run()
    assert rep.scanned == 1 and rep.removed == 1
    assert idx.ids_for_path(str(keep)) and not idx.ids_for_path(str(noisy))


def test_raising_min_bytes_prunes_small_images(tmp_path):
    root = tmp_path / "lib"
    big = _touch(root / "big.png", 5000, 1)
    small = _touch(root / "small.png", 50, 2)
    emb = FakeMediaEmbedder(dim=16)
    base = dict(memories_dir=tmp_path / "m", data_dir=tmp_path / "d", media_enabled=True, media_port=0,
                media_caption=False, media_dup_threshold=1.01, media_kinds=["image"], media_roots={str(root): "common"})
    idx = MediaIndex.open(Settings(**base, media_min_bytes=10).db_path, dim=emb.dim)
    assert Indexer(Settings(**base, media_min_bytes=10), emb, idx).run().added == 2
    rep = Indexer(Settings(**base, media_min_bytes=1000), emb, idx).run()
    assert rep.removed == 1 and idx.ids_for_path(str(big)) and not idx.ids_for_path(str(small))
