"""media_kinds を一時的に絞っても、対象外の種別の既存行を「消えた」扱いで落とさない(R8.10.8 の実測から)。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from engram.config import Settings
from engram.media_schema import MediaIndex
from engram.media_worker import FakeMediaEmbedder, Indexer


def _touch(path: Path, size: int = 100, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([(seed + i) % 256 for i in range(size)]))
    return path


def test_narrowed_kinds_keep_other_rows_but_still_drop_deleted_files(tmp_path):
    root = tmp_path / "lib"
    img = _touch(root / "a.png", 100, 1)
    vid = _touch(root / "b.mp4", 100, 2)
    img2 = _touch(root / "c.png", 100, 3)
    emb = FakeMediaEmbedder(dim=16)
    base = dict(memories_dir=tmp_path / "m", data_dir=tmp_path / "d", media_enabled=True, media_port=0,
                media_min_bytes=10, media_caption=False, media_dup_threshold=1.01, media_roots={str(root): "common"})
    s_all = Settings(**base, media_kinds=["image", "video"])
    idx = MediaIndex.open(s_all.db_path, dim=emb.dim)

    # 動画の復号は部品が要るので、Indexer の pieces_for を差し替えて1段だけ返す
    import engram.media_worker as mw
    orig = mw.pieces_for
    mw.pieces_for = lambda path, kind, settings: [mw.Piece(payload={"text": f"{kind}:{path.name}"})]
    try:
        rep = Indexer(s_all, emb, idx).run()
        assert rep.added == 3 and idx.count(kind="video") == 1
        # 種別を画像だけに絞って再開: 動画の行は残り、消した画像だけ落ちる
        img2.unlink()
        s_img = Settings(**base, media_kinds=["image"])
        rep2 = Indexer(s_img, emb, idx).run()
        assert rep2.removed == 1 and idx.count(kind="video") == 1 and idx.count(kind="image") == 1
        assert idx.ids_for_path(str(vid)) and not idx.ids_for_path(str(img2))
    finally:
        mw.pieces_for = orig
