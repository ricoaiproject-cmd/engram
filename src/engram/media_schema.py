"""媒体(画像・音声・動画・文書)の索引テーブル。本体と索引係(別 venv)の両方が使う。

依存は sqlite3 + sqlite_vec + numpy だけに留める(重い部品を本体側に持ち込まない)。
既存の memories / vec_memories / fts_memories には一切触れない。テーブルは3つ+メタ:

    media_items(id PK, path, root, room, kind, segment, page, sha256, bytes, mtime,
                duration, caption, text, source, source_meta, model, dim,
                created_at, tier, hits, last_hit, dup_of)
    vec_media  = vec0(item_id PK, embedding float[dim] distance_metric=cosine)
    fts_media  = fts5(item_id UNINDEXED, content, tokenize='trigram')
    media_meta(key PK, value)

1行=1ファイル(動画・音声は時間の段、PDF/Office はページ)。`dup_of` が付いた行は
ほぼ同一の別ファイルで、検索結果では代表(dup_of の先)に畳む。
`engram media uninstall --purge-index` は drop_all() でこの4表だけを落とす。
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np

MEDIA_KINDS = ("image", "video", "audio", "document")
MEDIA_TIERS = ("hot", "cold")
TABLES = ("media_items", "vec_media", "fts_media", "media_meta")

# 媒体の埋め込みモデル(索引係側の既定)。本体側は media_meta から読むだけ
DEFAULT_MEDIA_MODEL = "google/embeddinggemma-2"
DEFAULT_MEDIA_DIM = 768


def connect(path: Path) -> sqlite3.Connection:
    """index.db を開き sqlite-vec を読み込む(WAL・busy_timeout は本体と同じ流儀)。"""
    import sqlite_vec

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


class MediaIndex:
    """媒体索引の読み書き。本体(検索・夜間)と索引係(書き込み)で共用する。"""

    def __init__(self, conn: sqlite3.Connection, *, dim: int | None = None,
                 model: str | None = None) -> None:
        self._conn = conn
        self._init_schema(dim=dim, model=model)

    @classmethod
    def open(cls, db_path: Path, *, dim: int | None = None,
             model: str | None = None) -> "MediaIndex":
        return cls(connect(db_path), dim=dim, model=model)

    # ---- スキーマ -------------------------------------------------------
    def _init_schema(self, *, dim: int | None, model: str | None) -> None:
        c = self._conn
        exists = c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='media_meta'").fetchone() is not None
        if not exists and dim is None:
            # 本体側が索引未作成の DB を開いた場合: 何も作らず読み取り専用で振る舞う
            # (uninstall --purge-index の後に空の表を残さないため)
            self.dim = None
            return
        c.execute("CREATE TABLE IF NOT EXISTS media_meta (key TEXT PRIMARY KEY, value TEXT)")
        stored_dim = self.meta("dim")
        if stored_dim is None:
            if dim is None:
                self.dim = None
                return
            c.execute("INSERT INTO media_meta(key, value) VALUES ('dim', ?)", (str(dim),))
            c.execute("INSERT OR REPLACE INTO media_meta(key, value) VALUES ('model', ?)",
                      (model or DEFAULT_MEDIA_MODEL,))
            c.execute("INSERT OR REPLACE INTO media_meta(key, value) VALUES ('schema', '1')")
            self.dim = dim
        else:
            self.dim = int(stored_dim)
            if dim is not None and dim != self.dim:
                raise ValueError(f"媒体索引の次元が一致しません: 保存済み {self.dim} / 要求 {dim}。"
                                 "モデルを替えるなら `engram media index --rebuild` で作り直してください")
        c.execute(
            """CREATE TABLE IF NOT EXISTS media_items (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                root TEXT NOT NULL,
                room TEXT NOT NULL DEFAULT 'common',
                kind TEXT NOT NULL,
                segment INTEGER NOT NULL DEFAULT 0,
                page INTEGER,
                sha256 TEXT,
                bytes INTEGER,
                mtime REAL,
                duration REAL,
                caption TEXT,
                text TEXT,
                source TEXT NOT NULL DEFAULT 'folder',
                source_meta TEXT,
                model TEXT,
                dim INTEGER,
                created_at REAL NOT NULL,
                tier TEXT NOT NULL DEFAULT 'hot',
                hits INTEGER NOT NULL DEFAULT 0,
                last_hit REAL,
                dup_of TEXT
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_media_path ON media_items(path, segment)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_media_room_kind ON media_items(room, kind, tier)")
        c.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_media USING vec0("
            f"item_id TEXT PRIMARY KEY, embedding float[{self.dim}] distance_metric=cosine)"
        )
        c.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_media USING fts5("
            "item_id UNINDEXED, content, tokenize='trigram')"
        )
        c.commit()

    @property
    def ready(self) -> bool:
        return self.dim is not None

    def meta(self, key: str) -> str | None:
        try:
            row = self._conn.execute("SELECT value FROM media_meta WHERE key=?", (key,)).fetchone()
        except sqlite3.OperationalError:
            return None
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO media_meta(key, value) VALUES (?, ?)", (key, value))
        self._conn.commit()

    def drop_all(self) -> None:
        """媒体の4表だけを落とす(可逆性の要)。他の表には触れない。"""
        for t in TABLES:
            self._conn.execute(f"DROP TABLE IF EXISTS {t}")
        self._conn.commit()
        self.dim = None

    def close(self) -> None:
        self._conn.close()

    # ---- 書き込み -------------------------------------------------------
    def upsert(self, *, id: str, path: str, root: str, room: str, kind: str,
               embedding: np.ndarray, segment: int = 0, page: int | None = None,
               sha256: str | None = None, bytes: int | None = None, mtime: float | None = None,
               duration: float | None = None, caption: str | None = None, text: str | None = None,
               source: str = "folder", source_meta: dict | None = None,
               model: str | None = None, created_at: float | None = None,
               dup_of: str | None = None) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self.dim:
            raise ValueError(f"embedding dim {emb.shape[0]} != {self.dim}")
        content = " ".join(s for s in (caption or "", text or "", Path(path).name) if s)
        with self._conn:
            self._conn.execute(
                """INSERT INTO media_items(id, path, root, room, kind, segment, page, sha256, bytes, mtime,
                       duration, caption, text, source, source_meta, model, dim, created_at, tier, dup_of)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'hot',?)
                   ON CONFLICT(id) DO UPDATE SET
                       path=excluded.path, root=excluded.root, room=excluded.room, kind=excluded.kind,
                       segment=excluded.segment, page=excluded.page, sha256=excluded.sha256,
                       bytes=excluded.bytes, mtime=excluded.mtime, duration=excluded.duration,
                       caption=excluded.caption, text=excluded.text, source=excluded.source,
                       source_meta=excluded.source_meta, model=excluded.model, dim=excluded.dim,
                       dup_of=excluded.dup_of""",
                (id, path, root, room, kind, segment, page, sha256, bytes, mtime, duration,
                 caption, text, source, json.dumps(source_meta, ensure_ascii=False) if source_meta else None,
                 model, self.dim, created_at if created_at is not None else time.time(), dup_of),
            )
            self._conn.execute("DELETE FROM vec_media WHERE item_id=?", (id,))
            self._conn.execute("INSERT INTO vec_media(item_id, embedding) VALUES (?, ?)", (id, emb.tobytes()))
            self._conn.execute("DELETE FROM fts_media WHERE item_id=?", (id,))
            self._conn.execute("INSERT INTO fts_media(item_id, content) VALUES (?, ?)", (id, content))

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        ph = ",".join("?" * len(ids))
        with self._conn:
            self._conn.execute(f"DELETE FROM vec_media WHERE item_id IN ({ph})", ids)
            self._conn.execute(f"DELETE FROM fts_media WHERE item_id IN ({ph})", ids)
            cur = self._conn.execute(f"DELETE FROM media_items WHERE id IN ({ph})", ids)
        return cur.rowcount

    def set_caption(self, id: str, caption: str) -> None:
        row = self._conn.execute("SELECT path, text FROM media_items WHERE id=?", (id,)).fetchone()
        if row is None:
            return
        content = " ".join(s for s in (caption, row["text"] or "", Path(row["path"]).name) if s)
        with self._conn:
            self._conn.execute("UPDATE media_items SET caption=? WHERE id=?", (caption, id))
            self._conn.execute("DELETE FROM fts_media WHERE item_id=?", (id,))
            self._conn.execute("INSERT INTO fts_media(item_id, content) VALUES (?, ?)", (id, content))

    def record_hit(self, ids: list[str], now: float | None = None) -> None:
        if not ids:
            return
        ts = now if now is not None else time.time()
        with self._conn:
            for i in ids:
                self._conn.execute(
                    "UPDATE media_items SET hits=hits+1, last_hit=?, tier='hot' WHERE id=? OR dup_of=?",
                    (ts, i, i))

    def demote_cold(self, *, older_than_days: float, now: float | None = None) -> int:
        """一定期間使われなかった行を cold へ(消さない)。使われた瞬間に hot へ戻る(record_hit)。"""
        ts = now if now is not None else time.time()
        cutoff = ts - older_than_days * 86400.0
        with self._conn:
            cur = self._conn.execute(
                "UPDATE media_items SET tier='cold' WHERE tier='hot' AND "
                "COALESCE(last_hit, created_at) < ?", (cutoff,))
        return cur.rowcount

    # ---- 読み出し -------------------------------------------------------
    def known_files(self, root: str | None = None) -> dict[str, tuple[str | None, float | None, list[str]]]:
        """path -> (sha256, mtime, [ids]) 。差分索引の判定に使う。"""
        sql = "SELECT id, path, sha256, mtime FROM media_items"
        params: tuple = ()
        if root is not None:
            sql += " WHERE root=?"
            params = (root,)
        out: dict[str, tuple[str | None, float | None, list[str]]] = {}
        for r in self._conn.execute(sql, params):
            sha, mt, ids = out.get(r["path"], (r["sha256"], r["mtime"], []))
            ids.append(r["id"])
            out[r["path"]] = (sha, mt, ids)
        return out

    def ids_for_path(self, path: str) -> list[str]:
        return [r["id"] for r in self._conn.execute("SELECT id FROM media_items WHERE path=?", (path,))]

    def get(self, id: str) -> dict | None:
        r = self._conn.execute("SELECT * FROM media_items WHERE id=?", (id,)).fetchone()
        return None if r is None else _row_to_dict(r)

    def embedding_of(self, id: str) -> np.ndarray | None:
        r = self._conn.execute("SELECT embedding FROM vec_media WHERE item_id=?", (id,)).fetchone()
        if r is None:
            return None
        return np.frombuffer(r["embedding"], dtype=np.float32)

    def count(self, *, kind: str | None = None, room: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM media_items WHERE 1=1"
        params: list = []
        if kind:
            sql += " AND kind=?"; params.append(kind)
        if room:
            sql += " AND room=?"; params.append(room)
        return int(self._conn.execute(sql, params).fetchone()["n"])

    def stats(self) -> dict:
        if not self.ready:
            return {"ready": False, "items": 0}
        by_kind = {r["kind"]: r["n"] for r in self._conn.execute(
            "SELECT kind, COUNT(*) AS n FROM media_items GROUP BY kind")}
        by_room = {r["room"]: r["n"] for r in self._conn.execute(
            "SELECT room, COUNT(*) AS n FROM media_items GROUP BY room")}
        by_tier = {r["tier"]: r["n"] for r in self._conn.execute(
            "SELECT tier, COUNT(*) AS n FROM media_items GROUP BY tier")}
        dups = int(self._conn.execute("SELECT COUNT(*) AS n FROM media_items WHERE dup_of IS NOT NULL").fetchone()["n"])
        files = int(self._conn.execute("SELECT COUNT(DISTINCT path) AS n FROM media_items").fetchone()["n"])
        return {"ready": True, "items": sum(by_kind.values()), "files": files, "by_kind": by_kind,
                "by_room": by_room, "by_tier": by_tier, "near_duplicates": dups,
                "model": self.meta("model"), "dim": self.dim}

    # ---- 検索 -----------------------------------------------------------
    def vector_search(self, embedding: np.ndarray, k: int, *, kinds: list[str] | None = None,
                      rooms: list[str] | None = None, tiers: list[str] | None = None,
                      include_dups: bool = False) -> list[tuple[str, float]]:
        """(id, コサイン類似度) 降順 k 件。フィルタはオーバーフェッチ+JOIN(本体と同じ流儀)。"""
        if not self.ready:
            return []
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1).tobytes()
        over = max(k * 8 if (kinds or rooms or tiers or not include_dups) else k * 4, k)
        try:
            rows = self._conn.execute(
                "SELECT item_id, distance FROM vec_media WHERE embedding MATCH ? AND k = ?",
                (emb, over)).fetchall()
        except sqlite3.OperationalError:
            return []
        if not rows:
            return []
        allowed = self._allowed([r["item_id"] for r in rows], kinds, rooms, tiers, include_dups)
        out = []
        for r in rows:
            if r["item_id"] in allowed:
                out.append((r["item_id"], 1.0 - float(r["distance"])))
                if len(out) >= k:
                    break
        return out

    def keyword_search(self, query: str, k: int, *, kinds: list[str] | None = None,
                       rooms: list[str] | None = None, tiers: list[str] | None = None,
                       include_dups: bool = False) -> list[tuple[str, float]]:
        """説明文・抽出文字・ファイル名への全文検索。(id, bm25) ランク順。"""
        if not self.ready:
            return []
        tokens = [t for t in query.strip().split() if len(t) >= 3]
        if not tokens:
            return self._like_search(query.strip().split(), k, kinds, rooms, tiers, include_dups)
        fts = " ".join('"' + t.replace('"', '""') + '"' for t in tokens)
        try:
            rows = self._conn.execute(
                "SELECT item_id, bm25(fts_media) AS score FROM fts_media WHERE content MATCH ? "
                "ORDER BY score LIMIT ?", (fts, k * 8)).fetchall()
        except sqlite3.OperationalError:
            return []
        if not rows:
            return []
        allowed = self._allowed([r["item_id"] for r in rows], kinds, rooms, tiers, include_dups)
        out = [(r["item_id"], float(r["score"])) for r in rows if r["item_id"] in allowed]
        return out[:k]

    def _like_search(self, tokens, k, kinds, rooms, tiers, include_dups) -> list[tuple[str, float]]:
        tokens = [t for t in tokens if t]
        if not tokens:
            return []
        esc = lambda t: t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")  # noqa: E731
        where = " AND ".join("content LIKE ? ESCAPE '\\'" for _ in tokens)
        params = [f"%{esc(t)}%" for t in tokens]
        try:
            n = int(self._conn.execute("SELECT COUNT(*) AS n FROM fts_media").fetchone()["n"])
            rows = self._conn.execute(
                f"SELECT item_id FROM fts_media WHERE {where} LIMIT ?", params + [k * 8]).fetchall()
        except sqlite3.OperationalError:
            return []
        if not rows:
            return []
        df = len(rows)
        pseudo = -math.log(1.0 + n / max(df, 1))
        allowed = self._allowed([r["item_id"] for r in rows], kinds, rooms, tiers, include_dups)
        return [(r["item_id"], pseudo) for r in rows if r["item_id"] in allowed][:k]

    def _allowed(self, ids, kinds, rooms, tiers, include_dups) -> set[str]:
        if not ids:
            return set()
        ph = ",".join("?" * len(ids))
        sql = f"SELECT id FROM media_items WHERE id IN ({ph})"
        params: list = list(ids)
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"; params.extend(kinds)
        if rooms:
            sql += f" AND room IN ({','.join('?' * len(rooms))})"; params.extend(rooms)
        if tiers:
            sql += f" AND tier IN ({','.join('?' * len(tiers))})"; params.extend(tiers)
        if not include_dups:
            sql += " AND dup_of IS NULL"
        return {r["id"] for r in self._conn.execute(sql, params)}

    def rows(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        ph = ",".join("?" * len(ids))
        return {r["id"]: _row_to_dict(r) for r in
                self._conn.execute(f"SELECT * FROM media_items WHERE id IN ({ph})", ids)}

    def dup_count(self, id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM media_items WHERE dup_of=?", (id,)).fetchone()
        if row is None:
            return 0
        n = row["n"] if not isinstance(row, (tuple, list)) else row[0]
        return int(n or 0)


def _row_to_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    if d.get("source_meta"):
        try:
            d["source_meta"] = json.loads(d["source_meta"])
        except (TypeError, ValueError):
            pass
    return d


def rrf_merge(rankings, *, k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion(本体 dynamics.rrf_merge と同じ式。索引係側で import できるよう複製)。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_find(index: MediaIndex, *, qvec: np.ndarray | None, query: str, k: int,
                kinds: list[str] | None = None, rooms: list[str] | None = None,
                tiers: list[str] | None = None, now: float | None = None,
                w_relevance: float = 0.7, w_activation: float = 0.3,
                record_hits: bool = True) -> list[dict]:
    """ベクトル+全文のハイブリッド検索。結果は media_items の行に relevance/score を足した dict。

    activation は「最近使われたほど・回数が多いほど」の単純な指標(0..1)。本体の ACT-R 風活性度より
    粗いが、媒体は使用実績で前に出す/沈めるだけで十分(消さない忘却の最小形)。
    """
    ts = now if now is not None else time.time()
    cand_k = max(k * 10, 50)
    vec_hits = index.vector_search(qvec, cand_k, kinds=kinds, rooms=rooms, tiers=tiers) if qvec is not None else []
    kw_hits = index.keyword_search(query, cand_k, kinds=kinds, rooms=rooms, tiers=tiers) if query else []
    if not vec_hits and not kw_hits:
        return []
    fused = rrf_merge([[i for i, _ in vec_hits], [i for i, _ in kw_hits]])
    vec_rel = {i: s for i, s in vec_hits}
    kw_rel = {i: 1.0 - math.exp(s) for i, s in kw_hits}  # bm25(負)→0..1(本体と同じ写像)
    ids = sorted(fused, key=fused.get, reverse=True)[:cand_k]
    rows = index.rows(ids)
    out = []
    rels = {i: max(vec_rel.get(i, 0.0), kw_rel.get(i, 0.0)) for i in ids}
    lo, hi = (min(rels.values()), max(rels.values())) if rels else (0.0, 1.0)
    for i in ids:
        r = rows.get(i)
        if r is None:
            continue
        rel = rels[i]
        rel_norm = (rel - lo) / (hi - lo) if hi > lo else 1.0
        age_days = (ts - (r["last_hit"] or r["created_at"])) / 86400.0
        act = (1.0 - math.exp(-r["hits"] / 3.0)) * math.exp(-max(age_days, 0.0) / 30.0)
        r["relevance"] = round(rel, 4)
        r["cosine"] = round(vec_rel.get(i, 0.0), 4)
        r["activation"] = round(act, 4)
        r["score"] = round(w_relevance * rel_norm + w_activation * act, 4)
        r["duplicates"] = index.dup_count(i)
        out.append(r)
    out.sort(key=lambda d: d["score"], reverse=True)
    out = out[:k]
    if record_hits:
        index.record_hit([d["id"] for d in out], now=ts)
    return out
