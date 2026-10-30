"""媒体の索引係(別 venv で動く重い側)。`engram-media` コマンドの実体。

本体(Ruri・sentence-transformers 5.x)とは別の venv に `pip install engram[media]` で入れる。
EmbeddingGemma 2(文字・画像・音声・動画を同じ空間に写す)は sentence-transformers 6.1 以上を
要るため、本体の環境と同居させない。本体は media.py から TCP で問い合わせるだけ。

役割:
- 復号: 画像(そのまま)/動画(1秒1コマ・最大32コマ・約60秒ごとの段)/音声(16kHz・約5分ごと)/
  文書(PDF はページ画像+抽出文字。Office は soffice で PDF 化してから同じ道)
- 走査と差分索引: media_roots を歩き、sha256 と mtime の変化だけ埋め込み直す。消えたファイルの行は消す
- 門: min_bytes 未満の画像を取らない。近似重複(cos>=dup_threshold)は dup_of を付ける
- 説明文: Ollama の vision モデルで短い日本語の説明(固有名詞・数値を書かない)。不達なら付けない
- 常駐: 127.0.0.1:media_port の行区切り JSON(本体 daemon.py と同じ型)。op=ping/stop/status/
  embed_query/find/index/reindex_path
- CLI: engram-media daemon|index|find|stats|uninstall-index

依存の読み込みはすべて遅延 import(テストは FakeMediaEmbedder で部品なしに回す)。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Settings, get_settings, resolve_room
from .media_schema import (DEFAULT_MEDIA_DIM, DEFAULT_MEDIA_MODEL, MediaIndex, hybrid_find)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
DOC_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls", ".odt", ".odp", ".ods"}


def is_excluded(rel_parts, patterns) -> bool:
    """相対パスの各階層のフォルダ名が除外パターン(fnmatch)に当たれば True(配下ごと除外)。"""
    import fnmatch

    for part in rel_parts[:-1]:
        for pat in patterns:
            if fnmatch.fnmatch(part, pat):
                return True
    return False
PLACEHOLDER = {"image": "<|image|>", "video": "<|video|>", "audio": "<|audio|>"}
QUERY_TMPL = "task: search result | query: {q}"
DOC_TMPL = "title: {title} | text: {body}"
VIDEO_FPS = 1
VIDEO_MAX_FRAMES = 32
AUDIO_SR = 16000
DOC_MAX_PAGES = 40          # 1文書で埋め込むページの上限(超えた分は文字だけ)
DOC_RENDER_ZOOM = 1.5
_RECV_LIMIT = 256 * 1024


def kind_of(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in DOC_EXT:
        return "document"
    return None


def sha256_of(path: Path, *, head_only_bytes: int = 64 * 1024 * 1024) -> str:
    """ファイルの指紋。巨大ファイルは先頭 64MB+サイズで代用(差分検知が目的で厳密性は不要)。"""
    h = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as f:
        remaining = min(size, head_only_bytes)
        while remaining > 0:
            chunk = f.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    h.update(str(size).encode())
    return h.hexdigest()


def item_id(path: str, segment: int) -> str:
    return hashlib.sha1(f"{path}\x00{segment}".encode("utf-8")).hexdigest()[:26]


def log(settings: Settings, message: str) -> None:
    try:
        p = settings.media_dir / "media.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()}] {message}\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 埋め込み
# ---------------------------------------------------------------------------

class GemmaMediaEmbedder:
    """EmbeddingGemma 2(sentence-transformers 6.1+)。CPU/GPU 自動。float32(fp16 は NaN になる)。"""

    def __init__(self, model_id: str = DEFAULT_MEDIA_MODEL, *, device: str | None = None,
                 cache_dir: Path | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if cache_dir is not None:
            os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        self.model_id = model_id
        self.device = dev
        self._m = SentenceTransformer(model_id, device=dev, model_kwargs={"torch_dtype": dtype})
        self.dim = int(self._m.get_sentence_embedding_dimension() or DEFAULT_MEDIA_DIM)
        self._lock = threading.Lock()

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            v = self._m.encode(QUERY_TMPL.format(q=text), normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32).reshape(-1)

    def embed_doc(self, payload: dict) -> np.ndarray:
        """payload = {"text": "title: … | text: <|image|>", "image": [...]} 等(sentence-transformers の多モーダル入力)。"""
        with self._lock:
            v = self._m.encode(payload, normalize_embeddings=True)
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if not np.isfinite(v).all():
            raise ValueError("embedding has NaN/inf")
        return v


class FakeMediaEmbedder:
    """テスト用。文字列(payload の text+パス)の決定的ハッシュベクトル。部品なしで動く。"""

    model_id = "fake-media"
    device = "cpu"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vec(self, s: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for n in (1, 2, 3):
            for i in range(len(s) - n + 1):
                h = int(hashlib.md5(s[i:i + n].encode("utf-8")).hexdigest(), 16)
                v[h % self.dim] += 1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def embed_doc(self, payload: dict) -> np.ndarray:
        s = payload.get("text", "")
        for key in ("image", "video", "audio"):
            val = payload.get(key)
            if isinstance(val, list):
                s += " ".join(str(x) for x in val)
            elif isinstance(val, dict):
                s += key + str(val.get("array", np.zeros(1)).shape)
            elif val is not None:
                s += str(val)
        return self._vec(s)


# ---------------------------------------------------------------------------
# 復号(ファイル → 埋め込み入力の列)
# ---------------------------------------------------------------------------

@dataclass
class Piece:
    """1埋め込みぶんの入力。segment は時間の段/ページ番号(0 始まり)。"""
    payload: dict
    segment: int = 0
    page: int | None = None
    duration: float | None = None
    text: str | None = None
    thumb_png: bytes | None = None   # 説明文づくりに渡す縮小画像(無ければ元ファイル)


def _doc_text(title: str, placeholder: str) -> str:
    return DOC_TMPL.format(title=title or "none", body=placeholder)


def pieces_for_image(path: Path) -> list[Piece]:
    if path.suffix.lower() == ".gif":
        # アニメGIFは複数コマの配列になり画像処理が落ちる(R8.10.10 実測: too many values to unpack)。
        # 先頭のコマだけを RGB の PNG にして渡す
        from PIL import Image

        with Image.open(path) as im:
            im.seek(0)
            rgb = im.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="PNG")
        return [Piece(payload={"text": _doc_text("none", PLACEHOLDER["image"]), "image": [_PngBlob(buf.getvalue())]},
                      thumb_png=buf.getvalue())]
    return [Piece(payload={"text": _doc_text("none", PLACEHOLDER["image"]), "image": [str(path)]})]


def load_video_frames(path: Path, *, start: float, end: float, fps: int = VIDEO_FPS,
                      max_frames: int = VIDEO_MAX_FRAMES) -> dict | None:
    """PyAV で [start, end) 秒の範囲を 1秒1コマ(最大 max_frames)で取り出し VideoDict にする。

    torchcodec は ffmpeg の共有DLLが無い環境では読めない(R8.10.8 実測)ため復号は自前で行う。
    """
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        if start > 0:
            c.seek(int(start / float(stream.time_base)), stream=stream, backward=True, any_frame=False)
        next_t = start
        for fr in c.decode(stream):
            t = float(fr.pts * stream.time_base) if fr.pts is not None else None
            if t is None:
                continue
            if t >= end:
                break
            if t + 1e-6 >= next_t:
                frames.append(fr.to_ndarray(format="rgb24"))
                next_t += 1.0 / fps
    if not frames:
        return None
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    arr = np.stack(frames)
    n = arr.shape[0]
    return {"array": arr, "video_metadata": {"fps": fps, "total_num_frames": n, "duration": n / fps,
                                             "frames_indices": list(range(n))}}


def media_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=False).stdout
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return None


def pieces_for_video(path: Path, *, segment_sec: float) -> list[Piece]:
    dur = media_duration(path) or 0.0
    n_seg = max(1, int(np.ceil(dur / segment_sec))) if dur else 1
    out = []
    for s in range(n_seg):
        start, end = s * segment_sec, (s + 1) * segment_sec if dur else 1e9
        vd = load_video_frames(path, start=start, end=end)
        if vd is None:
            continue
        thumb = _png_bytes(vd["array"][len(vd["array"]) // 2]) if s == 0 else None
        out.append(Piece(payload={"text": _doc_text("none", PLACEHOLDER["video"]), "video": vd},
                         segment=s, duration=min(segment_sec, dur - start) if dur else None, thumb_png=thumb))
    return out


def pieces_for_audio(path: Path, *, segment_sec: float, sr: int = AUDIO_SR) -> list[Piece]:
    """librosa で 16kHz モノラル。torchcodec 経由の読み込みを避けるため配列で渡す。"""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    total = len(y) / sr
    n_seg = max(1, int(np.ceil(total / segment_sec)))
    out = []
    for s in range(n_seg):
        a = y[int(s * segment_sec * sr): int((s + 1) * segment_sec * sr)]
        if len(a) < sr // 10:
            continue
        out.append(Piece(payload={"text": _doc_text("none", PLACEHOLDER["audio"]),
                                  "audio": {"array": np.asarray(a, dtype=np.float32), "sampling_rate": sr}},
                         segment=s, duration=len(a) / sr))
    return out


def _png_bytes(rgb: np.ndarray, max_side: int = 768) -> bytes:
    from PIL import Image

    im = Image.fromarray(rgb)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def office_to_pdf(path: Path, out_dir: Path) -> Path | None:
    """LibreOffice(soffice)で PDF 化。無ければ None(文書は文字抽出なしで飛ばす)。"""
    exe = "soffice"
    for cand in (os.environ.get("SOFFICE"),
                 r"C:\Program Files\LibreOffice\program\soffice.exe",
                 r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"):
        if cand and Path(cand).is_file():
            exe = cand
            break
    try:
        subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(path)],
                       capture_output=True, timeout=180, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    pdf = out_dir / (path.stem + ".pdf")
    return pdf if pdf.is_file() else None


def pieces_for_document(path: Path, *, max_pages: int = DOC_MAX_PAGES) -> list[Piece]:
    """PDF/Office → ページごとの画像(見た目)+抽出文字。"""
    import pymupdf

    tmpdir = None
    pdf_path = path
    if path.suffix.lower() != ".pdf":
        tmpdir = tempfile.TemporaryDirectory(prefix="engram-media-")
        pdf_path = office_to_pdf(path, Path(tmpdir.name)) or path
        if pdf_path == path:
            return []
    out: list[Piece] = []
    try:
        doc = pymupdf.open(str(pdf_path))
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if i < max_pages:
                pix = page.get_pixmap(matrix=pymupdf.Matrix(DOC_RENDER_ZOOM, DOC_RENDER_ZOOM), alpha=False)
                png = pix.tobytes("png")
                out.append(Piece(payload={"text": _doc_text(path.stem, PLACEHOLDER["image"]),
                                          "image": [_PngBlob(png)]},
                                 segment=i, page=i + 1, text=text[:4000] or None, thumb_png=png))
            else:
                # 上限を超えたページは文字だけ(埋め込みは見出し文字で代用)
                out.append(Piece(payload={"text": DOC_TMPL.format(title=path.stem, body=text[:2000] or path.stem)},
                                 segment=i, page=i + 1, text=text[:4000] or None))
        doc.close()
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()
    return out


class _PngBlob:
    """sentence-transformers の image 入力に PIL.Image を渡すための遅延ラッパ。"""

    def __init__(self, png: bytes) -> None:
        self.png = png

    def to_pil(self):
        from PIL import Image

        return Image.open(io.BytesIO(self.png)).convert("RGB")


def _materialize(payload: dict) -> dict:
    imgs = payload.get("image")
    if imgs:
        payload = dict(payload)
        payload["image"] = [x.to_pil() if isinstance(x, _PngBlob) else x for x in imgs]
    return payload


def pieces_for(path: Path, kind: str, settings: Settings) -> list[Piece]:
    if kind == "image":
        return pieces_for_image(path)
    if kind == "video":
        return pieces_for_video(path, segment_sec=settings.media_video_segment_sec)
    if kind == "audio":
        return pieces_for_audio(path, segment_sec=settings.media_audio_segment_sec)
    if kind == "document":
        return pieces_for_document(path, max_pages=settings.media_doc_max_pages)
    return []


# ---------------------------------------------------------------------------
# 説明文(ローカルLLM・vision)
# ---------------------------------------------------------------------------

CAPTION_SYSTEM = ("あなたは画像の司書です。画像を日本語1文(40字以内)で説明します。"
                  "種類(スライド・写真・画面写真・図解・ポスター等)と主な被写体・色を述べ、"
                  "人名・会社名・数値・メールアドレス等の固有情報は書きません。")


class Captioner:
    def __init__(self, settings: Settings) -> None:
        from .localllm import LocalLLM

        llm = LocalLLM.from_settings(settings)
        if settings.media_caption_model:
            llm = LocalLLM(llm.url, settings.media_caption_model, timeout=llm.timeout)
        self._llm = llm
        self.enabled = settings.media_caption and llm.available(timeout=2.0)

    def caption(self, piece: Piece, path: Path) -> str | None:
        if not self.enabled:
            return None
        png = piece.thumb_png
        if png is None and kind_of(path) == "image":
            try:
                from PIL import Image

                im = Image.open(path).convert("RGB")
                im.thumbnail((768, 768))
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                png = buf.getvalue()
            except Exception:
                return None
        if png is None:
            return None
        text = self._llm.chat(CAPTION_SYSTEM, "この画像を説明してください。", temperature=0.1,
                              num_predict=80, images=[base64.b64encode(png).decode("ascii")])
        if not text:
            return None
        text = text.strip().splitlines()[0].strip()
        return text[:120] or None


# ---------------------------------------------------------------------------
# 索引係
# ---------------------------------------------------------------------------

@dataclass
class IndexReport:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    skipped: int = 0
    dups: int = 0
    failed: int = 0
    seconds: float = 0.0
    errors: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("scanned", "added", "updated", "removed", "skipped",
                                               "dups", "failed", "seconds")} | {"errors": self.errors[:20]}


class Indexer:
    def __init__(self, settings: Settings, embedder, index: MediaIndex, *, captioner=None) -> None:
        self.settings = settings
        self.embedder = embedder
        self.index = index
        self.captioner = captioner
        self.progress: dict = {"state": "idle"}

    # -- 走査 --
    def scan_root(self, root: Path) -> list[Path]:
        kinds = set(self.settings.media_kinds)
        out = []
        patterns = list(self.settings.media_exclude_dirs)
        for p in root.rglob("*"):
            if is_excluded(p.relative_to(root).parts, patterns):
                continue
            if not p.is_file():
                continue
            k = kind_of(p)
            if k is None or k not in kinds:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if k == "image" and size < self.settings.media_min_bytes:
                continue
            out.append(p)
        return out

    def room_for(self, path: Path, root: Path, root_room: str) -> str:
        return root_room or resolve_room(path, self.settings.room_paths)

    # -- 1ファイル --
    def index_file(self, path: Path, *, root: Path, room: str, source: str = "folder",
                   source_meta: dict | None = None, report: IndexReport | None = None) -> list[str]:
        kind = kind_of(path)
        if kind is None:
            return []
        sha = sha256_of(path)
        st = path.stat()
        pieces = pieces_for(path, kind, self.settings)
        ids: list[str] = []
        old_ids = set(self.index.ids_for_path(str(path)))
        for piece in pieces:
            vec = self.embedder.embed_doc(_materialize(piece.payload))
            iid = item_id(str(path), piece.segment)
            dup_of = self._find_dup(vec, exclude_path=str(path)) if kind == "image" and piece.segment == 0 else None
            caption = None
            if self.captioner is not None and (piece.segment == 0 or kind == "document"):
                prev = self.index.get(iid)
                caption = prev["caption"] if prev and prev.get("sha256") == sha and prev.get("caption") else \
                    self.captioner.caption(piece, path)
            self.index.upsert(
                id=iid, path=str(path), root=str(root), room=room, kind=kind, embedding=vec,
                segment=piece.segment, page=piece.page, sha256=sha, bytes=st.st_size, mtime=st.st_mtime,
                duration=piece.duration, caption=caption, text=piece.text, source=source,
                source_meta=source_meta, model=getattr(self.embedder, "model_id", None), dup_of=dup_of)
            ids.append(iid)
            if report is not None and dup_of:
                report.dups += 1
        stale = list(old_ids - set(ids))
        if stale:
            self.index.delete(stale)
        return ids

    def _find_dup(self, vec: np.ndarray, *, exclude_path: str) -> str | None:
        hits = self.index.vector_search(vec, 3, kinds=["image"], include_dups=False)
        for iid, cos in hits:
            if cos >= self.settings.media_dup_threshold:
                row = self.index.get(iid)
                if row and row["path"] != exclude_path:
                    return iid
        return None

    # -- 全体 --
    def run(self, *, roots: dict[str, str] | None = None, full: bool = False) -> IndexReport:
        t0 = time.time()
        rep = IndexReport()
        roots = roots if roots is not None else dict(self.settings.media_roots)
        kinds = set(self.settings.media_kinds)
        self.progress = {"state": "running", "started": t0, "done": 0, "total": 0}
        for root_s, room in roots.items():
            root = Path(root_s).expanduser()
            if not root.is_dir():
                rep.errors.append(f"root not found: {root}")
                continue
            files = self.scan_root(root)
            rep.scanned += len(files)
            self.progress["total"] += len(files)
            known = self.index.known_files(str(root))
            seen: set[str] = set()
            for p in files:
                sp = str(p)
                seen.add(sp)
                try:
                    st = p.stat()
                    prev = known.get(sp)
                    if prev is not None and not full:
                        prev_sha, prev_mtime, _ = prev
                        if prev_mtime is not None and abs(prev_mtime - st.st_mtime) < 1e-6:
                            rep.skipped += 1
                            self.progress["done"] += 1
                            continue
                        if prev_sha is not None and prev_sha == sha256_of(p):
                            rep.skipped += 1
                            self.progress["done"] += 1
                            continue
                    ids = self.index_file(p, root=root, room=room or "common", report=rep)
                    if not ids:
                        rep.skipped += 1
                    elif prev is None:
                        rep.added += 1
                    else:
                        rep.updated += 1
                except Exception as exc:  # 1件の失敗で全体を止めない
                    rep.failed += 1
                    rep.errors.append(f"{p.name}: {type(exc).__name__}: {str(exc)[:160]}")
                    log(self.settings, f"index failed {p}: {exc!r}")
                self.progress["done"] += 1
            # 消えたファイル。media_kinds を一時的に絞った(例: CPU機で動画を外した)ときに、走査対象外の
            # 種別の行まで「消えた」と誤判定して落とさないよう、今回の対象種別の行だけを見る
            # (R8.10.8 実測: 動画を外して再開したら動画117段が削除された)
            patterns = list(self.settings.media_exclude_dirs)
            gone = [ids for path, (_, _, ids) in known.items()
                    if path not in seen and kind_of(Path(path)) in kinds
                    and (not Path(path).is_file()
                         or is_excluded(Path(path).relative_to(root).parts, patterns)
                         or (kind_of(Path(path)) == "image" and Path(path).stat().st_size < self.settings.media_min_bytes))]
            for ids in gone:
                rep.removed += self.index.delete(ids)
        rep.seconds = round(time.time() - t0, 1)
        self.progress = {"state": "idle", "last": rep.as_dict(), "finished": time.time()}
        log(self.settings, f"index done {rep.as_dict()}")
        return rep


# ---------------------------------------------------------------------------
# 常駐サーバ
# ---------------------------------------------------------------------------

class _State:
    def __init__(self, settings: Settings, *, embedder=None) -> None:
        self.settings = settings
        self.last_seen = time.time()
        self.embedder = embedder or GemmaMediaEmbedder(cache_dir=settings.media_dir / "hf_cache")
        self.index = MediaIndex.open(settings.db_path, dim=self.embedder.dim,
                                     model=getattr(self.embedder, "model_id", None))
        cap = None
        try:
            cap = Captioner(settings)
            if not cap.enabled:
                cap = None
        except Exception:
            cap = None
        self.indexer = Indexer(settings, self.embedder, self.index, captioner=cap)
        self._index_thread: threading.Thread | None = None
        # sqlite3 の接続は複数スレッドで同時に使うと行の形が崩れる(R8.10.11 実測: 索引中の status で
        # IndexError)。索引スレッド用(self.index)と、問い合わせ用(self.index_q)で接続を分け、
        # 書き込みを伴う索引は lock で直列化する
        self.index_q = MediaIndex.open(settings.db_path)
        self.lock = threading.Lock()
        # 問い合わせ用の接続も、窓口が複数(ThreadingTCPServer)なので同時に使われる。
        # 接続1本を2スレッドで同時に使うと行の形が崩れるため、問い合わせも直列化する
        # (R8.10.23 実測: find が dup_count で TypeError: 'NoneType' object is not subscriptable)
        self.qlock = threading.RLock()

    def find(self, q: str, k: int, kinds, rooms, tiers, record_hits: bool) -> list[dict]:
        self.last_seen = time.time()
        qvec = self.embedder.embed_query(q) if q.strip() else None
        with self.qlock:
            return hybrid_find(self.index_q, qvec=qvec, query=q, k=k, kinds=kinds or None, rooms=rooms or None,
                               tiers=tiers or ["hot"], record_hits=record_hits)

    def stats(self) -> dict:
        with self.qlock:
            return self.index_q.stats()

    def demote(self, days: float) -> int:
        with self.qlock:
            return self.index_q.demote_cold(older_than_days=days)

    def start_index(self, roots, full: bool) -> dict:
        self.last_seen = time.time()
        if self._index_thread is not None and self._index_thread.is_alive():
            return {"ok": False, "error": "already running", "progress": self.indexer.progress}
        def _run() -> None:
            with self.lock:
                self.indexer.run(roots=roots, full=full)

        self._index_thread = threading.Thread(target=_run, daemon=True)
        self._index_thread.start()
        return {"ok": True, "started": True}

    def index_path(self, path: str, room: str | None, source: str, source_meta: dict | None) -> dict:
        self.last_seen = time.time()
        p = Path(path)
        if not p.is_file():
            return {"ok": False, "error": "not a file"}
        room = room or resolve_room(p, self.settings.room_paths)
        root = p.parent
        for r in self.settings.media_roots:
            try:
                p.relative_to(Path(r).expanduser())
                root = Path(r).expanduser()
                break
            except ValueError:
                continue
        if not self.lock.acquire(blocking=False):
            return {"ok": False, "error": "索引作成中のため後で(夜間の差分索引が終わると拾われる)"}
        try:
            ids = self.indexer.index_file(p, root=root, room=room, source=source, source_meta=source_meta)
        finally:
            self.lock.release()
        with self.qlock:
            return {"ok": True, "ids": ids, "items": [self.index_q.get(i) for i in ids]}


class _Handler(socketserver.StreamRequestHandler):
    timeout = 120

    def handle(self) -> None:
        try:
            raw = self.rfile.readline(_RECV_LIMIT)
            if not raw:
                return
            req = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            self._send({"error": "bad request"})
            return
        state: _State = self.server.state  # type: ignore[attr-defined]
        op = req.get("op", "find")
        # 別の記憶(data_dir)の窓口からの依頼は、停止も含めて何もしない。
        # 身元を名乗らない旧版の窓口の依頼は従来どおり受ける
        asked = req.get("home")
        if asked is not None and asked != state.settings.home_key:
            self._send({"error": "foreign home"})
            return
        try:
            if op == "ping":
                self._send({"ok": True, "pid": os.getpid(), "model": state.embedder.model_id,
                            "device": state.embedder.device})
            elif op == "stop":
                self._send({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            elif op == "status":
                self._send({"ok": True, "progress": state.indexer.progress, "stats": state.stats(),
                            "caption": state.indexer.captioner is not None})
            elif op == "embed_query":
                v = state.embedder.embed_query(str(req.get("q", "")))
                self._send({"ok": True, "vector": v.tolist()})
            elif op == "find":
                hits = state.find(str(req.get("q", "")), int(req.get("k", 5)), req.get("kinds"),
                                  req.get("rooms"), req.get("tiers"), bool(req.get("record_hits", True)))
                self._send({"ok": True, "hits": hits})
            elif op == "index":
                self._send(state.start_index(req.get("roots"), bool(req.get("full", False))))
            elif op == "index_path":
                self._send(state.index_path(str(req.get("path", "")), req.get("room"),
                                            str(req.get("source", "manual")), req.get("source_meta")))
            elif op == "demote":
                n = state.demote(float(req.get("days", state.settings.media_cold_days)))
                self._send({"ok": True, "demoted": n})
            else:
                self._send({"error": f"unknown op {op}"})
        except Exception as exc:
            import traceback

            log(state.settings, f"op={op} failed: {traceback.format_exc()[-800:]}")
            self._send({"error": f"{type(exc).__name__}: {exc}"})

    def _send(self, payload: dict) -> None:
        # どの応答にも身元を添える(窓口はこれで自分の索引係かを見分ける)
        payload = {**payload, "home": self.server.state.settings.home_key}  # type: ignore[attr-defined]
        try:
            self.wfile.write((json.dumps(payload, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
        except OSError:
            pass


class _Server(socketserver.ThreadingTCPServer):
    # Windows の SO_REUSEADDR は「使用中のポートへの重ね bind」まで許してしまい、
    # 索引係が二重に立つ(R8.10.23 共有デーモンで実測)。Windows では無効にする
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def port_file(settings: Settings) -> Path:
    return settings.media_dir / "daemon.json"


def serve(settings: Settings | None = None, *, embedder=None, idle_hours: float = 6.0) -> int:
    settings = settings or get_settings()
    if settings.media_port_effective:
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("127.0.0.1", settings.media_port_effective))
            probe.close()
        except OSError:
            log(settings, "既に起動済みのため終了(二重起動の抑止)")
            return 0
    log(settings, "起動開始(埋め込みモデルの読み込み)")
    try:
        state = _State(settings, embedder=embedder)
    except Exception as exc:
        log(settings, f"起動失敗: {type(exc).__name__}: {exc}")
        return 1
    server = _Server(("127.0.0.1", settings.media_port_effective), _Handler)
    server.state = state  # type: ignore[attr-defined]
    host, port = server.server_address
    pf = port_file(settings)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps({"port": port, "pid": os.getpid(), "started_at": time.time()}), encoding="utf-8")
    idle = max(60.0, idle_hours * 3600.0)

    def _watch() -> None:
        while True:
            time.sleep(30)
            busy = state._index_thread is not None and state._index_thread.is_alive()
            if not busy and time.time() - state.last_seen > idle:
                log(settings, "無操作のため終了")
                server.shutdown()
                return

    threading.Thread(target=_watch, daemon=True).start()
    log(settings, f"待受開始 port={port} model={state.embedder.model_id} device={state.embedder.device}")
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        try:
            pf.unlink()
        except OSError:
            pass
        log(settings, "終了")
    return 0


# ---------------------------------------------------------------------------
# CLI(engram-media)
# ---------------------------------------------------------------------------

def _print(obj, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps(obj, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="engram-media", description="媒体の索引係(別 venv)")
    ap.add_argument("--json", dest="as_json", action="store_true")
    ap.add_argument("--fake-embedder", action="store_true", help="部品なしで動かす(テスト用)")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("daemon", help="常駐して本体からの問い合わせに答える(前面)")
    p = sub.add_parser("index", help="media_roots を走査して差分を索引化(常駐係を使わず直接)")
    p.add_argument("--full", action="store_true", help="差分判定を無視して全部埋め込み直す")
    p.add_argument("--root", action="append", help="このフォルダだけ(複数可)。部屋は設定から")
    p = sub.add_parser("find", help="検索")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--kind", action="append")
    p.add_argument("--room", action="append")
    p.add_argument("--all-tiers", action="store_true")
    sub.add_parser("stats", help="索引の件数")
    p = sub.add_parser("uninstall-index", help="媒体の表(media_*)だけを落とす。記憶本体は無変更")
    p.add_argument("--yes", action="store_true")
    p = sub.add_parser("mail-fetch", help="メール添付の取り込み(Gmail 読み取り専用。engram[mail] が必要)")
    p.add_argument("--account", required=True)
    p.add_argument("--auth", action="store_true")
    p.add_argument("--max", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-index", action="store_true")
    a = ap.parse_args(argv)

    settings = get_settings()
    if a.command == "mail-fetch":
        from .media_mail import fetch

        rep = fetch(settings, a.account, interactive=a.auth, max_messages=a.max, dry_run=a.dry_run,
                    index=not a.no_index)
        _print(rep, True)
        return
    if a.command == "daemon":
        emb = FakeMediaEmbedder() if a.fake_embedder else None
        sys.exit(serve(settings, embedder=emb))
    if a.command == "stats":
        idx = MediaIndex.open(settings.db_path)
        _print(idx.stats(), a.as_json)
        return
    if a.command == "uninstall-index":
        if not a.yes:
            print("媒体の表(media_items / vec_media / fts_media / media_meta)を落とします。--yes を付けて実行してください。")
            sys.exit(2)
        idx = MediaIndex.open(settings.db_path)
        idx.drop_all()
        print("媒体の表を落としました(記憶本体は無変更)")
        return

    emb = FakeMediaEmbedder() if a.fake_embedder else GemmaMediaEmbedder(cache_dir=settings.media_dir / "hf_cache")
    idx = MediaIndex.open(settings.db_path, dim=emb.dim, model=getattr(emb, "model_id", None))
    if a.command == "index":
        cap = None
        try:
            cap = Captioner(settings)
            cap = cap if cap.enabled else None
        except Exception:
            cap = None
        roots = None
        if a.root:
            roots = {r: settings.media_roots.get(r) or resolve_room(Path(r), settings.room_paths) for r in a.root}
        rep = Indexer(settings, emb, idx, captioner=cap).run(roots=roots, full=a.full)
        _print(rep.as_dict(), a.as_json)
        return
    if a.command == "find":
        qvec = emb.embed_query(a.query)
        hits = hybrid_find(idx, qvec=qvec, query=a.query, k=a.k, kinds=a.kind, rooms=a.room,
                           tiers=None if a.all_tiers else ["hot"], record_hits=False)
        if a.as_json:
            _print(hits, True)
        else:
            for h in hits:
                print(f"{h['score']:.3f} cos={h['cosine']:.3f} [{h['kind']}/{h['room']}] {h['path']}"
                      + (f" p.{h['page']}" if h.get('page') else "")
                      + (f"  — {h['caption']}" if h.get('caption') else ""))


if __name__ == "__main__":
    main()
