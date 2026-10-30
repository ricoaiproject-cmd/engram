"""埋め込み層。本番は Ruri-v3(ローカル・日本語特化)、テストは FakeEmbedder。

実行系は2系統:
- OnnxRuriEmbedder — ONNX Runtime。import+ロードが1〜2秒で軽く、これが既定。
  `engram export-onnx` が生成したモデルディレクトリ(meta.json 付き)を読む。
- RuriEmbedder — sentence-transformers(torch)。import だけで warm 12〜24秒 /
  cold 50秒超かかる(server.py の ENGRAM_PRELOAD コメント参照)。ONNX モデルが
  未生成の環境でのフォールバック、および export-onnx のパリティ検証の基準。
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed_query(self, text: str) -> np.ndarray: ...
    def embed_docs(self, texts: list[str]) -> np.ndarray: ...


def mean_pool_normalize(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """attention mask で重み付き平均プーリングして L2 正規化する。

    sentence-transformers の Pooling(pooling_mode_mean_tokens=True) +
    encode(normalize_embeddings=True) と同じ計算。ONNX 経路でも torch 経路と
    同一のベクトル分布になることをパリティテストで保証する(db の dim 固定と
    既存ベクトルとの互換のため、この関数は安易に変更しないこと)。

    hidden: (batch, seq, dim) の最終隠れ状態
    mask:   (batch, seq) の attention mask(0/1)
    """
    m = mask.astype(np.float32)[:, :, np.newaxis]
    summed = (hidden.astype(np.float32) * m).sum(axis=1)
    counts = np.clip(m.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norms, 1e-12, None)


class RuriEmbedder:
    """cl-nagoya/ruri-v3 系。プレフィックス(検索クエリ:/検索文書:)必須。

    sentence-transformers は重いので遅延ロード。stdio MCP サーバーは常駐
    プロセスなのでロードは初回のみ。
    """

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-130m",
        query_prefix: str = "検索クエリ: ",
        doc_prefix: str = "検索文書: ",
    ) -> None:
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._model = None
        # FastMCP は同期ツールをワーカースレッドで実行する。background 先読みと
        # 初回ツール呼び出しが競合してもモデルを二重ロードしないよう保護する。
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                import contextlib
                import os
                import sys

                # HF Hub への接続確認はネットワーク不調時に無限に待つことがあり、
                # stdio MCP サーバーでは recall がハングしてセッションごと固まる。
                # キャッシュ済みならオフラインで即ロードし、無い時だけ取りに行く。
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
                from sentence_transformers import SentenceTransformer

                # stdio MCP では stdout は JSON-RPC 専用。ライブラリの迷い出力が
                # 混ざるとプロトコルが壊れるため、ロード中は stderr に退避する
                with contextlib.redirect_stdout(sys.stderr):
                    try:
                        self._model = SentenceTransformer(
                            self._model_name, local_files_only=True
                        )
                    except Exception:
                        # キャッシュ未取得の初回のみオンラインでダウンロード
                        self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        # sentence-transformers の新旧バージョン互換
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        return int(getter())

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._load().encode(
            [self._query_prefix + text], normalize_embeddings=True
        )[0]
        return np.asarray(vec, dtype=np.float32)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        vecs = self._load().encode(
            [self._doc_prefix + t for t in texts], normalize_embeddings=True
        )
        return np.asarray(vecs, dtype=np.float32)


class OnnxRuriEmbedder:
    """`engram export-onnx` が生成した ONNX モデルで埋め込む(既定の実行系)。

    torch を一切 import しないため、cold でも数秒で立ち上がる。モデル
    ディレクトリには model.onnx / tokenizer.json / meta.json が必要で、
    meta.json から dim を読むためモデルをロードせずに DB を開ける。
    """

    def __init__(
        self,
        model_dir: Path,
        query_prefix: str = "検索クエリ: ",
        doc_prefix: str = "検索文書: ",
    ) -> None:
        self._dir = Path(model_dir)
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        # (session, tokenizer, input_names) を1つのタプルで持つ。unload と
        # 実行中の _encode が競合しても、_encode はロード結果をローカル変数に
        # 受けてから使うため、参照が生きている限り安全に完走できる
        self._parts: tuple | None = None
        self._meta: dict | None = None
        self._lock = threading.Lock()

    @classmethod
    def is_available(cls, model_dir: Path) -> bool:
        # 実体は config.onnx_model_ready(軽量 import で使えるようそちらに置く)
        from .config import onnx_model_ready

        return onnx_model_ready(model_dir)

    def _load_meta(self) -> dict:
        if self._meta is None:
            with (self._dir / "meta.json").open(encoding="utf-8") as f:
                self._meta = json.load(f)
        return self._meta

    @property
    def dim(self) -> int:
        return int(self._load_meta()["dim"])

    def _load(self) -> tuple:
        """(session, tokenizer, input_names) を返す。

        呼び出し側は戻り値をローカル変数に受けてから使うこと。self の属性を
        直接参照すると、実行中に unload() された瞬間に None を踏む。
        """
        parts = self._parts
        if parts is not None:
            return parts
        with self._lock:
            if self._parts is None:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                meta = self._load_meta()
                tok = Tokenizer.from_file(str(self._dir / "tokenizer.json"))
                tok.enable_truncation(max_length=int(meta["max_seq_length"]))
                tok.enable_padding(
                    pad_id=int(meta["pad_token_id"]),
                    pad_token=str(meta["pad_token"]),
                )

                opts = ort.SessionOptions()
                # stdio MCP では stdout が JSON-RPC 専用のため、ORT の警告類も
                # 出力に混ぜない(3 = ERROR 以上のみ)
                opts.log_severity_level = 3
                session = ort.InferenceSession(
                    str(self._dir / "model.onnx"),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                input_names = [i.name for i in session.get_inputs()]
                self._parts = (session, tok, input_names)
            return self._parts

    @property
    def loaded(self) -> bool:
        """モデルがメモリ上にあるか(アイドル解放の判定に使う)。"""
        return self._parts is not None

    def unload(self) -> bool:
        """モデルをメモリから解放する。解放したら True。

        stdio MCP は常駐プロセスのため、Codex のように1クライアントが複数
        プロセスを起動・残留させる環境では、使われていないモデル(約1.1GB)が
        積み上がる(実機: 26.810 で4プロセス同時)。アイドル時に解放すれば
        次回利用時の再ロード(ONNX は数秒)だけで済み、記憶機能は失われない。
        実行中の _encode とは競合しない(_load の docstring 参照)。
        """
        with self._lock:
            was_loaded = self._parts is not None
            self._parts = None
        return was_loaded

    def _encode(self, texts: list[str]) -> np.ndarray:
        session, tokenizer, input_names = self._load()
        encodings = tokenizer.encode_batch(texts)
        ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)

        feeds: dict[str, np.ndarray] = {}
        for name in input_names:
            if name == "input_ids":
                feeds[name] = ids
            elif name == "attention_mask":
                feeds[name] = mask
            elif name == "token_type_ids":
                feeds[name] = np.zeros_like(ids)
        hidden = session.run(None, feeds)[0]  # last_hidden_state
        vecs = mean_pool_normalize(hidden, mask)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._query_prefix + text])[0]

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._doc_prefix + t for t in texts])


class DaemonEmbedder:
    """窓口用: 埋め込みの計算を共有デーモンに頼み、モデルを自分で持たない。

    背景(R8.10.23 実測): stdio MCP はセッションごとに1プロセス立ち、記憶を
    引いたプロセスがそれぞれ約800MBのモデルを抱えていた(同時に3つ)。
    モデルはデーモン1つに集め、各 MCP はここを通して数値だけを受け取る。

    デーモンが不在なら起動を仕掛けて少し待ち、それでも届かない・旧版で
    埋め込みに未対応・応答が壊れている場合は、手元の実行系(local)で
    計算する(従来どおり)。記憶機能は止めない。手元で計算したときの
    モデルはアイドル解放の対象(loaded / unload は local へ委譲)。
    """

    def __init__(self, settings, local, *, spawn_wait: float = 15.0,
                 retry_after: float = 60.0, request_timeout: float = 60.0
                 ) -> None:
        self._settings = settings
        self._local = local
        self._spawn_wait = spawn_wait
        self._retry_after = retry_after
        self._timeout = request_timeout
        # この時刻まではデーモンを試さず手元で計算する(不通の連打を防ぐ)
        self._skip_until = 0.0
        self.remote_calls = 0
        self.local_calls = 0

    # --- 手元の実行系へ委譲するもの(meta.json から読むのでロードは走らない)
    @property
    def dim(self) -> int:
        return int(self._local.dim)

    @property
    def loaded(self) -> bool:
        return bool(getattr(self._local, "loaded", False))

    def unload(self) -> bool:
        fn = getattr(self._local, "unload", None)
        return bool(fn()) if fn is not None else False

    # --- 埋め込み
    def embed_query(self, text: str) -> np.ndarray:
        vecs = self._remote("query", [text])
        if vecs is None:
            self.local_calls += 1
            return self._local.embed_query(text)
        return vecs[0]

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._remote("docs", list(texts))
        if vecs is None:
            self.local_calls += 1
            return self._local.embed_docs(texts)
        return vecs

    def _remote(self, kind: str, texts: list[str]) -> np.ndarray | None:
        import sys
        import time

        from . import daemon

        now = time.monotonic()
        if now < self._skip_until:
            return None
        raw = daemon.embed(self._settings, kind, texts, timeout=self._timeout)
        if raw is None:
            caps = daemon.capabilities(self._settings)
            if caps is not None and "embed" not in caps:
                # 旧版のデーモンが常駐している。入れ替わるまで手元で計算する
                self._skip_until = now + 10 * self._retry_after
                print("engram: daemon is an older version without embed; "
                      "embedding locally", file=sys.stderr)
                return None
            if caps is None:
                daemon.ensure_running(self._settings)
                deadline = time.monotonic() + self._spawn_wait
                while time.monotonic() < deadline:
                    caps = daemon.capabilities(self._settings)
                    if caps is not None:
                        break
                    time.sleep(0.25)
            if caps is not None and "embed" in caps:
                raw = daemon.embed(self._settings, kind, texts,
                                   timeout=self._timeout)
        if raw is None:
            self._skip_until = time.monotonic() + self._retry_after
            print("engram: daemon unavailable; embedding locally",
                  file=sys.stderr)
            return None
        vecs = np.asarray(raw, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape != (len(texts), self.dim):
            self._skip_until = time.monotonic() + self._retry_after
            print(f"engram: daemon returned shape {vecs.shape}; "
                  "embedding locally", file=sys.stderr)
            return None
        self.remote_calls += 1
        return vecs


def make_embedder(settings) -> Embedder:
    """settings.embed_backend に従って実行系を選ぶ(build_engine から使う)。

    auto  — ONNX モデルが生成済みならそれを使い、無ければ torch にフォールバック
    onnx  — ONNX を強制。モデル未生成ならエラー(暗黙の torch 起動 12〜24秒を防ぐ)
    torch — sentence-transformers を強制(export-onnx のパリティ基準もこれ)
    """
    backend = getattr(settings, "embed_backend", "auto")
    onnx_dir = settings.onnx_model_dir
    if backend not in ("auto", "onnx", "torch"):
        raise ValueError(
            f"embed_backend が不正です: {backend!r} (auto | onnx | torch)"
        )
    if backend in ("auto", "onnx") and OnnxRuriEmbedder.is_available(onnx_dir):
        return OnnxRuriEmbedder(
            onnx_dir,
            query_prefix=settings.query_prefix,
            doc_prefix=settings.doc_prefix,
        )
    if backend == "onnx":
        raise FileNotFoundError(
            f"ONNX モデルがありません: {onnx_dir}\n"
            "`engram export-onnx` で生成してください(追加の依存は不要)"
        )
    return RuriEmbedder(
        model_name=settings.embed_model,
        query_prefix=settings.query_prefix,
        doc_prefix=settings.doc_prefix,
    )


class FakeEmbedder:
    """テスト用の決定的埋め込み。文字 n-gram (1..3) の feature hashing。

    部分文字列を共有するテキスト同士はベクトルも近くなるため、
    類似度に依存するテスト(重複検知・近傍検索)がモデル無しで書ける。
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for n in (1, 2, 3):
            for i in range(len(text) - n + 1):
                gram = text[i : i + n]
                h = int.from_bytes(
                    hashlib.md5(gram.encode("utf-8")).digest()[:4], "little"
                )
                vec[h % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts])
