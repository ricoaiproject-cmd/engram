"""埋め込み層。本番は Ruri-v3(ローカル・日本語特化)、テストは FakeEmbedder。"""

from __future__ import annotations

import hashlib
import threading
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed_query(self, text: str) -> np.ndarray: ...
    def embed_docs(self, texts: list[str]) -> np.ndarray: ...


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
