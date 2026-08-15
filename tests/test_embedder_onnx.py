"""ONNX 埋め込み実行系のテスト(実モデル不使用)。

- mean_pool_normalize: torch 経路と同一のプーリング計算(純関数)の性質を検証
- make_embedder: embed_backend の選択ロジックを tmp_path Settings で検証
  (RuriEmbedder は isinstance 確認のみ。embed は呼ばない = モデルDLなし)
- ENGRAM_EMBED_BACKEND 環境変数が get_settings に届くことを検証
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from engram import config as cfg
from engram.config import Settings
from engram.embedder import (
    OnnxRuriEmbedder,
    RuriEmbedder,
    make_embedder,
    mean_pool_normalize,
)


# ---------------------------------------------------------------------------
# mean_pool_normalize(純関数)
# ---------------------------------------------------------------------------

class TestMeanPoolNormalize:
    def test_masked_mean_hand_computed(self):
        """マスク=1 のトークンだけの平均になること(手計算例)。

        hidden: (1, 3, 2)、mask=[1,1,0] → 3番目のトークンは無視され、
        平均は ([1,2] + [3,4]) / 2 = [2,3]。L2 正規化して [2,3]/sqrt(13)。
        """
        hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        mask = np.array([[1, 1, 0]])
        out = mean_pool_normalize(hidden, mask)
        expected = np.array([2.0, 3.0]) / np.sqrt(13.0)
        np.testing.assert_allclose(out[0], expected, rtol=1e-6)

    def test_output_is_l2_normalized(self):
        rng = np.random.default_rng(42)
        hidden = rng.normal(size=(4, 7, 16)).astype(np.float32)
        mask = np.ones((4, 7), dtype=np.int64)
        mask[2, 4:] = 0  # 一部パディングあり
        out = mean_pool_normalize(hidden, mask)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_all_zero_mask_does_not_crash_or_nan(self):
        """attention mask が全ゼロでも NaN やゼロ除算にならない(防御の検証)。"""
        hidden = np.ones((2, 3, 4), dtype=np.float32)
        mask = np.zeros((2, 3), dtype=np.int64)
        out = mean_pool_normalize(hidden, mask)
        assert out.shape == (2, 4)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_batch_shape(self):
        """(B, S, D) → (B, D) の形状変換。"""
        B, S, D = 5, 11, 8
        hidden = np.zeros((B, S, D), dtype=np.float32)
        hidden[:, :, 0] = 1.0
        mask = np.ones((B, S), dtype=np.int64)
        out = mean_pool_normalize(hidden, mask)
        assert out.shape == (B, D)


# ---------------------------------------------------------------------------
# make_embedder の選択ロジック
# ---------------------------------------------------------------------------

def _make_settings(tmp_path, backend: str) -> Settings:
    return Settings(
        data_dir=tmp_path,
        memories_dir=tmp_path / "memories",
        embed_backend=backend,
    )


def _fabricate_onnx_dir(settings: Settings) -> None:
    """export-onnx 済みに見えるディレクトリを捏造する(中身は空でよい。
    make_embedder はファイルの存在と meta.json しか見ないため)。"""
    d = settings.onnx_model_dir
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"")
    (d / "tokenizer.json").write_bytes(b"")
    (d / "meta.json").write_text(
        json.dumps({
            "dim": 512,
            "max_seq_length": 8192,
            "pad_token_id": 3,
            "pad_token": "<pad>",
        }),
        encoding="utf-8",
    )


class TestMakeEmbedder:
    def test_backend_torch_returns_ruri(self, tmp_path):
        s = _make_settings(tmp_path, "torch")
        emb = make_embedder(s)
        assert isinstance(emb, RuriEmbedder)  # embed は呼ばない(モデルDL回避)

    def test_auto_without_onnx_falls_back_to_torch(self, tmp_path):
        s = _make_settings(tmp_path, "auto")
        assert not s.onnx_model_dir.is_dir()
        emb = make_embedder(s)
        assert isinstance(emb, RuriEmbedder)

    def test_auto_with_onnx_dir_returns_onnx(self, tmp_path):
        s = _make_settings(tmp_path, "auto")
        _fabricate_onnx_dir(s)
        emb = make_embedder(s)
        assert isinstance(emb, OnnxRuriEmbedder)

    def test_onnx_dim_read_without_loading_session(self, tmp_path):
        """dim は meta.json から即答され、ONNX セッションはロードされない
        (model.onnx は空ファイルなのでロードが走れば必ず失敗する)。"""
        s = _make_settings(tmp_path, "auto")
        _fabricate_onnx_dir(s)
        emb = make_embedder(s)
        assert emb.dim == 512
        assert not emb.loaded  # セッション未ロードのまま

    def test_backend_onnx_forced_missing_dir_raises(self, tmp_path):
        s = _make_settings(tmp_path, "onnx")
        with pytest.raises(FileNotFoundError) as exc_info:
            make_embedder(s)
        assert "export-onnx" in str(exc_info.value)

    def test_backend_onnx_forced_with_dir_returns_onnx(self, tmp_path):
        s = _make_settings(tmp_path, "onnx")
        _fabricate_onnx_dir(s)
        emb = make_embedder(s)
        assert isinstance(emb, OnnxRuriEmbedder)

    def test_invalid_backend_raises_value_error(self, tmp_path):
        s = _make_settings(tmp_path, "tensorflow")
        with pytest.raises(ValueError):
            make_embedder(s)

    def test_partial_onnx_dir_is_not_available(self, tmp_path):
        """meta.json が欠けた不完全なディレクトリは ONNX とみなさない。"""
        s = _make_settings(tmp_path, "auto")
        d = s.onnx_model_dir
        d.mkdir(parents=True)
        (d / "model.onnx").write_bytes(b"")
        (d / "tokenizer.json").write_bytes(b"")
        # meta.json なし
        emb = make_embedder(s)
        assert isinstance(emb, RuriEmbedder)


# ---------------------------------------------------------------------------
# アイドル解放(unload)— Codex 多重プロセスのメモリ積み上がり対策(v0.12.0)
# ---------------------------------------------------------------------------

class TestOnnxUnload:
    def _loaded_embedder(self, tmp_path) -> OnnxRuriEmbedder:
        """ロード済みに見せかけた embedder(実モデル不使用)。"""
        s = _make_settings(tmp_path, "auto")
        _fabricate_onnx_dir(s)
        emb = make_embedder(s)
        emb._parts = (object(), object(), ["input_ids"])  # 擬似ロード状態
        return emb

    def test_unload_releases_and_reports(self, tmp_path):
        emb = self._loaded_embedder(tmp_path)
        assert emb.loaded
        assert emb.unload() is True  # 解放を実行した
        assert not emb.loaded

    def test_unload_when_not_loaded_is_noop(self, tmp_path):
        s = _make_settings(tmp_path, "auto")
        _fabricate_onnx_dir(s)
        emb = make_embedder(s)
        assert not emb.loaded
        assert emb.unload() is False  # 何も解放していない

    def test_unload_then_reload_on_next_use(self, tmp_path):
        """解放後の次回利用で _load が再ロードを試みること(モデルは空ファイル
        なのでロード失敗=例外。「ロードが走った」ことの証明として十分)。"""
        import pytest

        emb = self._loaded_embedder(tmp_path)
        emb.unload()
        with pytest.raises(Exception):
            emb.embed_query("再ロードを誘発")

    def test_encode_holds_local_refs_against_unload(self, tmp_path):
        """_encode がロード結果をローカル変数に受けているため、実行中に
        unload されても手元の参照で完走できる(競合安全性)。"""

        class FakeTokenizer:
            def encode_batch(self, texts):
                class E:
                    ids = [1, 2]
                    attention_mask = [1, 1]

                return [E() for _ in texts]

        class FakeSession:
            def run(self, _out, feeds):
                batch = len(feeds["input_ids"])
                return [np.ones((batch, 2, 4), dtype=np.float32)]

        emb = self._loaded_embedder(tmp_path)
        emb._parts = (FakeSession(), FakeTokenizer(), ["input_ids", "attention_mask"])

        original_load = emb._load

        def load_then_unload():
            parts = original_load()
            emb.unload()  # ロード直後に横から解放されたことを再現
            return parts

        emb._load = load_then_unload
        vecs = emb.embed_docs(["a", "b"])  # 例外なく完走すること
        assert vecs.shape == (2, 4)
        assert not emb.loaded


# ---------------------------------------------------------------------------
# ENGRAM_EMBED_BACKEND 環境変数 → Settings
# ---------------------------------------------------------------------------

class TestEmbedBackendEnv:
    def test_env_override_reaches_settings(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
        monkeypatch.setenv("ENGRAM_EMBED_BACKEND", "torch")
        s = cfg.get_settings()
        assert s.embed_backend == "torch"

    def test_default_is_auto(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
        monkeypatch.delenv("ENGRAM_EMBED_BACKEND", raising=False)
        s = cfg.get_settings()
        assert s.embed_backend == "auto"

    def test_onnx_model_dir_derived_from_model_name(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
        monkeypatch.delenv("ENGRAM_EMBED_BACKEND", raising=False)
        s = cfg.get_settings()
        assert s.onnx_model_dir == (
            tmp_path / "onnx" / s.embed_model.replace("/", "--")
        )
