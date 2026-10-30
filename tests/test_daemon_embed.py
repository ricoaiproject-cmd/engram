"""共有デーモンへの埋め込み依頼と、窓口用の DaemonEmbedder のテスト(R8.10.23)。

各セッションの MCP サーバーがモデルを持たず、デーモン1つに計算を頼む作りを
検証する。重いモデルは使わず FakeEmbedder を差し込む。
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import engram.server as server
from engram import daemon as daemon_mod
from engram.config import Settings
from engram.embedder import DaemonEmbedder, FakeEmbedder, OnnxRuriEmbedder


def _make_settings(tmp_path, **overrides) -> Settings:
    overrides.setdefault("daemon_port", 0)  # 空きポート(テスト同士の干渉を防ぐ)
    return Settings(
        memories_dir=tmp_path / "memories",
        data_dir=tmp_path / "data",
        **overrides,
    )


@pytest.fixture
def running_daemon(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "engram.embedder.make_embedder", lambda s: FakeEmbedder(dim=64)
    )
    t = threading.Thread(target=daemon_mod.serve, args=(settings,),
                         daemon=True)
    t.start()
    for _ in range(100):
        if daemon_mod.is_alive(settings):
            break
        time.sleep(0.05)
    yield settings
    daemon_mod.stop(settings)


class _CountingLocal(FakeEmbedder):
    """手元の実行系の代わり。呼ばれた回数と、ロード状態の委譲を見る。"""

    def __init__(self) -> None:
        super().__init__(dim=64)
        self.calls = 0
        self.loaded = False
        self.unload_calls = 0

    def embed_query(self, text):
        self.calls += 1
        self.loaded = True
        return super().embed_query(text)

    def embed_docs(self, texts):
        self.calls += 1
        self.loaded = True
        return super().embed_docs(texts)

    def unload(self):
        self.unload_calls += 1
        was, self.loaded = self.loaded, False
        return was


# --- デーモン側 -------------------------------------------------------------


def test_ping_advertises_embed(running_daemon):
    assert "embed" in daemon_mod.capabilities(running_daemon)


def test_daemon_embed_matches_local(running_daemon):
    ref = FakeEmbedder(dim=64)
    texts = ["予算要求の書式", "動画のサムネイル"]
    docs = daemon_mod.embed(running_daemon, "docs", texts)
    assert np.allclose(np.asarray(docs, dtype=np.float32),
                       ref.embed_docs(texts))
    q = daemon_mod.embed(running_daemon, "query", ["様式7"])
    assert np.allclose(np.asarray(q[0], dtype=np.float32),
                       ref.embed_query("様式7"))


def test_daemon_embed_rejects_bad_kind(running_daemon):
    assert daemon_mod.embed(running_daemon, "nope", ["x"]) is None


def test_daemon_embed_handles_large_text(running_daemon):
    """記憶の本文は 64KB を超えることがある(依頼の上限を広げた確認)。"""
    big = "記憶" * 50_000  # 約300KB(UTF-8)
    vecs = daemon_mod.embed(running_daemon, "docs", [big])
    assert vecs is not None and len(vecs) == 1


# --- 窓口側(DaemonEmbedder) ------------------------------------------------


def test_daemon_embedder_uses_daemon_not_local(running_daemon):
    local = _CountingLocal()
    emb = DaemonEmbedder(running_daemon, local, spawn_wait=0)
    v = emb.embed_query("様式7")
    d = emb.embed_docs(["予算要求の書式"])
    assert local.calls == 0 and not emb.loaded  # 手元のモデルは読まれていない
    assert emb.remote_calls == 2
    ref = FakeEmbedder(dim=64)
    assert np.allclose(v, ref.embed_query("様式7"))
    assert d.shape == (1, 64) and d.dtype == np.float32


def test_daemon_embedder_falls_back_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_DISABLE_DAEMON", "1")  # 起動を仕掛けない
    settings = _make_settings(tmp_path, daemon_port=1)  # 誰もいないポート
    local = _CountingLocal()
    emb = DaemonEmbedder(settings, local, spawn_wait=0, retry_after=60)
    v = emb.embed_query("様式7")
    assert local.calls == 1 and emb.local_calls == 1
    assert np.allclose(v, FakeEmbedder(dim=64).embed_query("様式7"))
    # 不通を覚えて、しばらくはデーモンを試さない(毎回の待ちを作らない)
    called = []
    monkeypatch.setattr(daemon_mod, "embed",
                        lambda *a, **k: called.append(1) or None)
    emb.embed_docs(["x"])
    assert called == [] and local.calls == 2


def test_daemon_embedder_skips_old_daemon(tmp_path, monkeypatch):
    """embed を知らない旧版のデーモンが常駐していたら、手元で計算する。"""
    settings = _make_settings(tmp_path)
    monkeypatch.setattr(daemon_mod, "embed", lambda *a, **k: None)
    monkeypatch.setattr(daemon_mod, "capabilities", lambda s: ["search"])
    spawned = []
    monkeypatch.setattr(daemon_mod, "ensure_running",
                        lambda s: spawned.append(1))
    local = _CountingLocal()
    emb = DaemonEmbedder(settings, local, spawn_wait=0)
    emb.embed_query("x")
    assert local.calls == 1
    assert spawned == []  # 生きているデーモンを二重に起こさない


def test_daemon_embedder_rejects_wrong_shape(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    monkeypatch.setattr(daemon_mod, "embed",
                        lambda *a, **k: [[0.0] * 10])  # 次元が違う
    local = _CountingLocal()
    emb = DaemonEmbedder(settings, local, spawn_wait=0)
    v = emb.embed_query("x")
    assert v.shape == (64,) and local.calls == 1


def test_daemon_embedder_spawns_and_waits(tmp_path, monkeypatch):
    """不在なら起動を仕掛け、上がってきたらデーモンで計算する。"""
    settings = _make_settings(tmp_path)
    state = {"up": False}

    def fake_embed(s, kind, texts, **k):
        if not state["up"]:
            return None
        return FakeEmbedder(dim=64).embed_docs(texts).tolist()

    monkeypatch.setattr(daemon_mod, "embed", fake_embed)
    monkeypatch.setattr(daemon_mod, "capabilities",
                        lambda s: ["search", "embed"] if state["up"] else None)
    monkeypatch.setattr(daemon_mod, "ensure_running",
                        lambda s: state.update(up=True))
    local = _CountingLocal()
    emb = DaemonEmbedder(settings, local, spawn_wait=2)
    emb.embed_docs(["x"])
    assert local.calls == 0 and emb.remote_calls == 1


def test_daemon_embedder_empty_docs(tmp_path):
    emb = DaemonEmbedder(_make_settings(tmp_path), _CountingLocal())
    out = emb.embed_docs([])
    assert out.shape == (0, 64)


def test_daemon_embedder_delegates_unload(tmp_path, monkeypatch):
    """手元で計算したときのモデルは、アイドル解放で返せる。"""
    monkeypatch.setenv("ENGRAM_DISABLE_DAEMON", "1")
    local = _CountingLocal()
    emb = DaemonEmbedder(_make_settings(tmp_path, daemon_port=1), local,
                         spawn_wait=0)
    emb.embed_query("x")
    assert emb.loaded
    assert emb.unload() is True and not emb.loaded


# --- MCP サーバーでの選び方 -------------------------------------------------


def _fake_onnx(tmp_path):
    return OnnxRuriEmbedder(tmp_path / "onnx")  # 生成だけならロードは走らない


def test_server_uses_daemon_embedder_for_onnx(tmp_path, monkeypatch):
    monkeypatch.delenv("ENGRAM_DISABLE_DAEMON", raising=False)
    monkeypatch.setattr("engram.embedder.make_embedder",
                        lambda s: _fake_onnx(tmp_path))
    assert isinstance(server._make_server_embedder(), DaemonEmbedder)


def test_server_keeps_local_when_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_DISABLE_DAEMON", "1")
    monkeypatch.setattr("engram.embedder.make_embedder",
                        lambda s: _fake_onnx(tmp_path))
    assert isinstance(server._make_server_embedder(), OnnxRuriEmbedder)


def test_server_keeps_local_when_setting_off(tmp_path, monkeypatch):
    monkeypatch.delenv("ENGRAM_DISABLE_DAEMON", raising=False)
    monkeypatch.setattr("engram.embedder.make_embedder",
                        lambda s: _fake_onnx(tmp_path))
    monkeypatch.setattr(
        "engram.config.get_settings",
        lambda: Settings(embed_via_daemon=False),
    )
    assert isinstance(server._make_server_embedder(), OnnxRuriEmbedder)


def test_server_keeps_local_for_torch(tmp_path, monkeypatch):
    monkeypatch.delenv("ENGRAM_DISABLE_DAEMON", raising=False)
    fake = FakeEmbedder(dim=64)  # ONNX 以外(torch 経路の代わり)
    monkeypatch.setattr("engram.embedder.make_embedder", lambda s: fake)
    assert server._make_server_embedder() is fake
