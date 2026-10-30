"""埋め込み常駐デーモンのテスト。

重い埋め込みモデルは使わず、FakeEmbedder を差し込んで通信・検索・縮退を検証する。
"""

from __future__ import annotations

import threading
import time

import pytest

from engram import daemon as daemon_mod
from engram.config import Settings
from engram.db import IndexDB
from engram.embedder import FakeEmbedder
from engram.engine import MemoryEngine
from engram.store import MarkdownStore


def _make_settings(tmp_path, **overrides) -> Settings:
    # daemon_port=0 で空きポートを取る。固定ポートだとテスト同士が
    # 同じポートを奪い合って干渉する(本番の単一インスタンス保証は
    # 固定ポートで行うが、それはテストの要件ではない)
    overrides.setdefault("daemon_port", 0)
    return Settings(
        memories_dir=tmp_path / "memories",
        data_dir=tmp_path / "data",
        **overrides,
    )


@pytest.fixture
def running_daemon(tmp_path, monkeypatch):
    """FakeEmbedder でデーモンを立てて、終了まで面倒を見る。"""
    settings = _make_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    # 記憶を2件用意する
    embedder = FakeEmbedder(dim=64)
    store = MarkdownStore(settings.memories_dir)
    db = IndexDB(settings.db_path, embedder.dim)
    engine = MemoryEngine(settings=settings, store=store, db=db,
                          embedder=embedder)
    hit = engine.remember("予算要求の書式は財務課の様式7を使うこと",
                          "knowledge", 6)
    other = engine.remember("動画のサムネイルはピンク基調でかわいく",
                            "knowledge", 6)
    db.close()

    monkeypatch.setattr(
        "engram.embedder.make_embedder", lambda s: FakeEmbedder(dim=64)
    )
    t = threading.Thread(target=daemon_mod.serve, args=(settings,),
                         daemon=True)
    t.start()
    for _ in range(100):  # 起動待ち(最大5秒)
        if daemon_mod.is_alive(settings):
            break
        time.sleep(0.05)
    yield settings, hit["id"], other["id"]
    daemon_mod.stop(settings)


def test_daemon_starts_and_answers_ping(running_daemon):
    settings, _hit, _other = running_daemon
    assert daemon_mod.is_alive(settings)
    assert daemon_mod.port_file(settings).is_file()


def test_daemon_returns_semantic_candidates(running_daemon):
    settings, hit_id, other_id = running_daemon
    cands = daemon_mod.semantic_candidates(
        settings, "予算要求の様式について確認したい", k=10
    )
    assert cands, "候補が返らない"
    assert hit_id in cands
    # 関連する記憶の方が、無関係な記憶より近い
    if other_id in cands:
        assert cands[hit_id] > cands[other_id]


def test_daemon_stop_removes_port_file(running_daemon):
    settings, _hit, _other = running_daemon
    assert daemon_mod.stop(settings)
    for _ in range(100):
        if not daemon_mod.is_alive(settings):
            break
        time.sleep(0.05)
    assert not daemon_mod.is_alive(settings)


def test_foreign_client_cannot_stop_or_read(running_daemon, tmp_path):
    """別の記憶(data_dir)の窓口は、同じ番号に繋いでも止められず、結果も使わない。

    R8.10.30 の事故: 試験用に入れた2つめの engram の `daemon --stop` が、
    同じ固定ポートで待つ本物のデーモンを止めた。
    """
    settings, hit_id, _other = running_daemon
    port = daemon_mod._read_port(settings)
    foreign = Settings(memories_dir=tmp_path / "other" / "memories",
                       data_dir=tmp_path / "other" / "data", daemon_port=port)
    assert foreign.home_key != settings.home_key

    assert daemon_mod.is_alive(foreign) is False
    assert daemon_mod.semantic_candidates(foreign, "予算要求の様式", k=10) == {}
    assert daemon_mod.stop(foreign) is False
    # 依頼そのものも拒まれる(窓口側の確認をすり抜けた場合の守り)
    assert daemon_mod.request(foreign, {"op": "stop"}) is None
    time.sleep(0.2)
    assert daemon_mod.is_alive(settings), "別の窓口の stop で本物が止まった"


def test_stop_skips_daemon_that_does_not_name_itself(tmp_path):
    """身元を名乗らない旧版のデーモンには stop を送らない。

    旧版は依頼の home を見ずに止まるので、ping の応答で自分のものと
    確かめられないときは stop を送ってはいけない。
    """
    import json
    import socket

    received: list[str] = []
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]

    def _old_daemon() -> None:
        srv.settimeout(5)
        try:
            while True:
                conn, _ = srv.accept()
                with conn:
                    req = json.loads(conn.makefile("rb").readline())
                    received.append(req.get("op"))
                    conn.sendall(b'{"ok": true, "pid": 1}\n')
        except OSError:
            pass

    threading.Thread(target=_old_daemon, daemon=True).start()
    settings = _make_settings(tmp_path, daemon_port=port)
    try:
        assert daemon_mod.stop(settings) is False
        assert "stop" not in received
        assert received == ["ping"]
    finally:
        srv.close()


def test_default_port_follows_data_dir(tmp_path):
    """既定の番号は data_dir ごとに決まり、同じ場所なら毎回同じ。"""
    a = Settings(data_dir=tmp_path / "a", memories_dir=tmp_path / "a" / "m")
    a2 = Settings(data_dir=tmp_path / "a", memories_dir=tmp_path / "a" / "m")
    others = {Settings(data_dir=tmp_path / f"b{i}").daemon_port_effective
              for i in range(5)}
    assert a.daemon_port is None
    assert a.daemon_port_effective == a2.daemon_port_effective
    assert len(others) > 1  # 場所が違えば番号もばらける
    assert 52000 <= a.daemon_port_effective < 60000
    assert a.media_port_effective == a.daemon_port_effective + 1
    # 明示した値はそのまま(0 は空きポート)
    assert Settings(data_dir=tmp_path / "a", daemon_port=0).daemon_port_effective == 0
    assert Settings(data_dir=tmp_path / "a", daemon_port=0).media_port_effective == 0
    assert Settings(data_dir=tmp_path / "a", daemon_port=51873).media_port_effective == 51874


def test_client_degrades_when_daemon_absent(tmp_path):
    """デーモンが居なければ空を返すだけで、例外にしない(字句経路へ縮退)。"""
    settings = _make_settings(tmp_path)
    assert daemon_mod.is_alive(settings) is False
    assert daemon_mod.semantic_candidates(settings, "何か", k=5) == {}
    assert daemon_mod.request(settings, {"op": "ping"}, timeout=0.5) is None


def test_surface_works_without_daemon(tmp_path):
    """デーモン不在でも surface は字句だけで従来どおり動く。"""
    from engram.surface import run_surface

    settings = _make_settings(
        tmp_path, surface_threshold=0.0, surface_min_relevance=0.0,
        surface_autostart_daemon=False,
    )
    embedder = FakeEmbedder(dim=64)
    store = MarkdownStore(settings.memories_dir)
    db = IndexDB(settings.db_path, embedder.dim)
    engine = MemoryEngine(settings=settings, store=store, db=db,
                          embedder=embedder)
    r = engine.remember("予算要求の書式は財務課の様式7を使うこと",
                        "knowledge", 6)
    db.close()

    result = run_surface("予算要求の書式を確認したい", settings=settings,
                         session_id="nodaemon")
    assert r["id"] in result["surfaced"]


def test_import_probe_does_not_inherit_stdin_and_retries_after_timeout(tmp_path, monkeypatch):
    """下見の子は stdin を引き継がない。打ち切りは覚えず、次の機会に確かめ直す。

    MCP サーバーの中から下見すると、子が標準入力の管を握ったまま止まり、
    打ち切り→「import できない」を覚えて常駐デーモンが二度と起動しなかった(R8.10.30)。
    """
    import subprocess
    import sys
    from pathlib import Path

    settings = _make_settings(tmp_path)
    seen = []

    def timeout_run(*args, **kwargs):
        seen.append(kwargs.get("stdin"))
        raise subprocess.TimeoutExpired(args[0], 20)

    monkeypatch.setattr(subprocess, "run", timeout_run)
    assert daemon_mod._can_import_engram(Path(sys.executable), settings) is False
    assert seen == [subprocess.DEVNULL]
    assert not (settings.data_dir / "daemon.import_ok.json").exists()

    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, b"", b""))
    assert daemon_mod._can_import_engram(Path(sys.executable), settings) is True
