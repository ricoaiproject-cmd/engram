"""MCP サーバー起動の回帰テスト。

守っているのは「起動タイムアウト」クラスの事故(2026-07-02〜03 に実際に発生):
- initialize ハンドシェイクが重い import に巻き込まれて数十秒ブロックする
- server モジュールの import 時点で torch 等の重量級が読み込まれてしまう

どちらも実モデル不要で検証できる(ENGRAM_PRELOAD=off ならエンジン構築なしで
ハンドシェイクが返る設計)。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")

# ハンドシェイクは本来 2〜3 秒。CI ランナーの遅さを見込んでも 15 秒あれば
# 十分で、これを超えるのは「重い import がハンドシェイクを塞ぐ」回帰。
HANDSHAKE_DEADLINE_SECONDS = 15


def _spawn_server(tmp_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR
    env["ENGRAM_PRELOAD"] = "off"
    env["ENGRAM_HOME"] = str(tmp_path)
    return subprocess.Popen(
        [sys.executable, "-c", "from engram.server import main; main()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def test_initialize_handshake_is_fast(tmp_path):
    """initialize が数秒で応答する(重い import に塞がれない)こと。"""
    proc = _spawn_server(tmp_path)
    try:
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "startup-test", "version": "0"},
            },
        }
        proc.stdin.write((json.dumps(request) + "\n").encode())
        proc.stdin.flush()

        # ハングしたサーバーで readline がテストごと固まらないよう、
        # 読み取りはスレッドに逃がして deadline で待つ
        lines: queue.Queue = queue.Queue()

        def _reader():
            for raw in proc.stdout:
                lines.put(raw)

        threading.Thread(target=_reader, daemon=True).start()

        response = None
        import time
        deadline = time.monotonic() + HANDSHAKE_DEADLINE_SECONDS
        while time.monotonic() < deadline:
            try:
                raw = lines.get(timeout=0.5)
            except queue.Empty:
                assert proc.poll() is None, "サーバーが起動中に死にました"
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == 1:
                response = msg
                break

        assert response is not None, (
            f"initialize が {HANDSHAKE_DEADLINE_SECONDS} 秒以内に応答しません"
            "(重い import がハンドシェイクを塞ぐ回帰の疑い)"
        )
        assert "result" in response
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_server_import_stays_light():
    """engram.server の import だけでは重量級モジュールを読み込まないこと。

    torch / sentence_transformers は言うまでもなく、onnxruntime も遅延ロード
    (初回 embed 時)に保つ。ここが崩れると全クライアントの起動が遅くなる。
    """
    code = (
        "import sys\n"
        "import engram.server\n"
        "heavy = [m for m in ('torch', 'sentence_transformers', 'onnxruntime')"
        " if m in sys.modules]\n"
        "print('HEAVY:' + ','.join(heavy))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    marker = [ln for ln in out.stdout.splitlines() if ln.startswith("HEAVY:")]
    assert marker, out.stdout
    heavy = marker[0].removeprefix("HEAVY:")
    assert heavy == "", f"import engram.server が重量級を読み込みました: {heavy}"
