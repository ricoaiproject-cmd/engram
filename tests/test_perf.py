"""観測性(perf ログ)のテスト。

append_perf / timed の単体テストに加え、server.py への計装が FastMCP の
ツールスキーマを壊していないことを回帰確認する(「計装したらツールが
消えた/壊れた」を検知するためのガード)。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from engram.config import Settings
from engram.perf import append_perf, timed


def _make_settings(tmp_path, **overrides) -> Settings:
    return Settings(
        memories_dir=tmp_path / "memories",
        data_dir=tmp_path / "data",
        **overrides,
    )


# ---------------------------------------------------------------------------
# append_perf
# ---------------------------------------------------------------------------

def test_append_perf_writes_valid_jsonl(tmp_path):
    settings = _make_settings(tmp_path)
    append_perf(settings, {"ts": 1.0, "kind": "tool", "name": "recall", "ms": 12.3, "ok": True})
    append_perf(settings, {"ts": 2.0, "kind": "preload", "name": "preload", "ms": 45.6, "ok": False})

    log = settings.data_dir / "perf" / "perf_log.jsonl"
    assert log.is_file()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    rec0 = json.loads(lines[0])
    assert rec0 == {"ts": 1.0, "kind": "tool", "name": "recall", "ms": 12.3, "ok": True}
    rec1 = json.loads(lines[1])
    assert rec1["kind"] == "preload"
    assert rec1["ok"] is False


def test_append_perf_rotates_when_over_5mb(tmp_path):
    settings = _make_settings(tmp_path)
    perf_dir = settings.data_dir / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)
    log = perf_dir / "perf_log.jsonl"

    # 5MB を超える既存ログを用意しておく
    log.write_bytes(b"x" * (5 * 1024 * 1024 + 1))

    append_perf(settings, {"ts": 3.0, "kind": "tool", "name": "stats", "ms": 1.0, "ok": True})

    old_log = perf_dir / "perf_log.jsonl.old"
    assert old_log.is_file()
    assert old_log.stat().st_size == 5 * 1024 * 1024 + 1

    # 新しいログには今回追記した1行だけが入っている
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["name"] == "stats"


def test_append_perf_disabled_when_perf_log_false(tmp_path):
    settings = _make_settings(tmp_path, perf_log=False)
    # append_perf 自体は perf_log を見ないので直接呼べば書き込まれる。
    # 「無効化」の実体は timed 側が append_perf を呼ばないこと(下のテストで検証)。
    # ここでは disabled 設定でもディレクトリが自動生成されないことを
    # timed 経由で確認する。
    with timed(settings, "tool", "recall"):
        pass
    perf_dir = settings.data_dir / "perf"
    assert not perf_dir.exists()


# ---------------------------------------------------------------------------
# timed
# ---------------------------------------------------------------------------

def test_timed_records_positive_ms(tmp_path):
    settings = _make_settings(tmp_path)
    with timed(settings, "tool", "recall"):
        pass

    log = settings.data_dir / "perf" / "perf_log.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "tool"
    assert rec["name"] == "recall"
    assert rec["ok"] is True
    assert rec["ms"] >= 0.0
    assert "ts" in rec


def test_timed_records_ok_false_on_exception_and_reraises(tmp_path):
    settings = _make_settings(tmp_path)

    with pytest.raises(ValueError):
        with timed(settings, "tool", "remember"):
            raise ValueError("boom")

    log = settings.data_dir / "perf" / "perf_log.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["ok"] is False
    assert rec["name"] == "remember"


def test_timed_noop_when_perf_log_disabled(tmp_path):
    settings = _make_settings(tmp_path, perf_log=False)
    with timed(settings, "tool", "recall"):
        pass
    assert not (settings.data_dir / "perf").exists()


# ---------------------------------------------------------------------------
# server.py への計装が FastMCP のツールスキーマを壊していないことの回帰確認
# ---------------------------------------------------------------------------

_EXPECTED_TOOLS = {
    "remember",
    "recall",
    "reinforce",
    "correct",
    "link",
    "forget",
    "consolidation_candidates",
    "mark_consolidated",
    "reindex",
    "stats",
}


def test_server_tool_registry_intact_after_instrumentation():
    from engram.server import mcp

    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert names == _EXPECTED_TOOLS

    # 各ツールがスキーマ(inputSchema)を保持しており、想定した引数が
    # 残っていることも確認する(recall を代表例に)
    by_name = {t.name: t for t in tools}
    recall_tool = by_name["recall"]
    assert recall_tool.inputSchema is not None
    props = recall_tool.inputSchema.get("properties", {})
    assert "query" in props
    assert "mode" in props
    assert "limit" in props
