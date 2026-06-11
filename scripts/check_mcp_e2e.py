"""MCP プロトコル経由の実機 E2E テスト。

Claude Code と同じ手順(spawn → initialize → tools/call)で engram-mcp.exe に
接続し、recall がタイムアウトせずに返ることを確認する。recall の初回呼び出しは
埋め込みモデルのロードを含むため、ここがハングしないことの証明になる。
"""

from __future__ import annotations

import asyncio
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 開発用 venv → uv tool → PATH の順で engram-mcp を探す(環境非依存)
import shutil
from pathlib import Path as _P

_CANDIDATES = [
    _P.home() / ".engram" / "venv" / "Scripts" / "engram-mcp.exe",
    _P.home() / ".local" / "bin" / "engram-mcp.exe",
]
SERVER = next(
    (str(p) for p in _CANDIDATES if p.is_file()),
    shutil.which("engram-mcp") or "engram-mcp",
)


async def main() -> int:
    params = StdioServerParameters(command=SERVER, args=[])
    t0 = time.perf_counter()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=30)
            print(f"initialize OK ({time.perf_counter() - t0:.1f}s)")

            tools = await asyncio.wait_for(session.list_tools(), timeout=30)
            print(f"tools/list OK: {len(tools.tools)} tools")

            t1 = time.perf_counter()
            res = await asyncio.wait_for(
                session.call_tool(
                    "recall",
                    {"query": "Windows サンドボックスでファイルをどこに置くべきか",
                     "limit": 3},
                ),
                timeout=120,
            )
            print(f"recall OK ({time.perf_counter() - t1:.1f}s)")
            text = res.content[0].text if res.content else "(empty)"
            print(text[:500])

            t2 = time.perf_counter()
            res2 = await asyncio.wait_for(
                session.call_tool("stats", {}), timeout=30
            )
            print(f"stats OK ({time.perf_counter() - t2:.1f}s)")
            print(res2.content[0].text[:300])
    print("\nE2E PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
