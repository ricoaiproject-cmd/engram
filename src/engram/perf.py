"""観測性(perf ログ)。

MCP ツール呼び出しと起動時の先読み(preload)の所要時間を
data_dir/perf/perf_log.jsonl に1行1 JSON で記録する。「調子が悪い」を
体感ではなくデータで診断するための常時計測(settings.perf_log で on/off)。

ログ形式(固定契約。変更しないこと):
    {"ts": epoch float, "kind": "tool" | "preload", "name": str, "ms": float, "ok": bool}

ローテーションやログ書き込み失敗時の振る舞いは surface.py の _append_log
(記憶の自発的想起ログ)と揃えている。
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import Settings

_LOG_ROTATE_BYTES = 5 * 1024 * 1024


def append_perf(settings: Settings, entry: dict) -> None:
    """perf ログへ1行追記する(ディレクトリ作成・ローテーション込み)。

    ログ書き込みの失敗(OSError)は記憶基盤の可用性を損なわないよう握りつぶす。
    """
    d = settings.data_dir / "perf"
    try:
        d.mkdir(parents=True, exist_ok=True)
        log = d / "perf_log.jsonl"
        if log.is_file() and log.stat().st_size > _LOG_ROTATE_BYTES:
            log.replace(log.with_suffix(".jsonl.old"))
        # 入力に不正なサロゲート等が紛れてもログ書き込みでは落とさない
        with log.open("a", encoding="utf-8", errors="replace") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


@contextmanager
def timed(settings: Settings, kind: str, name: str) -> Iterator[None]:
    """ブロックの所要時間を計測し、perf ログへ記録するコンテキストマネージャ。

    settings.perf_log が False ならタイマーすら回さず即座に何もしない(オーバー
    ヘッドは bool 判定1回のみ)。ブロック内で例外が発生した場合は ok=False で
    記録した上で、その例外を再送出する(呼び出し元の挙動は変えない)。
    """
    if not settings.perf_log:
        yield
        return

    start = time.perf_counter()
    ok = True
    try:
        yield
    except BaseException:
        ok = False
        raise
    finally:
        ms = (time.perf_counter() - start) * 1000.0
        append_perf(
            settings,
            {"ts": time.time(), "kind": kind, "name": name, "ms": ms, "ok": ok},
        )
