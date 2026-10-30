"""RSI ループの回転計 — 自律反復の回数と、提案が採用へ届いた率を測る。

RSI(再帰的自己改善)は「回転数 × 1回転あたりの改善率」。metabolism は記憶側の
物差しで、ループ全体の回転は測っていなかった。既存のログ・状態だけから
(何も書き換えず)数える。

1. 自律反復/日 — 眠り(night)の実行夜数・DMN の起床/日記・蒸留で保存した知見
2. 提案採用率 — 任意の台帳(rsi_ledger_path。Markdown 表)で、行の状態欄が
   rsi_ledger_done_pattern に合う割合。台帳が無ければ空欄のまま
3. DMN 朝送り率 — DMN の思いつきのうち for_user=true の割合(ノイズ度合いの目安)
"""

from __future__ import annotations

import datetime
import json
import re
import time
from pathlib import Path

from .config import Settings


def _lines_in_window(path: Path, pattern: str, since: datetime.datetime) -> int:
    if not path.is_file():
        return 0
    rx = re.compile(pattern)
    n = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\W*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if not m:
            continue
        try:
            ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ts >= since and rx.search(line):
            n += 1
    return n


def _distilled(path: Path, since: datetime.datetime) -> int:
    if not path.is_file():
        return 0
    total = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^- (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[distill\] 蒸留まとめ: (\d+)件", line)
        if m and datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") >= since:
            total += int(m.group(2))
    return total


def _dmn(settings: Settings, since_ts: float) -> dict:
    out = {"ticks": 0, "fired": 0, "wrote": 0, "for_user": 0, "for_user_rate": None}
    try:
        st = json.loads((settings.data_dir / "dmn" / "state.json").read_text(encoding="utf-8"))
    except Exception:
        return out
    for t in st.get("ticks", []):
        if t.get("ts", 0) < since_ts:
            continue
        out["ticks"] += 1
        out["fired"] += 1 if t.get("fired") else 0
        if t.get("result") == "wrote":
            out["wrote"] += 1
            out["for_user"] += 1 if t.get("for_user") else 0
    if out["wrote"]:
        out["for_user_rate"] = round(out["for_user"] / out["wrote"], 3)
    return out


def ledger_adoption(path: Path | None, done_pattern: str, *, state_col: int = 2) -> dict:
    """Markdown 表の各行(| id | 項目 | 状態 | ...)から採用率を出す。"""
    out = {"total": 0, "adopted": 0, "rate": None}
    if not path or not path.is_file():
        return out
    rx = re.compile(done_pattern)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) <= state_col or not re.match(r"^[A-Za-z]*\d+$", cells[0]):
            continue
        out["total"] += 1
        if rx.search(cells[state_col]):
            out["adopted"] += 1
    if out["total"]:
        out["rate"] = round(out["adopted"] / out["total"], 3)
    return out


def compute(settings: Settings, *, window_days: float = 7.0, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    since_ts = now - window_days * 86400
    since_dt = datetime.datetime.fromtimestamp(since_ts)
    sleep = settings.data_dir / "sleep"
    runs = _lines_in_window(sleep / "night_log.txt", r"sleep night start", since_dt)
    ok = _lines_in_window(sleep / "night_log.txt", r"agent OK", since_dt)
    distilled = _distilled(sleep / "night_agent_log.md", since_dt)
    dmn = _dmn(settings, since_ts)
    ledger = ledger_adoption(Path(settings.rsi_ledger_path).expanduser() if settings.rsi_ledger_path else None,
                             settings.rsi_ledger_done_pattern)
    loops = runs + dmn["wrote"]
    try:
        from .hypothesis import ledger_stats
        hyp = ledger_stats(settings, since_ts=since_ts, now=now)
        hyp.pop("by_kind", None)
    except Exception:
        hyp = None
    return {
        "computed_at": now, "window_days": window_days,
        "autonomous": {"night_runs": runs, "night_integration_ok": ok, "distilled_insights": distilled,
                       "dmn": dmn, "loops_total": loops, "loops_per_day": round(loops / window_days, 2)},
        "ledger": ledger,
        "hypotheses": hyp,
    }


def format_report(m: dict) -> str:
    a, d, g = m["autonomous"], m["autonomous"]["dmn"], m["ledger"]

    def pct(v):
        return f"{v * 100:.1f}%" if v is not None else "-"
    h = m.get("hypotheses")
    hyp_line = ("4. 仮説サイクル: 仮説なし" if not h or not h["created"] else
                f"4. 仮説サイクル: 立てた{h['created']}件 ／ 期限内に判定{h['resolved_by_due']}件"
                f"(答え合わせ率 {pct(h['answer_rate'])}) ／ 当たり{h['当たり']}・外れ{h['外れ']}"
                f"・判定不能{h['判定不能']}(当たり率 {pct(h['hit_rate'])}) ／ 未決{h['open']}(期限切れ{h['overdue']})")
    return "\n".join([
        f"[RSIループ回転計] 窓={m['window_days']:g}日",
        f"1. 自律反復: {a['loops_total']}回 ({a['loops_per_day']}回/日) = 眠り{a['night_runs']}夜"
        f"(統合OK {a['night_integration_ok']}) + DMN日記{d['wrote']}件(起床{d['fired']}/ティック{d['ticks']})"
        f" ／ 蒸留知見 {a['distilled_insights']}件",
        f"2. 提案採用率(台帳): {pct(g['rate'])} ({g['adopted']}/{g['total']})" if g["total"] else
        "2. 提案採用率(台帳): 台帳なし(config.toml の rsi_ledger_path で指定)",
        f"3. DMN朝送り率: {pct(d['for_user_rate'])} ({d['for_user']}/{d['wrote']})",
        hyp_line,
    ])
