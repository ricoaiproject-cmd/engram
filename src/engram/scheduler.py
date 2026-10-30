"""OS スケジューラへの登録(眠り=毎晩 / DMN=15分毎)。

Windows はタスクスケジューラ(schtasks)、それ以外は crontab に「# engram:<kind>」の
標識付きで1行足す。登録は利用者が `engram night --install` / `engram dmn --install` と
明示したときだけ行う(setup は勝手に登録しない=自動化に権限拡大を焼き込まない)。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TASK_NAMES = {"night": "EngramSleepNight", "dmn": "EngramDMN"}
MARKERS = {"night": "# engram:night", "dmn": "# engram:dmn"}


def _python_for_background() -> str:
    """Windows ではコンソール窓を出さない pythonw を使う(あれば)。"""
    exe = Path(sys.executable)
    if os.name == "nt":
        w = exe.with_name("pythonw.exe")
        if w.is_file():
            return str(w)
    return str(exe)


def command_for(kind: str) -> list[str]:
    """スケジューラに登録する実行コマンド(配列)。"""
    sub = {"night": ["night"], "dmn": ["dmn"]}[kind]
    return [_python_for_background(), "-m", "engram", *sub]


def _quote_win(parts: list[str]) -> str:
    return " ".join(f'"{p}"' if " " in p else p for p in parts)


def schedule_spec(kind: str, *, night_time: str = "02:30",
                  dmn_minutes: int = 15) -> dict:
    """OS 別の登録内容を組み立てて返す(実行はしない)。テストで検証できる形。"""
    cmd = command_for(kind)
    if os.name == "nt":
        tr = _quote_win(cmd)
        if kind == "night":
            args = ["schtasks", "/Create", "/TN", TASK_NAMES[kind], "/TR", tr,
                    "/SC", "DAILY", "/ST", night_time, "/F"]
        else:
            args = ["schtasks", "/Create", "/TN", TASK_NAMES[kind], "/TR", tr,
                    "/SC", "MINUTE", "/MO", str(dmn_minutes), "/F"]
        return {"os": "windows", "install": args,
                "uninstall": ["schtasks", "/Delete", "/TN", TASK_NAMES[kind], "/F"],
                "display": " ".join(args)}
    hh, mm = night_time.split(":")
    sched = f"{int(mm)} {int(hh)} * * *" if kind == "night" else f"*/{dmn_minutes} * * * *"
    line = f"{sched} {' '.join(cmd)} >/dev/null 2>&1 {MARKERS[kind]}"
    return {"os": "unix", "cron_line": line, "marker": MARKERS[kind],
            "display": f"crontab: {line}"}


def install(kind: str, **kw) -> tuple[bool, str]:
    spec = schedule_spec(kind, **kw)
    if spec["os"] == "windows":
        r = subprocess.run(spec["install"], capture_output=True, text=True)
        ok = r.returncode == 0
        return ok, (r.stdout or r.stderr or "").strip() or spec["display"]
    current = _crontab_read()
    lines = [ln for ln in current.splitlines() if spec["marker"] not in ln]
    lines.append(spec["cron_line"])
    return _crontab_write("\n".join(lines) + "\n"), spec["display"]


def uninstall(kind: str) -> tuple[bool, str]:
    spec = schedule_spec(kind)
    if spec["os"] == "windows":
        r = subprocess.run(spec["uninstall"], capture_output=True, text=True)
        return r.returncode == 0, (r.stdout or r.stderr or "").strip()
    current = _crontab_read()
    lines = [ln for ln in current.splitlines() if spec["marker"] not in ln]
    return _crontab_write("\n".join(lines) + ("\n" if lines else "")), "crontab から削除"


def _crontab_read() -> str:
    try:
        r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else ""
    except OSError:
        return ""


def _crontab_write(text: str) -> bool:
    try:
        r = subprocess.run(["crontab", "-"], input=text, capture_output=True, text=True)
        return r.returncode == 0
    except OSError:
        return False
