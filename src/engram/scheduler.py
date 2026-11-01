"""OS スケジューラへの登録(眠り=毎晩 / DMN=15分毎)。

Windows はタスクスケジューラ(schtasks)、それ以外は crontab に「# engram:<kind>」の
標識付きで1行足す。`engram setup` が既定で登録し、`engram night --uninstall` /
`engram dmn --uninstall` で外せる(外した意思は config の schedule_night / schedule_dmn に残る)。

登録するコマンドには `--home <data_dir>` を付ける。スケジューラから起動されると
ENGRAM_HOME などの環境が引き継がれないため、登録したのと別のホーム(既定の ~/.engram)を
触ってしまうのを防ぐ。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TASK_NAMES = {"night": "EngramSleepNight", "dmn": "EngramDMN"}
MARKERS = {"night": "# engram:night", "dmn": "# engram:dmn"}


def task_name(kind: str, suffix: str = "") -> str:
    return TASK_NAMES[kind] + suffix


def marker(kind: str, suffix: str = "") -> str:
    return MARKERS[kind] + suffix


def _python_for_background() -> str:
    """Windows ではコンソール窓を出さない pythonw を使う(あれば)。"""
    exe = Path(sys.executable)
    if os.name == "nt":
        w = exe.with_name("pythonw.exe")
        if w.is_file():
            return str(w)
    return str(exe)


def command_for(kind: str, home: Path | str | None = None) -> list[str]:
    """スケジューラに登録する実行コマンド(配列)。"""
    sub = {"night": ["night"], "dmn": ["dmn"]}[kind]
    cmd = [_python_for_background(), "-m", "engram", *sub]
    if home is not None:
        cmd += ["--home", str(home)]
    return cmd


def _quote_win(parts: list[str]) -> str:
    return " ".join(f'"{p}"' if " " in p else p for p in parts)


def schedule_spec(kind: str, *, night_time: str = "02:30",
                  dmn_minutes: int = 15, home: Path | str | None = None,
                  suffix: str = "") -> dict:
    """OS 別の登録内容を組み立てて返す(実行はしない)。テストで検証できる形。"""
    cmd = command_for(kind, home)
    name = task_name(kind, suffix)
    if os.name == "nt":
        tr = _quote_win(cmd)
        if kind == "night":
            args = ["schtasks", "/Create", "/TN", name, "/TR", tr,
                    "/SC", "DAILY", "/ST", night_time, "/F"]
        else:
            args = ["schtasks", "/Create", "/TN", name, "/TR", tr,
                    "/SC", "MINUTE", "/MO", str(dmn_minutes), "/F"]
        return {"os": "windows", "install": args,
                "uninstall": ["schtasks", "/Delete", "/TN", name, "/F"],
                "query": ["schtasks", "/Query", "/TN", name],
                "display": " ".join(args)}
    hh, mm = night_time.split(":")
    sched = f"{int(mm)} {int(hh)} * * *" if kind == "night" else f"*/{dmn_minutes} * * * *"
    line = f"{sched} {' '.join(cmd)} >/dev/null 2>&1 {marker(kind, suffix)}"
    return {"os": "unix", "cron_line": line, "marker": marker(kind, suffix),
            "display": f"crontab: {line}"}


def registered_command(kind: str, *, suffix: str = "") -> str | None:
    """いま登録されている実行コマンド(読むだけ)。無ければ None。"""
    spec = schedule_spec(kind, suffix=suffix)
    if spec["os"] == "windows":
        try:
            r = subprocess.run(spec["query"] + ["/XML"], capture_output=True)
        except OSError:
            return None
        if r.returncode != 0:
            return None
        raw = r.stdout
        text = None
        for enc in ("utf-8", "utf-16", "cp932"):
            try:
                text = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            if "<Exec>" in text:
                break
        if not text or "<Exec>" not in text:
            return ""
        import html
        import re
        cmd = re.search(r"<Command>(.*?)</Command>", text, re.S)
        args = re.search(r"<Arguments>(.*?)</Arguments>", text, re.S)
        return html.unescape(" ".join(m.group(1) for m in (cmd, args) if m))
    for ln in _crontab_read().splitlines():
        if ln.endswith(spec["marker"]):
            return ln
    return None


def owner(kind: str, home: Path | str, *, suffix: str = "") -> str:
    """その名前の登録が誰のものか: "none"(無い) / "ours"(このホームが登録) / "other"(別のホームか別の仕組み)。

    別のホームの試験や、利用者が同じ名前で自前の仕組みを登録している場合に、
    上書きも削除もしないための確かめ(R8.11.1、試験の後片付けが本物の登録を消した事故から)。
    """
    cmd = registered_command(kind, suffix=suffix)
    if cmd is None:
        return "none"
    norm = (lambda s: os.path.normcase(s)) if os.name == "nt" else (lambda s: s)
    marks = (f"--home {home}", f'--home "{home}"')
    return "ours" if any(norm(m) in norm(cmd) for m in marks) else "other"


def install(kind: str, *, force: bool = False, **kw) -> tuple[bool, str]:
    home, suffix = kw.get("home"), kw.get("suffix", "")
    if home is not None and not force and owner(kind, home, suffix=suffix) == "other":
        return False, (f"同じ名前の登録({task_name(kind, suffix)})が既にあり、このホームのものでは"
                       "ないため上書きしません(置き換えるなら --force-schedule)")
    spec = schedule_spec(kind, **kw)
    if spec["os"] == "windows":
        try:
            r = subprocess.run(spec["install"], capture_output=True, text=True)
        except OSError as exc:
            return False, f"schtasks を実行できません: {exc}"
        ok = r.returncode == 0
        return ok, (r.stdout or r.stderr or "").strip() or spec["display"]
    current = _crontab_read()
    lines = [ln for ln in current.splitlines() if not ln.endswith(spec["marker"])]
    lines.append(spec["cron_line"])
    return _crontab_write("\n".join(lines) + "\n"), spec["display"]


def uninstall(kind: str, *, suffix: str = "", home: Path | str | None = None,
              force: bool = False) -> tuple[bool, str]:
    if home is not None and not force:
        who = owner(kind, home, suffix=suffix)
        if who == "none":
            return True, f"登録はありません({task_name(kind, suffix)})"
        if who == "other":
            return False, (f"{task_name(kind, suffix)} はこのホームが登録したものではないため外しません"
                           "(別のホームか別の仕組みの登録。外すなら --force-schedule)")
    spec = schedule_spec(kind, suffix=suffix)
    if spec["os"] == "windows":
        try:
            r = subprocess.run(spec["uninstall"], capture_output=True, text=True)
        except OSError as exc:
            return False, f"schtasks を実行できません: {exc}"
        return r.returncode == 0, (r.stdout or r.stderr or "").strip()
    current = _crontab_read()
    lines = [ln for ln in current.splitlines() if not ln.endswith(spec["marker"])]
    return _crontab_write("\n".join(lines) + ("\n" if lines else "")), "crontab から削除"


def is_installed(kind: str, *, suffix: str = "") -> bool:
    """登録されているか(読むだけ)。"""
    spec = schedule_spec(kind, suffix=suffix)
    if spec["os"] == "windows":
        try:
            r = subprocess.run(spec["query"], capture_output=True, text=True)
        except OSError:
            return False
        return r.returncode == 0
    return any(ln.endswith(spec["marker"]) for ln in _crontab_read().splitlines())


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
