"""媒体の記憶・本体側(軽い側)。索引係(media_worker・別 venv)への窓口。

本体の venv には重い部品(torch・sentence-transformers 6.x)を入れない。ここは
- 索引係の生存確認・自動起動(daemon.py と同じ3段の暴走防止: 冷却・import 確認・固定ポート排他)
- 検索(find)・索引の開始・1件の即時索引・状態
- 夜間処理(night.d のプラグインから呼ばれる run_nightly)
- install / uninstall(night.d への配置・venv の用意・可逆な撤去)
だけを担う。TCP(127.0.0.1:media_port・行区切り JSON)で話す。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .config import Settings, get_settings, resolve_room

_SPAWN_COOLDOWN = 90.0
_RECV_LIMIT = 8 * 1024 * 1024
PLUGIN_NAME = "50_media_index.py"
PLUGIN_SOURCE = '''"""engram 夜間処理プラグイン(engram media install が配置)。媒体の差分索引と cold 降格。

本体 venv の Python で実行される。重い部品は索引係(別 venv)に任せ、ここは TCP で頼むだけ。
"""
import json, sys
from engram.config import get_settings
from engram.media import run_nightly

settings = get_settings()
res = run_nightly(settings)
try:
    p = settings.media_dir / "night_media_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, default=str) + "\\n")
except OSError:
    pass
sys.exit(0 if res.get("ok") else 1)
'''


# ---------------------------------------------------------------------------
# 通信
# ---------------------------------------------------------------------------

def port_file(settings: Settings) -> Path:
    return settings.media_dir / "daemon.json"


def _spawn_marker(settings: Settings) -> Path:
    return settings.media_dir / "daemon.spawn"


def log(settings: Settings, message: str) -> None:
    try:
        p = settings.media_dir / "media.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()} core] {message}\n")
    except OSError:
        pass


def _port(settings: Settings) -> int | None:
    if settings.media_port_effective:
        return settings.media_port_effective
    try:
        return int(json.loads(port_file(settings).read_text(encoding="utf-8"))["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def request(settings: Settings, payload: dict, *, timeout: float = 30.0) -> dict | None:
    """1往復。繋がらなければ None(索引係は必須ではない)。

    依頼に身元(home)を添え、応答の身元が自分と違えば None にする
    (同じ番号で待つ別の記憶の索引係・身元を名乗らない旧版は使わない)。
    """
    port = _port(settings)
    if not port:
        return None
    payload = {**payload, "home": settings.home_key}
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=min(timeout, 3.0)) as s:
            s.settimeout(timeout)
            s.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n") and len(buf) < _RECV_LIMIT:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        res = json.loads(buf.decode("utf-8", errors="replace")) if buf.strip() else None
    except (OSError, ValueError):
        return None
    if not isinstance(res, dict) or res.get("home") != settings.home_key:
        return None
    return res


def is_alive(settings: Settings) -> bool:
    res = request(settings, {"op": "ping"}, timeout=3.0)
    return bool(res and res.get("ok"))


def _recently_spawned(settings: Settings) -> bool:
    try:
        return time.time() - float(_spawn_marker(settings).read_text(encoding="utf-8")) < _SPAWN_COOLDOWN
    except (OSError, ValueError):
        return False


def worker_python(settings: Settings) -> Path | None:
    p = settings.media_python_path
    return p if p.is_file() else None


def _can_import_worker(python: Path, settings: Settings) -> tuple[bool, str]:
    """索引係を起動できるかを下見する。駄目なときは理由も返す(ログに残すため)。"""
    try:
        # 下見で sentence_transformers を実際に import すると torch の読み込みに
        # 30秒〜数分かかり(R8.10.23 実測33秒・Googleドライブ経由だと更に遅い)、120秒で
        # 打ち切られて「import できない」と誤判定して起動を見送っていた。
        # ここは「入っているか」だけ見れば足り、重い読み込みは索引係本体に任せる
        probe = ("import importlib.util, engram.media_worker; "
                 "raise SystemExit(0 if importlib.util.find_spec('sentence_transformers') else 1)")
        r = subprocess.run([str(python), "-c", probe], stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=180, check=False)
        if r.returncode == 0:
            return True, ""
        err = (r.stderr or b"").decode("utf-8", errors="replace").strip()
        return False, err[-400:] or f"exit={r.returncode}"
    except subprocess.TimeoutExpired:
        return False, "下見が180秒で終わらなかった(ドライブ不達の可能性)"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def ensure_running(settings: Settings, *, wait: float = 0.0) -> bool:
    """生きていなければ背後で起動する。wait 秒まで ping を待つ(0 なら待たない)。"""
    if os.environ.get("ENGRAM_DISABLE_DAEMON"):
        return False
    if is_alive(settings):
        return True
    if not _recently_spawned(settings):
        py = worker_python(settings)
        if py is None:
            log(settings, f"起動を見送り(索引係の Python が無い: {settings.media_python_path}。engram media setup を実行)")
            return False
        ok, why = _can_import_worker(py, settings)
        if not ok:
            log(settings, f"起動を見送り(索引係を import できない: {py}) 理由: {why}")
            return False
        try:
            _spawn_marker(settings).parent.mkdir(parents=True, exist_ok=True)
            _spawn_marker(settings).write_text(str(time.time()), encoding="utf-8")
        except OSError:
            pass
        pyw = py.with_name("pythonw.exe")
        runner = pyw if pyw.is_file() else py
        cmd = [str(runner), "-m", "engram.media_worker", "daemon"]
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0)
                                       | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        else:
            kwargs["start_new_session"] = True
        try:
            subprocess.Popen(cmd, **kwargs)
            log(settings, f"索引係の起動を仕掛けた: {runner}")
        except Exception as exc:
            log(settings, f"起動の仕掛けに失敗: {exc!r}")
            return False
    deadline = time.time() + wait
    while time.time() < deadline:
        time.sleep(1.0)
        if is_alive(settings):
            return True
    return is_alive(settings)


def stop(settings: Settings) -> bool:
    # 先に ping で自分の索引係と確かめてから止める(旧版は依頼の home を見ない)
    if not is_alive(settings):
        return False
    res = request(settings, {"op": "stop"}, timeout=5.0)
    return bool(res and res.get("ok"))


# ---------------------------------------------------------------------------
# 本体から使う機能
# ---------------------------------------------------------------------------

def enabled(settings: Settings) -> bool:
    return bool(settings.media_enabled)


def find(settings: Settings, query: str, *, k: int = 5, kinds: list[str] | None = None,
         rooms: list[str] | None = None, tiers: list[str] | None = None,
         record_hits: bool = True, wait: float | None = None) -> dict:
    """媒体の検索。索引係が寝ていれば起こして待つ(上限 media_find_timeout_sec)。"""
    if not enabled(settings):
        return {"hits": [], "mode": "media", "error": "media_enabled=false"}
    w = settings.media_find_timeout_sec if wait is None else wait
    if not ensure_running(settings, wait=w):
        return {"hits": [], "mode": "media",
                "error": "索引係が起動していません(初回はモデル読込に約40秒。もう一度試すか `engram media daemon --status`)"}
    res = request(settings, {"op": "find", "q": query, "k": k, "kinds": kinds, "rooms": rooms,
                             "tiers": tiers, "record_hits": record_hits}, timeout=120.0)
    if not res or not res.get("ok"):
        return {"hits": [], "mode": "media", "error": (res or {}).get("error", "no response")}
    return {"hits": [_as_hit(h) for h in res["hits"]], "mode": "media", "auto_deepened": False}


def _as_hit(h: dict) -> dict:
    """recall の hits と同じ顔(id/content/type/score/...)+媒体の情報。"""
    where = h.get("path", "")
    tag = f" p.{h['page']}" if h.get("page") else (f" #{h['segment']}" if h.get("segment") else "")
    content = f"[{h.get('kind')}] {Path(where).name}{tag}"
    if h.get("caption"):
        content += f" — {h['caption']}"
    if h.get("text"):
        content += f"\n{str(h['text'])[:300]}"
    sm = h.get("source_meta") or {}
    if isinstance(sm, dict) and sm.get("subject"):
        content += f"\n出所: メール「{sm.get('subject')}」 {sm.get('from', '')} {sm.get('date', '')}"
    return {
        "id": h.get("id"), "content": content, "type": "media", "kind": h.get("kind"),
        "path": where, "page": h.get("page"), "segment": h.get("segment"),
        "caption": h.get("caption"), "tags": ["media", str(h.get("kind"))], "tier": h.get("tier"),
        "score": h.get("score"), "relevance": h.get("relevance"), "cosine": h.get("cosine"),
        "activation": h.get("activation"), "importance": 0.3, "via": "media",
        "note": f"ほぼ同一の別ファイル{h['duplicates']}件" if h.get("duplicates") else "",
        "room": h.get("room"), "source": h.get("source"), "source_meta": sm or None,
    }


def status(settings: Settings) -> dict:
    alive = is_alive(settings)
    out: dict = {"enabled": enabled(settings), "alive": alive, "python": str(settings.media_python_path),
                 "python_exists": worker_python(settings) is not None, "roots": settings.media_roots,
                 "kinds": settings.media_kinds, "port": settings.media_port_effective}
    if alive:
        res = request(settings, {"op": "status"}, timeout=10.0) or {}
        out.update({k: res.get(k) for k in ("progress", "stats", "caption")})
    else:
        try:
            from .media_schema import MediaIndex
            out["stats"] = MediaIndex.open(settings.db_path).stats()
        except Exception as exc:
            out["stats"] = {"error": repr(exc)}
    plugin = settings.data_dir / "night.d" / PLUGIN_NAME
    out["night_plugin"] = plugin.is_file()
    return out


def start_index(settings: Settings, *, roots: dict | None = None, full: bool = False,
                wait: float = 90.0) -> dict:
    if not ensure_running(settings, wait=wait):
        return {"ok": False, "error": "索引係を起動できません"}
    return request(settings, {"op": "index", "roots": roots, "full": full}, timeout=30.0) or {"ok": False}


def wait_index(settings: Settings, *, timeout: float, poll: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        res = request(settings, {"op": "status"}, timeout=10.0) or {}
        last = res.get("progress") or {}
        if last.get("state") == "idle":
            return last
        time.sleep(poll)
    return last


def index_path(settings: Settings, path: str | Path, *, room: str | None = None, source: str = "manual",
               source_meta: dict | None = None, wait: float = 90.0) -> dict:
    """1件を即時に覚える(「このファイル覚えて」)。"""
    if not enabled(settings):
        return {"ok": False, "error": "media_enabled=false"}
    if not ensure_running(settings, wait=wait):
        return {"ok": False, "error": "索引係を起動できません"}
    p = Path(path).expanduser()
    room = room or resolve_room(p, settings.room_paths)
    return request(settings, {"op": "index_path", "path": str(p), "room": room, "source": source,
                              "source_meta": source_meta}, timeout=600.0) or {"ok": False}


def run_nightly(settings: Settings, *, timeout: float | None = None) -> dict:
    """夜間: 差分索引 → 完了待ち → cold 降格。索引係が無ければ何もしない(ok=False)。"""
    if not enabled(settings):
        return {"ok": False, "skipped": "media_enabled=false"}
    budget = timeout if timeout is not None else max(300.0, settings.night_agent_timeout_sec - 120.0)
    t0 = time.time()
    if not ensure_running(settings, wait=120.0):
        return {"ok": False, "error": "索引係を起動できません"}
    started = request(settings, {"op": "index", "roots": None, "full": False}, timeout=30.0) or {}
    prog = wait_index(settings, timeout=budget - (time.time() - t0))
    demoted = request(settings, {"op": "demote", "days": settings.media_cold_days}, timeout=60.0) or {}
    # 索引は常駐係の中で続くので、夜の持ち時間内に終わらなくても「依頼できた」なら成功扱い
    # (R8.10.11 実測: 5,600ファイルの差分走査が持ち時間 18 分を超え exit=1 と誤記録された)。
    # 終わったかどうかは finished で別に返す
    kicked = bool(started.get("ok")) or started.get("error") == "already running"
    return {"ok": kicked, "finished": prog.get("state") == "idle", "index": prog.get("last"),
            "progress": prog, "demoted": demoted.get("demoted"), "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# 自発的想起(発話フックから。文章の surface に相乗り)
# ---------------------------------------------------------------------------

def _surface_state(settings: Settings, session_id: str) -> Path:
    import re

    safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id or "unknown")[:80]
    return settings.media_dir / "surface" / f"session-{safe}.json"


def surface(settings: Settings, prompt: str, *, room: str = "common", session_id: str = "unknown",
            mode: str | None = None) -> dict:
    """発話に関連しそうな媒体を 0〜max_items 件返す。索引係が寝ていれば起こすだけ(待たない)。

    - 同じセッションで一度出した項目は二度出さない(文章の surface と同じ流儀)
    - cosine が media_surface_min_cos 未満なら出さない
    - shadow はログのみ。戻り値 items は active のときだけ埋まる
    """
    mode = mode or settings.media_surface_mode
    if not enabled(settings) or mode == "off":
        return {"mode": mode, "items": []}
    text = " ".join(prompt.split())
    if len(text) < settings.media_surface_min_prompt_chars:
        return {"mode": mode, "items": []}
    cues = list(getattr(settings, "media_surface_cues", []) or [])
    if cues and not any(c in text for c in cues):
        return {"mode": mode, "items": [], "skipped": "no cue"}
    if not is_alive(settings):
        ensure_running(settings, wait=0.0)
        return {"mode": mode, "items": [], "skipped": "worker not running"}
    rooms = sorted({room, "common"})
    res = request(settings, {"op": "find", "q": text[:500], "k": max(3, settings.media_surface_max_items), "kinds": None, "rooms": rooms,
                             "tiers": ["hot"], "record_hits": False}, timeout=4.0)
    if not res or not res.get("ok"):
        return {"mode": mode, "items": [], "skipped": (res or {}).get("error", "no response")}
    state_path = _surface_state(settings, session_id)
    try:
        seen = set(json.loads(state_path.read_text(encoding="utf-8")).get("surfaced", []))
    except (OSError, ValueError):
        seen = set()
    cands = [h for h in res["hits"] if float(h.get("cosine") or 0.0) >= settings.media_surface_min_cos
             and h.get("id") not in seen]
    if cands:
        top = float(cands[0].get("cosine") or 0.0)
        margin = float(getattr(settings, "media_surface_margin", 0.02))
        cands = [h for h in cands if top - float(h.get("cosine") or 0.0) <= margin]
    items = [_as_hit(h) for h in cands[: settings.media_surface_max_items]]
    log_entry = {"ts": time.time(), "session": session_id, "mode": mode, "prompt": text[:80],
                 "top": [(h.get("path"), round(float(h.get("cosine") or 0), 3)) for h in res["hits"][:3]],
                 "injected": [i["path"] for i in items] if mode == "active" else []}
    try:
        p = settings.media_dir / "surface_log.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    if mode != "active" or not items:
        return {"mode": mode, "items": []}
    try:
        seen.update(i["id"] for i in items if i.get("id"))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"surfaced": sorted(seen)}), encoding="utf-8")
    except OSError:
        pass
    return {"mode": mode, "items": items}


def format_surface(items: list[dict]) -> str:
    lines = ["(engram 媒体の想起)今の発言に関連しそうな画像・文書が媒体の棚にあります。"
             "役立ちそうなら場所を開いて中身を確かめてください。無関係なら無視して構いません。", ""]
    for it in items:
        cap = f" — {it['caption']}" if it.get("caption") else ""
        page = f" p.{it['page']}" if it.get("page") else ""
        lines.append(f"- [{it.get('kind')}] {it.get('path')}{page}{cap} (cos {it.get('cosine')})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# install / uninstall(可逆)
# ---------------------------------------------------------------------------

def install_plugin(settings: Settings) -> Path:
    d = settings.data_dir / "night.d"
    d.mkdir(parents=True, exist_ok=True)
    p = d / PLUGIN_NAME
    p.write_text(PLUGIN_SOURCE, encoding="utf-8")
    return p


def default_roots() -> dict:
    """対象フォルダが未設定のときの既定。画面写真があればそこ、無ければ画像フォルダ全体。"""
    pics = Path.home() / "Pictures"
    for cand in (pics / "Screenshots", pics):
        if cand.is_dir():
            return {str(cand): "common"}
    return {}


def enable(settings: Settings, config_file: Path | None = None) -> dict:
    """媒体の棚をオンにする: media_enabled=true、対象フォルダが無ければ既定を足す、夜間の差分索引を置く。

    設定ファイルは該当の行と区画だけを書き足す(利用者の他の設定には触れない)。
    """
    from .config import config_path
    from .setup import read_config_toml, set_config_scalar

    cfg = config_file or config_path()
    set_config_scalar(cfg, "media_enabled", True)
    out: dict = {"media_enabled": True}
    if not read_config_toml(cfg).get("media_roots") and not settings.media_roots:
        roots = default_roots()
        if roots:
            body = cfg.read_text(encoding="utf-8") if cfg.is_file() else ""
            if body and not body.endswith("\n"):
                body += "\n"
            body += "\n[media_roots]\n" + "".join(f"'{k}' = '{v}'\n" for k, v in roots.items())
            cfg.write_text(body, encoding="utf-8")
        out["media_roots"] = roots or "(未設定。config.toml の [media_roots] に対象フォルダを書いてください)"
    out["night_plugin"] = str(install_plugin(settings))
    return out


def setup_env(settings: Settings, *, source: str, python: str | None = None, upgrade: bool = False) -> dict:
    """索引係の venv を作り engram[media] を入れる(重い。数分・数GB)。source は pip が解釈する指定
    (ローカルのソースフォルダ、git+https://…、PyPI 名)。既にあれば upgrade のときだけ入れ直す。"""
    venv_dir = settings.media_dir / "venv"
    py = settings.media_python_path
    steps: list[str] = []
    base = python or sys.executable
    if not py.is_file():
        subprocess.run([base, "-m", "venv", str(venv_dir)], check=True)
        steps.append(f"venv 作成: {venv_dir}")
    elif not upgrade:
        return {"ok": True, "python": str(py), "steps": ["既存の venv を使用(--upgrade で入れ直し)"]}
    pip = [str(py), "-m", "pip", "install", "--upgrade"]
    r = subprocess.run(pip + [f"{source}[media]"] if not source.startswith("git+") and Path(source).exists()
                       else pip + [f"engram[media] @ {source}"], capture_output=True, text=True, check=False)
    steps.append(f"pip install: rc={r.returncode}")
    if r.returncode != 0:
        return {"ok": False, "python": str(py), "steps": steps, "stderr": r.stderr[-2000:]}
    return {"ok": True, "python": str(py), "steps": steps}


def uninstall(settings: Settings, *, purge_index: bool = False, purge_env: bool = False) -> dict:
    """撤去。既定は常駐停止+夜間プラグイン削除のみ(索引と venv は残る=再有効化が即座)。
    purge_index で媒体の表4つを落とす(記憶本体は無変更)。purge_env で venv・モデル・索引フォルダを消す。"""
    done: list[str] = []
    if stop(settings):
        done.append("索引係を停止")
        time.sleep(1.0)
    p = settings.data_dir / "night.d" / PLUGIN_NAME
    if p.is_file():
        p.unlink()
        done.append("夜間プラグインを削除")
    if purge_index:
        from .media_schema import MediaIndex
        try:
            idx = MediaIndex.open(settings.db_path)
            n = idx.stats().get("items", 0)
            idx.drop_all()
            idx.close()
            done.append(f"媒体の表を削除(行数 {n})")
        except Exception as exc:
            done.append(f"媒体の表の削除に失敗: {exc!r}")
    if purge_env:
        import shutil
        for sub in ("venv", "hf_cache", "index"):
            d = settings.media_dir / sub
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                done.append(f"削除: {d}")
    return {"ok": True, "done": done}


def format_status(s: dict) -> str:
    lines = [f"媒体の記憶: {'有効' if s.get('enabled') else '無効(media_enabled=false)'} / 索引係: {'稼働中' if s.get('alive') else '停止'}",
             f"索引係の Python: {s.get('python')} ({'あり' if s.get('python_exists') else '無し→ engram media setup'})",
             f"対象フォルダ: {len(s.get('roots') or {})}件 / 種別: {', '.join(s.get('kinds') or [])}",
             f"夜間プラグイン: {'配置済み' if s.get('night_plugin') else '未配置→ engram media install'}"]
    st = s.get("stats") or {}
    if st.get("ready"):
        lines.append(f"索引: {st.get('items')}行 / {st.get('files')}ファイル / 種別 {st.get('by_kind')} / 部屋 {st.get('by_room')} / 層 {st.get('by_tier')} / ほぼ同一 {st.get('near_duplicates')}")
        lines.append(f"モデル: {st.get('model')} ({st.get('dim')}次元) / 説明文: {'あり' if s.get('caption') else '無し(ローカルLLM不達か無効)'}")
    else:
        lines.append("索引: 未作成")
    pr = s.get("progress") or {}
    if pr.get("state") == "running":
        lines.append(f"索引作成中: {pr.get('done')}/{pr.get('total')}")
    return "\n".join(lines)
