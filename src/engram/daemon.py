"""埋め込み常駐デーモン(意味の網を毎発話で使えるようにする)。

背景(R8.8.27・別モデルの診断): 自発的想起(surface)が字句バイグラムなのは
「毎発話に埋め込みは重い」という理由だったが、その1〜2秒はモデルの**ロード**の
代金であって推論の代金ではない。温まったプロセスならクエリ1本の埋め込みは
数十ms で済む。ロードを1度だけ払って常駐させれば、毎発話でも意味検索が使える。

設計:
- localhost(127.0.0.1)のTCPで待つ小さなサーバー。1接続1リクエストの行JSON
- 起動時に埋め込みモデルと索引DBを開いて保持する(以後のロードは無し)
- リクエスト {"q": "発話", "k": 30, "rooms": [...]} に対し、
  意味の近い候補 [{"id", "relevance"}] を返す
- フック側は接続できなければ黙って字句経路へ縮退する(デーモンは必須ではない)
- リクエスト {"op": "embed", "kind": "query"|"docs", "texts": [...]} に対し、
  埋め込み {"vecs": [[...], ...]} を返す(R8.10.23)。各セッションの MCP
  サーバーはこれを使い、モデルを自分では読み込まない(embedder.DaemonEmbedder)。
  背景: MCP はセッションごとに1プロセス立ち、記憶を引いたプロセスが
  それぞれ約800MBのモデルを抱えていた。モデルはこのデーモン1つに集める
- 待受は 127.0.0.1 のみ。外部からは繋がらない
- 無操作が続いたら自動終了する(常駐しっぱなしでメモリを持たない)

使い方:
    engram daemon            # 前面で起動(動作確認用)
    engram daemon --status   # 生きているか確認
    engram daemon --stop     # 終了させる
フックからは自動起動される(surface.py の semantic_candidates)。
"""

from __future__ import annotations

import json
import os
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path

from .config import Settings, get_settings

_RECV_LIMIT = 64 * 1024
# 埋め込みの依頼と応答は記憶の本文やベクトル列を運ぶので大きい
# (512次元×件数の数値列)。127.0.0.1 限定なので上限を広めに取る
_EMBED_LIMIT = 16 * 1024 * 1024
# ping が返す「この版のデーモンが受け付ける依頼」。旧版のデーモンが
# 常駐したまま新版の窓口が繋ぎに来たとき、窓口が見分けて縮退するため
CAPS = ["search", "embed"]
_SPAWN_COOLDOWN = 90.0          # 起動試行からこの秒数は再試行しない


def port_file(settings: Settings) -> Path:
    return settings.data_dir / "daemon.json"


def _spawn_marker(settings: Settings) -> Path:
    return settings.data_dir / "daemon.spawn"


def log(settings: Settings, message: str) -> None:
    """起動・終了の経緯を残す(pythonw は標準出力が無く、静かに失敗するため)。"""
    try:
        path = settings.data_dir / "daemon.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{stamp} [{os.getpid()}] {message}\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# サーバー本体
# ---------------------------------------------------------------------------

class _State:
    """埋め込みモデルと索引を保持する(プロセス内で1つ)。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.last_seen = time.time()
        self.lock = threading.Lock()
        # ここでだけ重い依存をロードする(常駐の目的)
        from .embedder import make_embedder

        self.embedder = make_embedder(settings)
        self._db = None

    def db(self):
        if self._db is None:
            from .db import IndexDB

            self._db = IndexDB(self.settings.db_path, self.embedder.dim)
        return self._db

    def search(self, query: str, k: int, rooms: list[str] | None,
               tiers: list[str] | None) -> list[dict]:
        with self.lock:
            self.last_seen = time.time()
            vec = self.embedder.embed_query(query)
            hits = self.db().vector_search(
                vec, k, tiers=tiers or ["hot", "cold"],
                types=None, rooms=rooms or None,
            )
        return [{"id": mid, "relevance": float(cos)} for mid, cos in hits]

    def embed(self, kind: str, texts: list[str]) -> list[list[float]]:
        """窓口(各セッションの MCP)からの埋め込み依頼。モデルはここにしか無い。"""
        import numpy as np

        if kind not in ("query", "docs"):
            raise ValueError(f"kind が不正です: {kind!r} (query | docs)")
        with self.lock:
            self.last_seen = time.time()
            if not texts:
                return []
            if kind == "query":
                vecs = np.stack([self.embedder.embed_query(t) for t in texts])
            else:
                vecs = self.embedder.embed_docs(texts)
        # float32 → float の変換は JSON 経由でも往復で値が変わらない
        return np.asarray(vecs, dtype=np.float32).tolist()


class _Handler(socketserver.StreamRequestHandler):
    timeout = 30

    def handle(self) -> None:
        try:
            raw = self.rfile.readline(_EMBED_LIMIT)
            if not raw:
                return
            req = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            self._send({"error": "bad request"})
            return

        state: _State = self.server.state  # type: ignore[attr-defined]
        # 別の記憶(data_dir)の窓口からの依頼は、停止も含めて何もしない。
        # 身元を名乗らない旧版の窓口の依頼は従来どおり受ける
        asked = req.get("home")
        if asked is not None and asked != state.settings.home_key:
            self._send({"error": "foreign home"})
            return
        try:
            if req.get("op") == "ping":
                self._send({"ok": True, "pid": os.getpid(), "caps": CAPS})
                return
            if req.get("op") == "embed":
                texts = req.get("texts") or []
                if not isinstance(texts, list):
                    self._send({"error": "texts must be a list"})
                    return
                vecs = state.embed(str(req.get("kind", "")),
                                   [str(t) for t in texts])
                self._send({"vecs": vecs})
                return
            if req.get("op") == "stop":
                self._send({"ok": True})
                threading.Thread(
                    target=self.server.shutdown, daemon=True
                ).start()
                return
            query = str(req.get("q", ""))
            if not query.strip():
                self._send({"hits": []})
                return
            hits = state.search(
                query,
                int(req.get("k", 30)),
                req.get("rooms"),
                req.get("tiers"),
            )
            self._send({"hits": hits})
        except Exception as exc:
            self._send({"error": f"{type(exc).__name__}: {exc}"})

    def _send(self, payload: dict) -> None:
        # どの応答にも身元を添える(窓口はこれで自分のデーモンかを見分ける)
        payload = {**payload, "home": self.server.state.settings.home_key}  # type: ignore[attr-defined]
        try:
            self.wfile.write(
                (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            )
        except OSError:
            pass


class _Server(socketserver.ThreadingTCPServer):
    # Windows の SO_REUSEADDR は「使用中のポートに後から重ねて bind できる」
    # 意味になり、単一インスタンス保証(固定ポートの bind 失敗)が効かない。
    # R8.10.23 実測: 窓口3つが同時に起動を仕掛けると、同じポートで待ち受ける
    # デーモンが3つ立った。Windows では付けない(POSIX では再起動直後の
    # TIME_WAIT を避ける本来の意味なので付ける)
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def serve(settings: Settings | None = None) -> int:
    """デーモンを起動する(前面。呼び出し元が終わるまで戻らない)。"""
    settings = settings or get_settings()

    # 単一インスタンス保証: 決まったポートを握れなければ既に誰かが動いている。
    # モデルのロード(数秒・数百MB)より前に判定するのが要点。
    # daemon_port=0(テスト)のときは空きポートを取るので排他は効かない
    if settings.daemon_port_effective:
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("127.0.0.1", settings.daemon_port_effective))
        except OSError:
            log(settings, "既に起動済みのため終了(二重起動の抑止)")
            return 0
        probe.close()

    log(settings, "起動開始(埋め込みモデルの読み込み)")
    try:
        state = _State(settings)
    except Exception as exc:
        log(settings, f"起動失敗: {type(exc).__name__}: {exc}")
        return 1

    try:
        server = _Server(("127.0.0.1", settings.daemon_port_effective), _Handler)
    except OSError as exc:
        log(settings, f"待受に失敗: {exc}")
        return 1
    server.state = state  # type: ignore[attr-defined]
    host, port = server.server_address

    path = port_file(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"port": port, "pid": os.getpid(),
                    "started_at": time.time()}),
        encoding="utf-8",
    )

    idle_timeout = max(60.0, settings.daemon_idle_hours * 3600.0)

    def _idle_watch() -> None:
        while True:
            time.sleep(30)
            if time.time() - state.last_seen > idle_timeout:
                log(settings, f"無操作{settings.daemon_idle_hours:g}時間のため終了")
                server.shutdown()
                return

    threading.Thread(target=_idle_watch, daemon=True).start()
    log(settings, f"待受開始 port={port}")
    try:
        server.serve_forever(poll_interval=1.0)
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        log(settings, "終了")
    return 0


# ---------------------------------------------------------------------------
# クライアント(フック・CLI から使う)
# ---------------------------------------------------------------------------

def _read_port(settings: Settings) -> int | None:
    """待受ポート。ポートファイルがあればそれを、無ければ固定ポートを試す。

    ポートファイルはデーモンが終了時に消すが、強制終了だと残ることがある。
    固定ポートを既定にしておけば、ファイルが無くても繋がる。
    """
    try:
        data = json.loads(port_file(settings).read_text(encoding="utf-8"))
        port = int(data.get("port") or 0)
        if port:
            return port
    except Exception:
        pass
    return settings.daemon_port_effective or None


def request(settings: Settings, payload: dict, *,
            timeout: float = 2.0, limit: int = _RECV_LIMIT) -> dict | None:
    """デーモンへ1リクエスト。届かなければ None(呼び出し元は縮退する)。

    依頼には自分の身元(home)を添え、応答の身元が自分と違えば None にする。
    番号が同じでも別の記憶を持つデーモン(同じPCの2つめの導入・身元を
    名乗らない旧版)の応答は使わない。

    limit は応答の上限バイト数。埋め込みの依頼では _EMBED_LIMIT を渡す。
    """
    port = _read_port(settings)
    if not port:
        return None
    payload = {**payload, "home": settings.home_key}
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(
                (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            )
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(8192)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > limit:
                    return None
        res = json.loads(buf.decode("utf-8", errors="replace"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(res, dict) or res.get("home") != settings.home_key:
        return None
    return res


def is_alive(settings: Settings) -> bool:
    res = request(settings, {"op": "ping"}, timeout=1.0)
    return bool(res and res.get("ok"))


def capabilities(settings: Settings) -> list[str] | None:
    """生きているデーモンが受け付ける依頼の一覧。不在なら None。

    caps を返さない旧版のデーモンは検索だけを受け付ける(["search"])。
    """
    res = request(settings, {"op": "ping"}, timeout=1.0)
    if not (res and res.get("ok")):
        return None
    caps = res.get("caps")
    return list(caps) if isinstance(caps, list) else ["search"]


def embed(settings: Settings, kind: str, texts: list[str], *,
          timeout: float = 30.0) -> list[list[float]] | None:
    """デーモンに埋め込みを頼む。届かない・旧版で未対応・エラーなら None。"""
    res = request(settings, {"op": "embed", "kind": kind, "texts": texts},
                  timeout=timeout, limit=_EMBED_LIMIT)
    if not res or not isinstance(res.get("vecs"), list):
        return None
    return res["vecs"]


def stop(settings: Settings) -> bool:
    # 先に身元を確かめてから止める。身元を名乗らない旧版のデーモンは
    # 依頼の home を見ずに止まってしまうため、ping で自分のものと
    # 確かめられたときだけ stop を送る
    if not is_alive(settings):
        return False
    res = request(settings, {"op": "stop"}, timeout=2.0)
    return bool(res and res.get("ok"))


def _recently_spawned(settings: Settings) -> bool:
    """直前の起動試行から冷却時間内か(連打で積み上がるのを防ぐ)。"""
    try:
        last = _spawn_marker(settings).stat().st_mtime
        return (time.time() - last) < _SPAWN_COOLDOWN
    except OSError:
        return False


def _can_import_engram(python: Path, settings: Settings | None = None) -> bool:
    """その Python が engram を import できるか(別Pythonでの無言の滞留を防ぐ)。

    結果は環境の性質で変わらないのでファイルに覚える。毎発話の経路から
    呼ばれても子プロセスを起こさないようにするため(確認そのものが重い)。
    """
    cache_path = None
    if settings is not None:
        cache_path = settings.data_dir / "daemon.import_ok.json"
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            hit = cache.get(str(python))
            if isinstance(hit, bool):
                return hit
        except Exception:
            cache = {}
    else:
        cache = {}

    try:
        import subprocess

        # stdin を引き継がない。MCP サーバーの中から呼ぶと、標準入力の管を子が
        # 握ったまま止まり、20秒で打ち切られて「import できない」と誤判定していた
        # (R8.10.30 版比較の測定で発見。最初の recall が41秒かかり、常駐デーモンが起動しなかった)
        r = subprocess.run(
            [str(python), "-c", "import engram"],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=20,
        )
        ok = r.returncode == 0
    except Exception:
        # 打ち切り・起動失敗は環境の性質ではないので覚えない(次の機会に確かめ直す)
        return False

    if cache_path is not None:
        try:
            cache[str(python)] = ok
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass
    return ok


def ensure_running(settings: Settings) -> bool:
    """生きていなければ背後で起動する(起動を待たない)。

    ENGRAM_DISABLE_DAEMON=1 のときは何もしない(テストが実プロセスを
    作らないための安全弁)。

    起動には埋め込みモデルのロード(数秒)がかかるため、呼んだその回は
    間に合わない。次の発話から効く——毎発話の遅延を作らないための設計。

    多重起動の防止は3段構え: ①冷却時間内なら試みない ②起動する Python が
    engram を import できるか確かめる ③デーモン自身が固定ポートを握れなければ
    即終了する。実機で40プロセスの暴走を観測して追加した(R8.8.27)。
    """
    if os.environ.get("ENGRAM_DISABLE_DAEMON"):
        return False
    if is_alive(settings):
        return True
    if _recently_spawned(settings):
        return False
    try:
        import subprocess

        exe = Path(sys.executable)
        # pythonw があればコンソール窓を出さずに常駐させる
        pyw = exe.with_name("pythonw.exe")
        runner = pyw if pyw.is_file() else exe
        if not _can_import_engram(exe, settings):
            log(settings, f"起動を見送り(engram を import できない: {exe})")
            return False
        try:
            _spawn_marker(settings).write_text(str(time.time()),
                                               encoding="utf-8")
        except OSError:
            pass
        cmd = [str(runner), "-m", "engram.cli", "daemon"]
        kwargs: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "nt":
            # 親(フック)の終了に巻き込まれないよう切り離す
            kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(cmd, **kwargs)
        log(settings, f"起動を仕掛けた: {runner}")
        return False
    except Exception as exc:
        log(settings, f"起動の仕掛けに失敗: {exc!r}")
        return False


def semantic_candidates(settings: Settings, query: str, *, k: int,
                        rooms: list[str] | None = None,
                        tiers: list[str] | None = None) -> dict[str, float]:
    """意味の近い候補 {id: 関連度(コサイン)} を返す。デーモン不在なら空。"""
    res = request(settings, {"q": query, "k": k, "rooms": rooms,
                             "tiers": tiers})
    if not res or "hits" not in res:
        return {}
    return {h["id"]: float(h["relevance"]) for h in res["hits"]
            if h.get("id")}
