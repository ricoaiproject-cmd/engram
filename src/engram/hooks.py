"""エージェントのフックから呼ばれる入口(①自動符号化・②自発的想起)。

Claude Code の hooks(~/.claude/settings.json)に登録され、stdin の JSON を
受け取って動く:

- SessionEnd        → run_session_end : セッションを episode として自動保存
- UserPromptSubmit  → run_user_prompt : 関連記憶の自発的浮上(surface)

設計原則: フックは絶対にエージェントの動作を妨げない。
あらゆる失敗を握りつぶして exit 0 で返し、経緯は ~/.engram/hooks.log に残す。
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import Settings, get_settings, resolve_room

_ENCODED_SESSIONS_KEEP = 500

# 人間の発言ではない、エージェント/ハーネスが差し込むテキストの先頭部。
# これらに自発的想起を反応させると、システム通知のたびに無関係な記憶が
# 浮上してノイズになる(実ログで確認)。スラッシュコマンドも対象外。
_NON_HUMAN_PREFIXES = (
    "/",                       # スラッシュコマンド
    "<",                       # <task-notification> / <command-name> / <!-- ... 等
    "Caveat:",                 # ローカルコマンド実行時の注意書き
    "[Request interrupted",    # 中断通知
)


def _is_non_human(text: str) -> bool:
    """発言が人間由来でない(システム/ハーネスのテキスト)かを判定する。"""
    return any(text.startswith(p) for p in _NON_HUMAN_PREFIXES)


def _log(settings: Settings, message: str) -> None:
    try:
        path = settings.data_dir / "hooks.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{stamp} {message}\n")
    except OSError:
        pass


def _read_stdin_json(stdin_text: str | None) -> dict:
    """stdin の JSON を読む。

    フックの呼び出し元(Claude Code = Node)は UTF-8 で書き込むが、Windows の
    Python はリダイレクトされた stdin を cp932 で解釈してしまい、日本語の
    プロンプトやパス(例: マイドライブ)が壊れる(実機で発生)。テキスト層を
    経由せず、バイト列を直接 UTF-8 デコードする。
    """
    if stdin_text is None:
        try:
            stdin_text = sys.stdin.buffer.read().decode("utf-8",
                                                        errors="replace")
        except Exception:
            try:
                stdin_text = sys.stdin.read()
            except Exception:
                return {}
    try:
        data = json.loads(stdin_text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# ① 自動符号化(SessionEnd)
# ---------------------------------------------------------------------------

def _already_encoded(settings: Settings, session_id: str) -> bool:
    path = settings.data_dir / "encoded_sessions.txt"
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8").split()
        return session_id in lines
    except OSError:
        return False


def _mark_encoded(settings: Settings, session_id: str) -> None:
    path = settings.data_dir / "encoded_sessions.txt"
    try:
        lines: list[str] = []
        if path.is_file():
            lines = path.read_text(encoding="utf-8").split()
        lines.append(session_id)
        path.write_text(
            "\n".join(lines[-_ENCODED_SESSIONS_KEEP:]) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 統合の自動促し(consolidation nudge)
#
# 要約は LLM(エージェント)にしかできないため、engram 自身は統合を実行できない。
# 代わりに SessionEnd で候補クラスタ数を状態ファイルへ棚卸しし、次のセッションの
# UserPromptSubmit(軽量経路)が閾値超過を検知したらエージェントに統合作業を促す。
# 人間の「睡眠中の記憶整理」を、エージェントの手が空いた瞬間に移した設計。
# ---------------------------------------------------------------------------

def _consolidation_state_path(settings: Settings) -> Path:
    return settings.data_dir / "consolidation_state.json"


def _read_consolidation_state(settings: Settings) -> dict:
    try:
        return json.loads(
            _consolidation_state_path(settings).read_text(encoding="utf-8")
        )
    except Exception:
        return {}


def _write_consolidation_state(settings: Settings, updates: dict) -> None:
    state = _read_consolidation_state(settings)
    state.update(updates)
    path = _consolidation_state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _consolidation_nudge(settings: Settings, now: float | None = None) -> str | None:
    """統合候補が溜まっていたら促し文を返す(エンジン不使用の軽量判定)。

    促し過ぎはノイズなので、クラスタ数の閾値と促しの最短間隔の両方で絞る。
    状態ファイルへの記録に失敗したときは促さない(毎プロンプト促す事故防止)。
    """
    if not settings.consolidate_nudge:
        return None
    state = _read_consolidation_state(settings)
    clusters = int(state.get("clusters", 0) or 0)
    if clusters < settings.consolidate_nudge_min_clusters:
        return None
    ts = now if now is not None else time.time()
    last = float(state.get("last_nudged_at", 0.0) or 0.0)
    if ts - last < settings.consolidate_nudge_interval_days * 86400.0:
        return None
    try:
        _write_consolidation_state(settings, {"last_nudged_at": ts})
    except OSError:
        return None
    return (
        f"(engram 記憶整理) 統合候補の episode が {clusters} クラスタ溜まっています。"
        "区切りの良いところで consolidation_candidates を呼び、クラスタごとに"
        "内容を要約して remember(type=knowledge, 元クラスタの経緯を残す)で保存し、"
        "mark_consolidated(episode_ids, new_memory_id) で元 episode を降格して"
        "ください(人間の睡眠中の記憶整理に相当する保守作業です)。"
    )


# ---------------------------------------------------------------------------
# スキル化候補の自動促し(skill nudge)
#
# consolidation nudge と完全に相似形の機構。同じ形の作業を記録した episode が
# 一定件数(三度ルール)以上クラスタを成したら、手順として切り出す(スキル化)
# 価値があるとみて、エージェントにユーザーへの提案を促す。状態ファイルは
# consolidation と共用(consolidation_state.json)し、キーだけ分ける。
# ---------------------------------------------------------------------------

def _skill_nudge(settings: Settings, now: float | None = None) -> str | None:
    """スキル化候補が溜まっていたら促し文を返す(エンジン不使用の軽量判定)。

    consolidate_nudge と同じゲート構造: 設定 off / クラスタ数が閾値未満 /
    最短間隔未経過 / 状態ファイルへの書き込み失敗、のいずれかなら None。
    """
    if not settings.skill_nudge:
        return None
    state = _read_consolidation_state(settings)
    clusters = int(state.get("skill_clusters", 0) or 0)
    if clusters < settings.skill_nudge_min_clusters:
        return None
    ts = now if now is not None else time.time()
    last = float(state.get("last_skill_nudged_at", 0.0) or 0.0)
    if ts - last < settings.skill_nudge_interval_days * 86400.0:
        return None
    try:
        _write_consolidation_state(settings, {"last_skill_nudged_at": ts})
    except OSError:
        return None
    if settings.skill_autonomy:
        return (
            f"(engram スキル化候補) 同じ形の作業を記録した episode が {clusters} "
            f"クラスタ溜まっています(1クラスタ{settings.skill_min_count}件以上・"
            "三度ルール)。運用方針は自律作成です: 区切りの良いところで "
            "skill_candidates を呼び、手順として再利用価値があるクラスタは"
            "承認を待たずスキル(SKILL.md)を自動作成してよい。置き場所は"
            "自動生成枠(スキルフォルダの auto-<名前>/、frontmatter に "
            "auto: true と作成日を明記)。実需の反復か1案件内の反復(デモ・例示)"
            "かを内容から判断し、迷いは作る側に倒す(多少の誤りは許容・"
            "使われないものは代謝が引退させる)。作成/見送りの経緯を "
            "remember(type=knowledge) で記録し、mark_consolidated で"
            "元 episode を整理すること。"
        )
    return (
        f"(engram スキル化候補) 同じ形の作業を記録した episode が {clusters} "
        f"クラスタ溜まっています(1クラスタ{settings.skill_min_count}件以上・"
        "三度ルール)。区切りの良いところで skill_candidates を呼び、手順として"
        "再利用できそうならスキル化(手順書の切り出し)をユーザーに提案してください。"
        "勝手に作らず必ず承認を得ること。採用・見送りが決まったら、経緯を"
        "remember(type=knowledge) で記録し、mark_consolidated(episode_ids, "
        "new_memory_id) で元 episode を整理してください。"
    )


# ---------------------------------------------------------------------------
# 着手前 recall の促し(recall nudge)
#
# consolidation / skill nudge と同型の第3のナッジ。セッション最初の実質的な
# 人間発話で一度だけ「recall してから着手」を促す。マーカーは surface の
# セッション状態と同じ命名規則(session-*.json)に乗せ、_save_surfaced_ids の
# age ベース掃除に相乗りする。
# ---------------------------------------------------------------------------

def _recall_nudge(settings: Settings, session_id: str) -> str | None:
    """セッション最初の実質発話で一度だけ、着手前 recall の促し文を返す。

    マーカーファイルへの記録に失敗したときは促さない(毎プロンプト促す
    事故防止。consolidation nudge と同じ原則)。
    """
    if not settings.recall_nudge:
        return None
    from .surface import _SESSION_ID_SAFE, _surface_dir

    safe = _SESSION_ID_SAFE.sub("_", session_id) or "unknown"
    marker = _surface_dir(settings) / f"session-{safe}-nudged.json"
    if marker.is_file():
        return None
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}", encoding="utf-8")
    except OSError:
        return None
    return (
        "(engram 想起の促し) このセッションではまだ記憶を検索していません。"
        "作業に着手する前に、タスクの主題で recall を呼び、関連する過去の"
        "知見・流儀・文脈を確認してください。recall ツールが見当たらない"
        "場合は、ツール検索(ToolSearch 等)で読み込んでから呼ぶこと。"
    )


def _last_assistant_text(transcript_path, max_chars: int) -> str:
    """transcript(JSONL)の末尾から直前のアシスタント発話を取り出す。

    毎発話時に呼ばれるため全文は読まない(末尾 64KB のみ)。壊れた行は無視。
    """
    try:
        path = Path(transcript_path)
        if not transcript_path or not path.is_file():
            return ""
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > 65536:
                f.seek(size - 65536)
            tail = f.read().decode("utf-8", errors="replace")
        from .transcript import _texts_from_content

        for line in reversed(tail.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict) or obj.get("type") != "assistant":
                continue
            message = obj.get("message")
            if not isinstance(message, dict):
                continue
            texts = [
                x for x in _texts_from_content(message.get("content"))
                if x.strip()
            ]
            if texts:
                return texts[-1].strip()[-max_chars:]
        return ""
    except Exception:
        return ""


def _first_prompt_recall(settings: Settings, prompt: str, cwd) -> str | None:
    """セッション初回の実質発話をクエリに recall を代行実行し、注入文を返す。

    「ナッジ(促し)をやめて配管(代行)にする」(R8.8.12 の SessionStart 代行と
    同じ思想)の初回発話版。エンジン直呼びなのでモデル非依存。失敗時は None を
    返し、呼び出し元が従来の recall_nudge 文へ縮退する。
    """
    try:
        from .engine import build_engine

        # セッション最初の1回だけ、意味の網(常駐デーモン)の起動を仕掛ける。
        # 毎発話でやると起動失敗時にプロセスが積み上がる(R8.8.27の事故)
        if settings.surface_semantic and settings.surface_autostart_daemon:
            try:
                from . import daemon as _daemon

                _daemon.ensure_running(settings)
            except Exception:
                pass

        room = resolve_room(cwd, settings.room_paths)
        engine = build_engine(settings)
        try:
            result = engine.recall(
                query=prompt,
                mode="fast",
                limit=settings.first_prompt_recall_limit,
                room=room,
                # 機械の巡回で活性度を汲み上げない(R8.8.27)。代行recallが
                # 毎セッション recall_hit を打つと、活性度の富者益富ループが
                # 裏口から回り、古参が新参を押し出し続ける(実測:
                # recall_hit 17,038件のうち相当部分がこの経路)
                record_hits=False,
            )
        finally:
            engine.db.close()
        hits = result.get("hits", [])
        if not hits:
            return (
                "(engram 話題想起) 初回発話をクエリに recall を代行実行しましたが、"
                "関連する記憶は見つかりませんでした。工程が変わったら、その工程名で"
                " recall を自分で呼ぶこと。"
            )
        lines = [
            "(engram 話題想起) セッション初回の発話をクエリに recall を代行実行した"
            "結果です(字句一致の surface より深い意味検索)。検証してから使い、"
            "実際に役立った記憶は reinforce してください:",
            "",
        ]
        for h in hits:
            content = " ".join((h.get("content") or "").split())
            if len(content) > 300:
                content = content[:299] + "…"
            lines.append(
                f"- [{h['id']}] ({h.get('type', '?')}/{h.get('room', '?')} "
                f"score {h.get('score', 0):.2f}) {content}"
            )
        lines.append("")
        lines.append(
            "・これは初回発話の1回きりです。工程(調査→実装→レビュー等)が変わったら、"
            "その工程名で recall を呼び直すこと。"
        )
        return "\n".join(lines)
    except Exception as e:
        _log(settings, f"first-prompt recall 失敗(ナッジへ縮退): {e!r}")
        return None


def _detect_implicit_use(settings: Settings, session_id: str,
                         assistant_texts: list[str]) -> int:
    """浮上させた記憶が「申告なしに使われた」形跡を弱く記録する。

    reinforce は申告制なので「役立ったが報告し忘れた」を測れない。この穴が
    命中率を下限値にし、常連抑制の誤爆も生む。SessionEnd で、そのセッションに
    浮上させた記憶の特徴語がアシスタント発話に現れていたら implicit_use を打つ。

    特徴語は「記憶の本文にあり、かつ一般的すぎない語」を字句的に拾うだけ
    (埋め込み不使用)。戻り値は記録した件数。
    """
    if not settings.implicit_use_detect or not assistant_texts:
        return 0
    try:
        import sqlite3

        from .surface import _SESSION_ID_SAFE, _surface_dir

        safe = _SESSION_ID_SAFE.sub("_", session_id) or "unknown"
        state = _surface_dir(settings) / f"session-{safe}.json"
        ids = json.loads(state.read_text(encoding="utf-8")).get("surfaced", [])
        if not ids:
            return 0

        said = " ".join(assistant_texts)
        conn = sqlite3.connect(str(settings.db_path))
        try:
            conn.execute("PRAGMA busy_timeout=2000")
            ph = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT memory_id, content FROM fts_memories "
                f"WHERE memory_id IN ({ph})", ids,
            ).fetchall()
            hits = 0
            now = time.time()
            for mid, content in rows:
                terms = _distinctive_terms(content or "")
                if not terms:
                    continue
                matched = sum(1 for term in terms if term in said)
                if matched >= settings.implicit_use_min_terms:
                    with conn:
                        conn.execute(
                            "INSERT INTO access_events(memory_id, ts, kind, "
                            "weight) VALUES (?,?,?,?)",
                            (mid, now, "implicit_use",
                             settings.implicit_use_weight),
                        )
                    hits += 1
            return hits
        finally:
            conn.close()
    except Exception as e:
        _log(settings, f"implicit-use 検出失敗: {e!r}")
        return 0


def _distinctive_terms(content: str, *, limit: int = 12) -> list[str]:
    """本文から特徴語(長めのカタカナ・英数字・漢字列)を拾う。

    一般語を拾わないよう長さで足切りする。完全な形態素解析はしない
    (フックは軽量経路であり、ここでの目的は「同じ話をしていたか」の粗い判定)。
    """
    import re as _re

    terms = _re.findall(
        r"[A-Za-z][A-Za-z0-9_.-]{3,}|[ァ-ヴー]{4,}|[一-龥]{3,}", content
    )
    seen: list[str] = []
    for term in terms:
        if term not in seen:
            seen.append(term)
        if len(seen) >= limit:
            break
    return seen


def _progress_path(settings: Settings) -> Path:
    return settings.data_dir / "encode_progress.json"


def _read_progress(settings: Settings) -> dict:
    try:
        return json.loads(_progress_path(settings).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_progress(settings: Settings, session_id: str, n: int) -> None:
    state = _read_progress(settings)
    state[session_id] = n
    if len(state) > 500:  # 古い会話の分は捨てる(挿入順で古いもの)
        state = dict(list(state.items())[-500:])
    path = _progress_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def normalize_stop_input(data: dict, agent: str) -> dict | None:
    """エージェントごとの「応答が終わった」フックの入力を、session-end の形に読み替える。

    Antigravity の Stop は会話の途中(サブエージェント待ち fullyIdle=false など)でも呼ばれるので、
    手が空いたとき(fullyIdle=true)だけ記録する。記録しないときは None。
    """
    if agent == "antigravity":
        if data.get("fullyIdle") is not True:
            return None
        ws = data.get("workspacePaths") or []
        return {"session_id": str(data.get("conversationId", "")),
                "transcript_path": data.get("transcriptPath", ""),
                "cwd": ws[0] if isinstance(ws, list) and ws else "",
                "incremental": True}
    return None


def detach_session_end(payload: dict) -> bool:
    """要約を裏のプロセスに任せてすぐ返す。

    Codex は会話の終わりのフックを3秒で打ち切り、Gemini CLI と Antigravity は完了を待たない。
    埋め込みモデルの読み込みと要約はそれより長くかかるので、入力をファイルに置いて別のプロセスで行う。
    """
    try:
        settings = get_settings()
        q = settings.data_dir / "hook_queue"
        q.mkdir(parents=True, exist_ok=True)
        path = q / f"{int(time.time() * 1000)}_{os.getpid()}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        exe = Path(sys.executable)
        pyw = exe.with_name("pythonw.exe")
        runner = str(pyw if pyw.is_file() else exe)
        kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                        "stderr": subprocess.DEVNULL, "close_fds": True}
        if os.name == "nt":
            kwargs["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # 切り離し・新グループ・窓なし
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([runner, "-m", "engram", "hook", "session-end", "--from-file", str(path)], **kwargs)
        return True
    except Exception:
        return False


def run_session_end(stdin_text: str | None = None) -> int:
    """SessionEnd フック本体。常に 0 を返す(セッション終了を妨げない)。"""
    try:
        settings = get_settings()
    except Exception:
        return 0
    try:
        if not settings.auto_encode:
            return 0

        data = _read_stdin_json(stdin_text)
        session_id = str(data.get("session_id", "")) or "unknown"
        transcript_path = data.get("transcript_path", "")
        cwd = data.get("cwd", "")

        if not transcript_path or not Path(transcript_path).is_file():
            _log(settings, f"session-end {session_id}: transcript なし、スキップ")
            return 0

        from .transcript import (
            build_episodes,
            estimate_valence,
            extract_messages,
        )

        messages = extract_messages(transcript_path)
        # 会話の途中で何度も呼ばれるエージェント(Antigravity の Stop)は、前回までに記録した
        # 発言の数を覚えておき、増えた分だけを記録する
        progress_total = None
        if data.get("incremental") and session_id != "unknown":
            done = int(_read_progress(settings).get(session_id, 0))
            users = messages.get("user_texts", [])
            if len(users) <= done:
                _log(settings, f"session-end {session_id}: 新しい発言なし、スキップ")
                return 0
            messages = {**messages, "user_texts": users[done:]}
            progress_total = len(users)
            session_id = f"{session_id}@{done}"
        if session_id != "unknown" and _already_encoded(settings, session_id):
            _log(settings, f"session-end {session_id}: 符号化済み、スキップ")
            return 0
        if progress_total is not None:
            _write_progress(settings, session_id.split("@", 1)[0], progress_total)
        project = Path(cwd or messages.get("cwd") or "").name
        episodes = build_episodes(
            messages,
            date_str=datetime.date.today().isoformat(),
            project=project,
            min_chars=settings.auto_encode_min_chars,
            segment_user_turns=settings.encode_segment_user_turns,
        )
        if not episodes:
            _log(settings, f"session-end {session_id}: 中身が薄いため記録せず")
            return 0
        valence = estimate_valence(messages.get("user_texts", []))

        room = resolve_room(cwd or messages.get("cwd"), settings.room_paths)

        # (G) 暗黙の使用検出: 浮上させた記憶の特徴語がアシスタント発話に
        # 現れていたら、申告が無くても「使われた」として弱く記録する
        n_implicit = _detect_implicit_use(
            settings, session_id, messages.get("assistant_texts") or []
        )
        if n_implicit:
            _log(settings, f"session-end {session_id}: 暗黙の使用 {n_implicit}件")

        # (観察のみ)アシスタント発話を手がかりに shadow surface を流す。
        # assistant_cue_mode="shadow" のときだけ(active は毎発話時に本番注入
        # するため、ここでの二重ログは作らない)
        if (settings.assistant_cue_mode == "shadow"
                and settings.surface_mode != "off"):
            try:
                cue = " ".join(messages.get("assistant_texts") or [])[
                    -settings.assistant_cue_max_chars:
                ]
                if cue.strip():
                    from .surface import run_surface

                    run_surface(
                        cue,
                        settings=settings,
                        room=room,
                        session_id=f"{session_id}#assistant",
                        mode="shadow",
                    )
            except Exception as e:
                _log(settings, f"assistant-cue shadow 失敗: {e!r}")

        # ここで初めて重い依存(埋め込みモデル)をロードする。セッション終了後
        # なのでユーザーの操作は妨げない
        from .engine import build_engine

        engine = build_engine(settings)
        if len(episodes) == 1:
            results = [
                engine.remember(
                    content=episodes[0],
                    type="episode",
                    importance=settings.auto_episode_importance,
                    tags=["auto", "session"],
                    source="auto-encode",
                    room=room,
                    valence=valence,
                )
            ]
        else:
            # 分割パートは意図的に別記録なので重複検知を通さない(テンプレ骨格の
            # 類似で併合される実例あり)。パート同士は時系列リンクで鎖にする
            results = []
            prev_id: str | None = None
            for ep in episodes:
                r = engine._remember_direct(
                    content=ep,
                    type="episode",
                    importance=settings.auto_episode_importance,
                    tags=["auto", "session"],
                    source="auto-encode",
                    related_ids=[prev_id] if prev_id else None,
                    room=room,
                    now=time.time(),
                    valence=valence,
                )
                prev_id = r.get("id")
                results.append(r)
        result = results[-1]
        # 統合候補の棚卸し(エンジン構築済みなので追加コストは僅か)。
        # 候補数を状態ファイルへ書き、次セッションの user-prompt が促す
        if settings.consolidate_nudge:
            try:
                n_clusters = len(
                    engine.consolidation_candidates().get("clusters", [])
                )
                _write_consolidation_state(
                    settings,
                    {"clusters": n_clusters, "checked_at": time.time()},
                )
            except Exception as e:
                _log(settings, f"session-end 統合候補チェック失敗: {e!r}")

        # スキル化候補の棚卸し(consolidation と同じ理由で、ここでまとめて計算)。
        # 候補数を状態ファイルへ書き、次セッションの user-prompt が促す
        if settings.skill_nudge:
            try:
                n_skill_clusters = len(
                    engine.skill_candidates().get("clusters", [])
                )
                _write_consolidation_state(
                    settings,
                    {"skill_clusters": n_skill_clusters, "checked_at": time.time()},
                )
            except Exception as e:
                _log(settings, f"session-end スキル候補チェック失敗: {e!r}")

        engine.db.close()
        _mark_encoded(settings, session_id)
        _log(
            settings,
            f"session-end {session_id}: {result.get('status')} "
            f"episodes={len(results)} id={result.get('id')} "
            f"room={room} valence={valence:+.2f}",
        )
    except Exception as e:
        _log(settings, f"session-end エラー: {e!r}")
    return 0


# ---------------------------------------------------------------------------
# ② 自発的想起(UserPromptSubmit)
# ---------------------------------------------------------------------------

def run_user_prompt(stdin_text: str | None = None, event_name: str = "UserPromptSubmit") -> int:
    """UserPromptSubmit フック本体。常に 0 を返す(プロンプトを妨げない)。

    active モードでは hookSpecificOutput.additionalContext として浮上記憶を
    出力し、エージェントの文脈に差し込まれる。shadow モードではログのみ。
    """
    try:
        settings = get_settings()
    except Exception:
        return 0
    try:
        data = _read_stdin_json(stdin_text)
        prompt = str(data.get("prompt", ""))
        session_id = str(data.get("session_id", "")) or "unknown"
        cwd = data.get("cwd", "")

        stripped = prompt.strip()
        if len(stripped) < settings.surface_min_prompt_chars:
            return 0
        if _is_non_human(stripped):  # システム/ハーネスのテキストは対象外
            return 0

        context_parts: list[str] = []

        # 着手前 recall(セッション初回のみ。surface とは独立)。
        # 既定は「代行」(発話文で recall を実行して結果を注入)、失敗時は
        # 従来の「促し」へ縮退する。マーカー管理は _recall_nudge に相乗り
        recall_nudge = _recall_nudge(settings, session_id)
        if recall_nudge:
            injected = None
            if settings.first_prompt_recall:
                injected = _first_prompt_recall(settings, stripped, cwd)
            context_parts.append(injected or recall_nudge)

        if settings.surface_mode != "off":
            room = resolve_room(cwd, settings.room_paths)

            from .surface import format_context, run_surface

            result = run_surface(
                prompt,
                settings=settings,
                room=room,
                session_id=session_id,
            )

            if settings.surface_mode == "active" and (result.get("surfaced_items") or result.get("fringe_items")):
                context_parts.append(format_context(result.get("surfaced_items") or [],
                                                    fringe_items=result.get("fringe_items") or []))

            # 自分の思考をキューにする連想(assistant cue・本番)。直前の
            # アシスタント発話を手がかりに surface し、浮上分を追加注入する。
            # 同じ session_id を使うので発話由来との重複浮上は自動で抑止される
            if settings.assistant_cue_mode == "active":
                cue = _last_assistant_text(
                    data.get("transcript_path", ""),
                    settings.assistant_cue_max_chars,
                )
                if cue:
                    cue_result = run_surface(
                        cue,
                        settings=settings,
                        room=room,
                        session_id=session_id,
                        log_extra={"cue": "assistant"},
                    )
                    if (settings.surface_mode == "active"
                            and cue_result.get("surfaced_items")):
                        context_parts.append(format_context(
                            cue_result["surfaced_items"],
                            lead=(
                                "(engram 連想想起)以下は、直前のあなた自身の"
                                "思考(応答)から連想して浮上した記憶です。"
                            ),
                        ))

        # 媒体の自発的想起(0.24)。文章の surface とは別の棚(画像・文書)。索引係が
        # 寝ていれば起こすだけで今回は出さない。失敗しても発話を妨げない
        if getattr(settings, "media_enabled", False) and getattr(settings, "media_surface_mode", "off") != "off":
            try:
                from .media import format_surface, surface as media_surface

                room = resolve_room(cwd, settings.room_paths)
                mres = media_surface(settings, stripped, room=room, session_id=session_id)
                if settings.media_surface_mode == "active" and mres.get("items"):
                    context_parts.append(format_surface(mres["items"]))
            except Exception as e:  # 媒体は必須ではない
                _log(settings, f"media surface スキップ: {e!r}")

        # 統合の促し(surface とは独立。consolidate_nudge 設定でゲート)
        nudge = _consolidation_nudge(settings)
        if nudge:
            context_parts.append(nudge)

        # スキル化候補の促し(consolidation とは独立。skill_nudge 設定でゲート)
        skill_nudge = _skill_nudge(settings)
        if skill_nudge:
            context_parts.append(skill_nudge)

        if context_parts:
            payload = {
                "hookSpecificOutput": {
                    # Claude Code / Codex は UserPromptSubmit、Gemini CLI は BeforeAgent
                    "hookEventName": event_name,
                    "additionalContext": "\n\n".join(context_parts),
                }
            }
            print(json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        _log(settings, f"user-prompt エラー: {e!r}")
    return 0
