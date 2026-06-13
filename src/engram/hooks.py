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
import sys
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
        if session_id != "unknown" and _already_encoded(settings, session_id):
            _log(settings, f"session-end {session_id}: 符号化済み、スキップ")
            return 0

        from .transcript import build_episode, extract_messages

        messages = extract_messages(transcript_path)
        project = Path(cwd or messages.get("cwd") or "").name
        episode = build_episode(
            messages,
            date_str=datetime.date.today().isoformat(),
            project=project,
            min_chars=settings.auto_encode_min_chars,
        )
        if episode is None:
            _log(settings, f"session-end {session_id}: 中身が薄いため記録せず")
            return 0

        room = resolve_room(cwd or messages.get("cwd"), settings.room_paths)

        # ここで初めて重い依存(埋め込みモデル)をロードする。セッション終了後
        # なのでユーザーの操作は妨げない
        from .engine import build_engine

        engine = build_engine(settings)
        result = engine.remember(
            content=episode,
            type="episode",
            importance=settings.auto_episode_importance,
            tags=["auto", "session"],
            source="auto-encode",
            room=room,
        )
        engine.db.close()
        _mark_encoded(settings, session_id)
        _log(
            settings,
            f"session-end {session_id}: {result.get('status')} "
            f"id={result.get('id')} room={room}",
        )
    except Exception as e:
        _log(settings, f"session-end エラー: {e!r}")
    return 0


# ---------------------------------------------------------------------------
# ② 自発的想起(UserPromptSubmit)
# ---------------------------------------------------------------------------

def run_user_prompt(stdin_text: str | None = None) -> int:
    """UserPromptSubmit フック本体。常に 0 を返す(プロンプトを妨げない)。

    active モードでは hookSpecificOutput.additionalContext として浮上記憶を
    出力し、エージェントの文脈に差し込まれる。shadow モードではログのみ。
    """
    try:
        settings = get_settings()
    except Exception:
        return 0
    try:
        if settings.surface_mode == "off":
            return 0

        data = _read_stdin_json(stdin_text)
        prompt = str(data.get("prompt", ""))
        session_id = str(data.get("session_id", "")) or "unknown"
        cwd = data.get("cwd", "")

        stripped = prompt.strip()
        if len(stripped) < settings.surface_min_prompt_chars:
            return 0
        if _is_non_human(stripped):  # システム/ハーネスのテキストは対象外
            return 0

        room = resolve_room(cwd, settings.room_paths)

        from .surface import format_context, run_surface

        result = run_surface(
            prompt,
            settings=settings,
            room=room,
            session_id=session_id,
        )

        if settings.surface_mode == "active" and result.get("surfaced_items"):
            payload = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": format_context(
                        result["surfaced_items"]
                    ),
                }
            }
            print(json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        _log(settings, f"user-prompt エラー: {e!r}")
    return 0
