"""Claude Code の transcript(JSONL)からセッション要約を作る(①自動符号化)。

LLM を使わない決定的要約: セッションの背骨はユーザー発言なので、
「最初の依頼 + 途中の主な発言 + 最後の応答の要旨」を機械的に組み立てる。
粗い記録だが「セッションが終われば勝手にエピソード記憶になる」ことが価値。
精密な知見はエージェントがその場で remember する(役割分担)。
"""

from __future__ import annotations

import json
from pathlib import Path

# Claude Code がユーザー発言として transcript に混ぜ込む非発話テキストの先頭部
_NOISE_PREFIXES = (
    "<",            # <command-name> / <system-reminder> / <local-command-stdout> 等
    "Caveat:",      # ローカルコマンド実行時の注意書き
    "[Request interrupted",
)


def _texts_from_content(content) -> list[str]:
    """message.content(str または ブロックの list)からテキストを取り出す。"""
    if isinstance(content, str):
        return [content]
    texts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    texts.append(t)
    return texts


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return any(stripped.startswith(p) for p in _NOISE_PREFIXES)


def extract_messages(transcript_path: str | Path) -> dict:
    """transcript JSONL を走査して要約の材料を返す。

    返り値: {
        "user_texts": [ユーザーの実発言(時系列)],
        "last_assistant": 最後のアシスタント発言(無ければ ""),
        "summary": Claude Code が付けたセッション表題(無ければ ""),
        "cwd": transcript に記録された作業ディレクトリ(無ければ ""),
    }
    壊れた行・未知の形式は黙ってスキップする(フックを落とさない)。
    """
    user_texts: list[str] = []
    last_assistant = ""
    summary = ""
    cwd = ""

    path = Path(transcript_path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            if not cwd and isinstance(obj.get("cwd"), str):
                cwd = obj["cwd"]

            kind = obj.get("type")
            if kind == "summary":
                s = obj.get("summary")
                if isinstance(s, str) and not summary:
                    summary = s.strip()
                continue
            if obj.get("isMeta"):
                continue

            message = obj.get("message")
            if not isinstance(message, dict):
                continue

            if kind == "user" and message.get("role") == "user":
                for t in _texts_from_content(message.get("content")):
                    if not _is_noise(t):
                        user_texts.append(t.strip())
            elif kind == "assistant":
                for t in _texts_from_content(message.get("content")):
                    if t.strip():
                        last_assistant = t.strip()

    return {
        "user_texts": user_texts,
        "last_assistant": last_assistant,
        "summary": summary,
        "cwd": cwd,
    }


def _clip(text: str, limit: int) -> str:
    """1行に潰して limit 文字に丸める。"""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def build_episode(
    messages: dict,
    *,
    date_str: str,
    project: str = "",
    min_chars: int = 24,
    max_user_items: int = 6,
) -> str | None:
    """extract_messages の結果から episode 本文を組み立てる。

    記録に値しないセッション(実発言なし・極端に短い)は None を返す。
    """
    user_texts: list[str] = messages.get("user_texts", [])
    if not user_texts:
        return None
    if sum(len(t) for t in user_texts) < min_chars:
        return None

    title = messages.get("summary") or _clip(user_texts[0], 60)
    where = f"、{project}" if project else ""

    lines: list[str] = [f"セッション体験({date_str}{where}): {_clip(title, 80)}", ""]
    lines.append("ユーザーの依頼・発言:")
    lines.append(f"1. {_clip(user_texts[0], 200)}")

    # 2件目以降: 件数が多ければ前半と後半から拾う(中盤の相づちより情報が濃い)
    rest = user_texts[1:]
    if len(rest) > max_user_items - 1:
        head_n = (max_user_items - 1) // 2
        tail_n = max_user_items - 1 - head_n
        picked = rest[:head_n] + rest[-tail_n:]
        omitted = len(rest) - len(picked)
    else:
        picked = rest
        omitted = 0
    for i, t in enumerate(picked, start=2):
        lines.append(f"{i}. {_clip(t, 110)}")
    if omitted > 0:
        lines.append(f"(ほか {omitted} 件の発言)")

    last_assistant = messages.get("last_assistant", "")
    if last_assistant:
        lines.append("")
        lines.append(f"結末(最後の応答の要旨): {_clip(last_assistant, 280)}")

    return "\n".join(lines)
