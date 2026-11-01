"""Claude Code の transcript(JSONL)からセッション要約を作る(①自動符号化)。

LLM を使わない決定的要約: セッションの背骨はユーザー発言なので、
「最初の依頼 + 途中の主な発言 + 最後の応答の要旨」を機械的に組み立てる。
粗い記録だが「セッションが終われば勝手にエピソード記憶になる」ことが価値。
精密な知見はエージェントがその場で remember する(役割分担)。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Claude Code がユーザー発言として transcript に混ぜ込む非発話テキストの先頭部
_NOISE_PREFIXES = (
    "<",            # <command-name> / <system-reminder> / <local-command-stdout> 等
    "Caveat:",      # ローカルコマンド実行時の注意書き
    "[Request interrupted",
    "# AGENTS.md instructions",  # Codex が利用者の発言として差し込む指示ファイル
)

_AGY_REQUEST = re.compile(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", re.S)


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
    assistant_texts: list[str] = []   # 直近のアシスタント発言(最大3件)
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

            # Antigravity の記録(brain/<id>/.system_generated/logs/transcript_full.jsonl):
            # 1行1ステップ。利用者の発言は USER_INPUT の content(<USER_REQUEST> で囲まれる)、
            # 応答は PLANNER_RESPONSE の content(本文の無いステップもある)
            if kind in ("USER_INPUT", "PLANNER_RESPONSE") and isinstance(obj.get("content"), str):
                text = obj["content"]
                if kind == "USER_INPUT":
                    m = _AGY_REQUEST.search(text)
                    text = m.group(1) if m else text
                    if not _is_noise(text):
                        user_texts.append(text.strip())
                elif text.strip():
                    last_assistant = text.strip()
                    assistant_texts.append(last_assistant)
                    if len(assistant_texts) > 3:
                        assistant_texts.pop(0)
                continue

            # Codex の記録: {"type": "response_item", "payload": {"type": "message", "role": ...,
            # "content": [{"type": "input_text" | "output_text", "text": ...}]}}。作業ディレクトリは
            # session_meta / turn_context の payload.cwd にある。Claude Code と同じ形に読み替える
            payload = obj.get("payload")
            if isinstance(payload, dict):
                if not cwd and isinstance(payload.get("cwd"), str):
                    cwd = payload["cwd"]
                if kind == "response_item" and payload.get("type") == "message":
                    blocks = [{"type": "text", "text": b.get("text")}
                              for b in payload.get("content") or []
                              if isinstance(b, dict) and b.get("type") in ("input_text", "output_text")]
                    kind = {"user": "user", "assistant": "assistant"}.get(payload.get("role"))
                    obj = {"message": {"role": kind, "content": blocks}}

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
                        assistant_texts.append(last_assistant)
                        if len(assistant_texts) > 3:
                            assistant_texts.pop(0)

    return {
        "user_texts": user_texts,
        "assistant_texts": assistant_texts,
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

# 感情価の推定に使う語(決定的ヒューリスティック。LLM 不使用)
_NEGATIVE_MARKERS = (
    "訂正", "間違", "違う", "ダメ", "だめ", "やり直", "事故", "失敗",
    "エラー", "おかしい", "忘れすぎ", "できていない", "止まって",
)
_POSITIVE_MARKERS = (
    "ありがとう", "助かる", "助かった", "いいね", "完璧", "素晴らし",
    "成功", "合格", "その通り", "まさにそれ",
)


def estimate_valence(user_texts: list[str]) -> float:
    """ユーザー発言から episode の感情価(-0.8..0.8)を推定する。

    人間の符号化優先度は感情が自動でつける(扁桃体)。その最小版として、
    訂正・失敗の語(負)と称賛・成功の語(正)の出現数で符号化する。
    粗い代理指標であり、精密な感情価はエージェントが remember 時に付ける。
    """
    text = " ".join(user_texts)
    neg = sum(text.count(m) for m in _NEGATIVE_MARKERS)
    pos = sum(text.count(m) for m in _POSITIVE_MARKERS)
    return max(-0.8, min(0.8, 0.2 * pos - 0.25 * neg))


def build_episodes(
    messages: dict,
    *,
    date_str: str,
    project: str = "",
    min_chars: int = 24,
    max_user_items: int = 6,
    segment_user_turns: int = 0,
) -> list[str]:
    """セッションを episode 本文のリストに組み立てる(粒度の設定つき)。

    segment_user_turns > 0 かつユーザー発言がそれを超える長いセッションは、
    発言をその件数ごとに区切って複数の episode にする(パート i/n を明記)。
    1セッション1要約では長丁場の中盤が捨てられるため(「代謝する脳」Phase 3)。
    それ以外は従来どおり build_episode 1件。
    """
    user_texts: list[str] = messages.get("user_texts", [])
    if not user_texts:
        return []
    if sum(len(t) for t in user_texts) < min_chars:
        return []

    n_seg = segment_user_turns
    if n_seg <= 0 or len(user_texts) <= n_seg:
        ep = build_episode(
            messages,
            date_str=date_str,
            project=project,
            min_chars=min_chars,
            max_user_items=max_user_items,
        )
        return [ep] if ep else []

    title = messages.get("summary") or _clip(user_texts[0], 60)
    chunks = [
        user_texts[i:i + n_seg] for i in range(0, len(user_texts), n_seg)
    ]
    total = len(chunks)
    episodes: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        part_messages = {
            "user_texts": chunk,
            "summary": f"{_clip(title, 60)} (パート{idx}/{total})",
            # 結末はセッション最後のパートにだけ書く
            "last_assistant": (
                messages.get("last_assistant", "") if idx == total else ""
            ),
        }
        ep = build_episode(
            part_messages,
            date_str=date_str,
            project=project,
            min_chars=0,
            # パート内は全発言を残す(粒度向上の本体)
            max_user_items=n_seg,
        )
        if ep:
            episodes.append(ep)
    return episodes

