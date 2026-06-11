"""transcript 要約(①自動符号化)のテスト。"""

from __future__ import annotations

import json

from engram.transcript import build_episode, extract_messages


def _write_jsonl(path, objs):
    with path.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _user(text):
    return {"type": "user", "cwd": "C:/proj/demo",
            "message": {"role": "user", "content": text}}


def _assistant(text):
    return {"type": "assistant",
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": text}]}}


def test_extract_messages_basic(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [
        {"type": "summary", "summary": "計画書の作成"},
        _user("計画書のたたき台を作って"),
        _assistant("承知しました。構成案はこうです。"),
        _user("2章をもっと短くして"),
        _assistant("短くしました。完成です。"),
    ])
    m = extract_messages(p)
    assert m["user_texts"] == ["計画書のたたき台を作って", "2章をもっと短くして"]
    assert m["last_assistant"] == "短くしました。完成です。"
    assert m["summary"] == "計画書の作成"
    assert m["cwd"] == "C:/proj/demo"


def test_extract_messages_skips_noise(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [
        _user("<command-name>/clear</command-name>"),
        _user("Caveat: The messages below were generated..."),
        {"type": "user", "isMeta": True,
         "message": {"role": "user", "content": "メタ行"}},
        # tool_result だけのユーザー行(テキストブロックなし)
        {"type": "user",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "x", "content": "result"}
         ]}},
        _user("本物の発言です"),
        "壊れた行",  # json だが dict でない
    ])
    # 壊れた JSON 行も混ぜる
    with p.open("a", encoding="utf-8") as f:
        f.write("{not json}\n")
    m = extract_messages(p)
    assert m["user_texts"] == ["本物の発言です"]


def test_build_episode_normal(tmp_path):
    messages = {
        "user_texts": ["計画書のたたき台を作ってほしい",
                       "2章をもっと短くして",
                       "これで完成にしよう"],
        "last_assistant": "計画書を完成させました。様式は標準フォーマットに従っています。",
        "summary": "計画書の作成",
        "cwd": "C:/proj/demo",
    }
    ep = build_episode(messages, date_str="2026-06-12", project="demo")
    assert ep is not None
    assert "2026-06-12" in ep
    assert "demo" in ep
    assert "計画書の作成" in ep            # summary が表題になる
    assert "計画書のたたき台" in ep
    assert "結末" in ep


def test_build_episode_skips_trivial():
    # 発言なし
    assert build_episode({"user_texts": [], "last_assistant": "x"},
                         date_str="2026-06-12") is None
    # 発言が短すぎる
    assert build_episode({"user_texts": ["ok"], "last_assistant": ""},
                         date_str="2026-06-12") is None


def test_build_episode_caps_user_items():
    texts = [f"発言その{i}、内容はそれなりに長いものとする" for i in range(20)]
    ep = build_episode({"user_texts": texts, "last_assistant": "done",
                        "summary": ""},
                       date_str="2026-06-12", max_user_items=6)
    assert ep is not None
    assert "(ほか" in ep
    # 最初と最後は必ず拾う
    assert "発言その0" in ep
    assert "発言その19" in ep
