"""メール添付の門(純粋関数)のテスト。Gmail API は呼ばない。"""
from __future__ import annotations

from pathlib import Path

from engram.config import Settings
from engram.media_mail import Account, is_human_sender, safe_name, wanted_attachment


def test_sender_gate():
    assert is_human_sender("山田 太郎 <taro@example.co.jp>")
    assert not is_human_sender("Google <no-reply@accounts.google.com>")
    assert not is_human_sender("Newsletter <news@example.com>")
    assert not is_human_sender("notification@github.com")


def _part(filename, size=100_000, disposition="attachment", content_id=None):
    headers = [{"name": "Content-Disposition", "value": f'{disposition}; filename="{filename}"'}]
    if content_id:
        headers.append({"name": "Content-ID", "value": content_id})
    return {"filename": filename, "headers": headers, "body": {"size": size, "attachmentId": "x"}}


def test_attachment_gate():
    assert wanted_attachment(_part("見積書.pdf"))[0]
    assert wanted_attachment(_part("photo.jpg", size=500_000))[0]
    assert not wanted_attachment(_part("logo.png", size=4_000))[0]              # 署名画像の大きさ
    assert not wanted_attachment(_part("image001.png", content_id="<a@b>"))[0]  # 本文埋め込み
    assert not wanted_attachment(_part("banner.jpg", disposition="inline"))[0]
    assert not wanted_attachment(_part("invite.ics"))[0]
    assert not wanted_attachment(_part("smime.p7s"))[0]
    assert not wanted_attachment({"filename": "", "headers": [], "body": {}})[0]


def test_safe_name_and_account_defaults(tmp_path):
    assert safe_name('a/b:c*d?"e<f>g|h.pdf') == "a_b_c_d_e_f_g_h.pdf"
    assert safe_name("   ") == "attachment"
    s = Settings(memories_dir=tmp_path / "kb" / "memories", data_dir=tmp_path / "data",
                 media_mail={"personal": {"room": "common"},
                             "work": {"room": "仕事", "save_dir": str(tmp_path / "H" / "media" / "work")}})
    acc = Account.from_settings(s, "personal")
    assert acc.save_dir == tmp_path / "data" / "media" / "inbox" / "personal"   # 未指定ならローカル
    assert acc.token == tmp_path / "data" / "media" / "mail" / "token_personal.json"
    assert "has:attachment" in acc.query and acc.room == "common"
    work = Account.from_settings(s, "work")
    assert work.save_dir == tmp_path / "H" / "media" / "work" and work.room == "仕事"  # アカウントのドライブに合わせる
