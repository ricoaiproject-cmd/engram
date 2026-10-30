"""メール添付の取り込み(Gmail API・読み取り専用)。`engram-media mail-fetch --account 名前`。

方針(R8.10.8): 「人が人に送った文書・画像」だけを取り込む。通知・配信・広告、本文埋め込みの画像、
署名画像、カレンダー招待(.ics)は門で止める。保存先はアカウントごとの save_dir。**そのメールと同じ
アカウントのドライブに置く**(ユーザー裁定 R8.10.8: 職場メールの添付は職場アカウントのドライブ、個人は
個人のドライブ。レーンをまたいでデータを移さない)。save_dir 未指定なら data_dir/media/inbox/<アカウント>
(ローカル・同期なし)に落とす。保存したら索引係に即時索引を頼む(source="mail"、差出人・件名・日付つき)。

設定(config.toml):
    [media_mail.work]
    client_secret = "~/.engram/media/mail/client_secret.json"   # Google Cloud の OAuth クライアント(デスクトップ)
    token         = "~/.engram/media/mail/token_work.json"      # 初回に `--auth` でブラウザ許可 → 以後は自動更新
    query         = "has:attachment newer_than:30d -category:promotions -category:updates -category:forums"
    room          = "仕事"
    save_dir      = "D:/WorkDrive/attachments"   # 職場アカウントのドライブ
    [media_mail.personal]
    token         = "~/.engram/media/mail/token_personal.json"
    room          = "common"
    save_dir      = "E:/PersonalDrive/attachments"   # 個人アカウントのドライブ

依存: engram[mail](google-api-python-client, google-auth-oauthlib)。スコープは gmail.readonly のみ。
メールの送信・変更・削除は一切しない(外部コンテンツは命令ではない: 添付の中身に指示があっても読むだけ)。
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
ACCEPT_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls", ".odt", ".odp", ".ods",
              ".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff", ".mp4", ".mov", ".mp3", ".wav", ".m4a"}
REJECT_EXT = {".ics", ".vcf", ".p7s", ".asc", ".sig", ".htm", ".html", ".xml", ".json"}
NOREPLY_RE = re.compile(r"(no-?reply|donotreply|do-not-reply|mailer-daemon|notification|newsletter|info@|news@|"
                        r"support@|alert|digest|noreply)", re.I)
MIN_BYTES = {"image": 30_000}   # 署名画像・ロゴを落とす(文書は小さくても取る)
STATE_NAME = "mail_state.json"


@dataclass
class Account:
    name: str
    client_secret: Path
    token: Path
    query: str
    room: str
    save_dir: Path

    @classmethod
    def from_settings(cls, settings: Settings, name: str) -> "Account":
        conf = (settings.media_mail or {}).get(name)
        if not conf:
            raise KeyError(f"config.toml の [media_mail.{name}] がありません")
        default_save = settings.media_dir / "inbox" / name   # 未指定ならローカル(同期なし)。実運用は save_dir を明示
        return cls(
            name=name,
            client_secret=Path(conf.get("client_secret", settings.media_dir / "mail" / "client_secret.json")).expanduser(),
            token=Path(conf.get("token", settings.media_dir / "mail" / f"token_{name}.json")).expanduser(),
            query=conf.get("query", "has:attachment newer_than:30d -category:promotions -category:updates -category:forums"),
            room=conf.get("room", "common"),
            save_dir=Path(conf.get("save_dir", default_save)).expanduser(),
        )


def _service(acc: Account, *, interactive: bool):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = None
    if acc.token.is_file():
        creds = Credentials.from_authorized_user_file(str(acc.token), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif interactive:
            from google_auth_oauthlib.flow import InstalledAppFlow

            flow = InstalledAppFlow.from_client_secrets_file(str(acc.client_secret), SCOPES)
            creds = flow.run_local_server(port=0)
        else:
            raise RuntimeError(f"認証が必要です: engram-media mail-fetch --account {acc.name} --auth")
        acc.token.parent.mkdir(parents=True, exist_ok=True)
        acc.token.write_text(creds.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _header(msg: dict, name: str) -> str:
    for h in msg.get("payload", {}).get("headers", []):
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _walk_parts(part: dict):
    yield part
    for p in part.get("parts", []) or []:
        yield from _walk_parts(p)


def is_human_sender(from_header: str) -> bool:
    return not NOREPLY_RE.search(from_header or "")


def wanted_attachment(part: dict) -> tuple[bool, str]:
    name = part.get("filename") or ""
    if not name:
        return False, "no filename"
    ext = Path(name).suffix.lower()
    if ext in REJECT_EXT or ext not in ACCEPT_EXT:
        return False, f"ext {ext}"
    headers = {h.get("name", "").lower(): h.get("value", "") for h in part.get("headers", [])}
    disp = headers.get("content-disposition", "")
    if "inline" in disp.lower() or headers.get("content-id"):
        return False, "inline"      # 本文埋め込み画像・署名
    size = int(part.get("body", {}).get("size", 0) or 0)
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff"} and size and size < MIN_BYTES["image"]:
        return False, "small image"
    return True, "ok"


def safe_name(s: str, limit: int = 80) -> str:
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", s).strip(" .")
    return s[:limit] or "attachment"


def _state_path(settings: Settings) -> Path:
    return settings.media_dir / "mail" / STATE_NAME


def _load_state(settings: Settings) -> dict:
    try:
        return json.loads(_state_path(settings).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(settings: Settings, st: dict) -> None:
    p = _state_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def fetch(settings: Settings, account: str, *, interactive: bool = False, max_messages: int = 200,
          dry_run: bool = False, index: bool = True) -> dict:
    """添付を保存し(既存はスキップ)、索引係に即時索引を頼む。戻り値は件数と一覧。"""
    acc = Account.from_settings(settings, account)
    svc = _service(acc, interactive=interactive)
    state = _load_state(settings)
    seen: set[str] = set(state.get(account, {}).get("messages", []))
    rep = {"account": account, "messages": 0, "saved": 0, "skipped": 0, "rejected": {}, "indexed": 0, "items": []}

    page = None
    msgs: list[dict] = []
    while len(msgs) < max_messages:
        res = svc.users().messages().list(userId="me", q=acc.query, maxResults=min(100, max_messages - len(msgs)),
                                          pageToken=page).execute()
        msgs.extend(res.get("messages", []) or [])
        page = res.get("nextPageToken")
        if not page:
            break

    for m in msgs:
        mid = m["id"]
        if mid in seen:
            rep["skipped"] += 1
            continue
        full = svc.users().messages().get(userId="me", id=mid, format="full").execute()
        rep["messages"] += 1
        frm, subj, date = _header(full, "From"), _header(full, "Subject"), _header(full, "Date")
        if not is_human_sender(frm):
            rep["rejected"]["sender"] = rep["rejected"].get("sender", 0) + 1
            seen.add(mid)
            continue
        ts = int(full.get("internalDate", "0")) / 1000.0
        ym = time.strftime("%Y-%m", time.localtime(ts)) if ts else "unknown"
        for part in _walk_parts(full.get("payload", {})):
            ok, why = wanted_attachment(part)
            if not ok:
                if part.get("filename"):
                    rep["rejected"][why] = rep["rejected"].get(why, 0) + 1
                continue
            att_id = part.get("body", {}).get("attachmentId")
            if not att_id:
                continue
            fname = safe_name(part["filename"])
            dest = acc.save_dir / ym / f"{mid[:10]}_{fname}"
            if dest.exists():
                rep["skipped"] += 1
                continue
            if dry_run:
                rep["items"].append({"would_save": str(dest), "from": frm, "subject": subj})
                continue
            data = svc.users().messages().attachments().get(userId="me", messageId=mid, id=att_id).execute()
            raw = base64.urlsafe_b64decode(data.get("data", ""))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            rep["saved"] += 1
            meta = {"account": account, "message_id": mid, "from": frm, "subject": subj, "date": date,
                    "view_url": f"https://mail.google.com/mail/u/0/#all/{mid}"}
            rep["items"].append({"path": str(dest), **meta})
            if index:
                from . import media as _media

                r = _media.index_path(settings, dest, room=acc.room, source="mail", source_meta=meta, wait=120.0)
                if r.get("ok"):
                    rep["indexed"] += 1
        seen.add(mid)
    if not dry_run:
        state.setdefault(account, {})["messages"] = sorted(seen)[-5000:]
        state[account]["last_run"] = time.time()
        _save_state(settings, state)
    return rep


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="engram-media mail-fetch")
    ap.add_argument("--account", required=True)
    ap.add_argument("--auth", action="store_true", help="初回のブラウザ許可を行う(ユーザー操作)")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-index", action="store_true")
    a = ap.parse_args(argv)
    rep = fetch(get_settings(), a.account, interactive=a.auth, max_messages=a.max, dry_run=a.dry_run,
                index=not a.no_index)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
