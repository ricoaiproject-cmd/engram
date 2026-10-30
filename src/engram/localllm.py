"""ローカル LLM(Ollama 互換 API)の薄い窓口。

engram 本体は LLM を持たない。眠りの統合・蒸留・DMN(自発思考)のような
「文章を生成する係」だけを、利用者のPCで動くローカルモデルに任せる。
モデル・URL は Settings(config.toml)で差し替える。到達できなければ
呼び出し側が静かに縮退する(壊れない)のが約束。
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from .config import Settings


class LocalLLM:
    def __init__(self, url: str, model: str, *, timeout: int = 600) -> None:
        self.url = url
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> "LocalLLM":
        return cls(settings.llm_url, settings.llm_model,
                   timeout=settings.llm_timeout_sec)

    def available(self, timeout: int = 10) -> bool:
        try:
            with urllib.request.urlopen(
                self.url.replace("/api/chat", "/api/tags"), timeout=timeout
            ) as resp:
                resp.read()
            return True
        except Exception:
            return False

    def chat(self, system: str, user: str, *, temperature: float = 0.2,
             num_predict: int = 1200, images: list[str] | None = None) -> str | None:
        """Ollama /api/chat。images は base64 文字列の列(vision 対応モデルのみ有効。
        媒体の説明文づくりで使う。非対応モデルは無視するか失敗→None で縮退)。"""
        user_msg: dict = {"role": "user", "content": user}
        if images:
            user_msg["images"] = list(images)
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, user_msg],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return (data.get("message") or {}).get("content", "").strip()
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return None


def strip_fences(text: str) -> str:
    """```json ... ``` の囲いを外す。"""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)


def extract_json(text: str, kind: str = "object"):
    """本文中の最初の JSON オブジェクト/配列を取り出す。壊れていれば None。"""
    text = strip_fences(text)
    pat = r"\{.*\}" if kind == "object" else r"\[.*\]"
    m = re.search(pat, text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
