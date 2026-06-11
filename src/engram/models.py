"""共有データ型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

MEMORY_TYPES = ("knowledge", "preference", "project", "episode")
TIERS = ("hot", "cold", "superseded", "trash")
EVENT_KINDS = ("create", "recall_hit", "reinforce", "correction")
LINK_KINDS = ("explicit", "co_recall", "derived_from", "superseded_by")


@dataclass
class MemoryRecord:
    """記憶1件。正本は Markdown ファイル、DB はインデックス。"""

    id: str                      # ULID
    type: str                    # MEMORY_TYPES のいずれか
    created: str                 # ISO 8601(ローカルTZ付き)
    importance: int              # 1-10。呼び出し元エージェントが文脈から採点
    tags: list[str] = field(default_factory=list)
    source: str = "unknown"      # claude-code / codex / antigravity 等
    tier: str = "hot"
    links: list[str] = field(default_factory=list)  # リンク先 id のリスト
    content: str = ""            # 本文(frontmatter を除く)
    path: Path | None = None     # Markdown ファイルの絶対パス
    content_hash: str = ""       # 本文の sha256(手編集検知用)
    room: str = "common"         # 記憶の部屋(仕事/個人の文脈分離。既定は共通)


@dataclass
class RecallHit:
    """recall の結果1件。スコア内訳付きで返し、エージェントが判断できるようにする。"""

    id: str
    content: str
    type: str
    tags: list[str]
    tier: str
    score: float                 # 最終スコア
    relevance: float             # クエリとの意味的関連度(0-1)
    activation: float            # 活性度(0-1 正規化)
    importance: float            # importance/10
    via: str = "direct"          # "direct" | "associative"(連想リンク経由)
    note: str = ""               # 例: "→ [id] により訂正済み"
    room: str = "common"         # 記憶の部屋
