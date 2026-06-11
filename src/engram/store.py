"""Markdown 正本ストア(担当: Agent A)。

1記憶 = 1ファイル(Zettelkasten 式に原子的)。Obsidian でそのまま閲覧・編集できる。

配置:
    memories/knowledge/   memories/preferences/   memories/projects/
    memories/episodes/YYYY/MM/    memories/_trash/
type → サブディレクトリ対応: knowledge→knowledge, preference→preferences,
project→projects, episode→episodes/YYYY/MM(created 由来)。

ファイル形式(python-frontmatter で読み書き):
    ---
    id: 01JXXXX...            # ULID
    type: knowledge
    created: 2026-06-11T09:00:00+09:00
    tags: [sqlite, windows]
    importance: 7
    source: claude-code
    tier: hot
    links: ["[[01JYYY...]]"]   # Obsidian wiki-link 形式で保存
    ---
    本文(プレーン Markdown)

実装要件:
- ファイル名: YYYYMMDD-{slug}-{idの末尾6文字}.md
  slug = 本文先頭行から生成(最大40文字、Windows 禁止文字 <>:"/\\|?* と改行を除去、
  空白は '-' に。日本語はそのまま可)。
- links は frontmatter 上は "[[id]]" 文字列、MemoryRecord.links 上は素の id リスト。
  読み書きで相互変換する。
- content_hash = 本文(frontmatter 除く、strip 後)の sha256 hex。
- tier 変更や本文更新はファイルを書き換える。episode 以外のディレクトリ移動は
  伴わない(tier は frontmatter のみ)。forget は _trash/ へファイル移動。
- scan_all() は _trash を除く全 .md を MemoryRecord で yield(reindex 用)。
- frontmatter が壊れたファイルは警告として収集しスキップ(例外で落とさない)。
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import warnings
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import frontmatter

from .models import MemoryRecord

logger = logging.getLogger(__name__)

# Windows forbidden filename characters + control chars
_WIN_FORBIDDEN = re.compile(r'[<>:"/\\|?*\r\n\x00-\x1f]')

# Mapping type → subdirectory name (non-episode)
_TYPE_TO_DIR: dict[str, str] = {
    "knowledge": "knowledge",
    "preference": "preferences",
    "project": "projects",
}


def content_hash(content: str) -> str:
    """本文の正規化ハッシュ(strip 後 sha256 hex)。"""
    normalized = content.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _make_slug(content: str) -> str:
    """本文先頭行から slug を生成。"""
    first_line = content.strip().splitlines()[0] if content.strip() else ""
    # Remove Windows forbidden chars and control chars
    slug = _WIN_FORBIDDEN.sub("", first_line)
    # Replace spaces with hyphens
    slug = slug.replace(" ", "-")
    # Truncate to 40 chars
    slug = slug[:40]
    # If empty, use a fallback
    if not slug:
        slug = "memory"
    return slug


def _parse_created_date(created: str) -> datetime:
    """ISO 8601 文字列から datetime を返す。"""
    # Handle timezone offset like +09:00
    try:
        # Python 3.7+ fromisoformat handles most cases but not all timezone formats
        return datetime.fromisoformat(created)
    except (ValueError, TypeError):
        return datetime.now(tz=timezone.utc)


def _links_to_frontmatter(links: list[str]) -> list[str]:
    """素の id リスト → "[[id]]" 形式のリスト。"""
    return [f"[[{lid}]]" for lid in links]


def _links_from_frontmatter(raw: list | None) -> list[str]:
    """frontmatter の links → 素の id リスト("[[id]]" を剥がす)。"""
    if not raw:
        return []
    result = []
    for item in raw:
        s = str(item).strip()
        if s.startswith("[[") and s.endswith("]]"):
            result.append(s[2:-2])
        else:
            result.append(s)
    return result


def _subdir_for_type(root: Path, type: str, created: str) -> Path:
    """type と created から保存先ディレクトリを決定。"""
    if type == "episode":
        dt = _parse_created_date(created)
        return root / "episodes" / dt.strftime("%Y") / dt.strftime("%m")
    return root / _TYPE_TO_DIR.get(type, type)


def _filename_from_record(id: str, type: str, created: str, content: str) -> str:
    """YYYYMMDD-{slug}-{id末尾6文字}.md"""
    dt = _parse_created_date(created)
    date_str = dt.strftime("%Y%m%d")
    slug = _make_slug(content)
    short_id = id[-6:]
    return f"{date_str}-{slug}-{short_id}.md"


class MarkdownStore:
    def __init__(self, root: Path) -> None:
        """root = memories ディレクトリ。無ければサブディレクトリごと作成。"""
        self._root = Path(root)
        # Create all standard subdirectories
        for subdir in ("knowledge", "preferences", "projects", "_trash"):
            (self._root / subdir).mkdir(parents=True, exist_ok=True)
        # episodes dir
        (self._root / "episodes").mkdir(parents=True, exist_ok=True)

    def _write_record(self, record: MemoryRecord) -> None:
        """MemoryRecord をファイルに書き出す。"""
        post = frontmatter.Post(
            content=record.content,
            id=record.id,
            type=record.type,
            created=record.created,
            tags=record.tags,
            importance=record.importance,
            source=record.source,
            tier=record.tier,
            links=_links_to_frontmatter(record.links),
        )
        text = frontmatter.dumps(post)
        assert record.path is not None
        record.path.parent.mkdir(parents=True, exist_ok=True)
        record.path.write_text(text, encoding="utf-8")

    def create(
        self,
        *,
        content: str,
        type: str,
        importance: int,
        tags: list[str] | None = None,
        source: str = "unknown",
        links: list[str] | None = None,
        id: str | None = None,
        created: str | None = None,
    ) -> MemoryRecord:
        """新規記憶ファイルを書き、MemoryRecord を返す。"""
        from ulid import ULID

        if id is None:
            id = str(ULID())
        if created is None:
            created = datetime.now().astimezone().isoformat()

        subdir = _subdir_for_type(self._root, type, created)
        subdir.mkdir(parents=True, exist_ok=True)
        filename = _filename_from_record(id, type, created, content)
        path = subdir / filename

        ch = content_hash(content)
        record = MemoryRecord(
            id=id,
            type=type,
            created=created,
            importance=importance,
            tags=tags or [],
            source=source,
            tier="hot",
            links=links or [],
            content=content,
            path=path,
            content_hash=ch,
        )
        self._write_record(record)
        return record

    def read(self, path: Path) -> MemoryRecord:
        path = Path(path)
        post = frontmatter.load(str(path))
        raw_links = post.get("links", [])
        return MemoryRecord(
            id=str(post["id"]),
            type=str(post["type"]),
            created=str(post["created"]),
            importance=int(post["importance"]),
            tags=list(post.get("tags") or []),
            source=str(post.get("source", "unknown")),
            tier=str(post.get("tier", "hot")),
            links=_links_from_frontmatter(raw_links),
            content=post.content,
            path=path,
            content_hash=content_hash(post.content),
        )

    def find_by_id(self, id: str) -> MemoryRecord | None:
        """全走査せず、まず DB 側の path を使うのが本筋。store 単体では全走査でよい。"""
        for record in self.scan_all():
            if record.id == id:
                return record
        return None

    def update(self, record: MemoryRecord) -> MemoryRecord:
        """record.path のファイルを record の内容で書き直す(content_hash 再計算)。"""
        updated = MemoryRecord(
            id=record.id,
            type=record.type,
            created=record.created,
            importance=record.importance,
            tags=record.tags,
            source=record.source,
            tier=record.tier,
            links=record.links,
            content=record.content,
            path=record.path,
            content_hash=content_hash(record.content),
        )
        self._write_record(updated)
        return updated

    def add_link(self, record: MemoryRecord, target_id: str) -> MemoryRecord:
        """links に target_id を追加(重複は無視)してファイル更新。"""
        if target_id in record.links:
            return record
        updated = MemoryRecord(
            id=record.id,
            type=record.type,
            created=record.created,
            importance=record.importance,
            tags=record.tags,
            source=record.source,
            tier=record.tier,
            links=record.links + [target_id],
            content=record.content,
            path=record.path,
            content_hash=record.content_hash,
        )
        self._write_record(updated)
        return updated

    def set_tier(self, record: MemoryRecord, tier: str) -> MemoryRecord:
        updated = MemoryRecord(
            id=record.id,
            type=record.type,
            created=record.created,
            importance=record.importance,
            tags=record.tags,
            source=record.source,
            tier=tier,
            links=record.links,
            content=record.content,
            path=record.path,
            content_hash=record.content_hash,
        )
        self._write_record(updated)
        return updated

    def move_to_trash(self, record: MemoryRecord) -> MemoryRecord:
        """_trash/ へ移動し tier=trash に。物理削除はしない。"""
        trash_dir = self._root / "_trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        assert record.path is not None
        new_path = trash_dir / record.path.name
        # Handle name collision in trash
        if new_path.exists() and new_path != record.path:
            stem = record.path.stem
            suffix = record.path.suffix
            new_path = trash_dir / f"{stem}-{record.id[-4:]}{suffix}"
        shutil.move(str(record.path), str(new_path))
        updated = MemoryRecord(
            id=record.id,
            type=record.type,
            created=record.created,
            importance=record.importance,
            tags=record.tags,
            source=record.source,
            tier="trash",
            links=record.links,
            content=record.content,
            path=new_path,
            content_hash=record.content_hash,
        )
        self._write_record(updated)
        return updated

    def scan_all(self) -> Iterator[MemoryRecord]:
        """_trash を除く全 .md を MemoryRecord で yield(reindex 用)。
        frontmatter が壊れたファイルは警告としてスキップ。
        """
        trash_dir = self._root / "_trash"
        for md_file in self._root.rglob("*.md"):
            # Skip files inside _trash
            try:
                md_file.relative_to(trash_dir)
                continue  # it's inside _trash
            except ValueError:
                pass  # not in _trash, proceed

            try:
                record = self.read(md_file)
                yield record
            except Exception as exc:
                warnings.warn(
                    f"Skipping broken frontmatter in {md_file}: {exc}",
                    stacklevel=2,
                )
                continue
